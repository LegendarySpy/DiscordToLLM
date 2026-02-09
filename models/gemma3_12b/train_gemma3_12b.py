from __future__ import annotations

import os

# must be set ASAP (before torch/unsloth import)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# IMPORTANT: unsloth wants to be imported before transformers
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import EarlyStoppingCallback, set_seed
from transformers.utils import logging as hf_logging
from trl import SFTTrainer, SFTConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("gemma 3 12b sft (unsloth + trl)")

    p.add_argument("--dataset", type=str, required=True, help="path to dataset.jsonl (ShareGPT-style)")
    p.add_argument("--output_dir", type=str, default="models/gemma3_12b/output/lora")

    p.add_argument("--model", type=str, default="google/gemma-3-12b-it")
    p.add_argument("--max_seq_len", type=int, default=2048)

    # A40 defaults
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)

    p.add_argument("--epochs", type=float, default=2.5)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.0)

    p.add_argument("--four_bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--packing", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--eval_split", type=float, default=0.05)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--early_stopping_patience", type=int, default=3)

    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=False)

    # dataset hygiene: drop assistant turns before first user
    p.add_argument("--drop_leading_assistant", action=argparse.BooleanOptionalAction, default=True)

    # chat template control
    p.add_argument("--chat_template", type=str, default="auto", help="auto, gemma-3, chatml, ...")
    p.add_argument("--response_only_loss", action=argparse.BooleanOptionalAction, default=True)

    return p.parse_args()


def _clean_messages(messages: Any, drop_leading_assistant: bool) -> Optional[List[Dict[str, str]]]:
    if not isinstance(messages, list) or not messages:
        return None

    out: List[Dict[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        content = m.get("content")
        if content is None:
            continue
        content = str(content).strip()
        if not content:
            continue
        out.append({"role": role, "content": content})

    # require last = assistant (so response-only loss always has a target)
    if not out or out[-1]["role"] != "assistant":
        return None

    if drop_leading_assistant:
        sys_msgs: List[Dict[str, str]] = []
        i = 0
        while i < len(out) and out[i]["role"] == "system":
            sys_msgs.append(out[i])
            i += 1
        j = i
        while j < len(out) and out[j]["role"] == "assistant":
            j += 1
        out = sys_msgs + out[j:]
        if not out or out[-1]["role"] != "assistant":
            return None

    return out


def load_sharegpt_jsonl(path: str) -> Dataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    ds = load_dataset("json", data_files=str(p), split="train")
    if "messages" not in ds.column_names:
        if "conversations" in ds.column_names:
            ds = ds.rename_column("conversations", "messages")
        else:
            raise ValueError("dataset must contain a 'messages' or 'conversations' column")
    return ds


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    hf_logging.set_verbosity_warning()

    # ampere speed
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    report_to = ["wandb"] if args.wandb_project else []
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    print(f"gpu: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"four_bit: {args.four_bit} | max_seq_len: {args.max_seq_len}")
    print(f"batch: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum} effective")
    print(f"trust_remote_code: {args.trust_remote_code}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,  # auto
        load_in_4bit=args.four_bit,
        trust_remote_code=args.trust_remote_code,
    )

    # Prefer the model's built-in template; fallback to llama-3 if missing.
    if args.chat_template != "auto":
        tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)
    elif getattr(tokenizer, "chat_template", None) is None:
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    ds = load_sharegpt_jsonl(args.dataset)

    def to_text(batch: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = [_clean_messages(m, args.drop_leading_assistant) for m in batch["messages"]]
        keep = [c is not None for c in cleaned]
        msgs_for_template = [c if c is not None else [{"role": "user", "content": "skip"}] for c in cleaned]
        texts = tokenizer.apply_chat_template(msgs_for_template, tokenize=False, add_generation_prompt=False)
        return {"text": texts, "keep": keep}

    # single-proc map avoids multiprocess pickling weirdness with tokenizers
    ds = ds.map(to_text, batched=True, desc="formatting with chat template")
    ds = ds.filter(lambda ex: ex["keep"], desc="drop invalid rows")
    ds = ds.remove_columns([c for c in ds.column_names if c != "text"])

    if args.eval_split and args.eval_split > 0:
        split = ds.train_test_split(test_size=args.eval_split, seed=args.seed, shuffle=True)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds, None

    cfg = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_length=args.max_seq_len,
        packing=args.packing,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,

        optim="adamw_8bit",
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),

        logging_steps=args.logging_steps,
        report_to=report_to,
        run_name=args.run_name,

        do_eval=eval_ds is not None,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=True,

        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    # response-only loss masking (gemma style turns)
    if args.response_only_loss:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )

    if eval_ds is not None and args.early_stopping_patience > 0:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    print("\n--- sample text ---")
    print(train_ds[0]["text"][:800])
    print("--- end sample ---\n")

    trainer.train()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    info = {
        "base_model": args.model,
        "max_seq_len": args.max_seq_len,
        "four_bit": args.four_bit,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "effective_batch": args.batch_size * args.grad_accum,
        "epochs": args.epochs,
        "lr": args.lr,
        "dataset": args.dataset,
        "train_rows": len(train_ds),
        "eval_rows": len(eval_ds) if eval_ds is not None else 0,
        "chat_template": args.chat_template,
        "response_only_loss": args.response_only_loss,
        "trust_remote_code": args.trust_remote_code,
    }
    with open(out / "training_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"\nsaved lora adapter -> {out}")


if __name__ == "__main__":
    main()
