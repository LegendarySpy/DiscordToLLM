
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import unsloth  # must be imported before peft/transformers for patches

import torch
from peft import PeftModel

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("export lora -> gguf (unsloth)")
    p.add_argument("--base_model", type=str, default="arcee-ai/Trinity-Nano-Preview")
    p.add_argument("--lora_dir", type=str, default="models/trinity/output/lora")
    p.add_argument("--out_dir", type=str, default="models/trinity/output/gguf")
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--chat_template", type=str, default="chatml")
    p.add_argument("--quant", type=str, default="q4_k_m", help="f16, q8_0, q4_k_m, q5_k_m, ...")
    p.add_argument("--maximum_memory_usage", type=float, default=0.75, help="lower if you oom while saving")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    lora_dir = Path(args.lora_dir)
    if not lora_dir.exists():
        raise FileNotFoundError(f"lora_dir not found: {lora_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # export path: load full precision (not 4bit), then quantize to gguf
    dtype = torch.float16

    # If lora_dir is a PEFT adapter, let Unsloth load it directly.
    # This ensures the returned model is a PeftModel for proper merging.
    use_peft_dir = (lora_dir / "adapter_config.json").exists()
    model_name = str(lora_dir) if use_peft_dir else args.base_model

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_len,
        dtype=dtype,
        load_in_4bit=False,
        trust_remote_code=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)

    if not use_peft_dir:
        model = PeftModel.from_pretrained(model, str(lora_dir), is_trainable=False)

    model.save_pretrained_gguf(
        str(out_dir),
        tokenizer,
        quantization_method=args.quant,
        maximum_memory_usage=args.maximum_memory_usage,
    )

    print(f"✅ wrote gguf to {out_dir}")
    print("note: make sure your gguf runner is using GPU offload (n_gpu_layers / -ngl), otherwise it will be CPU-slow.")


if __name__ == "__main__":
    main()
