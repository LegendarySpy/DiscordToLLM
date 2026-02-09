
from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import unsloth  # must be imported before peft/transformers for patches

import torch
from peft import PeftModel

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


@contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def resolve_path(raw_path: str, repo_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def relocate_generated_artifacts(result: object, repo_root: Path, out_dir: Path) -> None:
    if not isinstance(result, dict):
        return

    candidates = []
    gguf_files = result.get("gguf_files")
    if isinstance(gguf_files, list):
        candidates.extend(gguf_files)

    modelfile = result.get("modelfile")
    if isinstance(modelfile, str):
        candidates.append(modelfile)

    for item in candidates:
        source = Path(item)
        if not source.is_absolute():
            source = repo_root / source
        if not source.exists():
            continue

        destination = out_dir / source.name
        if source.resolve() == destination.resolve():
            continue
        if destination.exists():
            destination.unlink()
        source.replace(destination)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("export lora -> gguf (gemma 3 + unsloth)")
    p.add_argument("--base_model", type=str, default="google/gemma-3-12b-it")
    p.add_argument("--lora_dir", type=str, default="models/gemma3_12b/output/lora")
    p.add_argument("--out_dir", type=str, default="models/gemma3_12b/output/gguf")
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--chat_template", type=str, default="auto", help="auto, gemma-3, chatml, ...")
    p.add_argument("--quant", type=str, default="q4_k_m", help="f16, q8_0, q4_k_m, q5_k_m, ...")
    p.add_argument("--maximum_memory_usage", type=float, default=0.75, help="lower if you oom while saving")
    p.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    lora_dir = resolve_path(args.lora_dir, repo_root)
    if not lora_dir.exists():
        raise FileNotFoundError(f"lora_dir not found: {lora_dir}")

    out_dir = resolve_path(args.out_dir, repo_root)
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
        trust_remote_code=args.trust_remote_code,
    )
    if args.chat_template != "auto":
        tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)
    elif getattr(tokenizer, "chat_template", None) is None:
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    if not use_peft_dir:
        model = PeftModel.from_pretrained(model, str(lora_dir), is_trainable=False)

    with pushd(repo_root):
        result = model.save_pretrained_gguf(
            str(out_dir),
            tokenizer,
            quantization_method=args.quant,
            maximum_memory_usage=args.maximum_memory_usage,
        )
    relocate_generated_artifacts(result, repo_root, out_dir)

    print(f"✅ wrote gguf to {out_dir}")
    print("note: make sure your gguf runner is using GPU offload (n_gpu_layers / -ngl), otherwise it will be CPU-slow.")


if __name__ == "__main__":
    main()
