#!/usr/bin/env python3
"""Build prompt_embeddings.pt from label_vocab.json using Qwen embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from body_tell.data.vocabulary import (  # noqa: E402
    EMBEDDING_DIM,
    INSTRUCTION,
    TEXT_ENCODER,
    file_sha256,
    flatten_prompt_records,
    load_label_vocab,
)


def wrap_with_instruction(text_prompts: List[str]) -> List[str]:
    return [f"Instruct: {INSTRUCTION}\nQuery: {text}" for text in text_prompts]


def last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    import torch

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(dtype_arg: str, device: str) -> Any:
    import torch

    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


def build_embeddings(args: argparse.Namespace, prompt_texts: List[str]) -> Any:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="left",
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        args.model_name,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).eval()
    model = model.to(device)

    embeddings = []
    wrapped = wrap_with_instruction(prompt_texts)
    with torch.inference_mode():
        for start in range(0, len(wrapped), args.batch_size):
            batch = wrapped[start : start + args.batch_size]
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            output = model(**tokens)
            pooled = last_token_pool(output.last_hidden_state, tokens["attention_mask"])
            embeddings.append(pooled.detach().to("cpu", dtype=torch.float32))
            print(f"embedded {min(start + args.batch_size, len(wrapped))}/{len(wrapped)}", flush=True)

    result = torch.cat(embeddings, dim=0)
    if result.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Expected embedding dim {EMBEDDING_DIM}, got {result.shape[1]}")
    if not torch.isfinite(result).all().item():
        raise ValueError("Embedding tensor contains NaN or Inf")
    return result


def build_deterministic_fallback(prompt_records: List[Dict[str, Any]]) -> Any:
    """Build a deterministic non-Qwen cache for offline smoke tests only."""

    import hashlib
    import torch

    rows = []
    for record in prompt_records:
        digest = hashlib.sha256(record["text"].encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        vector = torch.randn(EMBEDDING_DIM, generator=generator, dtype=torch.float32)
        vector = vector / vector.norm().clamp_min(1e-12)
        rows.append(vector)
    return torch.stack(rows, dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=Path, default=ROOT / "configs" / "label_vocab.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "text_embeddings" / "prompt_embeddings.pt",
    )
    parser.add_argument("--model-name", default=TEXT_ENCODER)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--allow-deterministic-fallback",
        action="store_true",
        help="Write deterministic non-Qwen vectors if model loading fails. For smoke tests only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vocab = load_label_vocab(args.vocab)
    prompt_records = flatten_prompt_records(vocab)
    prompt_texts = [record["text"] for record in prompt_records]
    is_qwen_cache = True
    fallback_error = None

    try:
        embeddings = build_embeddings(args, prompt_texts)
    except Exception as exc:
        if not args.allow_deterministic_fallback:
            raise
        fallback_error = repr(exc)
        print(f"Qwen embedding failed; writing deterministic fallback: {fallback_error}")
        embeddings = build_deterministic_fallback(prompt_records)
        is_qwen_cache = False

    import torch

    cache = {
        "model_name": args.model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "instruction": INSTRUCTION,
        "vocab_version": vocab["version"],
        "vocab_hash": file_sha256(args.vocab),
        "num_prompts": len(prompt_records),
        "prompt_records": prompt_records,
        "is_qwen_cache": is_qwen_cache,
        "fallback_error": fallback_error,
        "embeddings": embeddings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.output)
    print(f"wrote {args.output}")
    print(f"num_prompts={len(prompt_records)} embedding_shape={tuple(embeddings.shape)}")


if __name__ == "__main__":
    main()

