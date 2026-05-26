from __future__ import annotations

from pathlib import Path

import torch

from body_tell.data.vocabulary import (
    EMBEDDING_DIM,
    INSTRUCTION,
    TEXT_ENCODER,
    file_sha256,
    flatten_prompt_records,
    load_label_vocab,
    validate_prompt_embedding_cache,
)


ROOT = Path(__file__).resolve().parents[1]


def test_static_prompt_embedding_cache_matches_label_vocab() -> None:
    vocab_path = ROOT / "configs" / "label_vocab.json"
    cache_path = ROOT / "artifacts" / "text_embeddings" / "prompt_embeddings.pt"
    vocab = load_label_vocab(vocab_path)
    prompt_records = flatten_prompt_records(vocab)

    validation = validate_prompt_embedding_cache(cache_path, vocab_path)

    assert validation["errors"] == []
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    embeddings = cache["embeddings"]
    assert cache["model_name"] == TEXT_ENCODER
    assert cache["instruction"] == INSTRUCTION
    assert cache["vocab_hash"] == file_sha256(vocab_path)
    assert cache["num_prompts"] == len(prompt_records)
    assert cache["prompt_records"] == prompt_records
    assert tuple(embeddings.shape) == (len(prompt_records), EMBEDDING_DIM)
    assert cache["is_qwen_cache"] is True
