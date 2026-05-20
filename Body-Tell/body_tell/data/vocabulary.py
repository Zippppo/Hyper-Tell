"""Vocabulary utilities for Phase 0 prompt-conditioned segmentation.

This module intentionally stays independent from model training code. Phase 1
datasets can load the Phase 0 artifacts here, validate that they agree, and
sample prompt records for target-mask construction.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np


VOCAB_VERSION = "phase0-2026-05-20"
TEXT_ENCODER = "Qwen/Qwen3-Embedding-4B"
EMBEDDING_DIM = 2560
INSTRUCTION = (
    "Given an anatomical term query, retrieve the precise anatomical entity "
    "and location it represents"
)


class VocabularyValidationError(ValueError):
    """Raised when a vocabulary artifact fails strict validation."""


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write("\n")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_label_vocab(path: str | Path) -> Dict[str, Any]:
    vocab = read_json(path)
    if not isinstance(vocab, dict):
        raise VocabularyValidationError(f"Expected JSON object in {path}")
    return vocab


def class_by_id(vocab: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(item["id"]): dict(item) for item in vocab.get("classes", [])}


def train_class_ids(vocab: Mapping[str, Any]) -> List[int]:
    return [
        int(item["id"])
        for item in vocab.get("classes", [])
        if bool(item.get("train_as_positive", False))
    ]


def foreground_eval_class_ids(vocab: Mapping[str, Any]) -> List[int]:
    return [
        int(item["id"])
        for item in vocab.get("classes", [])
        if bool(item.get("eval_as_foreground", False))
    ]


def flatten_prompt_records(
    vocab: Mapping[str, Any],
    include_classes: bool = True,
    include_aggregates: bool = True,
) -> List[Dict[str, Any]]:
    """Return prompt records with stable integer indexes for embedding lookup."""

    records: List[Dict[str, Any]] = []

    if include_classes:
        for cls in vocab.get("classes", []):
            class_id = int(cls["id"])
            canonical = str(cls["canonical"])
            for prompt_offset, text in enumerate(cls.get("prompts", [])):
                records.append(
                    {
                        "index": len(records),
                        "prompt_id": f"class_{class_id:03d}_prompt_{prompt_offset:03d}",
                        "source_type": "class",
                        "class_id": class_id,
                        "aggregate_id": None,
                        "text": str(text),
                        "is_canonical": prompt_offset == 0 and str(text) == canonical,
                    }
                )

    if include_aggregates:
        for aggregate in vocab.get("aggregates", []):
            aggregate_id = str(aggregate["id"])
            canonical = str(aggregate["canonical"])
            for prompt_offset, text in enumerate(aggregate.get("prompts", [])):
                records.append(
                    {
                        "index": len(records),
                        "prompt_id": f"{aggregate_id}_prompt_{prompt_offset:03d}",
                        "source_type": "aggregate",
                        "class_id": None,
                        "aggregate_id": aggregate_id,
                        "text": str(text),
                        "is_canonical": prompt_offset == 0 and str(text) == canonical,
                    }
                )

    return records


def validate_label_vocab(
    vocab: Mapping[str, Any],
    dataset_info: Optional[Mapping[str, Any]] = None,
    strict: bool = False,
) -> Dict[str, List[str]]:
    """Validate the Phase 0 label vocabulary.

    Returns a dict with ``errors`` and ``warnings``. If ``strict`` is true, any
    error raises ``VocabularyValidationError`` after all checks run.
    """

    errors: List[str] = []
    warnings: List[str] = []

    classes = list(vocab.get("classes", []))
    if not classes:
        errors.append("classes is empty or missing")
    ids = [int(item.get("id", -1)) for item in classes]
    expected_ids = list(range(len(classes)))
    if ids != expected_ids:
        errors.append(f"class ids must be contiguous 0..{len(classes) - 1}, got {ids[:8]}...")

    if dataset_info is not None:
        source_names = list(dataset_info.get("class_names", []))
        num_classes = int(dataset_info.get("num_classes", len(source_names)))
        if len(classes) != num_classes:
            errors.append(f"vocab has {len(classes)} classes, dataset_info has {num_classes}")
        for cls in classes:
            class_id = int(cls.get("id", -1))
            if 0 <= class_id < len(source_names):
                expected = source_names[class_id]
                if cls.get("source_name") != expected:
                    errors.append(
                        f"class {class_id} source_name={cls.get('source_name')} "
                        f"does not match dataset_info={expected}"
                    )

    ignored = set(int(x) for x in vocab.get("ignore_as_positive", []))
    train_false = {
        int(cls["id"])
        for cls in classes
        if not bool(cls.get("train_as_positive", False))
    }
    if ignored != train_false:
        warnings.append(
            "ignore_as_positive does not exactly match classes with train_as_positive=false"
        )

    canonical_seen: Dict[str, int] = {}
    prompt_seen: Dict[str, List[str]] = defaultdict(list)
    valid_ids = set(ids)

    for cls in classes:
        class_id = int(cls.get("id", -1))
        source_name = str(cls.get("source_name", ""))
        canonical = str(cls.get("canonical", "")).strip()
        prompts = [str(p).strip() for p in cls.get("prompts", [])]

        if not canonical:
            errors.append(f"class {class_id} has empty canonical")
        if not prompts:
            errors.append(f"class {class_id} has no prompts")
        elif prompts[0] != canonical:
            errors.append(f"class {class_id} first prompt must equal canonical")
        if any(not prompt for prompt in prompts):
            errors.append(f"class {class_id} has an empty prompt")

        canonical_key = canonical.casefold()
        if canonical_key in canonical_seen:
            errors.append(
                f"duplicate canonical prompt {canonical!r} in classes "
                f"{canonical_seen[canonical_key]} and {class_id}"
            )
        canonical_seen[canonical_key] = class_id

        for prompt in prompts:
            prompt_seen[prompt.casefold()].append(f"class:{class_id}")

        side = _source_laterality(source_name)
        if side is not None:
            for prompt in prompts:
                words = prompt.casefold().replace("-", " ").split()
                if side not in words:
                    errors.append(f"class {class_id} prompt lacks laterality {side!r}: {prompt!r}")
                if len([word for word in words if word != side]) == 0:
                    errors.append(f"class {class_id} prompt is only a side word: {prompt!r}")

    for aggregate in vocab.get("aggregates", []):
        aggregate_id = str(aggregate.get("id", ""))
        prompts = [str(p).strip() for p in aggregate.get("prompts", [])]
        canonical = str(aggregate.get("canonical", "")).strip()
        components = [int(x) for x in aggregate.get("component_class_ids", [])]
        if not aggregate_id:
            errors.append("aggregate has empty id")
        if not prompts:
            errors.append(f"aggregate {aggregate_id} has no prompts")
        elif prompts[0] != canonical:
            errors.append(f"aggregate {aggregate_id} first prompt must equal canonical")
        for component in components:
            if component not in valid_ids:
                errors.append(f"aggregate {aggregate_id} references unknown class {component}")
        for prompt in prompts:
            prompt_seen[prompt.casefold()].append(f"aggregate:{aggregate_id}")

    duplicates = {prompt: owners for prompt, owners in prompt_seen.items() if len(owners) > 1}
    if duplicates:
        warnings.append(f"duplicate prompt texts require audit: {duplicates}")

    result = {"errors": errors, "warnings": warnings}
    if strict and errors:
        raise VocabularyValidationError("; ".join(errors))
    return result


def load_class_presence(path: str | Path) -> Dict[str, Any]:
    presence = read_json(path)
    if not isinstance(presence, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return presence


def validate_prompt_embedding_cache(
    cache_path: str | Path,
    vocab_path: str | Path,
    strict: bool = False,
) -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    import torch

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    vocab = load_label_vocab(vocab_path)
    records = flatten_prompt_records(vocab)
    expected_hash = file_sha256(vocab_path)
    embeddings = cache.get("embeddings")

    if cache.get("vocab_hash") != expected_hash:
        errors.append("embedding cache vocab_hash does not match label_vocab.json")
    if cache.get("vocab_version") != vocab.get("version"):
        errors.append("embedding cache vocab_version does not match label_vocab.json")
    if int(cache.get("num_prompts", -1)) != len(records):
        errors.append("embedding cache num_prompts does not match prompt records")
    if cache.get("prompt_records") != records:
        errors.append("embedding cache prompt_records do not match vocabulary flattening")
    if embeddings is None:
        errors.append("embedding cache has no embeddings tensor")
    else:
        if tuple(embeddings.shape) != (len(records), EMBEDDING_DIM):
            errors.append(
                f"embedding shape {tuple(embeddings.shape)} != {(len(records), EMBEDDING_DIM)}"
            )
        if not torch.isfinite(embeddings).all().item():
            errors.append("embedding tensor contains NaN or Inf")

    if cache.get("is_qwen_cache") is not True:
        warnings.append("embedding cache is not marked as a Qwen last-token cache")

    result = {"errors": errors, "warnings": warnings}
    if strict and errors:
        raise VocabularyValidationError("; ".join(errors))
    return result


def prompt_records_by_class(vocab: Mapping[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for record in flatten_prompt_records(vocab, include_aggregates=False):
        grouped[int(record["class_id"])].append(record)
    return dict(grouped)


def sample_prompts_for_case(
    case_record: Mapping[str, Any],
    vocab: Mapping[str, Any],
    num_positive: int = 2,
    num_negative: int = 1,
    min_voxels: int = 1,
    rng: Optional[random.Random] = None,
) -> List[Dict[str, Any]]:
    """Sample prompt records for a case.

    Positive records are sampled from present foreground classes with at least
    ``min_voxels``. Negative records are sampled from absent trainable classes
    and are marked with ``target_empty=true``.
    """

    rng = rng or random.Random()
    counts = {int(k): int(v) for k, v in case_record.get("voxel_counts", {}).items()}
    trainable = set(train_class_ids(vocab))
    present = {
        class_id
        for class_id, count in counts.items()
        if class_id in trainable and count >= min_voxels
    }
    absent = trainable - present
    grouped = prompt_records_by_class(vocab)

    sampled: List[Dict[str, Any]] = []
    positive_classes = _sample_without_replacement(sorted(present), num_positive, rng)
    negative_classes = _sample_without_replacement(sorted(absent), num_negative, rng)

    for class_id in positive_classes:
        record = dict(rng.choice(grouped[class_id]))
        record["target_empty"] = False
        record["target_class_ids"] = [class_id]
        sampled.append(record)

    for class_id in negative_classes:
        record = dict(rng.choice(grouped[class_id]))
        record["target_empty"] = True
        record["target_class_ids"] = [class_id]
        sampled.append(record)

    return sampled


def build_target_mask(labels: np.ndarray, class_ids: Iterable[int]) -> np.ndarray:
    """Construct a binary target mask from integer labels and class ids."""

    class_ids = list(int(x) for x in class_ids)
    if not class_ids:
        return np.zeros_like(labels, dtype=bool)
    return np.isin(labels, class_ids)


def _sample_without_replacement(
    values: Sequence[int],
    count: int,
    rng: random.Random,
) -> List[int]:
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return list(values)
    return rng.sample(list(values), count)


def _source_laterality(source_name: str) -> Optional[str]:
    parts = source_name.split("_")
    if "left" in parts:
        return "left"
    if "right" in parts:
        return "right"
    return None


__all__ = [
    "EMBEDDING_DIM",
    "INSTRUCTION",
    "TEXT_ENCODER",
    "VOCAB_VERSION",
    "VocabularyValidationError",
    "build_target_mask",
    "class_by_id",
    "file_sha256",
    "flatten_prompt_records",
    "foreground_eval_class_ids",
    "load_class_presence",
    "load_label_vocab",
    "prompt_records_by_class",
    "read_json",
    "sample_prompts_for_case",
    "train_class_ids",
    "validate_label_vocab",
    "validate_prompt_embedding_cache",
    "write_json",
]

