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
    validation = load_and_validate_prompt_cache(cache_path, vocab_path, strict=strict)
    return {"errors": validation["errors"], "warnings": validation["warnings"]}


def load_and_validate_prompt_cache(
    cache_path: str | Path,
    vocab_path: str | Path,
    strict: bool = False,
) -> Dict[str, Any]:
    import torch

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    vocab = load_label_vocab(vocab_path)
    validation = validate_prompt_embedding_cache_data(cache, vocab, vocab_path, strict=strict)
    return {"cache": cache, "errors": validation["errors"], "warnings": validation["warnings"]}


def validate_prompt_embedding_cache_data(
    cache: Mapping[str, Any],
    vocab: Mapping[str, Any],
    vocab_path: str | Path,
    strict: bool = False,
) -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    import torch

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


def prompt_records_by_concept(vocab: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Return trainable prompt records grouped by base or aggregate concept.

    Base-class concepts use a single target class. Aggregate concepts are kept
    separate from base-class records and target the union of their configured
    component classes. Aggregate ``train_as_positive`` flags are intentionally
    not used here because the current vocabulary stores existing aggregates as
    inference/evaluation-disabled records while still caching their embeddings.
    """

    trainable = set(train_class_ids(vocab))
    aggregate_components = _aggregate_components_by_id(vocab)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for record in flatten_prompt_records(vocab):
        source_type = str(record.get("source_type", ""))
        if source_type == "class":
            class_id = int(record["class_id"])
            if class_id not in trainable:
                continue
            concept_id = _class_concept_id(class_id)
            target_class_ids = [class_id]
        elif source_type == "aggregate":
            aggregate_id = str(record["aggregate_id"])
            target_class_ids = aggregate_components.get(aggregate_id, [])
            if not target_class_ids or any(
                class_id not in trainable for class_id in target_class_ids
            ):
                continue
            concept_id = _aggregate_concept_id(aggregate_id)
        else:
            continue

        enriched = dict(record)
        enriched["concept_id"] = concept_id
        enriched["target_class_ids"] = list(target_class_ids)
        grouped[concept_id].append(enriched)

    return dict(grouped)


def sample_prompts_for_case(
    case_record: Mapping[str, Any],
    vocab: Mapping[str, Any],
    num_positive: int = 2,
    num_negative: int = 1,
    min_voxels: int = 1,
    canonical_prob: float = 0.25,
    rng: Optional[random.Random] = None,
    prompt_index: Optional[Mapping[Any, Sequence[Mapping[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Sample prompt records for a case.

    Positive records are sampled from present concepts with at least
    ``min_voxels`` voxels. Base-class concepts target one class; aggregate
    concepts are present when the union of their component classes reaches the
    threshold and target that same union. Positive prompt text uses
    ``canonical_prob`` for canonical terms and otherwise samples non-canonical
    variants. If a positive concept has no variants, the canonical record is
    used and marked as an explicit fallback. Negative records are sampled from
    absent trainable concepts, keep ordinary random prompt selection, and are
    marked with ``target_empty=true``.
    """

    canonical_prob = float(canonical_prob)
    if not 0.0 <= canonical_prob <= 1.0:
        raise ValueError("canonical_prob must be between 0 and 1")

    rng = rng or random.Random()
    counts = {int(k): int(v) for k, v in case_record.get("voxel_counts", {}).items()}
    grouped = prompt_index if prompt_index is not None else prompt_records_by_concept(vocab)
    concepts = _trainable_concepts_from_prompt_index(grouped, vocab)
    present = {
        concept_id
        for concept_id, concept in concepts.items()
        if _concept_voxel_count(concept["target_class_ids"], counts) >= min_voxels
    }
    absent = set(concepts) - present

    sampled: List[Dict[str, Any]] = []
    positive_concepts = _sample_without_replacement(
        sorted(present, key=_concept_sort_key),
        num_positive,
        rng,
    )
    negative_concepts = _sample_without_replacement(
        sorted(absent, key=_concept_sort_key),
        num_negative,
        rng,
    )

    for concept_id in positive_concepts:
        concept = concepts[concept_id]
        record = _sample_positive_prompt_record(concept["records"], canonical_prob, rng)
        record["target_empty"] = False
        record["target_class_ids"] = list(concept["target_class_ids"])
        sampled.append(record)

    for concept_id in negative_concepts:
        concept = concepts[concept_id]
        record = dict(rng.choice(concept["records"]))
        record["prompt_sampling_role"] = "negative"
        record["prompt_sampling_mode"] = "random"
        record["has_variants"] = _has_noncanonical_variant(concept["records"])
        record["canonical_fallback"] = False
        record["target_empty"] = True
        record["target_class_ids"] = list(concept["target_class_ids"])
        sampled.append(record)

    return sampled


def build_target_mask(labels: np.ndarray, class_ids: Iterable[int]) -> np.ndarray:
    """Construct a binary target mask from integer labels and class ids."""

    class_ids = list(int(x) for x in class_ids)
    if not class_ids:
        return np.zeros_like(labels, dtype=bool)
    return np.isin(labels, class_ids)


def _sample_positive_prompt_record(
    records: Sequence[Mapping[str, Any]],
    canonical_prob: float,
    rng: random.Random,
) -> Dict[str, Any]:
    canonical_records = [
        record for record in records if bool(record.get("is_canonical", False))
    ]
    variant_records = [
        record for record in records if not bool(record.get("is_canonical", False))
    ]
    canonical_record = canonical_records[0] if canonical_records else records[0]
    use_canonical = rng.random() < canonical_prob

    if use_canonical:
        selected = dict(canonical_record)
        mode = "canonical"
    elif variant_records:
        selected = dict(rng.choice(variant_records))
        mode = "variant"
    else:
        selected = dict(canonical_record)
        mode = "canonical_fallback"

    selected["prompt_sampling_role"] = "positive"
    selected["prompt_sampling_mode"] = mode
    selected["has_variants"] = bool(variant_records)
    selected["canonical_fallback"] = mode == "canonical_fallback"
    return selected


def _has_noncanonical_variant(records: Sequence[Mapping[str, Any]]) -> bool:
    return any(not bool(record.get("is_canonical", False)) for record in records)


def _aggregate_components_by_id(vocab: Mapping[str, Any]) -> Dict[str, List[int]]:
    return {
        str(aggregate["id"]): _unique_ints(aggregate.get("component_class_ids", []))
        for aggregate in vocab.get("aggregates", [])
    }


def _trainable_concepts_from_prompt_index(
    prompt_index: Mapping[Any, Sequence[Mapping[str, Any]]],
    vocab: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    trainable = set(train_class_ids(vocab))
    aggregate_components = _aggregate_components_by_id(vocab)
    concepts: Dict[str, Dict[str, Any]] = {}

    for key, records in prompt_index.items():
        if not records:
            continue
        first = records[0]
        source_type = str(first.get("source_type", "class"))
        target_class_ids = _concept_target_class_ids(key, first, aggregate_components)
        if not _is_trainable_concept(source_type, target_class_ids, trainable):
            continue
        concept_id = str(first.get("concept_id") or _concept_id_from_record(key, first))

        enriched_records: List[Dict[str, Any]] = []
        for record in records:
            enriched = dict(record)
            enriched.setdefault("concept_id", concept_id)
            enriched["target_class_ids"] = list(target_class_ids)
            enriched_records.append(enriched)

        concepts[concept_id] = {
            "records": enriched_records,
            "target_class_ids": list(target_class_ids),
        }

    return concepts


def _concept_target_class_ids(
    key: Any,
    record: Mapping[str, Any],
    aggregate_components: Mapping[str, Sequence[int]],
) -> List[int]:
    if "target_class_ids" in record:
        return _unique_ints(record.get("target_class_ids", []))
    if record.get("source_type") == "aggregate":
        aggregate_id = str(record["aggregate_id"])
        return _unique_ints(aggregate_components.get(aggregate_id, []))
    class_id = record.get("class_id", key)
    return [int(class_id)]


def _concept_id_from_record(key: Any, record: Mapping[str, Any]) -> str:
    if record.get("source_type") == "aggregate":
        return _aggregate_concept_id(str(record["aggregate_id"]))
    class_id = record.get("class_id", key)
    return _class_concept_id(int(class_id))


def _is_trainable_concept(
    source_type: str,
    target_class_ids: Sequence[int],
    trainable: set[int],
) -> bool:
    if not target_class_ids:
        return False
    if source_type == "aggregate":
        return all(class_id in trainable for class_id in target_class_ids)
    return len(target_class_ids) == 1 and int(target_class_ids[0]) in trainable


def _concept_voxel_count(
    target_class_ids: Sequence[int],
    counts: Mapping[int, int],
) -> int:
    return sum(
        int(counts.get(int(class_id), 0))
        for class_id in _unique_ints(target_class_ids)
    )


def _unique_ints(values: Iterable[Any]) -> List[int]:
    result: List[int] = []
    seen: set[int] = set()
    for value in values:
        class_id = int(value)
        if class_id in seen:
            continue
        result.append(class_id)
        seen.add(class_id)
    return result


def _class_concept_id(class_id: int) -> str:
    return f"class:{int(class_id)}"


def _aggregate_concept_id(aggregate_id: str) -> str:
    return f"aggregate:{aggregate_id}"


def _concept_sort_key(value: Any) -> tuple[int, int, str]:
    text = str(value)
    if text.startswith("class:"):
        try:
            return (0, int(text.split(":", 1)[1]), "")
        except ValueError:
            return (0, 0, text)
    if isinstance(value, int):
        return (0, int(value), "")
    if text.startswith("aggregate:"):
        return (1, 0, text)
    return (2, 0, text)


def _sample_without_replacement(
    values: Sequence[Any],
    count: int,
    rng: random.Random,
) -> List[Any]:
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
    "load_and_validate_prompt_cache",
    "prompt_records_by_class",
    "prompt_records_by_concept",
    "read_json",
    "sample_prompts_for_case",
    "train_class_ids",
    "validate_label_vocab",
    "validate_prompt_embedding_cache",
    "validate_prompt_embedding_cache_data",
    "write_json",
]
