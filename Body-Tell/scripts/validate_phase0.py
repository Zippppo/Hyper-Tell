#!/usr/bin/env python3
"""Validate Phase 0 vocabulary artifacts and run a sampler dry run."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from body_tell.data.vocabulary import (  # noqa: E402
    build_target_mask,
    load_class_presence,
    load_label_vocab,
    read_json,
    sample_prompts_for_case,
    validate_label_vocab,
    validate_prompt_embedding_cache,
)


def find_dry_run_case(
    presence: Dict[str, Any],
    vocab: Dict[str, Any],
    num_positive: int,
    num_negative: int,
) -> Tuple[str, Dict[str, Any]]:
    trainable = {
        int(item["id"])
        for item in vocab["classes"]
        if bool(item.get("train_as_positive", False))
    }
    for case_id, case in presence["cases"].items():
        present = set(int(x) for x in case.get("foreground_present_class_ids", []))
        absent = trainable - present
        if len(present) >= num_positive and len(absent) >= num_negative:
            return case_id, case
    raise RuntimeError("No case has enough positive and negative classes for dry run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-info", type=Path, default=ROOT / "Dataset" / "dataset_info.json")
    parser.add_argument("--voxel-dir", type=Path, default=ROOT / "Dataset" / "voxel_data")
    parser.add_argument("--vocab", type=Path, default=ROOT / "configs" / "label_vocab.json")
    parser.add_argument(
        "--presence",
        type=Path,
        default=ROOT / "artifacts" / "data_stats" / "class_presence.json",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=ROOT / "artifacts" / "text_embeddings" / "prompt_embeddings.pt",
    )
    parser.add_argument("--num-positive", type=int, default=2)
    parser.add_argument("--num-negative", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_info = read_json(args.dataset_info)
    vocab = load_label_vocab(args.vocab)
    presence = load_class_presence(args.presence)

    vocab_validation = validate_label_vocab(vocab, dataset_info=dataset_info, strict=True)
    cache_validation = validate_prompt_embedding_cache(args.embedding_cache, args.vocab, strict=True)
    if presence["num_cases"] != len(presence["cases"]):
        raise ValueError("class_presence num_cases does not match cases length")
    if presence["num_classes"] != dataset_info["num_classes"]:
        raise ValueError("class_presence num_classes does not match dataset_info")

    case_id, case = find_dry_run_case(
        presence,
        vocab,
        args.num_positive,
        args.num_negative,
    )
    rng = random.Random(args.seed)
    sampled = sample_prompts_for_case(
        case,
        vocab,
        num_positive=args.num_positive,
        num_negative=args.num_negative,
        rng=rng,
    )
    labels_path = args.voxel_dir / case["filename"]
    with np.load(labels_path) as data:
        labels = data["voxel_labels"]

    positive_count = 0
    negative_count = 0
    for record in sampled:
        mask = build_target_mask(labels, record["target_class_ids"])
        voxels = int(mask.sum())
        if record["target_empty"]:
            negative_count += 1
            if voxels != 0:
                raise ValueError(f"negative prompt {record['prompt_id']} produced non-empty target")
        else:
            positive_count += 1
            if voxels <= 0:
                raise ValueError(f"positive prompt {record['prompt_id']} produced empty target")

    print("phase0 validation passed")
    print(f"vocab warnings: {vocab_validation['warnings']}")
    print(f"cache warnings: {cache_validation['warnings']}")
    print(f"dry_run_case: {case_id} {case['filename']} shape={case['shape']}")
    print(f"sampled_prompts: {sampled}")
    print(f"positive_count={positive_count} negative_count={negative_count}")


if __name__ == "__main__":
    main()

