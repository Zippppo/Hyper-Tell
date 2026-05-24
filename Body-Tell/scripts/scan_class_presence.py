#!/usr/bin/env python3
"""Scan Body-Tell voxel labels and write class_presence.json."""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from body_tell.data.vocabulary import (  # noqa: E402
    VOCAB_VERSION,
    load_label_vocab,
    read_json,
    validate_label_vocab,
    write_json,
)


def split_lookup(split_data: Dict[str, Any]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for split in ("train", "val", "test"):
        for filename in split_data.get(split, []):
            lookup[Path(filename).name] = split
    return lookup


def summarize_counts(values: List[int]) -> Dict[str, float | int]:
    if not values:
        return {
            "min_voxels_per_present_case": 0,
            "median_voxels_per_present_case": 0,
            "mean_voxels_per_present_case": 0.0,
            "max_voxels_per_present_case": 0,
        }
    return {
        "min_voxels_per_present_case": int(min(values)),
        "median_voxels_per_present_case": int(statistics.median(values)),
        "mean_voxels_per_present_case": float(statistics.fmean(values)),
        "max_voxels_per_present_case": int(max(values)),
    }


def scan_npz(path: Path, num_classes: int) -> Tuple[List[int], Dict[str, int], List[int]]:
    with np.load(path) as data:
        labels = data["voxel_labels"]
        shape = [int(x) for x in labels.shape]
        counts = np.bincount(labels.reshape(-1), minlength=num_classes)
    present = [int(i) for i, count in enumerate(counts[:num_classes]) if int(count) > 0]
    voxel_counts = {
        str(i): int(count)
        for i, count in enumerate(counts[:num_classes])
        if int(count) > 0
    }
    return shape, voxel_counts, present


def build_presence(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_info = read_json(args.dataset_info)
    split_data = read_json(args.split_file)
    vocab = load_label_vocab(args.vocab)
    validate_label_vocab(vocab, dataset_info=dataset_info, strict=True)

    num_classes = int(dataset_info["num_classes"])
    trainable = {
        int(item["id"])
        for item in vocab["classes"]
        if bool(item.get("train_as_positive", False))
    }
    split_by_file = split_lookup(split_data)
    npz_files = sorted(args.voxel_dir.glob("*.npz"))
    if args.limit is not None:
        npz_files = npz_files[: args.limit]

    cases: Dict[str, Any] = {}
    class_present_counts: Dict[int, List[int]] = defaultdict(list)
    class_total_voxels = Counter()
    class_case_count = Counter()
    shape_counter = Counter()
    split_present_counts = Counter()
    warnings: List[str] = []

    for index, path in enumerate(npz_files, start=1):
        shape, voxel_counts, present = scan_npz(path, num_classes)
        shape_key = "x".join(str(x) for x in shape)
        shape_counter[shape_key] += 1
        case_id = path.stem
        split = split_by_file.get(path.name)
        if split is None:
            warnings.append(f"{path.name} is not listed in dataset_split.json")
            split = "unassigned"
        split_present_counts[split] += 1

        foreground_present = [
            class_id
            for class_id in present
            if class_id in trainable and int(voxel_counts[str(class_id)]) >= args.min_positive_voxels
        ]
        cases[case_id] = {
            "filename": path.name,
            "split": split,
            "shape": shape,
            "present_class_ids": present,
            "foreground_present_class_ids": foreground_present,
            "voxel_counts": voxel_counts,
        }

        for class_id_text, count in voxel_counts.items():
            class_id = int(class_id_text)
            count = int(count)
            class_total_voxels[class_id] += count
            class_case_count[class_id] += 1
            class_present_counts[class_id].append(count)

        if index % args.progress_every == 0:
            print(f"scanned {index}/{len(npz_files)} files", flush=True)

    split_expected = {
        split: {Path(name).name for name in split_data.get(split, [])}
        for split in ("train", "val", "test")
    }
    present_files = {path.name for path in npz_files}
    split_coverage = {
        split: {
            "declared_count": len(files),
            "present_count": len(files & present_files),
            "missing_count": len(files - present_files),
            "missing_files_preview": sorted(files - present_files)[:20],
        }
        for split, files in split_expected.items()
    }

    classes: Dict[str, Any] = {}
    num_cases = len(cases)
    for class_id, cls in enumerate(vocab["classes"]):
        values = class_present_counts.get(class_id, [])
        case_count = int(class_case_count.get(class_id, 0))
        case_fraction = float(case_count / num_cases) if num_cases else 0.0
        summary = summarize_counts(values)
        classes[str(class_id)] = {
            "source_name": cls["source_name"],
            "canonical": cls["canonical"],
            "train_as_positive": bool(cls.get("train_as_positive", False)),
            "eval_as_foreground": bool(cls.get("eval_as_foreground", False)),
            "case_count": case_count,
            "case_fraction": case_fraction,
            "total_voxels": int(class_total_voxels.get(class_id, 0)),
            **summary,
            "is_rare": bool(case_fraction < args.rare_case_fraction_threshold),
            "is_small_structure": bool(
                summary["median_voxels_per_present_case"] < args.small_median_voxel_threshold
            ),
        }

    shape_tuples = [
        tuple(int(part) for part in shape_key.split("x"))
        for shape_key in shape_counter
    ]
    if shape_tuples:
        recommended_volume_size = [
            max(shape[axis] for shape in shape_tuples)
            for axis in range(3)
        ]
    else:
        recommended_volume_size = []

    return {
        "version": VOCAB_VERSION,
        "source_dataset_info": str(args.dataset_info),
        "source_split_file": str(args.split_file),
        "source_voxel_dir": str(args.voxel_dir),
        "label_vocab": str(args.vocab),
        "num_cases": num_cases,
        "num_classes": num_classes,
        "min_positive_voxels": int(args.min_positive_voxels),
        "shape_summary": {
            "unique_shape_count": len(shape_counter),
            "shape_counts": dict(sorted(shape_counter.items())),
            "recommended_volume_size": recommended_volume_size,
        },
        "split_summary": {
            "declared": {
                split: len(split_data.get(split, []))
                for split in ("train", "val", "test")
            },
            "scanned": dict(split_present_counts),
            "coverage": split_coverage,
        },
        "cases": cases,
        "classes": classes,
        "warnings": warnings[:200],
        "warning_count": len(warnings),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-info", type=Path, default=ROOT / "Dataset" / "dataset_info.json")
    parser.add_argument("--split-file", type=Path, default=ROOT / "Dataset" / "dataset_split.json")
    parser.add_argument("--voxel-dir", type=Path, default=ROOT / "Dataset" / "voxel_data")
    parser.add_argument("--vocab", type=Path, default=ROOT / "configs" / "label_vocab.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "data_stats" / "class_presence.json",
    )
    parser.add_argument("--min-positive-voxels", type=int, default=1)
    parser.add_argument("--small-median-voxel-threshold", type=int, default=1000)
    parser.add_argument("--rare-case-fraction-threshold", type=float, default=0.05)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    presence = build_presence(args)
    write_json(presence, args.output)
    print(f"wrote {args.output}")
    print(
        f"cases={presence['num_cases']} "
        f"unique_shapes={presence['shape_summary']['unique_shape_count']} "
        f"recommended_volume_size={presence['shape_summary']['recommended_volume_size']}"
    )
    if presence["warning_count"]:
        print(f"warnings={presence['warning_count']}")


if __name__ == "__main__":
    main()
