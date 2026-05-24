#!/usr/bin/env python3
"""Build a deterministic train/val/test split from a voxel npz directory."""

from __future__ import annotations

import argparse
import random
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from body_tell.data.vocabulary import write_json  # noqa: E402


def build_split(args: argparse.Namespace) -> Dict[str, Any]:
    files = sorted(path.name for path in args.voxel_dir.glob("*.npz"))
    if not files:
        raise ValueError(f"No .npz files found in {args.voxel_dir}")
    holdout_count = args.val_count + args.test_count
    if len(files) <= holdout_count:
        raise ValueError(
            f"Need more files than val+test holdout ({holdout_count}), got {len(files)}"
        )

    shuffled: List[str] = list(files)
    random.Random(args.seed).shuffle(shuffled)
    val = shuffled[: args.val_count]
    test = shuffled[args.val_count : holdout_count]
    train = shuffled[holdout_count:]
    split_date = args.split_date or date.today().isoformat()
    data_directory = args.data_directory or str(args.voxel_dir)

    return {
        "split_info": {
            "total_samples": len(files),
            "train_count": len(train),
            "val_count": len(val),
            "test_count": len(test),
            "random_seed": args.seed,
            "split_date": split_date,
            "data_directory": data_directory,
        },
        "train": train,
        "val": val,
        "test": test,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voxel-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--test-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-date", type=str, default=None)
    parser.add_argument("--data-directory", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = build_split(args)
    write_json(split, args.output)
    print(f"wrote {args.output}")
    print(
        f"total={split['split_info']['total_samples']} "
        f"train={split['split_info']['train_count']} "
        f"val={split['split_info']['val_count']} "
        f"test={split['split_info']['test_count']}"
    )


if __name__ == "__main__":
    main()
