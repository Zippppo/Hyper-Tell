#!/usr/bin/env python3
"""Copy and crop the S2I dataset package to a fixed voxel grid size."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


METADATA_FILES = ("dataset_info.json", "dataset_split.json", "class_presence.json")


@dataclass(frozen=True)
class AxisCropPlan:
    src: slice
    dst: slice
    grid_offset: int
    crop_before: int
    crop_after: int
    pad_before: int
    pad_after: int


@dataclass(frozen=True)
class CropStats:
    filename: str
    source_shape: tuple[int, int, int]
    target_shape: tuple[int, int, int]
    cropped: bool
    padded: bool
    axis_crop_before: tuple[int, int, int]
    axis_crop_after: tuple[int, int, int]
    axis_pad_before: tuple[int, int, int]
    axis_pad_after: tuple[int, int, int]
    input_bytes: int
    output_bytes: int
    source_sensor_points: int
    kept_sensor_points: int


@dataclass(frozen=True)
class DatasetCropStats:
    total_files: int
    target_shape: tuple[int, int, int]
    cropped_files: int
    padded_files: int
    full_coverage_files: int
    full_coverage_fraction: float
    input_bytes: int
    output_bytes: int
    estimated_label_bytes: int
    source_sensor_points: int
    kept_sensor_points: int
    axis_crop_before: tuple[int, int, int]
    axis_crop_after: tuple[int, int, int]
    axis_pad_before: tuple[int, int, int]
    axis_pad_after: tuple[int, int, int]

    @classmethod
    def from_cases(
        cls,
        case_stats: Sequence[CropStats],
        target_shape: Sequence[int],
    ) -> "DatasetCropStats":
        total = len(case_stats)
        cropped = sum(1 for item in case_stats if item.cropped)
        padded = sum(1 for item in case_stats if item.padded)
        target = tuple(int(x) for x in target_shape)
        return cls(
            total_files=total,
            target_shape=target,
            cropped_files=cropped,
            padded_files=padded,
            full_coverage_files=total - cropped,
            full_coverage_fraction=float((total - cropped) / total) if total else 0.0,
            input_bytes=sum(item.input_bytes for item in case_stats),
            output_bytes=sum(item.output_bytes for item in case_stats),
            estimated_label_bytes=total * int(np.prod(target)),
            source_sensor_points=sum(item.source_sensor_points for item in case_stats),
            kept_sensor_points=sum(item.kept_sensor_points for item in case_stats),
            axis_crop_before=_sum_axes(item.axis_crop_before for item in case_stats),
            axis_crop_after=_sum_axes(item.axis_crop_after for item in case_stats),
            axis_pad_before=_sum_axes(item.axis_pad_before for item in case_stats),
            axis_pad_after=_sum_axes(item.axis_pad_after for item in case_stats),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def summary_lines(self, mode: str) -> list[str]:
        shape_text = "x".join(str(x) for x in self.target_shape)
        return [
            f"mode={mode}",
            f"target_shape={shape_text}",
            f"total_files={self.total_files}",
            f"cropped_files={self.cropped_files}",
            f"padded_files={self.padded_files}",
            f"full_coverage_files={self.full_coverage_files}",
            f"full_coverage_fraction={self.full_coverage_fraction:.6f}",
            f"input_bytes={self.input_bytes}",
            f"output_bytes={self.output_bytes}",
            f"estimated_label_bytes={self.estimated_label_bytes}",
            f"sensor_points_kept={self.kept_sensor_points}/{self.source_sensor_points}",
            f"axis_crop_before={self.axis_crop_before}",
            f"axis_crop_after={self.axis_crop_after}",
            f"axis_pad_before={self.axis_pad_before}",
            f"axis_pad_after={self.axis_pad_after}",
        ]


@dataclass(frozen=True)
class _Task:
    src: Path
    dst: Path | None
    target_shape: tuple[int, int, int]
    write: bool


def _sum_axes(values: Iterable[Sequence[int]]) -> tuple[int, int, int]:
    total = [0, 0, 0]
    for value in values:
        for axis, axis_value in enumerate(value):
            total[axis] += int(axis_value)
    return tuple(total)  # type: ignore[return-value]


def _shape_tuple(shape: Sequence[int]) -> tuple[int, int, int]:
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D shape, got {tuple(shape)}")
    return tuple(int(x) for x in shape)  # type: ignore[return-value]


def plan_crop(
    source_shape: Sequence[int],
    target_shape: Sequence[int],
) -> tuple[AxisCropPlan, AxisCropPlan, AxisCropPlan]:
    source = _shape_tuple(source_shape)
    target = _shape_tuple(target_shape)
    plan = []
    for src_size, dst_size in zip(source, target):
        if src_size >= dst_size:
            src_start = (src_size - dst_size) // 2
            dst_start = 0
            copy_size = dst_size
        else:
            src_start = 0
            dst_start = (dst_size - src_size) // 2
            copy_size = src_size
        src_end = src_start + copy_size
        dst_end = dst_start + copy_size
        plan.append(
            AxisCropPlan(
                src=slice(src_start, src_end),
                dst=slice(dst_start, dst_end),
                grid_offset=src_start - dst_start,
                crop_before=src_start,
                crop_after=max(0, src_size - src_end),
                pad_before=dst_start,
                pad_after=max(0, dst_size - dst_end),
            )
        )
    return tuple(plan)  # type: ignore[return-value]


def crop_labels(
    labels: np.ndarray,
    target_shape: Sequence[int],
    plan: Sequence[AxisCropPlan] | None = None,
    pad_value: int = 0,
) -> np.ndarray:
    if labels.ndim != 3:
        raise ValueError(f"Expected voxel_labels to be 3D, got shape {labels.shape}")
    target = _shape_tuple(target_shape)
    plan = tuple(plan) if plan is not None else plan_crop(labels.shape, target)
    output = np.full(target, pad_value, dtype=labels.dtype)
    src_slices = tuple(item.src for item in plan)
    dst_slices = tuple(item.dst for item in plan)
    output[dst_slices] = labels[src_slices]
    return output


def crop_npz_file(
    src_path: Path,
    dst_path: Path | None,
    target_shape: Sequence[int],
    write: bool = True,
) -> CropStats:
    src_path = Path(src_path)
    dst_path = Path(dst_path) if dst_path is not None else None
    target = _shape_tuple(target_shape)
    input_bytes = src_path.stat().st_size

    with np.load(src_path, allow_pickle=False) as data:
        labels = data["voxel_labels"]
        source_shape = _shape_tuple(labels.shape)
        plan = plan_crop(source_shape, target)
        grid_world_min = np.asarray(data["grid_world_min"], dtype=np.float32)
        grid_voxel_size = np.asarray(data["grid_voxel_size"], dtype=np.float32)
        sensor_pc = np.asarray(data["sensor_pc"], dtype=np.float32)

        if grid_voxel_size.ndim == 0:
            grid_voxel_size = np.repeat(grid_voxel_size, 3)
        if grid_world_min.shape[0] < 3 or grid_voxel_size.shape[0] < 3:
            raise ValueError(f"{src_path} has invalid grid metadata")
        if sensor_pc.ndim != 2 or sensor_pc.shape[1] < 3:
            raise ValueError(f"{src_path} has invalid sensor_pc shape {sensor_pc.shape}")

        offsets = np.asarray([item.grid_offset for item in plan], dtype=np.float32)
        new_world_min = grid_world_min[:3] + offsets * grid_voxel_size[:3]
        new_world_max = new_world_min + np.asarray(target, dtype=np.float32) * grid_voxel_size[:3]
        keep = np.all(sensor_pc[:, :3] >= new_world_min, axis=1) & np.all(
            sensor_pc[:, :3] < new_world_max,
            axis=1,
        )
        kept_sensor_pc = sensor_pc[keep]
        output_bytes = 0

        if write:
            if dst_path is None:
                raise ValueError("dst_path is required when write=True")
            arrays = {key: data[key] for key in data.files}
            arrays["voxel_labels"] = crop_labels(np.asarray(labels), target, plan)
            arrays["sensor_pc"] = kept_sensor_pc.astype(data["sensor_pc"].dtype, copy=False)
            arrays["grid_world_min"] = new_world_min.astype(data["grid_world_min"].dtype, copy=False)
            arrays["grid_world_max"] = new_world_max.astype(data["grid_world_max"].dtype, copy=False)
            arrays["grid_occ_size"] = np.asarray(target, dtype=data["grid_occ_size"].dtype)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dst_path, **arrays)
            output_bytes = dst_path.stat().st_size

    return CropStats(
        filename=src_path.name,
        source_shape=source_shape,
        target_shape=target,
        cropped=any(item.crop_before or item.crop_after for item in plan),
        padded=any(item.pad_before or item.pad_after for item in plan),
        axis_crop_before=tuple(int(item.crop_before) for item in plan),  # type: ignore[return-value]
        axis_crop_after=tuple(int(item.crop_after) for item in plan),  # type: ignore[return-value]
        axis_pad_before=tuple(int(item.pad_before) for item in plan),  # type: ignore[return-value]
        axis_pad_after=tuple(int(item.pad_after) for item in plan),  # type: ignore[return-value]
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        source_sensor_points=int(sensor_pc.shape[0]),
        kept_sensor_points=int(kept_sensor_pc.shape[0]),
    )


def _process_task(task: _Task) -> CropStats:
    return crop_npz_file(task.src, task.dst, task.target_shape, write=task.write)


def _resolve_input(input_root: Path) -> tuple[Path, Path]:
    input_root = Path(input_root)
    if (input_root / "data").is_dir():
        return input_root, input_root / "data"
    if input_root.is_dir() and any(input_root.glob("*.npz")):
        return input_root.parent, input_root
    raise FileNotFoundError(f"Could not find dataset data directory under {input_root}")


def _copy_metadata(input_package_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for filename in METADATA_FILES:
        src = input_package_root / filename
        if src.exists():
            shutil.copy2(src, output_root / filename)


def crop_dataset(
    input_root: Path,
    output_root: Path,
    target_shape: Sequence[int],
    dry_run: bool = False,
    workers: int = 1,
    overwrite: bool = False,
    progress_every: int = 100,
) -> DatasetCropStats:
    target = _shape_tuple(target_shape)
    input_package_root, input_data_dir = _resolve_input(Path(input_root))
    output_root = Path(output_root)
    output_data_dir = output_root / "data"

    if input_package_root.resolve() == output_root.resolve():
        raise ValueError("Refusing in-place crop: output_root must differ from input_root")
    if input_data_dir.resolve() == output_data_dir.resolve():
        raise ValueError("Refusing in-place crop: output data directory matches input data directory")

    npz_files = sorted(input_data_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {input_data_dir}")

    if not dry_run:
        if output_data_dir.exists() and any(output_data_dir.glob("*.npz")) and not overwrite:
            raise FileExistsError(
                f"{output_data_dir} already contains .npz files; pass --overwrite to replace them"
            )
        _copy_metadata(input_package_root, output_root)

    tasks = [
        _Task(
            src=path,
            dst=None if dry_run else output_data_dir / path.name,
            target_shape=target,
            write=not dry_run,
        )
        for path in npz_files
    ]

    workers = max(1, int(workers))
    progress_every = max(1, int(progress_every))
    stats: list[CropStats] = []
    if workers == 1:
        for index, task in enumerate(tasks, start=1):
            stats.append(_process_task(task))
            if index % progress_every == 0 or index == len(tasks):
                print(f"processed {index}/{len(tasks)} files", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_task, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), start=1):
                stats.append(future.result())
                if index % progress_every == 0 or index == len(futures):
                    print(f"processed {index}/{len(futures)} files", flush=True)

    return DatasetCropStats.from_cases(stats, target)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-shape", type=int, nargs=3, required=True, metavar=("D", "H", "W"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stats = crop_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        target_shape=args.target_shape,
        dry_run=args.dry_run,
        workers=args.workers,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
    )
    mode = "dry-run" if args.dry_run else "write"
    for line in stats.summary_lines(mode):
        print(line)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(stats.to_json() + "\n", encoding="utf-8")
        print(f"wrote_summary_json={args.summary_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
