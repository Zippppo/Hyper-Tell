"""Prompt-conditioned HyperBody voxel dataset for Body-Tell Phase 1."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .vocabulary import (
    build_target_mask,
    load_class_presence,
    load_and_validate_prompt_cache,
    load_label_vocab,
    prompt_records_by_class,
    read_json,
    sample_prompts_for_case,
    train_class_ids,
)


_EPOCH_SEED_STRIDE = 1_000_003


class HyperBodyPromptDataset(Dataset):
    """Read ``Body-Tell/Dataset/voxel_data`` as prompt-conditioned masks.

    Prompt sampling is deterministic for a given ``seed``, epoch, and global
    dataset index. In DDP training, call ``DistributedSampler.set_epoch(epoch)``
    and then ``dataset.set_epoch(epoch)`` before iterating the DataLoader. The
    prompt RNG intentionally does not include rank, so a case has the same
    prompt sample for a given epoch/index regardless of which rank receives it.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        volume_size: Optional[Sequence[int]] = None,
        patch_size: Optional[Sequence[int]] = None,
        foreground_oversample_prob: float = 0.0,
        num_positive: int = 2,
        num_negative: int = 1,
        min_voxels: int = 1,
        seed: int = 0,
        vocab_path: str | Path | None = None,
        split_path: str | Path | None = None,
        presence_path: str | Path | None = None,
        voxel_dir: str | Path | None = None,
        embedding_cache_path: str | Path | None = None,
        strict_embedding_cache: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.volume_size = tuple(int(x) for x in volume_size) if volume_size else None
        self.patch_size = self._normalize_shape("patch_size", patch_size)
        self.foreground_oversample_prob = float(foreground_oversample_prob)
        if not 0.0 <= self.foreground_oversample_prob <= 1.0:
            raise ValueError("foreground_oversample_prob must be between 0 and 1")
        self.num_positive = int(num_positive)
        self.num_negative = int(num_negative)
        self.min_voxels = int(min_voxels)
        self.seed = int(seed)
        self._epoch = 0

        self.vocab_path = (
            self._resolve_root_path(vocab_path)
            if vocab_path
            else self.root / "configs" / "label_vocab.json"
        )
        self.split_path = (
            self._resolve_root_path(split_path)
            if split_path
            else self.root / "Dataset" / "dataset_split.json"
        )
        self.presence_path = (
            self._resolve_root_path(presence_path)
            if presence_path
            else self.root / "artifacts" / "data_stats" / "class_presence.json"
        )
        self.embedding_cache_path = (
            self._resolve_root_path(embedding_cache_path)
            if embedding_cache_path
            else self.root / "artifacts" / "text_embeddings" / "prompt_embeddings.pt"
        )
        self.voxel_dir = (
            self._resolve_root_path(voxel_dir)
            if voxel_dir
            else self.root / "Dataset" / "voxel_data"
        )

        self.vocab = load_label_vocab(self.vocab_path)
        self._train_class_ids = tuple(train_class_ids(self.vocab))
        self.presence = load_class_presence(self.presence_path)
        split_data = read_json(self.split_path)
        if split not in split_data:
            raise ValueError(f"Unknown split {split!r}; expected one of train, val, test")
        self.case_files = [self._resolve_case_path(name) for name in split_data[split]]
        if not self.case_files:
            raise ValueError(f"Split {split!r} contains no cases")

        validation = load_and_validate_prompt_cache(
            self.embedding_cache_path,
            self.vocab_path,
            strict=strict_embedding_cache,
        )
        self.embedding_warnings = validation["warnings"]
        self.embedding_cache = validation["cache"]
        self.prompt_embeddings = self.embedding_cache["embeddings"].float().contiguous()
        self._prompt_records_by_class = prompt_records_by_class(self.vocab)

    @staticmethod
    def _normalize_shape(
        name: str,
        shape: Optional[Sequence[int]],
    ) -> Optional[Tuple[int, int, int]]:
        if shape is None:
            return None
        values = tuple(int(x) for x in shape)
        if len(values) != 3:
            raise ValueError(f"{name} must contain exactly 3 dimensions")
        if any(value <= 0 for value in values):
            raise ValueError(f"{name} dimensions must be positive")
        return values

    def _resolve_root_path(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        return self.root / path

    def _resolve_case_path(self, split_entry: str | Path) -> Path:
        path = Path(split_entry)
        if path.is_absolute():
            return path
        if path.parent == Path("."):
            return self.voxel_dir / path
        return self.root / path

    def __len__(self) -> int:
        return len(self.case_files)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used by prompt sampling.

        The per-sample seed is ``base_seed + epoch * large_prime + index``.
        This makes repeated access deterministic within an epoch while allowing
        the same case to draw different prompt text across epochs. Call this
        before creating the DataLoader iterator; persistent workers need their
        own worker-side epoch propagation.
        """

        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch = epoch

    def get_epoch(self) -> int:
        return self._epoch

    def _prompt_rng(self, index: int) -> random.Random:
        return random.Random(self.seed + self._epoch * _EPOCH_SEED_STRIDE + int(index))

    def _patch_case_record(
        self,
        case_record: Mapping[str, Any],
        labels: np.ndarray,
    ) -> Dict[str, Any]:
        class_ids, counts = np.unique(labels, return_counts=True)
        voxel_counts = {
            int(class_id): int(count)
            for class_id, count in zip(class_ids, counts)
        }
        present_class_ids = sorted(voxel_counts)
        trainable = set(self._train_class_ids)
        patch_record = dict(case_record)
        patch_record.update(
            {
                "shape": [int(axis) for axis in labels.shape],
                "present_class_ids": present_class_ids,
                "foreground_present_class_ids": [
                    class_id for class_id in present_class_ids if class_id in trainable
                ],
                "voxel_counts": {
                    str(class_id): int(count) for class_id, count in voxel_counts.items()
                },
            }
        )
        return patch_record

    def _sample_foreground_voxel(
        self,
        labels: np.ndarray,
        rng: random.Random,
    ) -> Optional[Tuple[int, int, int]]:
        if not self._train_class_ids:
            return None
        foreground_mask = np.isin(labels, self._train_class_ids)
        foreground_flat = np.flatnonzero(foreground_mask.reshape(-1))
        if foreground_flat.size == 0:
            return None
        flat_index = int(foreground_flat[rng.randrange(int(foreground_flat.size))])
        return tuple(int(axis) for axis in np.unravel_index(flat_index, labels.shape))

    @staticmethod
    def _random_crop_start(
        shape: Sequence[int],
        patch_size: Sequence[int],
        rng: random.Random,
    ) -> Tuple[int, int, int]:
        starts: List[int] = []
        for axis_size, patch_axis in zip(shape, patch_size):
            max_start = max(0, int(axis_size) - int(patch_axis))
            starts.append(rng.randint(0, max_start) if max_start else 0)
        return starts[0], starts[1], starts[2]

    @staticmethod
    def _foreground_crop_start(
        shape: Sequence[int],
        patch_size: Sequence[int],
        voxel: Sequence[int],
        rng: random.Random,
    ) -> Tuple[int, int, int]:
        starts: List[int] = []
        for axis_size, patch_axis, voxel_axis in zip(shape, patch_size, voxel):
            axis_size = int(axis_size)
            patch_axis = int(patch_axis)
            voxel_axis = int(voxel_axis)
            max_start = max(0, axis_size - patch_axis)
            if max_start == 0:
                starts.append(0)
                continue
            low = max(0, voxel_axis - patch_axis + 1)
            high = min(voxel_axis, max_start)
            starts.append(rng.randint(low, high) if high > low else low)
        return starts[0], starts[1], starts[2]

    def _sample_patch(
        self,
        labels: np.ndarray,
        occupancy: np.ndarray,
        rng: random.Random,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        input_shape = tuple(int(axis) for axis in labels.shape)
        patch_size = self.patch_size
        if patch_size is None:
            metadata = {
                "branch": "whole_case",
                "start": [0, 0, 0],
                "end": list(input_shape),
                "input_shape": list(input_shape),
                "patch_size": list(input_shape),
                "output_shape": list(input_shape),
                "foreground_oversample_prob": self.foreground_oversample_prob,
            }
            return labels, occupancy, metadata

        needs_crop = any(
            patch_axis < axis_size
            for patch_axis, axis_size in zip(patch_size, input_shape)
        )
        if not needs_crop:
            patch_labels = labels
            patch_occupancy = occupancy
            if tuple(patch_labels.shape) != patch_size:
                patch_labels = fit_array_to_shape(patch_labels, patch_size, pad_value=0)
                patch_occupancy = fit_array_to_shape(patch_occupancy, patch_size, pad_value=False)
            metadata = {
                "branch": "whole_case",
                "start": [0, 0, 0],
                "end": list(input_shape),
                "input_shape": list(input_shape),
                "patch_size": list(patch_size),
                "output_shape": [int(axis) for axis in patch_labels.shape],
                "foreground_oversample_prob": self.foreground_oversample_prob,
            }
            return patch_labels, patch_occupancy, metadata

        foreground_voxel = None
        if (
            self.foreground_oversample_prob > 0.0
            and rng.random() < self.foreground_oversample_prob
        ):
            foreground_voxel = self._sample_foreground_voxel(labels, rng)

        if foreground_voxel is not None:
            branch = "foreground"
            start = self._foreground_crop_start(input_shape, patch_size, foreground_voxel, rng)
        else:
            branch = "random"
            start = self._random_crop_start(input_shape, patch_size, rng)

        end = [
            min(int(axis_size), int(axis_start) + int(patch_axis))
            for axis_size, axis_start, patch_axis in zip(input_shape, start, patch_size)
        ]
        slices = tuple(slice(axis_start, axis_end) for axis_start, axis_end in zip(start, end))
        patch_labels = labels[slices]
        patch_occupancy = occupancy[slices]
        if tuple(patch_labels.shape) != patch_size:
            patch_labels = fit_array_to_shape(patch_labels, patch_size, pad_value=0)
            patch_occupancy = fit_array_to_shape(patch_occupancy, patch_size, pad_value=False)

        metadata = {
            "branch": branch,
            "start": [int(axis) for axis in start],
            "end": [int(axis) for axis in end],
            "input_shape": list(input_shape),
            "patch_size": list(patch_size),
            "output_shape": [int(axis) for axis in patch_labels.shape],
            "foreground_oversample_prob": self.foreground_oversample_prob,
        }
        return patch_labels, patch_occupancy, metadata

    def __getitem__(self, index: int) -> Dict[str, Any]:
        case_path = self.case_files[index]
        case_id = case_path.stem
        case_record = self.presence.get("cases", {}).get(case_id)
        if case_record is None:
            raise KeyError(f"No class_presence record for {case_id}")

        with np.load(case_path) as data:
            labels = np.asarray(data["voxel_labels"], dtype=np.int64)
            occupancy = voxelize_sensor_points(
                np.asarray(data["sensor_pc"], dtype=np.float32),
                labels.shape,
                np.asarray(data["grid_world_min"], dtype=np.float32),
                np.asarray(data["grid_voxel_size"], dtype=np.float32),
            )

        if self.volume_size is not None:
            labels = fit_array_to_shape(labels, self.volume_size, pad_value=0)
            occupancy = fit_array_to_shape(occupancy, self.volume_size, pad_value=False)

        rng = self._prompt_rng(index)
        labels, occupancy, crop_metadata = self._sample_patch(labels, occupancy, rng)
        patch_record = self._patch_case_record(case_record, labels)
        prompt_records = sample_prompts_for_case(
            patch_record,
            self.vocab,
            num_positive=self.num_positive,
            num_negative=self.num_negative,
            min_voxels=self.min_voxels,
            rng=rng,
            prompt_index=self._prompt_records_by_class,
        )
        if not prompt_records:
            raise ValueError(f"No prompts could be sampled for {case_id}")

        prompt_ids = torch.tensor([int(record["index"]) for record in prompt_records], dtype=torch.long)
        target_empty = torch.tensor(
            [bool(record.get("target_empty", False)) for record in prompt_records],
            dtype=torch.bool,
        )

        masks: List[np.ndarray] = []
        for record in prompt_records:
            if bool(record.get("target_empty", False)):
                masks.append(np.zeros_like(labels, dtype=np.float32))
            else:
                masks.append(
                    build_target_mask(labels, record.get("target_class_ids", [])).astype(np.float32)
                )

        return {
            "case_id": case_id,
            "case_path": str(case_path),
            "occupancy": torch.from_numpy(occupancy.astype(np.float32, copy=False)).unsqueeze(0),
            "voxel_labels": torch.from_numpy(labels.astype(np.int64, copy=False)),
            "text_embeddings": self.prompt_embeddings[prompt_ids].clone(),
            "prompt_ids": prompt_ids,
            "prompt_texts": [str(record["text"]) for record in prompt_records],
            "target_class_ids": [list(record.get("target_class_ids", [])) for record in prompt_records],
            "target_empty": target_empty,
            "target_masks": torch.from_numpy(np.stack(masks, axis=0)),
            "crop": crop_metadata,
        }


def voxelize_sensor_points(
    sensor_pc: np.ndarray,
    shape: Sequence[int],
    grid_world_min: np.ndarray,
    grid_voxel_size: np.ndarray,
) -> np.ndarray:
    """Convert surface point coordinates into a binary occupancy grid."""

    shape_tuple = tuple(int(x) for x in shape)
    occupancy = np.zeros(shape_tuple, dtype=bool)
    if sensor_pc.size == 0:
        return occupancy

    voxel_size = np.asarray(grid_voxel_size, dtype=np.float32)
    if voxel_size.ndim == 0:
        voxel_size = np.repeat(voxel_size, 3)
    voxel_size = np.where(voxel_size == 0, 1.0, voxel_size)
    world_min = np.asarray(grid_world_min, dtype=np.float32)
    indices = np.floor((sensor_pc[:, :3] - world_min[:3]) / voxel_size[:3]).astype(np.int64)

    valid = np.ones(indices.shape[0], dtype=bool)
    for axis, axis_size in enumerate(shape_tuple):
        valid &= (indices[:, axis] >= 0) & (indices[:, axis] < axis_size)
    if valid.any():
        valid_indices = indices[valid]
        occupancy[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = True
    return occupancy


def fit_array_to_shape(
    array: np.ndarray,
    output_shape: Sequence[int],
    pad_value: int | float | bool = 0,
) -> np.ndarray:
    """Center-crop or center-pad a 3D array to ``output_shape``."""

    if array.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {array.shape}")
    output_shape = tuple(int(x) for x in output_shape)
    output = np.full(output_shape, pad_value, dtype=array.dtype)

    src_slices = []
    dst_slices = []
    for src_size, dst_size in zip(array.shape, output_shape):
        if src_size >= dst_size:
            src_start = (src_size - dst_size) // 2
            src_slices.append(slice(src_start, src_start + dst_size))
            dst_slices.append(slice(0, dst_size))
        else:
            dst_start = (dst_size - src_size) // 2
            src_slices.append(slice(0, src_size))
            dst_slices.append(slice(dst_start, dst_start + src_size))

    output[tuple(dst_slices)] = array[tuple(src_slices)]
    return output


def prompt_collate_fn(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Collate samples, padding variable prompt counts within each batch."""

    collated: Dict[str, Any] = {}
    for key in ("occupancy", "voxel_labels"):
        collated[key] = torch.stack([item[key] for item in batch], dim=0)

    prompt_lengths = [int(item["prompt_ids"].shape[0]) for item in batch]
    max_prompts = max(prompt_lengths)
    prompt_valid = torch.zeros(len(batch), max_prompts, dtype=torch.bool)
    for batch_index, num_prompts in enumerate(prompt_lengths):
        prompt_valid[batch_index, :num_prompts] = True

    def pad_prompt_tensor(key: str, pad_value: int | float | bool) -> torch.Tensor:
        first = batch[0][key]
        output = first.new_full((len(batch), max_prompts, *first.shape[1:]), pad_value)
        for batch_index, item in enumerate(batch):
            tensor = item[key]
            num_prompts = int(tensor.shape[0])
            output[batch_index, :num_prompts] = tensor
        return output

    collated["text_embeddings"] = pad_prompt_tensor("text_embeddings", 0.0)
    collated["prompt_ids"] = pad_prompt_tensor("prompt_ids", -1)
    collated["target_empty"] = pad_prompt_tensor("target_empty", True)
    collated["target_masks"] = pad_prompt_tensor("target_masks", 0.0)
    collated["prompt_valid"] = prompt_valid
    for key in ("case_id", "case_path", "prompt_texts", "target_class_ids", "crop"):
        collated[key] = [item[key] for item in batch]
    return collated


__all__ = [
    "HyperBodyPromptDataset",
    "fit_array_to_shape",
    "prompt_collate_fn",
    "voxelize_sensor_points",
]
