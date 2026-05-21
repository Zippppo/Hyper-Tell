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
    load_label_vocab,
    read_json,
    sample_prompts_for_case,
    validate_prompt_embedding_cache,
)


class HyperBodyPromptDataset(Dataset):
    """Read ``Body-Tell/Dataset/voxel_data`` as prompt-conditioned masks."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        volume_size: Optional[Sequence[int]] = None,
        num_positive: int = 2,
        num_negative: int = 1,
        min_voxels: int = 1,
        seed: int = 0,
        vocab_path: str | Path | None = None,
        split_path: str | Path | None = None,
        presence_path: str | Path | None = None,
        embedding_cache_path: str | Path | None = None,
        strict_embedding_cache: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.volume_size = tuple(int(x) for x in volume_size) if volume_size else None
        self.num_positive = int(num_positive)
        self.num_negative = int(num_negative)
        self.min_voxels = int(min_voxels)
        self.seed = int(seed)

        self.vocab_path = Path(vocab_path) if vocab_path else self.root / "configs" / "label_vocab.json"
        self.split_path = (
            Path(split_path) if split_path else self.root / "Dataset" / "dataset_split.json"
        )
        self.presence_path = (
            Path(presence_path)
            if presence_path
            else self.root / "artifacts" / "data_stats" / "class_presence.json"
        )
        self.embedding_cache_path = (
            Path(embedding_cache_path)
            if embedding_cache_path
            else self.root / "artifacts" / "text_embeddings" / "prompt_embeddings.pt"
        )
        self.voxel_dir = self.root / "Dataset" / "voxel_data"

        self.vocab = load_label_vocab(self.vocab_path)
        self.presence = load_class_presence(self.presence_path)
        split_data = read_json(self.split_path)
        if split not in split_data:
            raise ValueError(f"Unknown split {split!r}; expected one of train, val, test")
        self.case_files = [self.voxel_dir / str(name) for name in split_data[split]]
        if not self.case_files:
            raise ValueError(f"Split {split!r} contains no cases")

        validation = validate_prompt_embedding_cache(
            self.embedding_cache_path,
            self.vocab_path,
            strict=strict_embedding_cache,
        )
        self.embedding_warnings = validation["warnings"]
        self.embedding_cache = torch.load(
            self.embedding_cache_path,
            map_location="cpu",
            weights_only=False,
        )
        self.prompt_embeddings = self.embedding_cache["embeddings"].float().contiguous()

    def __len__(self) -> int:
        return len(self.case_files)

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

        rng = random.Random(self.seed + index)
        prompt_records = sample_prompts_for_case(
            case_record,
            self.vocab,
            num_positive=self.num_positive,
            num_negative=self.num_negative,
            min_voxels=self.min_voxels,
            rng=rng,
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
    for key in ("case_id", "case_path", "prompt_texts", "target_class_ids"):
        collated[key] = [item[key] for item in batch]
    return collated


__all__ = [
    "HyperBodyPromptDataset",
    "fit_array_to_shape",
    "prompt_collate_fn",
    "voxelize_sensor_points",
]
