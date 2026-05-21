from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from body_tell.data.dataset import HyperBodyPromptDataset, prompt_collate_fn
from body_tell.data.vocabulary import EMBEDDING_DIM, file_sha256, flatten_prompt_records


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _make_tiny_body_tell_root(tmp_path: Path) -> Path:
    root = tmp_path / "Body-Tell"
    voxel_dir = root / "Dataset" / "voxel_data"
    voxel_dir.mkdir(parents=True)

    labels = np.zeros((4, 5, 6), dtype=np.uint8)
    labels[1:3, 2:4, 1:4] = 1
    labels[3, 1:4, 3:5] = 2
    sensor_pc = np.array(
        [
            [0.2, 0.2, 0.2],
            [1.1, 2.1, 1.1],
            [3.6, 4.6, 5.6],
            [9.0, 9.0, 9.0],
        ],
        dtype=np.float32,
    )
    np.savez(
        voxel_dir / "CASE_0001.npz",
        sensor_pc=sensor_pc,
        voxel_labels=labels,
        grid_world_min=np.zeros(3, dtype=np.float32),
        grid_world_max=np.array([4, 5, 6], dtype=np.float32),
        grid_voxel_size=np.ones(3, dtype=np.float32),
        grid_occ_size=np.array([4, 5, 6], dtype=np.int32),
    )

    dataset_info = {
        "num_classes": 4,
        "class_names": ["inside_body_empty", "liver", "spleen", "pancreas"],
    }
    split = {
        "split_info": {"seed": 7},
        "train": ["CASE_0001.npz"],
        "val": [],
        "test": [],
    }
    vocab = {
        "version": "phase0-2026-05-20",
        "source_dataset_info": "Body-Tell/Dataset/dataset_info.json",
        "language": "en",
        "text_encoder": "Qwen/Qwen3-Embedding-4B",
        "instruction": "test instruction",
        "classes": [
            {
                "id": 0,
                "source_name": "inside_body_empty",
                "canonical": "inside body empty space",
                "prompts": ["inside body empty space"],
                "train_as_positive": False,
                "eval_as_foreground": False,
            },
            {
                "id": 1,
                "source_name": "liver",
                "canonical": "liver",
                "prompts": ["liver"],
                "train_as_positive": True,
                "eval_as_foreground": True,
            },
            {
                "id": 2,
                "source_name": "spleen",
                "canonical": "spleen",
                "prompts": ["spleen"],
                "train_as_positive": True,
                "eval_as_foreground": True,
            },
            {
                "id": 3,
                "source_name": "pancreas",
                "canonical": "pancreas",
                "prompts": ["pancreas"],
                "train_as_positive": True,
                "eval_as_foreground": True,
            },
        ],
        "aggregates": [],
        "ignore_as_positive": [0],
    }
    presence = {
        "version": "phase0-2026-05-20",
        "num_cases": 1,
        "num_classes": 4,
        "shape_summary": {"recommended_volume_size": [4, 5, 6]},
        "cases": {
            "CASE_0001": {
                "filename": "CASE_0001.npz",
                "split": "train",
                "shape": [4, 5, 6],
                "present_class_ids": [0, 1, 2],
                "foreground_present_class_ids": [1, 2],
                "voxel_counts": {"0": 102, "1": 12, "2": 6},
            }
        },
        "classes": {},
    }

    _write_json(root / "Dataset" / "dataset_info.json", dataset_info)
    _write_json(root / "Dataset" / "dataset_split.json", split)
    _write_json(root / "configs" / "label_vocab.json", vocab)
    _write_json(root / "artifacts" / "data_stats" / "class_presence.json", presence)

    records = flatten_prompt_records(vocab)
    embeddings = torch.arange(len(records) * EMBEDDING_DIM, dtype=torch.float32).view(
        len(records), EMBEDDING_DIM
    )
    cache = {
        "model_name": "Qwen/Qwen3-Embedding-4B",
        "embedding_dim": EMBEDDING_DIM,
        "instruction": "test instruction",
        "vocab_version": vocab["version"],
        "vocab_hash": file_sha256(root / "configs" / "label_vocab.json"),
        "num_prompts": len(records),
        "prompt_records": records,
        "is_qwen_cache": True,
        "embeddings": embeddings,
    }
    cache_path = root / "artifacts" / "text_embeddings" / "prompt_embeddings.pt"
    cache_path.parent.mkdir(parents=True)
    torch.save(cache, cache_path)
    return root


def test_prompt_dataset_builds_binary_masks_and_negative_prompt(tmp_path: Path) -> None:
    root = _make_tiny_body_tell_root(tmp_path)

    dataset = HyperBodyPromptDataset(
        root=root,
        split="train",
        volume_size=(4, 5, 6),
        num_positive=2,
        num_negative=1,
        seed=3,
    )

    sample = dataset[0]

    assert sample["case_id"] == "CASE_0001"
    assert sample["occupancy"].shape == (1, 4, 5, 6)
    assert sample["occupancy"].dtype == torch.float32
    assert sample["target_masks"].shape == (3, 4, 5, 6)
    assert sample["text_embeddings"].shape == (3, EMBEDDING_DIM)
    assert sample["prompt_ids"].tolist() == [1, 2, 3]
    assert sample["target_empty"].tolist() == [False, False, True]
    assert sample["target_masks"][0].sum().item() == 12
    assert sample["target_masks"][1].sum().item() == 6
    assert sample["target_masks"][2].sum().item() == 0
    assert sample["occupancy"].sum().item() == 3


def test_prompt_dataset_pads_to_requested_volume_size(tmp_path: Path) -> None:
    root = _make_tiny_body_tell_root(tmp_path)

    dataset = HyperBodyPromptDataset(
        root=root,
        split="train",
        volume_size=(6, 7, 8),
        num_positive=1,
        num_negative=0,
        seed=11,
    )

    sample = dataset[0]

    assert sample["occupancy"].shape == (1, 6, 7, 8)
    assert sample["target_masks"].shape == (1, 6, 7, 8)
    assert sample["target_masks"].sum().item() in {6, 12}


def test_prompt_collate_pads_mixed_prompt_counts() -> None:
    spatial_shape = (2, 3, 4)
    base = {
        "occupancy": torch.ones(1, *spatial_shape),
        "voxel_labels": torch.zeros(spatial_shape, dtype=torch.long),
        "case_path": "case.npz",
    }
    sample_two = {
        **base,
        "case_id": "CASE_TWO",
        "text_embeddings": torch.ones(2, 5),
        "prompt_ids": torch.tensor([11, 12]),
        "prompt_texts": ["liver", "spleen"],
        "target_class_ids": [[1], [2]],
        "target_empty": torch.tensor([False, False]),
        "target_masks": torch.ones(2, *spatial_shape),
    }
    sample_three = {
        **base,
        "case_id": "CASE_THREE",
        "text_embeddings": torch.full((3, 5), 2.0),
        "prompt_ids": torch.tensor([21, 22, 23]),
        "prompt_texts": ["liver", "spleen", "pancreas"],
        "target_class_ids": [[1], [2], [3]],
        "target_empty": torch.tensor([False, False, True]),
        "target_masks": torch.full((3, *spatial_shape), 2.0),
    }

    batch = prompt_collate_fn([sample_two, sample_three])

    assert batch["text_embeddings"].shape == (2, 3, 5)
    assert batch["prompt_ids"].tolist() == [[11, 12, -1], [21, 22, 23]]
    assert batch["target_empty"].tolist() == [[False, False, True], [False, False, True]]
    assert batch["target_masks"].shape == (2, 3, *spatial_shape)
    assert batch["prompt_valid"].tolist() == [[True, True, False], [True, True, True]]
    assert torch.equal(batch["text_embeddings"][0, 2], torch.zeros(5))
    assert batch["target_masks"][0, 2].sum().item() == 0
