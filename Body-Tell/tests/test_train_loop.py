from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from body_tell.data.dataset import HyperBodyPromptDataset, prompt_collate_fn
from body_tell.data.vocabulary import EMBEDDING_DIM, file_sha256, flatten_prompt_records
from body_tell.losses.prompt_losses import PromptSegmentationLoss
from body_tell.metrics.prompt_metrics import compute_prompt_metrics
from body_tell.models.voxtell_body_model import VoxTellBodyConfig, VoxTellBodyModel


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
        [[0.2, 0.2, 0.2], [1.1, 2.1, 1.1], [3.6, 4.6, 5.6]],
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
    embeddings = torch.randn(len(records), EMBEDDING_DIM)
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


def test_mini_overfit_loss_decreases(tmp_path: Path) -> None:
    """A tiny model trained on 1 sample for several steps must show loss decrease."""
    torch.manual_seed(42)
    root = _make_tiny_body_tell_root(tmp_path)

    dataset = HyperBodyPromptDataset(
        root=root,
        split="train",
        volume_size=(4, 5, 6),
        num_positive=2,
        num_negative=1,
        seed=0,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, collate_fn=prompt_collate_fn
    )

    model = VoxTellBodyModel(
        VoxTellBodyConfig(
            input_channels=1,
            encoder_channels=(4, 8, 16),
            text_embedding_dim=EMBEDDING_DIM,
            query_dim=16,
            text_projection_hidden_dim=16,
            transformer_num_heads=4,
            transformer_layers=1,
            transformer_feedforward_dim=32,
            num_maskformer_stages=3,
            num_heads=2,
            deep_supervision=False,
        )
    )
    model.train()

    criterion = PromptSegmentationLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for step in range(20):
        for batch in loader:
            occupancy = batch["occupancy"]
            text_embeddings = batch["text_embeddings"]
            target_masks = batch["target_masks"]

            logits = model(occupancy, text_embeddings)
            result = criterion(logits, target_masks)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )
    assert losses[-1] < 0.85 * losses[0], (
        f"Loss did not decrease enough: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


def test_mini_overfit_dice_improves(tmp_path: Path) -> None:
    """After overfitting on 1 sample, foreground Dice should be non-trivial."""
    torch.manual_seed(42)
    root = _make_tiny_body_tell_root(tmp_path)

    dataset = HyperBodyPromptDataset(
        root=root,
        split="train",
        volume_size=(4, 5, 6),
        num_positive=2,
        num_negative=1,
        seed=0,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, collate_fn=prompt_collate_fn
    )

    model = VoxTellBodyModel(
        VoxTellBodyConfig(
            input_channels=1,
            encoder_channels=(4, 8, 16),
            text_embedding_dim=EMBEDDING_DIM,
            query_dim=16,
            text_projection_hidden_dim=16,
            transformer_num_heads=4,
            transformer_layers=1,
            transformer_feedforward_dim=32,
            num_maskformer_stages=3,
            num_heads=2,
            deep_supervision=False,
        )
    )
    model.train()

    criterion = PromptSegmentationLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(40):
        for batch in loader:
            logits = model(batch["occupancy"], batch["text_embeddings"])
            loss = criterion(logits, batch["target_masks"])["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["occupancy"], batch["text_embeddings"])
            metrics = compute_prompt_metrics(
                logits, batch["target_masks"], target_empty=batch["target_empty"]
            )

    assert metrics["foreground_mean_dice"] > 0.3, (
        f"Dice too low after overfit: {metrics['foreground_mean_dice']:.4f}"
    )
