from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import train as train_module
from body_tell.data.dataset import HyperBodyPromptDataset, prompt_collate_fn
from body_tell.data.vocabulary import EMBEDDING_DIM, file_sha256, flatten_prompt_records
from body_tell.losses.prompt_losses import PromptSegmentationLoss
from body_tell.metrics.prompt_metrics import compute_prompt_metrics
from body_tell.models.voxtell_body_model import VoxTellBodyConfig, VoxTellBodyModel


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _make_train_loop_batch() -> dict[str, torch.Tensor]:
    target_masks = torch.zeros((1, 2, 4, 4, 4), dtype=torch.float32)
    target_masks[:, 0, 1:3, 1:3, 1:3] = 1.0
    return {
        "occupancy": torch.ones((1, 1, 4, 4, 4), dtype=torch.float32),
        "text_embeddings": torch.ones((1, 2, EMBEDDING_DIM), dtype=torch.float32),
        "target_masks": target_masks,
        "target_empty": torch.tensor([[False, True]]),
        "prompt_valid": torch.tensor([[True, True]]),
    }


class _DeepSupervisionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        occupancy: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> list[torch.Tensor]:
        batch_size = occupancy.shape[0]
        num_prompts = text_embeddings.shape[1]
        spatial_shape = occupancy.shape[2:]
        primary = self.weight * torch.ones(
            (batch_size, num_prompts, *spatial_shape),
            device=occupancy.device,
            dtype=occupancy.dtype,
        )
        aux = self.weight * torch.ones(
            (batch_size, num_prompts, 2, 2, 2),
            device=occupancy.device,
            dtype=occupancy.dtype,
        )
        return [primary, aux]


class _DeepSupervisionCriterion:
    def __init__(self) -> None:
        self.seen_output_shapes: list[tuple[int, ...]] | None = None

    def __call__(
        self,
        model_outputs: object,
        target_masks: torch.Tensor,
        prompt_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        assert isinstance(model_outputs, list)
        assert len(model_outputs) == 2
        self.seen_output_shapes = [tuple(output.shape) for output in model_outputs]
        primary = model_outputs[0]
        loss = (primary - target_masks).square().mean()
        return {"loss": loss, "bce_loss": loss.detach(), "dice_loss": loss.detach()}


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
        "shape_summary": {"recommended_volume_size": [8, 8, 8]},
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


def test_build_dataset_forwards_s2i_path_config(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyDataset:
        pass

    def fake_dataset(**kwargs):
        captured.update(kwargs)
        return DummyDataset()

    monkeypatch.setattr(train_module, "HyperBodyPromptDataset", fake_dataset)

    cfg = {
        "data": {
            "root": "Body-Tell",
            "voxel_dir": "S2I-Dataset-70cls/data",
            "split_path": "S2I-Dataset-70cls/dataset_split.json",
            "presence_path": "S2I-Dataset-70cls/class_presence.json",
            "vocab_path": "configs/label_vocab.json",
            "embedding_cache_path": "artifacts/text_embeddings/prompt_embeddings.pt",
            "strict_embedding_cache": False,
            "volume_size": [4, 5, 6],
            "patch_size": [4, 5, 3],
            "foreground_oversample_prob": 0.85,
            "num_positive": 2,
            "num_negative": 1,
            "min_voxels": 1,
        }
    }

    dataset = train_module.build_dataset(cfg, split="train")

    assert isinstance(dataset, DummyDataset)
    assert captured == {
        "root": "Body-Tell",
        "split": "train",
        "volume_size": (4, 5, 6),
        "patch_size": (4, 5, 3),
        "foreground_oversample_prob": 0.85,
        "num_positive": 2,
        "num_negative": 1,
        "min_voxels": 1,
        "voxel_dir": "S2I-Dataset-70cls/data",
        "split_path": "S2I-Dataset-70cls/dataset_split.json",
        "presence_path": "S2I-Dataset-70cls/class_presence.json",
        "vocab_path": "configs/label_vocab.json",
        "embedding_cache_path": "artifacts/text_embeddings/prompt_embeddings.pt",
        "strict_embedding_cache": False,
    }


def test_phase1_canonical_config_targets_cropped_s2i_package() -> None:
    canonical_config = Path("Body-Tell/configs/phase1_voxtell_aligned.yaml")
    cfg = train_module.load_config(canonical_config)

    assert cfg["data"]["voxel_dir"] == "S2I-Dataset-70cls/data"
    assert cfg["data"]["split_path"] == "S2I-Dataset-70cls/dataset_split.json"
    assert cfg["data"]["presence_path"] == "S2I-Dataset-70cls/class_presence.json"
    assert cfg["data"]["volume_size"] == [128, 128, 256]
    assert cfg["data"]["patch_size"] == [128, 128, 128]
    assert cfg["data"]["foreground_oversample_prob"] == 0.85


def test_train_one_epoch_accepts_deep_supervision_outputs(monkeypatch) -> None:
    batch = _make_train_loop_batch()
    loader = torch.utils.data.DataLoader([batch], batch_size=None)
    model = _DeepSupervisionModel()
    criterion = _DeepSupervisionCriterion()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    metrics_seen: dict[str, tuple[int, ...]] = {}

    def fake_compute_prompt_metrics(
        logits: torch.Tensor,
        targets: torch.Tensor,
        target_empty: torch.Tensor | None = None,
        prompt_valid: torch.Tensor | None = None,
    ) -> dict[str, float]:
        assert isinstance(logits, torch.Tensor)
        metrics_seen["logits"] = tuple(logits.shape)
        metrics_seen["targets"] = tuple(targets.shape)
        return {"foreground_mean_dice": 1.0, "negative_fp_rate": 0.0}

    monkeypatch.setattr(train_module, "compute_prompt_metrics", fake_compute_prompt_metrics)

    result = train_module.train_one_epoch(
        model,
        loader,
        criterion,  # type: ignore[arg-type]
        optimizer,
        torch.device("cpu"),
        epoch=1,
    )

    assert criterion.seen_output_shapes == [(1, 2, 4, 4, 4), (1, 2, 2, 2, 2)]
    assert metrics_seen == {"logits": (1, 2, 4, 4, 4), "targets": (1, 2, 4, 4, 4)}
    assert result["foreground_mean_dice"] == 1.0


def test_evaluate_accepts_deep_supervision_outputs(monkeypatch) -> None:
    batch = _make_train_loop_batch()
    loader = torch.utils.data.DataLoader([batch], batch_size=None)
    model = _DeepSupervisionModel()
    criterion = _DeepSupervisionCriterion()
    metrics_seen: dict[str, tuple[int, ...]] = {}

    def fake_compute_prompt_metrics(
        logits: torch.Tensor,
        targets: torch.Tensor,
        target_empty: torch.Tensor | None = None,
        prompt_valid: torch.Tensor | None = None,
    ) -> dict[str, float]:
        assert isinstance(logits, torch.Tensor)
        metrics_seen["logits"] = tuple(logits.shape)
        metrics_seen["targets"] = tuple(targets.shape)
        return {"foreground_mean_dice": 0.5, "negative_fp_rate": 0.25}

    monkeypatch.setattr(train_module, "compute_prompt_metrics", fake_compute_prompt_metrics)

    result = train_module.evaluate(
        model,
        loader,
        criterion,  # type: ignore[arg-type]
        torch.device("cpu"),
    )

    assert criterion.seen_output_shapes == [(1, 2, 4, 4, 4), (1, 2, 2, 2, 2)]
    assert metrics_seen == {"logits": (1, 2, 4, 4, 4), "targets": (1, 2, 4, 4, 4)}
    assert result["loss"] == pytest.approx(result["bce_loss"])
    assert result["loss"] == pytest.approx(result["dice_loss"])
    assert result["foreground_mean_dice"] == 0.5
    assert result["negative_fp_rate"] == 1.0


def test_primary_logits_for_metrics_rejects_shape_mismatch() -> None:
    targets = torch.zeros((1, 2, 4, 4, 4))
    model_outputs = [torch.zeros((1, 2, 2, 2, 2)), torch.zeros((1, 2, 1, 1, 1))]

    with pytest.raises(ValueError, match="primary logits shape"):
        train_module._primary_logits_for_metrics(model_outputs, targets)


def test_ddp_metric_reduce_helper(monkeypatch) -> None:
    local_totals = {
        "loss_sum": 2.0,
        "bce_loss_sum": 1.0,
        "dice_loss_sum": 1.5,
        "foreground_dice_sum": 0.4,
        "foreground_dice_count": 1.0,
        "negative_fp_sum": 3.0,
        "negative_fp_count": 10.0,
        "batch_count": 2.0,
    }
    other_rank_totals = torch.tensor(
        [6.0, 5.0, 4.5, 1.6, 3.0, 1.0, 10.0, 1.0],
        dtype=torch.float64,
    )
    all_reduce_calls = []

    def fake_all_reduce(tensor: torch.Tensor, op: object | None = None) -> None:
        all_reduce_calls.append((tensor.clone(), op))
        tensor += other_rank_totals.to(device=tensor.device)

    monkeypatch.setattr(train_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(train_module.dist, "all_reduce", fake_all_reduce)

    reduced = train_module._reduce_ddp_metric_totals(
        local_totals,
        torch.device("cpu"),
    )
    metrics = train_module._metric_averages_from_totals(reduced)

    assert len(all_reduce_calls) == 1
    assert metrics == {
        "loss": pytest.approx(8.0 / 3.0),
        "bce_loss": pytest.approx(2.0),
        "dice_loss": pytest.approx(2.0),
        "foreground_mean_dice": pytest.approx(0.5),
        "negative_fp_rate": pytest.approx(0.2),
    }
    assert local_totals["foreground_dice_sum"] / local_totals["foreground_dice_count"] == 0.4
    assert metrics["foreground_mean_dice"] > 0.45


def test_mini_overfit_loss_decreases(tmp_path: Path) -> None:
    """A tiny model trained on 1 sample for several steps must show loss decrease."""
    torch.manual_seed(42)
    root = _make_tiny_body_tell_root(tmp_path)

    dataset = HyperBodyPromptDataset(
        root=root,
        split="train",
        volume_size=(8, 8, 8),
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
        volume_size=(8, 8, 8),
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
