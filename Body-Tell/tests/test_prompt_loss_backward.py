from __future__ import annotations

import math

import torch
import torch.nn as nn

import train as train_module

from body_tell.losses.prompt_losses import PromptSegmentationLoss
from body_tell.metrics.prompt_metrics import compute_prompt_metrics


def test_prompt_segmentation_loss_backpropagates() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 6, 5, 4, requires_grad=True)
    targets = (torch.rand(2, 3, 6, 5, 4) > 0.7).float()
    targets[:, 2] = 0.0

    criterion = PromptSegmentationLoss(bce_weight=0.5, dice_weight=0.5)
    result = criterion(logits, targets)

    assert set(result) == {"loss", "bce_loss", "dice_loss"}
    assert result["loss"].ndim == 0
    result["loss"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_prompt_metrics_report_positive_dice_and_negative_fp_rate() -> None:
    logits = torch.full((1, 3, 4, 4, 4), -8.0)
    targets = torch.zeros_like(logits)
    targets[:, 0, 1:3, 1:3, 1:3] = 1.0
    logits[:, 0, 1:3, 1:3, 1:3] = 8.0
    logits[:, 2, 0, 0, 0] = 8.0
    target_empty = torch.tensor([[False, True, True]])

    metrics = compute_prompt_metrics(logits, targets, target_empty=target_empty)

    assert metrics["foreground_mean_dice"] > 0.99
    assert metrics["negative_fp_rate"] == 1.0 / (2 * 4 * 4 * 4)


def test_prompt_loss_ignores_invalid_padding_prompts() -> None:
    torch.manual_seed(1)
    logits = torch.randn(1, 3, 4, 4, 4, requires_grad=True)
    targets = (torch.rand(1, 3, 4, 4, 4) > 0.65).float()
    prompt_valid = torch.tensor([[True, True, False]])
    with torch.no_grad():
        logits[:, 2] = 20.0
    targets[:, 2] = 0.0

    criterion = PromptSegmentationLoss(bce_weight=0.5, dice_weight=0.5)
    masked = criterion(logits, targets, prompt_valid=prompt_valid)
    sliced = criterion(logits[:, :2], targets[:, :2])

    assert torch.allclose(masked["loss"], sliced["loss"])
    masked["loss"].backward()
    assert logits.grad is not None
    assert logits.grad[:, 2].abs().sum().item() == 0.0


def test_prompt_metrics_ignore_invalid_padding_prompts() -> None:
    logits = torch.full((1, 3, 4, 4, 4), -8.0)
    targets = torch.zeros_like(logits)
    target_empty = torch.tensor([[False, True, True]])
    prompt_valid = torch.tensor([[True, True, False]])

    targets[:, 0, 1:3, 1:3, 1:3] = 1.0
    logits[:, 0, 1:3, 1:3, 1:3] = 8.0
    logits[:, 2] = 8.0

    metrics = compute_prompt_metrics(
        logits,
        targets,
        target_empty=target_empty,
        prompt_valid=prompt_valid,
    )

    assert metrics["foreground_mean_dice"] > 0.99
    assert metrics["negative_fp_rate"] == 0.0


def test_prompt_metrics_nan_safe_foreground_dice_aggregation() -> None:
    positive_targets = torch.zeros((1, 1, 4, 4, 4), dtype=torch.float32)
    positive_targets[:, :, 1:3, 1:3, 1:3] = 1.0
    positive_logits = torch.full_like(positive_targets, -8.0)
    positive_logits[:, :, 1:3, 1:3, 1:3] = 8.0

    negative_targets = torch.zeros_like(positive_targets)
    negative_logits = torch.full_like(positive_targets, -8.0)
    negative_logits[:, :, 0, 0, 0] = 8.0

    class SequencedLogitModel(nn.Module):
        def __init__(self, logits_sequence: list[torch.Tensor]) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.0))
            self.logits_sequence = logits_sequence
            self.index = 0

        def forward(
            self,
            occupancy: torch.Tensor,
            text_embeddings: torch.Tensor,
        ) -> torch.Tensor:
            logits = self.logits_sequence[self.index].to(
                device=occupancy.device,
                dtype=occupancy.dtype,
            )
            self.index += 1
            return logits + self.weight * 0.0

    def make_batch(target_masks: torch.Tensor, target_empty: bool) -> dict[str, torch.Tensor]:
        return {
            "occupancy": torch.ones((1, 1, 4, 4, 4), dtype=torch.float32),
            "text_embeddings": torch.ones((1, 1, 2), dtype=torch.float32),
            "target_masks": target_masks,
            "target_empty": torch.tensor([[target_empty]]),
            "prompt_valid": torch.tensor([[True]]),
        }

    loader = torch.utils.data.DataLoader(
        [
            make_batch(positive_targets, target_empty=False),
            make_batch(negative_targets, target_empty=True),
        ],
        batch_size=None,
    )
    model = SequencedLogitModel([positive_logits, negative_logits])
    criterion = PromptSegmentationLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    metrics = train_module.train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        torch.device("cpu"),
        epoch=1,
    )

    assert math.isfinite(metrics["foreground_mean_dice"])
    assert metrics["foreground_mean_dice"] > 0.99
    assert metrics["negative_fp_rate"] > 0.0
