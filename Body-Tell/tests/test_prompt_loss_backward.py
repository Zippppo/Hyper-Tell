from __future__ import annotations

import torch

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
