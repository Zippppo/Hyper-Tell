"""Metrics for prompt-conditioned binary mask prediction."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def compute_prompt_metrics(
    logits: Tensor,
    targets: Tensor,
    target_empty: Tensor | None = None,
    prompt_valid: Tensor | None = None,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Compute foreground Dice and negative prompt false-positive rate."""

    if logits.shape != targets.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} != targets {tuple(targets.shape)}")
    predictions = (torch.sigmoid(logits) > threshold).float()
    targets = targets.float()
    flat_predictions = predictions.flatten(start_dim=2)
    flat_targets = targets.flatten(start_dim=2)
    expected_prompt_shape = tuple(flat_targets.shape[:2])
    target_has_foreground = flat_targets.sum(dim=-1) > 0

    if target_empty is None:
        target_empty = ~target_has_foreground
    else:
        if tuple(target_empty.shape) != expected_prompt_shape:
            raise ValueError(
                f"target_empty shape {tuple(target_empty.shape)} != expected "
                f"{expected_prompt_shape}"
            )
        target_empty = target_empty.to(device=logits.device, dtype=torch.bool)
    if prompt_valid is None:
        prompt_valid = torch.ones_like(target_empty, dtype=torch.bool)
    else:
        if tuple(prompt_valid.shape) != expected_prompt_shape:
            raise ValueError(
                f"prompt_valid shape {tuple(prompt_valid.shape)} != expected "
                f"{expected_prompt_shape}"
            )
        prompt_valid = prompt_valid.to(device=logits.device, dtype=torch.bool)

    positive = prompt_valid & ~target_empty & target_has_foreground
    if positive.any():
        pred_pos = flat_predictions[positive]
        target_pos = flat_targets[positive]
        intersection = (pred_pos * target_pos).sum(dim=-1)
        denominator = pred_pos.sum(dim=-1) + target_pos.sum(dim=-1)
        dice_values = (2.0 * intersection + eps) / (denominator + eps)
        dice_sum = dice_values.sum().item()
        dice_count = float(dice_values.numel())
        dice = dice_sum / dice_count
    else:
        dice_sum = 0.0
        dice_count = 0.0
        dice = float("nan")

    negative = prompt_valid & target_empty
    if negative.any():
        negative_fp_rate = flat_predictions[negative].mean().item()
    else:
        negative_fp_rate = 0.0

    return {
        "foreground_mean_dice": float(dice),
        "foreground_dice_sum": float(dice_sum),
        "foreground_dice_count": float(dice_count),
        "negative_fp_rate": float(negative_fp_rate),
    }


__all__ = ["compute_prompt_metrics"]
