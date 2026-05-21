"""Metrics for prompt-conditioned binary mask prediction."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def compute_prompt_metrics(
    logits: Tensor,
    targets: Tensor,
    target_empty: Tensor | None = None,
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

    if target_empty is None:
        target_empty = flat_targets.sum(dim=-1) == 0
    else:
        target_empty = target_empty.to(device=logits.device, dtype=torch.bool)

    positive = ~target_empty & (flat_targets.sum(dim=-1) > 0)
    if positive.any():
        pred_pos = flat_predictions[positive]
        target_pos = flat_targets[positive]
        intersection = (pred_pos * target_pos).sum(dim=-1)
        denominator = pred_pos.sum(dim=-1) + target_pos.sum(dim=-1)
        dice = ((2.0 * intersection + eps) / (denominator + eps)).mean().item()
    else:
        dice = float("nan")

    if target_empty.any():
        negative_fp_rate = flat_predictions[target_empty].mean().item()
    else:
        negative_fp_rate = 0.0

    return {
        "foreground_mean_dice": float(dice),
        "negative_fp_rate": float(negative_fp_rate),
    }


__all__ = ["compute_prompt_metrics"]

