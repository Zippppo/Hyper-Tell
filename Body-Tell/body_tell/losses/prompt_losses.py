"""Loss functions for prompt-conditioned binary mask prediction."""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class BinaryDiceLoss(nn.Module):
    """Dice loss over non-empty target prompts."""

    def __init__(self, eps: float = 1e-6, include_empty: bool = False) -> None:
        super().__init__()
        self.eps = float(eps)
        self.include_empty = bool(include_empty)

    def forward(
        self,
        probabilities: Tensor,
        targets: Tensor,
        prompt_valid: Tensor | None = None,
    ) -> Tensor:
        probabilities = probabilities.float()
        targets = targets.float()
        probs_flat = probabilities.flatten(start_dim=2)
        targets_flat = targets.flatten(start_dim=2)
        target_has_fg = targets_flat.sum(dim=-1) > 0
        if self.include_empty:
            valid = torch.ones_like(target_has_fg, dtype=torch.bool)
        else:
            valid = target_has_fg
        if prompt_valid is not None:
            prompt_valid = _validate_prompt_valid(prompt_valid, target_has_fg, probabilities.device)
            valid = valid & prompt_valid
        if not valid.any():
            return probabilities.sum() * 0.0

        probs_flat = probs_flat[valid]
        targets_flat = targets_flat[valid]
        intersection = (probs_flat * targets_flat).sum(dim=-1)
        denominator = probs_flat.sum(dim=-1) + targets_flat.sum(dim=-1)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - dice.mean()


class PromptSegmentationLoss(nn.Module):
    """BCEWithLogits plus binary Dice, with optional deep supervision support."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        deep_supervision_weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.deep_supervision_weights = list(deep_supervision_weights or [])
        self.dice_loss = BinaryDiceLoss()

    def forward(
        self,
        logits: Tensor | Sequence[Tensor],
        targets: Tensor,
        prompt_valid: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        if isinstance(logits, (list, tuple)):
            return self._forward_deep_supervision(list(logits), targets, prompt_valid=prompt_valid)
        return self._loss_for_scale(logits, targets, prompt_valid=prompt_valid)

    def _forward_deep_supervision(
        self,
        logits: List[Tensor],
        targets: Tensor,
        prompt_valid: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        if not logits:
            raise ValueError("deep supervision logits list is empty")
        weights = self.deep_supervision_weights
        if not weights:
            weights = [1.0 / (2**idx) for idx in range(len(logits))]
        if len(weights) < len(logits):
            raise ValueError("deep_supervision_weights shorter than logits list")

        total = None
        bce_total = None
        dice_total = None
        normalizer = float(sum(weights[: len(logits)]))
        for weight, scale_logits in zip(weights, logits):
            scale_targets = resize_targets_like(targets, scale_logits)
            scale_loss = self._loss_for_scale(
                scale_logits,
                scale_targets,
                prompt_valid=prompt_valid,
            )
            weighted = float(weight) / normalizer
            total = scale_loss["loss"] * weighted if total is None else total + scale_loss["loss"] * weighted
            bce_total = (
                scale_loss["bce_loss"] * weighted
                if bce_total is None
                else bce_total + scale_loss["bce_loss"] * weighted
            )
            dice_total = (
                scale_loss["dice_loss"] * weighted
                if dice_total is None
                else dice_total + scale_loss["dice_loss"] * weighted
            )
        return {"loss": total, "bce_loss": bce_total, "dice_loss": dice_total}

    def _loss_for_scale(
        self,
        logits: Tensor,
        targets: Tensor,
        prompt_valid: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        if logits.shape != targets.shape:
            raise ValueError(f"logits shape {tuple(logits.shape)} != targets {tuple(targets.shape)}")
        targets = targets.float()
        prompt_valid = _validate_prompt_valid(prompt_valid, logits, logits.device)

        bce_per_voxel = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if prompt_valid is None:
            bce = bce_per_voxel.mean()
        elif prompt_valid.any():
            view_shape = (*prompt_valid.shape, *([1] * (logits.ndim - prompt_valid.ndim)))
            valid_mask = prompt_valid.reshape(view_shape)
            denominator = valid_mask.expand_as(bce_per_voxel).sum().clamp_min(1)
            bce = (bce_per_voxel * valid_mask).sum() / denominator
        else:
            bce = logits.sum() * 0.0

        dice = self.dice_loss(torch.sigmoid(logits), targets, prompt_valid=prompt_valid)
        loss = self.bce_weight * bce + self.dice_weight * dice
        return {"loss": loss, "bce_loss": bce, "dice_loss": dice}


def resize_targets_like(targets: Tensor, logits: Tensor) -> Tensor:
    if targets.shape == logits.shape:
        return targets
    batch_size, num_prompts = targets.shape[:2]
    flat = targets.reshape(batch_size * num_prompts, 1, *targets.shape[2:]).float()
    resized = F.interpolate(flat, size=logits.shape[2:], mode="nearest")
    return resized.reshape(batch_size, num_prompts, *logits.shape[2:])


def _validate_prompt_valid(
    prompt_valid: Tensor | None,
    reference: Tensor,
    device: torch.device,
) -> Tensor | None:
    if prompt_valid is None:
        return None
    expected_shape = tuple(reference.shape[:2])
    if tuple(prompt_valid.shape) != expected_shape:
        raise ValueError(
            f"prompt_valid shape {tuple(prompt_valid.shape)} != expected {expected_shape}"
        )
    return prompt_valid.to(device=device, dtype=torch.bool)


__all__ = ["BinaryDiceLoss", "PromptSegmentationLoss", "resize_targets_like"]
