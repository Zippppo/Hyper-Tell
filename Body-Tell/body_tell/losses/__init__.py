"""Losses for Body-Tell prompt segmentation."""

from .prompt_losses import BinaryDiceLoss, PromptSegmentationLoss

__all__ = ["BinaryDiceLoss", "PromptSegmentationLoss"]

