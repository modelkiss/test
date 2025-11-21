"""Shared helper utilities for attack implementations."""
from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping

import torch


def normalize_signal(param_diff: Mapping[str, torch.Tensor], eps: float = 1e-8) -> MutableMapping[str, torch.Tensor]:
    """Scale parameter diffs to unit norm for stable matching."""

    normalized: MutableMapping[str, torch.Tensor] = {}
    for name, tensor in param_diff.items():
        norm = tensor.norm().clamp_min(eps)
        normalized[name] = tensor / norm
    return normalized


def clone_signal(param_diff: Mapping[str, torch.Tensor]) -> MutableMapping[str, torch.Tensor]:
    """Clone a parameter diff dictionary."""

    return {k: v.clone() for k, v in param_diff.items()}


def denormalize_image(image: torch.Tensor, mean: Iterable[float] | None = None, std: Iterable[float] | None = None) -> torch.Tensor:
    """Invert common normalization for visualization or saving."""

    mean = mean or [0.5] * image.shape[0]
    std = std or [0.5] * image.shape[0]
    mean_tensor = torch.tensor(mean, device=image.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, device=image.device).view(-1, 1, 1)
    return image * std_tensor + mean_tensor
