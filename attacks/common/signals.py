"""Utilities for constructing gradient or parameter difference target signals."""
from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping, Sequence

import torch


def compute_param_diff(
    model_state_before: Mapping[str, torch.Tensor],
    model_state_after: Mapping[str, torch.Tensor],
) -> MutableMapping[str, torch.Tensor]:
    """Compute parameter-wise differences between two model state dicts.

    Args:
        model_state_before: State dict prior to unlearning or removal.
        model_state_after: State dict after unlearning or removal.

    Returns:
        A dictionary where each tensor is ``after - before`` for the
        corresponding parameter name.
    """

    diff: MutableMapping[str, torch.Tensor] = {}
    for name, after_tensor in model_state_after.items():
        before_tensor = model_state_before.get(name)
        if before_tensor is None:
            continue
        diff[name] = after_tensor - before_tensor
    return diff


def average_param_diffs(
    diff_list: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float] | None = None,
) -> MutableMapping[str, torch.Tensor]:
    """Compute a (possibly weighted) average over multiple diff dictionaries."""

    if not diff_list:
        return {}

    if weights is None:
        weights = [1.0 for _ in diff_list]
    if len(weights) != len(diff_list):
        raise ValueError("Length of weights must match diff_list length.")

    weight_tensors = [float(w) for w in weights]
    total_weight = sum(weight_tensors)
    if total_weight == 0:
        total_weight = 1.0

    aggregated: MutableMapping[str, torch.Tensor] = {}
    for diff, weight in zip(diff_list, weight_tensors):
        for name, tensor in diff.items():
            weighted = tensor * (weight / total_weight)
            if name in aggregated:
                aggregated[name] = aggregated[name] + weighted
            else:
                aggregated[name] = weighted.clone()
    return aggregated


def select_layers(
    param_diff: Mapping[str, torch.Tensor], layer_name_patterns: Iterable[str] | None
) -> MutableMapping[str, torch.Tensor]:
    """Filter the diff dictionary by keeping parameters matching any pattern."""

    if not layer_name_patterns:
        return {k: v.clone() for k, v in param_diff.items()}

    selected: MutableMapping[str, torch.Tensor] = {}
    for name, tensor in param_diff.items():
        if any(pattern in name for pattern in layer_name_patterns):
            selected[name] = tensor.clone()
    return selected


def flatten_param_diff(param_diff: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a parameter diff dictionary into a single vector."""

    if not param_diff:
        return torch.tensor([])

    flattened = [tensor.reshape(-1) for tensor in param_diff.values()]
    return torch.cat(flattened, dim=0)
