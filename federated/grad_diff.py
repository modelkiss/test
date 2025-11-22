"""Utilities for computing multi-round parameter or gradient differences."""
from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Tuple

import torch

StateDict = Mapping[str, torch.Tensor]
ParamDiff = Dict[str, torch.Tensor]


def compute_param_diff(state_a: StateDict, state_b: StateDict) -> ParamDiff:
    """Compute the parameter difference ``state_a - state_b`` for each tensor."""

    return {name: state_a[name] - state_b[name] for name in state_a}


def build_weighted_param_diff(
    snapshots: Mapping[int, StateDict], rounds: Iterable[int], alpha: float
) -> ParamDiff:
    """Build a weighted parameter difference signal from multiple rounds.

    Args:
        snapshots: Mapping from round index to model state. ``snapshots[0]`` is
            expected to be the state before unlearning.
        rounds: Iterable of round offsets to include. Positive values reference
            later rounds, negative values reference earlier rounds.
        alpha: Decay factor controlling the exponential weighting by distance
            from round ``0``.

    Returns:
        Aggregated parameter difference with the same keys as ``snapshots[0]``.
    """

    diffs: Dict[int, ParamDiff] = {}
    weights: Dict[int, float] = {}

    for t in rounds:
        if t == 0:
            continue
        if t > 0:
            diff_t = compute_param_diff(snapshots[0], snapshots[t])
        else:
            diff_t = compute_param_diff(snapshots[t], snapshots[0])
        w_t = math.exp(-alpha * abs(t))
        diffs[t] = diff_t
        weights[t] = w_t

    if not weights:
        return {}

    normalization = sum(weights.values())
    for t in weights:
        weights[t] /= normalization

    target_signal: ParamDiff = {}
    for t, diff_t in diffs.items():
        weight = weights[t]
        for name, tensor in diff_t.items():
            if name not in target_signal:
                target_signal[name] = weight * tensor
            else:
                target_signal[name] += weight * tensor

    return target_signal


def split_param_diff_into_groups(param_diff: ParamDiff) -> Tuple[ParamDiff, ParamDiff]:
    """Split parameter differences into semantic and appearance groups."""

    semantic_group: ParamDiff = {}
    appearance_group: ParamDiff = {}

    for name, tensor in param_diff.items():
        if name.startswith(("layer3", "layer4", "fc")):
            semantic_group[name] = tensor
        elif name.startswith(("conv1", "layer1", "layer2")):
            appearance_group[name] = tensor

    return semantic_group, appearance_group
