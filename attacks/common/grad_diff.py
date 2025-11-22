"""Utilities for computing gradient-like parameter differences for attacks."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple

import torch
from torch import Tensor

StateDict = Mapping[str, Tensor]


def compute_param_diff(state_a: StateDict, state_b: StateDict) -> MutableMapping[str, Tensor]:
    """Compute element-wise differences between two model state dictionaries.

    Args:
        state_a: The minuend state dictionary.
        state_b: The subtrahend state dictionary.

    Returns:
        A dictionary mapping parameter names to ``state_a[name] - state_b[name]``.
        Only parameters present in both dictionaries are included.
    """

    diff: MutableMapping[str, Tensor] = {}
    for name, tensor_a in state_a.items():
        tensor_b = state_b.get(name)
        if tensor_b is None:
            continue
        diff[name] = tensor_a - tensor_b
    return diff


def build_weighted_param_diff(
    snapshots: Dict[int, StateDict], rounds: List[int], alpha: float
) -> MutableMapping[str, Tensor]:
    """Build a weighted parameter difference signal across multiple rounds.

    Args:
        snapshots: Mapping from round index to model state dict. Round ``0`` is
            treated as the baseline (last round before unlearning).
        rounds: List of round indices to include in the weighted aggregation.
        alpha: Exponential decay factor controlling the weight for distant rounds.

    Returns:
        A dictionary containing the weighted sum of parameter differences.
    """

    if 0 not in snapshots:
        raise ValueError("snapshots must include the baseline round 0 state.")

    baseline = snapshots[0]

    initial_weights = [math.exp(-alpha * abs(t)) for t in rounds]
    weight_sum = sum(initial_weights)
    if weight_sum == 0:
        raise ValueError("Sum of weights is zero; check alpha and rounds inputs.")
    normalized_weights = [w / weight_sum for w in initial_weights]

    aggregated: MutableMapping[str, Tensor] = {}
    for t, weight in zip(rounds, normalized_weights):
        snapshot = snapshots.get(t)
        if snapshot is None:
            raise KeyError(f"Missing snapshot for round {t}.")

        if t > 0:
            diff = compute_param_diff(baseline, snapshot)
        elif t < 0:
            diff = compute_param_diff(snapshot, baseline)
        else:
            # Round 0 contributes no difference.
            continue

        for name, tensor in diff.items():
            weighted_tensor = tensor * weight
            if name in aggregated:
                aggregated[name] = aggregated[name] + weighted_tensor
            else:
                aggregated[name] = weighted_tensor.clone()

    return aggregated


def split_param_diff_into_groups(
    param_diff: Mapping[str, Tensor]
) -> Tuple[MutableMapping[str, Tensor], MutableMapping[str, Tensor]]:
    """Split parameter differences into semantic and appearance groups.

    Args:
        param_diff: Full parameter difference dictionary.

    Returns:
        A tuple ``(semantic_group, appearance_group)`` where the semantic group
        contains parameters whose names start with ``layer3``, ``layer4`` or
        ``fc``, and the appearance group contains parameters whose names start
        with ``conv1``, ``layer1`` or ``layer2``.
    """

    semantic_prefixes: Iterable[str] = ("layer3", "layer4", "fc")
    appearance_prefixes: Iterable[str] = ("conv1", "layer1", "layer2")

    semantic_group: MutableMapping[str, Tensor] = {}
    appearance_group: MutableMapping[str, Tensor] = {}

    for name, tensor in param_diff.items():
        if name.startswith(semantic_prefixes):
            semantic_group[name] = tensor
        elif name.startswith(appearance_prefixes):
            appearance_group[name] = tensor

    return semantic_group, appearance_group
