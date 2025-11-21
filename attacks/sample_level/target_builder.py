"""Construct target signals for single-sample reconstruction."""
from __future__ import annotations

from typing import Any, Mapping, MutableMapping

from attacks.common import signals
from federated.state import TrainingLog, UnlearningLog


def build_sample_target_signal(
    training_log: TrainingLog,
    unlearning_log: UnlearningLog,
    forgotten_sample_id: Any,
    config: Mapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    """Approximate the gradient contribution of a forgotten sample."""

    _ = forgotten_sample_id
    config = config or {}
    layers = config.get("layers")
    snapshots_before = training_log.global_model_snapshots
    snapshots_after = unlearning_log.global_model_snapshots_after_unlearning

    diffs = []
    if snapshots_before and snapshots_after:
        before = snapshots_before[-1]
        after = snapshots_after[-1]
        diff = signals.compute_param_diff(before, after)
        if layers:
            diff = signals.select_layers(diff, layers)
        diffs.append(diff)

    return signals.average_param_diffs(diffs)
