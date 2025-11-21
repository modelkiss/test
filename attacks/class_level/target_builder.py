"""Construct target signals for class-level unlearning attacks."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, MutableMapping

from attacks.common import signals
from federated.state import TrainingLog, UnlearningLog


def build_class_target_signal(
    training_log: TrainingLog,
    unlearning_log: UnlearningLog,
    forgotten_classes: Iterable[Any],
    config: Mapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    """Extract parameter differences for the specified classes."""

    config = config or {}
    layers = config.get("layers")
    diffs = []

    if training_log.global_model_snapshots and unlearning_log.global_model_snapshots_after_unlearning:
        before = training_log.global_model_snapshots[-1]
        after = unlearning_log.global_model_snapshots_after_unlearning[-1]
        full_diff = signals.compute_param_diff(before, after)

        class_patterns = [str(cls_id) for cls_id in forgotten_classes]
        if layers:
            class_patterns = list(layers) + class_patterns
        filtered = signals.select_layers(full_diff, class_patterns)
        diffs.append(filtered)

    return signals.average_param_diffs(diffs)
