"""Utilities to construct client-level target signals from training logs."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, MutableMapping

from attacks.common import signals
from federated.state import TrainingLog, UnlearningLog


def _rounds_with_clients(round_client_ids: Iterable[Iterable[Any]], target_clients: Iterable[Any]) -> list[int]:
    target_set = set(target_clients)
    rounds: list[int] = []
    for idx, clients in enumerate(round_client_ids):
        if target_set.intersection(set(clients)):
            rounds.append(idx)
    return rounds


def build_client_target_signal(
    training_log: TrainingLog,
    unlearning_log: UnlearningLog,
    forgotten_clients: Iterable[Any],
    config: Mapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    """Aggregate parameter differences associated with forgotten clients."""

    config = config or {}
    layers = config.get("layers")
    weights = config.get("round_weights")

    candidate_rounds = _rounds_with_clients(training_log.round_client_ids, forgotten_clients)
    paired_rounds = zip(candidate_rounds, candidate_rounds)

    diffs = []
    for train_idx, forget_idx in paired_rounds:
        if train_idx >= len(training_log.global_model_snapshots):
            continue
        if forget_idx >= len(unlearning_log.global_model_snapshots_after_unlearning):
            continue
        before = training_log.global_model_snapshots[train_idx]
        after = unlearning_log.global_model_snapshots_after_unlearning[forget_idx]
        diff = signals.compute_param_diff(before, after)
        if layers:
            diff = signals.select_layers(diff, layers)
        diffs.append(diff)

    if not diffs and training_log.global_model_snapshots and unlearning_log.global_model_snapshots_after_unlearning:
        before = training_log.global_model_snapshots[-1]
        after = unlearning_log.global_model_snapshots_after_unlearning[-1]
        base_diff = signals.compute_param_diff(before, after)
        diffs.append(signals.select_layers(base_diff, layers) if layers else base_diff)

    aggregated = signals.average_param_diffs(diffs, weights=weights)
    return aggregated
