"""Data structures for logging federated training and unlearning."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class TrainingLog:
    """Log structure that captures the full federated training trace."""

    global_model_snapshots: List[Mapping[str, Any]] = field(default_factory=list)
    round_client_ids: List[List[Any]] = field(default_factory=list)
    client_updates: Dict[int, Dict[Any, Any]] = field(default_factory=dict)
    train_metrics: Dict[str, List[float]] = field(default_factory=dict)

    def add_snapshot(self, state_dict: Mapping[str, Any]) -> None:
        self.global_model_snapshots.append(state_dict)

    def add_round_clients(self, round_clients: List[Any]) -> None:
        self.round_client_ids.append(round_clients)

    def add_client_updates(self, round_idx: int, updates: Dict[Any, Any]) -> None:
        self.client_updates[round_idx] = updates


@dataclass
class UnlearningLog:
    """Log structure for different unlearning procedures."""

    global_model_snapshots_after_unlearning: List[Mapping[str, Any]] = field(
        default_factory=list
    )
    forgotten_clients: List[Any] = field(default_factory=list)
    forgotten_classes: List[Any] = field(default_factory=list)
    forgotten_samples: List[Any] = field(default_factory=list)
    unlearning_metrics: Dict[str, List[float]] = field(default_factory=dict)

    def add_snapshot(self, state_dict: Mapping[str, Any]) -> None:
        self.global_model_snapshots_after_unlearning.append(state_dict)
