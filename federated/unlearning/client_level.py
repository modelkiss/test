"""Client-level unlearning strategies such as FedEraser and FedRecovery."""
from __future__ import annotations

import random
from typing import Any, Iterable, List, Mapping, Sequence

import torch

from .base_unlearning import BaseUnlearning
from ..aggregator import Aggregator
from ..client import FederatedClient


class ClientLevelUnlearning(BaseUnlearning):
    """Implements client-level unlearning scaffolding."""

    def __init__(
        self,
        model: torch.nn.Module,
        training_log,
        config: Any,
        aggregator: Aggregator,
        clients: Iterable[FederatedClient],
    ) -> None:
        super().__init__(model, training_log, config)
        self.aggregator = aggregator
        self.clients = list(clients)
        self.method = _get_config_attr(config, "unlearning_method", "FedEraser")
        allowed_methods = {"FedEraser", "FedRecovery"}
        if self.method not in allowed_methods:
            raise ValueError(
                f"Unsupported client-level unlearning method: {self.method}. "
                f"Choose one of {sorted(allowed_methods)}."
            )

    def select_targets(self) -> None:
        self.forgotten_clients = list(_get_config_attr(self.config, "forgotten_clients", []))

    def perform_unlearning(self) -> Sequence[Mapping[str, torch.Tensor]]:
        self._load_training_terminal_model()
        snapshots: List[Mapping[str, torch.Tensor]] = [self._clone_current_state()]

        # Remove historical contributions for the forgotten clients when available.
        for round_idx in sorted(self.training_log.client_updates.keys()):
            updates = self.training_log.client_updates[round_idx]
            for client_id in self.forgotten_clients:
                if client_id in updates:
                    self.model = self.aggregator.remove_client_contribution(
                        self.model, updates[client_id]
                    )
            snapshots.append(self._clone_current_state())

        # Optional recovery or compensation phase.
        recovery_rounds = _get_config_attr(self.config, "recovery_rounds", 0)
        client_fraction = _get_config_attr(self.config, "client_fraction", 1.0)
        for _ in range(recovery_rounds):
            selected = self._sample_clients(client_fraction)
            updates = [client.local_train(self.model, self.config) for client in selected]
            self.model = self.aggregator.aggregate(self.model, updates)
            snapshots.append(self._clone_current_state())

        return snapshots

    def _sample_clients(self, fraction: float) -> List[FederatedClient]:
        if fraction >= 1.0:
            return self.clients
        sample_size = max(1, int(len(self.clients) * fraction))
        return random.sample(self.clients, sample_size)


def _get_config_attr(config: Any, name: str, default: Any) -> Any:
    if hasattr(config, name):
        return getattr(config, name)
    if isinstance(config, dict) and name in config:
        return config[name]
    return default
