"""Federated training driver."""
from __future__ import annotations

import copy
import math
import random
from typing import Any, Iterable, List

import torch

from .aggregator import Aggregator
from .client import ClientUpdate, FederatedClient
from .state import TrainingLog
from . import utils


class FederatedTrainer:
    """Run the full federated training loop."""

    def __init__(
        self,
        model: torch.nn.Module,
        clients: Iterable[FederatedClient],
        aggregator: Aggregator,
        config: Any,
    ) -> None:
        self.global_model = copy.deepcopy(model)
        self.clients = list(clients)
        self.aggregator = aggregator
        self.config = config
        seed = _get_config_attr(config, "seed", None)
        if seed is not None:
            utils.set_random_seed(seed)

    def run(self) -> TrainingLog:
        num_rounds = _get_config_attr(self.config, "num_rounds", 1)
        client_fraction = _get_config_attr(self.config, "client_fraction", 1.0)
        snapshot_interval = _get_config_attr(self.config, "snapshot_interval", 1)
        track_client_updates = _get_config_attr(
            self.config, "track_client_updates", True
        )
        tracked_client_ids = _get_config_attr(self.config, "tracked_client_ids", None)

        training_log = TrainingLog()

        for round_idx in range(num_rounds):
            selected_clients = self._sample_clients(client_fraction)
            round_updates = {}
            client_updates = []

            for client in selected_clients:
                update = client.local_train(self.global_model, self.config)
                round_updates[client.client_id] = update
                client_updates.append(update)

            self.global_model = self.aggregator.aggregate(
                self.global_model, client_updates
            )

            training_log.add_round_clients([c.client_id for c in selected_clients])
            if track_client_updates:
                logged_updates = _prepare_logged_updates(round_updates, tracked_client_ids)
                if logged_updates:
                    training_log.add_client_updates(round_idx, logged_updates)

            if (round_idx + 1) % snapshot_interval == 0:
                training_log.add_snapshot(_clone_state_to_cpu(self.global_model))

        # Ensure the terminal global model is always recorded for downstream steps.
        training_log.add_snapshot(_clone_state_to_cpu(self.global_model))
        training_log.final_model_state = _clone_state_to_cpu(self.global_model)

        return training_log

    def _sample_clients(self, client_fraction: float) -> List[FederatedClient]:
        num_clients = len(self.clients)
        if client_fraction >= 1.0:
            return self.clients
        sample_size = max(1, math.ceil(num_clients * client_fraction))
        return random.sample(self.clients, sample_size)


def _get_config_attr(config: Any, name: str, default: Any) -> Any:
    if hasattr(config, name):
        return getattr(config, name)
    if isinstance(config, dict) and name in config:
        return config[name]
    return default


def _prepare_logged_updates(
    round_updates: dict[Any, ClientUpdate], tracked_client_ids: list[Any] | None
) -> dict[Any, ClientUpdate]:
    """Prepare a lightweight, CPU-friendly copy of client updates for logging."""

    filtered = {}
    for client_id, update in round_updates.items():
        if tracked_client_ids is not None and client_id not in tracked_client_ids:
            continue
        filtered[client_id] = ClientUpdate(
            delta_params={k: v.detach().cpu() for k, v in update.delta_params.items()},
            num_samples=update.num_samples,
            metrics=update.metrics,
        )
    return filtered


def _clone_state_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone the model state and move it to CPU for storage."""

    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
