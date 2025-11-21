"""Sample-level unlearning routines (FedAF, FedAU)."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch

from .base_unlearning import BaseUnlearning
from ..aggregator import Aggregator
from ..client import FederatedClient
from .. import utils


class SampleLevelUnlearning(BaseUnlearning):
    """Unlearning driver that targets specific samples across clients."""

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
        self.method = _get_config_attr(config, "unlearning_method", "FedAF")
        allowed_methods = {"FedAF", "FedAU"}
        if self.method not in allowed_methods:
            raise ValueError(
                f"Unsupported sample-level unlearning method: {self.method}. "
                f"Choose one of {sorted(allowed_methods)}."
            )

    def select_targets(self) -> None:
        target = _get_config_attr(self.config, "forgotten_samples", [])
        if isinstance(target, (list, tuple)):
            self.forgotten_samples = list(target)
        else:
            self.forgotten_samples = [target]

    def perform_unlearning(self) -> Sequence[Mapping[str, torch.Tensor]]:
        self._load_training_terminal_model()
        snapshots: List[Mapping[str, torch.Tensor]] = [self._clone_current_state()]

        unlearning_rounds = _get_config_attr(self.config, "unlearning_rounds", 1)
        client_fraction = _get_config_attr(self.config, "client_fraction", 1.0)
        loss_fn = _get_config_attr(self.config, "unlearning_loss_fn", torch.nn.CrossEntropyLoss())
        optimizer_cls = _get_config_attr(self.config, "optimizer_cls", torch.optim.SGD)
        optimizer_kwargs = _get_config_attr(self.config, "optimizer_kwargs", {"lr": 0.01})

        for _ in range(unlearning_rounds):
            affected_clients = self._collect_clients_for_targets()
            if client_fraction < 1.0 and affected_clients:
                affected_clients = affected_clients[: max(1, int(len(affected_clients) * client_fraction))]

            updates = []
            for client in affected_clients:
                # Reuse the standard local training but allow specialized loss functions.
                config_override = {
                    **({} if not isinstance(self.config, dict) else self.config),
                    "loss_fn": loss_fn,
                    "optimizer_cls": optimizer_cls,
                    "optimizer_kwargs": optimizer_kwargs,
                }
                updates.append(client.local_train(self.model, config_override))

            self.model = self.aggregator.aggregate(self.model, updates)
            snapshots.append(utils.clone_model_state_to_cpu(self.model))

        return snapshots

    def _collect_clients_for_targets(self) -> List[FederatedClient]:
        mapping: Dict[Any, List[Any]] = _get_config_attr(
            self.config, "sample_client_map", {}
        )
        located_clients: List[FederatedClient] = []
        if mapping:
            for sample in self.forgotten_samples:
                target_ids = mapping.get(sample, [])
                located_clients.extend(
                    [client for client in self.clients if client.client_id in target_ids]
                )
        else:
            locator = _get_config_attr(self.config, "sample_locator", None)
            if callable(locator):
                for sample in self.forgotten_samples:
                    located_clients.extend(locator(self.clients, sample))
            else:
                located_clients = self.clients
        # Remove duplicates while preserving order.
        seen = set()
        unique_clients: List[FederatedClient] = []
        for client in located_clients:
            if client.client_id not in seen:
                unique_clients.append(client)
                seen.add(client.client_id)
        return unique_clients


def _get_config_attr(config: Any, name: str, default: Any) -> Any:
    if hasattr(config, name):
        return getattr(config, name)
    if isinstance(config, dict) and name in config:
        return config[name]
    return default
