"""Server-side aggregation utilities."""
from __future__ import annotations

import copy
from typing import Iterable, List

import torch

from .client import ClientUpdate


class Aggregator:
    """Simple weighted averaging aggregator with extensibility hooks."""

    def __init__(self, method: str = "fedavg") -> None:
        self.method = method

    def aggregate(
        self, global_model: torch.nn.Module, client_updates: List[ClientUpdate]
    ) -> torch.nn.Module:
        if not client_updates:
            return global_model

        total_samples = sum(update.num_samples for update in client_updates)
        aggregated_delta = self._initialize_zero_delta(client_updates[0])

        for update in client_updates:
            weight = update.num_samples / max(total_samples, 1)
            for name, delta in update.delta_params.items():
                if not delta.is_floating_point():
                    continue
                aggregated_delta[name] += weight * delta

        new_model = copy.deepcopy(global_model)
        with torch.no_grad():
            for name, param in new_model.named_parameters():
                param.add_(aggregated_delta[name])
        return new_model

    def remove_client_contribution(
        self, global_model: torch.nn.Module, target_update: ClientUpdate
    ) -> torch.nn.Module:
        """Remove one client's contribution from the global model."""

        new_model = copy.deepcopy(global_model)
        with torch.no_grad():
            for name, param in new_model.named_parameters():
                if name in target_update.delta_params:
                    param.sub_(target_update.delta_params[name])
        return new_model

    def apply_weighted_updates(
        self, global_model: torch.nn.Module, updates: Iterable[ClientUpdate], weights
    ) -> torch.nn.Module:
        """Apply arbitrary weighted updates to the global model."""

        new_model = copy.deepcopy(global_model)
        weight_map = list(weights)
        updates_list = list(updates)
        with torch.no_grad():
            for name, param in new_model.named_parameters():
                for update, weight in zip(updates_list, weight_map):
                    if name in update.delta_params:
                        param.add_(weight * update.delta_params[name])
        return new_model

    def _initialize_zero_delta(self, example_update: ClientUpdate):
        return {
            name: torch.zeros_like(delta)
            for name, delta in example_update.delta_params.items()
            if delta.is_floating_point()
        }
