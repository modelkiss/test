"""Class-level unlearning routines (e.g., FUCRT, FUCP)."""
from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Sequence

import torch

from .base_unlearning import BaseUnlearning
from .. import utils


class ClassLevelUnlearning(BaseUnlearning):
    """Unlearning driver that targets specific label classes."""

    def __init__(self, model: torch.nn.Module, training_log, config: Any):
        super().__init__(model, training_log, config)
        self.method = _get_config_attr(config, "unlearning_method", "FUCRT")
        allowed_methods = {"FUCRT", "FUCP"}
        if self.method not in allowed_methods:
            raise ValueError(
                f"Unsupported class-level unlearning method: {self.method}. "
                f"Choose one of {sorted(allowed_methods)}."
            )

    def select_targets(self) -> None:
        self.forgotten_classes = list(_get_config_attr(self.config, "forgotten_classes", []))

    def perform_unlearning(self) -> Sequence[Mapping[str, torch.Tensor]]:
        self._load_training_terminal_model()
        snapshots: List[Mapping[str, torch.Tensor]] = [self._clone_current_state()]

        retrain_rounds = _get_config_attr(self.config, "retrain_rounds", 0)
        dataloader: Iterable = _get_config_attr(self.config, "retrain_dataloader", None)
        device = _get_config_attr(self.config, "device", "cpu")
        loss_fn = _get_config_attr(self.config, "loss_fn", torch.nn.CrossEntropyLoss())

        if dataloader is None or retrain_rounds <= 0:
            return snapshots

        optimizer_cls = _get_config_attr(self.config, "optimizer_cls", torch.optim.SGD)
        optimizer_kwargs = _get_config_attr(self.config, "optimizer_kwargs", {"lr": 0.01})
        optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)

        for _ in range(retrain_rounds):
            epoch_loss = 0.0
            seen_samples = 0
            self.model.train()
            for batch in dataloader:
                inputs, targets = batch
                mask = ~torch.isin(targets, torch.tensor(self.forgotten_classes, device=targets.device))
                if mask.sum() == 0:
                    continue
                inputs = inputs[mask].to(device)
                targets = targets[mask].to(device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * targets.size(0)
                seen_samples += targets.size(0)

            average_loss = epoch_loss / max(seen_samples, 1)
            self.metrics.setdefault("retrain_loss", []).append(average_loss)
            snapshots.append(utils.clone_model_state(self.model))

        return snapshots


def _get_config_attr(config: Any, name: str, default: Any) -> Any:
    if hasattr(config, name):
        return getattr(config, name)
    if isinstance(config, dict) and name in config:
        return config[name]
    return default
