"""Client-side training abstractions for federated learning."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import torch

from . import utils


@dataclass
class ClientUpdate:
    """Container for a single client's contribution to a training round."""

    delta_params: Dict[str, torch.Tensor]
    num_samples: int
    metrics: Dict[str, Any] = field(default_factory=dict)


class FederatedClient:
    """Implements local training logic for a single federated client."""

    def __init__(
        self,
        client_id: Any,
        train_loader: Iterable,
        val_loader: Optional[Iterable] = None,
        local_epochs: int = 1,
        lr: float = 0.01,
        device: str = "cpu",
    ) -> None:
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = device

    def local_train(self, model: torch.nn.Module, config: Any) -> ClientUpdate:
        """Run local training and return the resulting update."""

        model = copy.deepcopy(model).to(self.device)
        initial_state = utils.clone_model_state(model)

        loss_fn = _get_config_attr(config, "loss_fn", torch.nn.CrossEntropyLoss())
        optimizer_cls = _get_config_attr(config, "optimizer_cls", torch.optim.SGD)
        optimizer_kwargs = _get_config_attr(config, "optimizer_kwargs", {"lr": self.lr})
        optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

        model.train()
        total_loss = 0.0
        total_samples = 0
        for _ in range(self.local_epochs):
            for batch in self.train_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * targets.size(0)
                total_samples += targets.size(0)

        updated_state = model.state_dict()
        delta = {k: updated_state[k] - initial_state[k] for k in initial_state}
        metrics = {"loss": total_loss / max(total_samples, 1)}

        if self.val_loader is not None:
            val_loss, val_acc = utils.evaluate_model(
                model, self.val_loader, loss_fn, device=self.device
            )
            metrics.update({"val_loss": val_loss, "val_acc": val_acc})

        return ClientUpdate(delta_params=delta, num_samples=total_samples, metrics=metrics)


def _get_config_attr(config: Any, name: str, default: Any) -> Any:
    """Safe helper to extract configuration values from objects or dictionaries."""

    if hasattr(config, name):
        return getattr(config, name)
    if isinstance(config, dict) and name in config:
        return config[name]
    return default
