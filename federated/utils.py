"""Utility helpers for federated training and unlearning."""
from __future__ import annotations

import copy
import random
from typing import Iterable, Mapping, Tuple

import torch


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_model_state(model: torch.nn.Module) -> Mapping[str, torch.Tensor]:
    """Return a deep copy of a model's state dict."""

    return {k: v.clone().detach() for k, v in model.state_dict().items()}


def save_model_snapshot(state_dict: Mapping[str, torch.Tensor], path: str) -> None:
    """Persist a model snapshot to disk."""

    torch.save(state_dict, path)


def load_model_snapshot(path: str) -> Mapping[str, torch.Tensor]:
    """Load a model snapshot from disk."""

    return torch.load(path, map_location="cpu")


def model_parameters_difference(
    new_state: Mapping[str, torch.Tensor],
    old_state: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    """Compute parameter differences between two state dicts."""

    return {k: new_state[k] - old_state[k] for k in new_state}


def apply_delta_to_model(
    model: torch.nn.Module, delta: Mapping[str, torch.Tensor]
) -> torch.nn.Module:
    """Return a new model with ``delta`` applied to its parameters."""

    updated = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in updated.named_parameters():
            if name in delta:
                param.add_(delta[name])
    return updated


def evaluate_model(
    model: torch.nn.Module,
    dataloader: Iterable,
    loss_fn: torch.nn.Module,
    device: str = "cpu",
) -> Tuple[float, float]:
    """Compute loss and accuracy over a dataloader."""

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)
    average_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return average_loss, accuracy
