"""Gradient computation utilities shared across attack variants."""
from __future__ import annotations

from typing import Callable, Mapping, MutableMapping, Sequence

import torch
import torch.nn.functional as F

from .signals import flatten_param_diff


Gradients = Mapping[str, torch.Tensor]


def _infer_device(*sources: Mapping[str, torch.Tensor]) -> torch.device:
    """Return a reasonable device even when dictionaries are empty."""

    for mapping in sources:
        for tensor in mapping.values():
            return tensor.device
    return torch.device("cpu")


def compute_model_gradients(
    model: torch.nn.Module,
    images: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor] | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    target_labels: torch.Tensor | None = None,
    layers_to_use: Sequence[str] | None = None,
) -> MutableMapping[str, torch.Tensor]:
    """Compute gradients of a provided loss with respect to model parameters."""

    model.zero_grad(set_to_none=True)
    outputs = model(images)
    if target_labels is not None:
        loss = loss_fn(outputs, target_labels)
    else:
        loss = loss_fn(outputs)
    loss.backward()

    gradients: MutableMapping[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if layers_to_use and not any(pattern in name for pattern in layers_to_use):
            continue
        gradients[name] = param.grad.detach().clone()
    return gradients


def gradient_matching_loss(
    current_grads: Gradients,
    target_signal: Mapping[str, torch.Tensor],
    match_type: str = "l2",
) -> torch.Tensor:
    """Compare current gradients with a target signal and return a scalar loss."""

    if not current_grads or not target_signal:
        return torch.tensor(0.0, device=_infer_device(current_grads, target_signal))

    keys = [k for k in current_grads if k in target_signal]
    if not keys:
        return torch.tensor(0.0, device=_infer_device(current_grads, target_signal))

    current_flat = flatten_param_diff({k: current_grads[k] for k in keys})
    target_flat = flatten_param_diff({k: target_signal[k] for k in keys}).to(current_flat.device)

    if match_type == "l2":
        return F.mse_loss(current_flat, target_flat)
    if match_type == "cosine":
        similarity = F.cosine_similarity(current_flat, target_flat, dim=0, eps=1e-8)
        return 1 - similarity
    raise ValueError(f"Unsupported match_type: {match_type}")


def build_attack_loss(
    model: torch.nn.Module,
    images: torch.Tensor,
    target_signal: Mapping[str, torch.Tensor],
    loss_config: Mapping[str, object],
) -> torch.Tensor:
    """Assemble the total loss used for guided sampling."""

    match_type = loss_config.get("match_type", "l2")
    layers = loss_config.get("layers")
    grad_loss_weight = float(loss_config.get("grad_loss_weight", 1.0))

    loss_fn = loss_config.get("loss_fn", lambda out, labels=None: out.mean())
    target_labels = loss_config.get("target_labels")

    gradients = compute_model_gradients(
        model=model,
        images=images,
        loss_fn=loss_fn,
        target_labels=target_labels,
        layers_to_use=layers,
    )
    loss_grad = gradient_matching_loss(gradients, target_signal, match_type=match_type)

    total_loss = grad_loss_weight * loss_grad

    classification_loss_fn = loss_config.get("classification_loss_fn")
    if classification_loss_fn is not None and target_labels is not None:
        outputs = model(images)
        cls_loss = classification_loss_fn(outputs, target_labels)
        total_loss = total_loss + float(loss_config.get("classification_weight", 1.0)) * cls_loss

    prior_loss_fn = loss_config.get("prior_loss_fn")
    if prior_loss_fn is not None:
        prior_loss = prior_loss_fn(images)
        total_loss = total_loss + float(loss_config.get("prior_weight", 1.0)) * prior_loss

    return total_loss
