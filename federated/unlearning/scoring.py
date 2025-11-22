"""Loss and scoring utilities for unlearning evaluation."""
from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


TensorOrFloat = Union[torch.Tensor, float]


def _as_weighted_models(models: Iterable[Union[torch.nn.Module, Tuple[torch.nn.Module, float]]]):
    model_weights: list[Tuple[torch.nn.Module, float]] = []
    for entry in models:
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], (int, float)):
            model_weights.append((entry[0], float(entry[1])))
        else:
            model_weights.append((entry, 1.0))
    return model_weights


def compute_confidences(
    pre_models: Iterable[Union[torch.nn.Module, Tuple[torch.nn.Module, float]]],
    post_models: Iterable[Union[torch.nn.Module, Tuple[torch.nn.Module, float]]],
    x: torch.Tensor,
    target_id: int,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute average confidence of ``target_id`` before and after unlearning.

    Args:
        pre_models: Sequence of models (or ``(model, weight)`` tuples) before unlearning.
        post_models: Sequence of models (or ``(model, weight)`` tuples) after unlearning.
        x: Input batch.
        target_id: Target class/client/sample index in the softmax output.
        device: Optional device override for inference.

    Returns:
        Tuple of weighted average confidences ``(p_before, p_after)``.
    """

    def _average_confidence(models: Iterable[Tuple[torch.nn.Module, float]]) -> torch.Tensor:
        weighted_probs: list[torch.Tensor] = []
        weight_sum = 0.0
        for model, weight in models:
            model_device = device or next(model.parameters()).device
            model.eval()
            with torch.no_grad():
                logits = model(x.to(model_device, non_blocking=True))
                probs = F.softmax(logits, dim=-1)[..., target_id]
                weighted_probs.append(probs.detach() * weight)
                weight_sum += weight
        if not weighted_probs:
            return torch.tensor(0.0)
        stacked = torch.stack(weighted_probs)
        return stacked.sum(dim=0) / max(weight_sum, 1e-8)

    pre_weighted = _as_weighted_models(pre_models)
    post_weighted = _as_weighted_models(post_models)
    return _average_confidence(pre_weighted), _average_confidence(post_weighted)


def grad_match_loss(
    x: torch.Tensor,
    model: torch.nn.Module,
    target_signal: Mapping[str, torch.Tensor],
    layer_names: Sequence[str],
    *,
    loss_fn: Optional[torch.nn.Module] = None,
    target: Optional[torch.Tensor] = None,
    reduction: str = "l2",
) -> torch.Tensor:
    """Match parameter gradients with a target signal.

    The function computes gradients induced by ``x`` on selected layers and compares
    them against ``target_signal`` using either L2 distance or cosine dissimilarity.

    Args:
        x: Input tensor.
        model: Model to evaluate.
        target_signal: Mapping from layer name to target gradient tensor.
        layer_names: Iterable of parameter names to consider.
        loss_fn: Optional supervised loss. If omitted the sum of logits is used.
        target: Optional labels required when ``loss_fn`` is provided.
        reduction: Either ``"l2"`` or ``"cos"``.

    Returns:
        A scalar tensor representing the gradient matching loss.
    """

    params = dict(model.named_parameters())
    selected_params = [params[name] for name in layer_names if name in params]
    if not selected_params:
        return torch.tensor(0.0, device=x.device)

    model.zero_grad(set_to_none=True)
    model.train()
    outputs = model(x)
    if loss_fn is not None:
        if target is None:
            raise ValueError("target labels are required when loss_fn is provided")
        loss = loss_fn(outputs, target)
    else:
        loss = outputs.sum()

    grads = torch.autograd.grad(loss, selected_params, retain_graph=False, create_graph=False)

    losses: list[torch.Tensor] = []
    for name, grad in zip(layer_names, grads):
        if grad is None or name not in target_signal:
            continue
        tgt = target_signal[name].to(grad.device)
        if reduction == "cos":
            grad_flat = grad.flatten()
            tgt_flat = tgt.flatten()
            denom = grad_flat.norm(p=2) * tgt_flat.norm(p=2) + 1e-8
            similarity = (grad_flat * tgt_flat).sum() / denom
            losses.append(1 - similarity)
        else:
            losses.append(F.mse_loss(grad, tgt))

    if not losses:
        return torch.tensor(0.0, device=x.device)
    return torch.stack(losses).mean()


def semantic_align_loss(
    f_x: torch.Tensor, f_target_semantic: torch.Tensor, reduction: str = "l2"
) -> torch.Tensor:
    """Align semantic features with a target representation."""

    if reduction == "cos":
        numerator = (f_x * f_target_semantic).sum(dim=-1)
        denom = (
            f_x.norm(p=2, dim=-1) * f_target_semantic.norm(p=2, dim=-1) + 1e-8
        )
        return (1 - numerator / denom).mean()
    return F.mse_loss(f_x, f_target_semantic)


def appearance_align_loss(
    g_x: torch.Tensor, g_target_appearance: torch.Tensor, reduction: str = "l2"
) -> torch.Tensor:
    """Align appearance-level representations."""

    if reduction == "cos":
        numerator = (g_x * g_target_appearance).sum(dim=-1)
        denom = (
            g_x.norm(p=2, dim=-1) * g_target_appearance.norm(p=2, dim=-1) + 1e-8
        )
        return (1 - numerator / denom).mean()
    return F.mse_loss(g_x, g_target_appearance)


def _unpack_logits_and_features(
    outputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(outputs, tuple) and len(outputs) == 2:
        return outputs[0], outputs[1]
    return outputs, None


def class_prior_loss(
    x: torch.Tensor,
    model: torch.nn.Module,
    class_prior: Union[torch.Tensor, Mapping[int, torch.Tensor], None],
    class_id: int,
    is_forget_class: bool,
    *,
    loss_fn: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """Combine cross-entropy with class prior regularisation."""

    model_device = next(model.parameters()).device
    outputs = model(x.to(model_device, non_blocking=True))
    logits, features = _unpack_logits_and_features(outputs)
    labels = torch.full((logits.size(0),), class_id, device=logits.device, dtype=torch.long)
    ce_loss = (loss_fn or torch.nn.CrossEntropyLoss())(logits, labels)

    proto_loss = torch.tensor(0.0, device=logits.device)
    suppression_loss = torch.tensor(0.0, device=logits.device)

    if class_prior is not None:
        if is_forget_class:
            if isinstance(class_prior, Mapping) and class_id in class_prior and features is not None:
                prototype = class_prior[class_id].to(features.device)
                proto_loss = F.mse_loss(features.mean(dim=0), prototype)
            elif isinstance(class_prior, torch.Tensor) and features is not None:
                proto_loss = F.mse_loss(features.mean(dim=0), class_prior[class_id])
        else:
            if isinstance(class_prior, Mapping) and "forgotten_class_id" in class_prior:
                forgotten_id = int(class_prior["forgotten_class_id"])
                suppression_loss = F.softmax(logits, dim=-1)[:, forgotten_id].mean()
            elif isinstance(class_prior, torch.Tensor) and class_prior.numel() > class_id:
                suppression_loss = F.softmax(logits, dim=-1)[:, class_id].mean()

    return ce_loss + proto_loss + suppression_loss


def client_prior_loss(
    x: torch.Tensor,
    model: torch.nn.Module,
    client_prior: Union[torch.Tensor, Mapping[int, torch.Tensor]],
    client_id: int,
    *,
    reduction: str = "l2",
) -> torch.Tensor:
    """Regularise client-specific representations using a prior signal."""

    outputs = model(x)
    logits, features = _unpack_logits_and_features(outputs)
    target = (
        client_prior[client_id]
        if isinstance(client_prior, Mapping)
        else client_prior[client_id]
    )
    anchor = features if features is not None else logits
    if reduction == "cos":
        numerator = (anchor * target.to(anchor.device)).sum(dim=-1)
        denom = anchor.norm(p=2, dim=-1) * target.to(anchor.device).norm(p=2, dim=-1) + 1e-8
        return (1 - numerator / denom).mean()
    return F.mse_loss(anchor, target.to(anchor.device))


def sample_prior_loss(
    x: torch.Tensor,
    model: torch.nn.Module,
    sample_prior: Mapping[int, torch.Tensor],
    sample_id: int,
) -> torch.Tensor:
    """Encourage consistency with sample-level priors."""

    outputs = model(x)
    logits, features = _unpack_logits_and_features(outputs)
    anchor = features if features is not None else logits
    return F.mse_loss(anchor, sample_prior[sample_id].to(anchor.device))


def conf_gap_loss(p_before: TensorOrFloat, p_after: TensorOrFloat) -> torch.Tensor:
    """Confidence gap that is larger for stronger forgetting."""

    return F.relu(torch.as_tensor(p_before) - torch.as_tensor(p_after)).mean()


def conf_stable_loss(p_before: TensorOrFloat, p_after: TensorOrFloat) -> torch.Tensor:
    """Confidence stability penalty that prefers minimal drift."""

    return torch.abs(torch.as_tensor(p_before) - torch.as_tensor(p_after)).mean()


def aggregate_score(
    level: str,
    component_values: Mapping[str, TensorOrFloat],
    weights: Mapping[str, float],
    *,
    stats: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> float:
    """Combine loss components into a single scalar score.

    Args:
        level: Current unlearning level (e.g., ``"client"`` / ``"class"`` / ``"sample"``).
        component_values: Computed loss values keyed by component name.
        weights: Weighting configuration. If it contains a nested mapping for the
            provided ``level``, that sub-mapping will be used.
        stats: Optional mapping from component name to ``(mean, std)`` for
            normalisation.

    Returns:
        Weighted, normalised score where lower is better.
    """

    level_weights: MutableMapping[str, float]
    if level in weights and isinstance(weights[level], Mapping):
        level_weights = dict(weights[level])  # type: ignore[index]
    else:
        level_weights = dict(weights)

    score = 0.0
    total_weight = 0.0

    for name, raw_weight in level_weights.items():
        if name not in component_values:
            continue
        value = component_values[name]
        value_t = torch.as_tensor(value, dtype=torch.float32)
        if stats and name in stats:
            mean, std = stats[name]
            if std > 0:
                value_t = (value_t - mean) / std
        score += raw_weight * float(value_t.detach().cpu())
        total_weight += abs(raw_weight)

    if total_weight == 0:
        return score
    return score / total_weight
