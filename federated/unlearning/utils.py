"""Utility helpers for unlearning scoring pipelines."""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

from ..state import TrainingLog, UnlearningLog

ModelSnapshot = Mapping[str, torch.Tensor]


def _clone_state_dict(snapshot: ModelSnapshot) -> ModelSnapshot:
    """Return a detached CPU copy of a state dict-like mapping."""

    return {k: v.clone().detach().cpu() for k, v in snapshot.items()}


def collect_pre_post_models(
    training_log: TrainingLog,
    unlearning_log: UnlearningLog,
    pre_rounds: int,
    post_rounds: int,
) -> Tuple[List[ModelSnapshot], List[ModelSnapshot]]:
    """Return recent snapshots from training and unlearning traces."""

    def _tail_snapshots(snapshots: Sequence[ModelSnapshot], count: int) -> List[ModelSnapshot]:
        if count <= 0:
            return []
        if len(snapshots) >= count:
            return list(snapshots[-count:])
        return list(snapshots)

    pre_snapshots = _tail_snapshots(training_log.global_model_snapshots, pre_rounds)
    if not pre_snapshots and training_log.final_model_state is not None:
        pre_snapshots = [training_log.final_model_state]

    post_snapshots = _tail_snapshots(
        unlearning_log.global_model_snapshots_after_unlearning, post_rounds
    )
    if not post_snapshots and unlearning_log.final_model_state is not None:
        post_snapshots = [unlearning_log.final_model_state]

    return [_clone_state_dict(s) for s in pre_snapshots], [
        _clone_state_dict(s) for s in post_snapshots
    ]


def extract_features(
    model: torch.nn.Module, x: torch.Tensor, layers: Sequence[str]
) -> Dict[str, torch.Tensor]:
    """Collect forward activations for specified module names."""

    features: Dict[str, torch.Tensor] = {}
    hooks = []
    modules = dict(model.named_modules())
    for name in layers:
        if name not in modules:
            raise ValueError(f"Layer {name} not found in model modules")

        def _hook(module, inputs, output, key=name):  # type: ignore[override]
            features[key] = output.detach()

        hooks.append(modules[name].register_forward_hook(_hook))

    model.eval()
    with torch.no_grad():
        model(x)

    for hook in hooks:
        hook.remove()
    return features


def batched_forward(
    model: torch.nn.Module,
    dataloader: Iterable,
    fn: Callable[[torch.nn.Module, Tuple], torch.Tensor],
    *,
    device: str = "cpu",
) -> List[torch.Tensor]:
    """Apply ``fn`` across a dataloader with device management."""

    results: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch_on_device = to_device(batch, device)
            results.append(fn(model, batch_on_device))
    return results


def to_device(batch, device: str):
    """Recursively move a batch structure to ``device``."""

    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, Mapping):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        converted = [to_device(v, device) for v in batch]
        return type(batch)(converted)
    return batch


def detach_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Detach tensor and move to CPU."""

    return tensor.detach().cpu()
