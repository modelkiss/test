"""Coordinate class-level reconstruction attacks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import torch

from attacks.class_level.guidance import ClassLevelGuidance
from attacks.class_level.target_builder import build_class_target_signal
from attacks.common.diffusion_prior import DiffusionPrior
from federated.state import TrainingLog, UnlearningLog


@dataclass
class AttackResult:
    reconstructed_images: torch.Tensor
    metrics: Dict[str, float]
    extra_info: Dict[str, Any] | None = None


def run_class_level_attack(
    config: Mapping[str, Any],
    model: torch.nn.Module,
    training_log: TrainingLog,
    unlearning_log: UnlearningLog,
) -> Dict[Any, AttackResult]:
    """Run a class-level attack for all forgotten classes."""

    forgotten_classes = config.get("forgotten_classes", [])
    target_signal = build_class_target_signal(
        training_log, unlearning_log, forgotten_classes, config.get("target_signal", {})
    )

    results: Dict[Any, AttackResult] = {}
    for cls_id in forgotten_classes:
        guidance = ClassLevelGuidance(
            model=model,
            target_signal=target_signal,
            target_classes=[cls_id],
            config=config.get("guidance", {}),
        )
        prior = DiffusionPrior(
            latent_shape=tuple(config.get("latent_shape", (3, 32, 32))),
            decoder=config.get("decoder"),
            device=config.get("device", "cpu"),
        )
        steps = int(config.get("num_steps", 10))
        batch_size = int(config.get("batch_size", 4))
        images, history = prior.guided_sampling(
            guidance_fn=guidance,
            num_steps=steps,
            batch_size=batch_size,
            opt_config=config.get("optim", {}),
            save_history=config.get("save_history", False),
        )
        metrics: Dict[str, float] = {"history_length": float(len(history))}
        results[cls_id] = AttackResult(
            reconstructed_images=images, metrics=metrics, extra_info={"history": history}
        )

    return results
