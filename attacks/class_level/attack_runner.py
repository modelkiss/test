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
        diffusion_cfg: Mapping[str, Any] = config.get("diffusion", {})
        prior = DiffusionPrior(
            model=config.get("diffusion_model"),
            model_name=str(diffusion_cfg.get("model_name", "openai_guided_diffusion")),
            image_size=int(diffusion_cfg.get("image_size", 32)),
            num_inference_steps=int(diffusion_cfg.get("num_inference_steps", 10)),
            guidance_scale=float(diffusion_cfg.get("guidance_scale", 1.0)),
            device=config.get("device", "cpu"),
        )
        steps = int(
            diffusion_cfg.get("num_inference_steps", config.get("num_steps", prior.num_inference_steps))
        )
        batch_size = int(diffusion_cfg.get("batch_size", config.get("batch_size", 4)))
        step_size = float(diffusion_cfg.get("step_size", config.get("step_size", 0.02)))
        images, history = prior.guided_sampling(
            guidance_fn=guidance,
            batch_size=batch_size,
            step_size=step_size,
            num_inference_steps=steps,
            save_history=config.get("save_history", False),
        )
        metrics: Dict[str, float] = {"history_length": float(len(history))}
        results[cls_id] = AttackResult(
            reconstructed_images=images, metrics=metrics, extra_info={"history": history}
        )

    return results
