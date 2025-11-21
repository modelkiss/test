"""Entry point for client-level reconstruction attacks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import torch

from attacks.common.diffusion_prior import DiffusionPrior
from attacks.client_level.guidance import ClientLevelGuidance
from attacks.client_level.target_builder import build_client_target_signal
from federated.state import TrainingLog, UnlearningLog


@dataclass
class AttackResult:
    reconstructed_images: torch.Tensor
    metrics: Dict[str, float]
    extra_info: Dict[str, Any] | None = None


def run_client_level_attack(
    config: Mapping[str, Any],
    model: torch.nn.Module,
    training_log: TrainingLog,
    unlearning_log: UnlearningLog,
) -> AttackResult:
    """High-level orchestration for a client-level attack run."""

    forgotten_clients = config.get("forgotten_clients", [])
    target_signal = build_client_target_signal(
        training_log, unlearning_log, forgotten_clients, config.get("target_signal", {})
    )

    guidance = ClientLevelGuidance(model, target_signal, config.get("guidance", {}))
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
    return AttackResult(reconstructed_images=images, metrics=metrics, extra_info={"history": history})
