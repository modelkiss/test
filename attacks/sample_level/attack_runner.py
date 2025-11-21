"""Coordinate sample-level reconstruction attacks and candidate selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import torch

from attacks.common.diffusion_prior import DiffusionPrior
from attacks.common import scoring
from attacks.sample_level.guidance import SampleLevelGuidance
from attacks.sample_level.target_builder import build_sample_target_signal
from federated.state import TrainingLog, UnlearningLog


@dataclass
class AttackResult:
    top_candidates: list[scoring.CandidateRecord]
    metrics: Dict[str, float]
    extra_info: Dict[str, Any] | None = None


def run_sample_level_attack(
    config: Mapping[str, Any],
    model: torch.nn.Module,
    training_log: TrainingLog,
    unlearning_log: UnlearningLog,
) -> AttackResult:
    """Generate and score candidate reconstructions for a forgotten sample."""

    forgotten_sample_id = config.get("forgotten_sample")
    target_signal = build_sample_target_signal(
        training_log, unlearning_log, forgotten_sample_id, config.get("target_signal", {})
    )

    candidate_pool: list[scoring.CandidateRecord] = []
    guidance = SampleLevelGuidance(
        model=model,
        target_signal=target_signal,
        config=config.get("guidance", {}),
        candidate_pool=candidate_pool,
    )
    prior = DiffusionPrior(
        latent_shape=tuple(config.get("latent_shape", (3, 32, 32))),
        decoder=config.get("decoder"),
        device=config.get("device", "cpu"),
    )

    total_candidates = int(config.get("total_candidates", 64))
    batch_size = int(config.get("batch_size", 8))
    steps = int(config.get("num_steps", 8))

    generated = 0
    while generated < total_candidates:
        current_batch = min(batch_size, total_candidates - generated)
        images, _ = prior.guided_sampling(
            guidance_fn=guidance,
            num_steps=steps,
            batch_size=current_batch,
            opt_config=config.get("optim", {}),
            save_history=False,
        )
        guidance.postprocess(images)
        generated += current_batch

    top_k = int(config.get("top_k", 5))
    best = scoring.select_best_candidates(candidate_pool, k=top_k)
    metrics: Dict[str, float] = {"generated": float(generated)}
    return AttackResult(top_candidates=best, metrics=metrics, extra_info={})
