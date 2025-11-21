"""Sample-level guidance with candidate scoring."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from attacks.common import gradients, scoring
from attacks.common.guidance_base import BaseGuidance


class SampleLevelGuidance(BaseGuidance):
    """Compute losses for candidate reconstructions of a single sample."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_signal: Mapping[str, torch.Tensor],
        config: Mapping[str, Any],
        candidate_pool: list[scoring.CandidateRecord] | None = None,
    ) -> None:
        super().__init__(model, target_signal, config)
        self.layers: Sequence[str] | None = config.get("layers")
        self.match_type: str = config.get("match_type", "l2")
        self.grad_weight: float = float(config.get("grad_weight", 1.0))
        self.cls_weight: float = float(config.get("cls_weight", 0.0))
        self.target_label = config.get("target_label")
        self.candidate_pool = candidate_pool if candidate_pool is not None else []
        self.score_cfg = config.get("scoring", {})

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if self.target_label is not None:
            target_labels = torch.tensor(
                [self.target_label] * images.size(0), device=images.device, dtype=torch.long
            )
        else:
            target_labels = None

        grads = gradients.compute_model_gradients(
            model=self.model,
            images=images,
            loss_fn=F.cross_entropy if target_labels is not None else torch.mean,
            target_labels=target_labels,
            layers_to_use=self.layers,
        )
        loss_grad = gradients.gradient_matching_loss(
            grads, self.target_signal, match_type=self.match_type
        )

        loss_cls = None
        if target_labels is not None:
            logits = self.model(images)
            loss_cls = F.cross_entropy(logits, target_labels)

        total_loss = self.grad_weight * loss_grad
        if loss_cls is not None:
            total_loss = total_loss + self.cls_weight * loss_cls

        candidate_records: list[scoring.CandidateRecord] = []
        for idx in range(images.size(0)):
            grad_component = float(loss_grad.detach().cpu())
            cls_component = float(loss_cls.detach().cpu()) if loss_cls is not None else None
            score = scoring.compute_candidate_score(
                grad_component, cls_component, cfg=self.score_cfg
            )
            candidate_records.append(
                scoring.CandidateRecord(
                    image=images[idx].detach().cpu(),
                    loss_gradient_match=grad_component,
                    loss_classification=cls_component,
                    score_total=score,
                )
            )
        scoring.update_topk_candidates(
            self.candidate_pool, candidate_records, k=int(self.score_cfg.get("top_k", 10))
        )

        return total_loss
