"""Class-level guidance mixing classification and gradient matching losses."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from attacks.common import gradients
from attacks.common.guidance_base import BaseGuidance


class ClassLevelGuidance(BaseGuidance):
    """Drive samples toward forgotten classes and gradient signatures."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_signal: Mapping[str, torch.Tensor],
        target_classes: Iterable[Any],
        config: Mapping[str, Any],
    ) -> None:
        super().__init__(model, target_signal, config)
        self.target_classes = list(target_classes)
        self.layers: Sequence[str] | None = config.get("layers")
        self.match_type: str = config.get("match_type", "l2")
        self.grad_weight: float = float(config.get("grad_weight", 1.0))
        self.cls_weight: float = float(config.get("cls_weight", 1.0))

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.model(images)
        target_labels = torch.tensor(
            [self.target_classes[0]] * images.size(0), device=images.device, dtype=torch.long
        )
        cls_loss = F.cross_entropy(logits, target_labels)

        grads = gradients.compute_model_gradients(
            model=self.model,
            images=images,
            loss_fn=F.cross_entropy,
            target_labels=target_labels,
            layers_to_use=self.layers,
        )
        grad_loss = gradients.gradient_matching_loss(
            grads, self.target_signal, match_type=self.match_type
        )

        total_loss = self.cls_weight * cls_loss + self.grad_weight * grad_loss
        return total_loss
