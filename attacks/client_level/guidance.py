"""Client-level guidance combining gradient matching and diversity regularizers."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from attacks.common import gradients
from attacks.common.guidance_base import BaseGuidance


class ClientLevelGuidance(BaseGuidance):
    """Guidance that aligns generated images with forgotten client signals."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_signal: Mapping[str, torch.Tensor],
        config: Mapping[str, Any],
    ) -> None:
        super().__init__(model, target_signal, config)
        self.layers: Sequence[str] | None = config.get("layers")
        self.match_type: str = config.get("match_type", "l2")
        self.grad_weight: float = float(config.get("grad_weight", 1.0))
        self.diversity_weight: float = float(config.get("diversity_weight", 0.0))

    def _diversity_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        cov = centered.T @ centered / max(1, embeddings.shape[0])
        return -cov.diag().mean()

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model(images)

        def surrogate_loss(logits: torch.Tensor) -> torch.Tensor:
            return logits.mean()

        grads = gradients.compute_model_gradients(
            model=self.model,
            images=images,
            loss_fn=surrogate_loss,
            layers_to_use=self.layers,
        )
        loss_grad = gradients.gradient_matching_loss(
            grads, self.target_signal, match_type=self.match_type
        )

        total_loss = self.grad_weight * loss_grad

        if self.diversity_weight > 0:
            embeddings = outputs
            if embeddings.dim() > 2:
                embeddings = embeddings.flatten(start_dim=1)
            total_loss = total_loss + self.diversity_weight * self._diversity_loss(embeddings)

        return total_loss
