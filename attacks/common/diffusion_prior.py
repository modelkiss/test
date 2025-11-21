"""Lightweight diffusion-style prior with generic guided sampling."""
from __future__ import annotations

from typing import Callable, List, Mapping, Tuple

import torch


class DiffusionPrior:
    """Encapsulate a diffusion-like latent and guided optimization loop."""

    def __init__(
        self,
        latent_shape: Tuple[int, ...] = (3, 32, 32),
        decoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.latent_shape = latent_shape
        self.decoder = decoder
        self.device = torch.device(device)

    def sample_latent(self, batch_size: int) -> torch.Tensor:
        """Return a batch of initial latent tensors."""

        latent = torch.randn((batch_size, *self.latent_shape), device=self.device)
        latent.requires_grad_(True)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors into image tensors."""

        if self.decoder is None:
            return latent
        return self.decoder(latent)

    def guided_sampling(
        self,
        guidance_fn: Callable[[torch.Tensor], torch.Tensor],
        num_steps: int,
        batch_size: int,
        opt_config: Mapping[str, object] | None = None,
        save_history: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run gradient-based guidance over latents to reconstruct images."""

        opt_config = opt_config or {}
        latent = self.sample_latent(batch_size)
        optimizer_name = opt_config.get("optimizer", "adam").lower()
        lr = float(opt_config.get("lr", 1e-2))

        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD([latent], lr=lr)
        else:
            optimizer = torch.optim.Adam([latent], lr=lr)

        history: List[torch.Tensor] = []
        for _ in range(num_steps):
            optimizer.zero_grad()
            images = self.decode(latent)
            loss = guidance_fn(images)
            loss.backward()
            optimizer.step()

            if save_history:
                history.append(images.detach().cpu())

        final_images = self.decode(latent).detach().cpu()
        return final_images, history
