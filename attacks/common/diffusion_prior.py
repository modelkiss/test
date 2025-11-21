"""Diffusion prior that performs gradient guidance in pixel/noise space."""
from __future__ import annotations

from typing import Callable, List, Sequence

import torch


class DiffusionPrior:
    """Lightweight guided sampling loop for OpenAI-style guided diffusion."""

    def __init__(
        self,
        *,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        model_name: str = "openai_guided_diffusion_cifar",
        image_size: int = 32,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        total_diffusion_steps: int = 1000,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.total_diffusion_steps = total_diffusion_steps
        self.device = torch.device(device)

        self.betas = torch.linspace(1e-4, 0.02, total_diffusion_steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.timesteps = self._subsample_timesteps(num_inference_steps)

    def _subsample_timesteps(self, num_steps: int) -> Sequence[int]:
        step_indices = torch.linspace(
            self.total_diffusion_steps - 1, 0, num_steps, dtype=torch.long
        )
        return step_indices.tolist()

    def _predict_noise(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        if self.model is None:
            return torch.zeros_like(x_t)
        t_batch = torch.full((x_t.shape[0],), t, device=self.device, dtype=torch.long)
        return self.model(x_t, t_batch)

    def _predict_x0(self, x_t: torch.Tensor, eps: torch.Tensor, t: int) -> torch.Tensor:
        alpha_bar = self.alpha_bars[t]
        return (x_t - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)

    def _ddpm_step(
        self, x_t: torch.Tensor, eps: torch.Tensor, t: int, next_t: int, noise: torch.Tensor
    ) -> torch.Tensor:
        if next_t < 0:
            return self._predict_x0(x_t, eps, t)

        alpha_bar_next = self.alpha_bars[next_t]
        mean = torch.sqrt(alpha_bar_next) * self._predict_x0(x_t, eps, t) + torch.sqrt(
            1 - alpha_bar_next
        ) * eps

        beta_t = self.betas[t]
        alpha_bar_t = self.alpha_bars[t]
        sigma_t = torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar_t) * beta_t)
        return mean + sigma_t * noise

    def _to_images(self, latents: torch.Tensor) -> torch.Tensor:
        return ((latents.clamp(-1, 1) + 1) / 2).clamp(0, 1)

    def guided_sampling(
        self,
        *,
        guidance_fn: Callable[[torch.Tensor], torch.Tensor],
        batch_size: int,
        step_size: float = 0.02,
        num_inference_steps: int | None = None,
        save_history: bool = False,
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        """Run classifier/classifier-free guidance directly in pixel space."""

        steps = num_inference_steps or self.num_inference_steps
        timesteps: List[int] = list(self._subsample_timesteps(steps))
        x_t = torch.randn(
            (batch_size, 3, self.image_size, self.image_size), device=self.device
        )
        history: List[torch.Tensor] = []

        for idx, t in enumerate(timesteps):
            x_t = x_t.detach().requires_grad_(True)
            pred_eps = self._predict_noise(x_t, int(t))
            current_images = self._to_images(self._predict_x0(x_t, pred_eps, int(t)))

            loss = guidance_fn(current_images) * self.guidance_scale
            loss.backward()

            with torch.no_grad():
                grad = x_t.grad if x_t.grad is not None else torch.zeros_like(x_t)
                guided = x_t - step_size * grad

                next_t = int(timesteps[idx + 1]) if idx + 1 < len(timesteps) else -1
                noise = torch.randn_like(guided) if next_t >= 0 else torch.zeros_like(guided)
                x_t = self._ddpm_step(guided, pred_eps, int(t), next_t, noise)

                if save_history:
                    history.append(current_images.detach().cpu())

        final_images = self._to_images(x_t).detach().cpu()
        return final_images, history
