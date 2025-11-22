"""Wrapper utilities for running Stable Diffusion with optional external guidance."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
from diffusers import StableDiffusionPipeline


class StableDiffusionWrapper:
    """Thin wrapper around a Stable Diffusion pipeline.

    The wrapper exposes hooks to sample and decode latents while allowing
    custom losses to intervene in the denoising loop.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str,
        image_size: int,
        use_lora: bool = False,
        lora_weights: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        cache_text_embeddings: bool = True,
    ) -> None:
        """Load a Stable Diffusion checkpoint.

        Args:
            ckpt_path: Path to diffusers checkpoint or local weights.
            device: Target device for the underlying modules.
            image_size: Target image size for sampling/decoding.
            use_lora: Whether to load and train LoRA/adapter modules only.
            lora_weights: Optional path to LoRA weights when ``use_lora`` is ``True``.
            dtype: Precision for the model weights.
            cache_text_embeddings: Whether to cache computed text embeddings.
        """

        self.device = torch.device(device)
        self.image_size = image_size
        self.dtype = dtype
        self.cache_text_embeddings = cache_text_embeddings
        self._text_cache: Dict[str, torch.Tensor] = {}

        self.pipeline: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
            ckpt_path, torch_dtype=dtype
        ).to(self.device)
        self.pipeline.vae.eval()
        self.pipeline.unet.eval()
        self.pipeline.text_encoder.eval()
        for module in [self.pipeline.unet, self.pipeline.vae, self.pipeline.text_encoder]:
            module.requires_grad_(False)

        if use_lora:
            if lora_weights:
                try:
                    self.pipeline.load_lora_weights(lora_weights)
                except Exception as exc:  # pragma: no cover - defensive
                    raise RuntimeError(f"Failed to load LoRA weights from {lora_weights}") from exc
            self._freeze_non_lora_parameters()

        self.vae_scale_factor = getattr(self.pipeline, "vae_scale_factor", 8)
        self.latent_channels = self.pipeline.unet.in_channels

    def _freeze_non_lora_parameters(self) -> None:
        """Freeze base model weights, leaving LoRA/adapter parameters trainable."""

        for module in [self.pipeline.unet, self.pipeline.text_encoder, self.pipeline.vae]:
            module.requires_grad_(False)

        for module in [self.pipeline.unet, self.pipeline.text_encoder]:
            for name, parameter in module.named_parameters():
                if "lora" in name or "adapter" in name:
                    parameter.requires_grad = True

    def encode_text(self, prompt: str) -> torch.Tensor:
        """Encode text prompts into conditioning embeddings."""

        if self.cache_text_embeddings and prompt in self._text_cache:
            return self._text_cache[prompt]

        inputs = self.pipeline.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            return_tensors="pt",
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_embeddings = self.pipeline.text_encoder(**inputs)[0].to(self.device, self.dtype)

        if self.cache_text_embeddings:
            self._text_cache[prompt] = text_embeddings
        return text_embeddings

    def sample_latents(self, batch_size: int) -> torch.Tensor:
        """Sample Gaussian noise latents."""

        height = self.image_size // self.vae_scale_factor
        width = self.image_size // self.vae_scale_factor
        return torch.randn(
            (batch_size, self.latent_channels, height, width),
            device=self.device,
            dtype=self.dtype,
        )

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents into images within the ``[0, 1]`` range."""

        scaling_factor = getattr(self.pipeline.vae.config, "scaling_factor", 0.18215)
        latents = latents / scaling_factor
        images = self.pipeline.vae.decode(latents).sample
        images = (images.clamp(-1, 1) + 1) / 2
        return images

    def guided_sampling_with_loss(
        self,
        guidance_fn: Callable[[torch.Tensor], torch.Tensor],
        num_steps: int,
        batch_size: int,
        step_size: float,
        text_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run sampling with an external guidance loss applied at each step."""

        if text_embed is None:
            text_embed = self.encode_text("")
        text_embed = text_embed.to(device=self.device, dtype=self.dtype)
        self.pipeline.scheduler.set_timesteps(num_steps, device=self.device)
        latents = self.sample_latents(batch_size)
        init_noise_sigma = getattr(self.pipeline.scheduler, "init_noise_sigma", 1.0)
        latents = latents * init_noise_sigma

        for timestep in self.pipeline.scheduler.timesteps:
            latents.requires_grad_(True)
            with torch.no_grad():
                latent_model_input = self.pipeline.scheduler.scale_model_input(latents, timestep)
                noise_pred = self.pipeline.unet(
                    latent_model_input, timestep, encoder_hidden_states=text_embed
                ).sample
                latents = self.pipeline.scheduler.step(noise_pred, timestep, latents).prev_sample

            decoded = self.decode_latents(latents)
            loss = guidance_fn(decoded)
            loss.backward()

            with torch.no_grad():
                grad = latents.grad
                if grad is None:
                    raise RuntimeError("No gradient computed for latents during guidance.")
                latents = latents - step_size * grad
            latents = latents.detach()

        images = self.decode_latents(latents)
        return images
