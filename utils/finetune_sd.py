"""Two-stage Stable Diffusion fine-tuning utilities.

This module implements a pair of high-level routines for adapting a
``StableDiffusionWrapper`` with guidance from a frozen feature extractor ``F``.
The routines follow a coarse-to-fine recipe:

1. ``stage1_finetune_sd`` performs global bias correction with a mix of
   distribution regularization, semantic/appearance alignment, prior matching,
   and confidence-based guidance.
2. ``stage2_refine_sd_with_pseudo`` refines the model on high-quality pseudo
   samples with a smaller learning rate while keeping the diffusion prior in
   check.

Both stages are intentionally flexible: optional targets or weights simply
result in zeroed losses, allowing callers to supply partial guidance without
code changes.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import torch
import torch.nn.functional as F

from .diffusion_sd import StableDiffusionWrapper


LossDict = MutableMapping[str, torch.Tensor]


def stage1_finetune_sd(
    sd: StableDiffusionWrapper,
    F_model: torch.nn.Module,
    targets: Mapping[str, Any],
    config: Mapping[str, Any],
) -> StableDiffusionWrapper:
    """Run the first-stage coarse fine-tuning loop for Stable Diffusion.

    Args:
        sd: Trainable Stable Diffusion wrapper. LoRA parameters or selected
            U-Net blocks should already be marked as trainable via
            ``requires_grad``.
        F_model: Frozen feature extractor/classifier used to provide guidance.
        targets: Target statistics including semantic and appearance anchors as
            well as optional prior distributions.
        config: Hyperparameters controlling optimization (learning rate, steps,
            batch size) and weighting for the various loss components.

    Returns:
        The same Stable Diffusion wrapper after in-place updates.
    """

    device = sd.device
    num_steps = int(config.get("num_steps", 100))
    batch_size = int(config.get("batch_size", 1))
    learning_rate = float(config.get("learning_rate", 1e-4))
    level = config.get("level", "class")

    loss_weights = _extract_loss_weights(config)
    optimizer = torch.optim.Adam(_collect_trainable_parameters(sd), lr=learning_rate)

    pre_models = list(config.get("pre_models", []))
    post_models = list(config.get("post_models", []))

    sd.pipeline.unet.train()

    for step in range(num_steps):
        optimizer.zero_grad()

        latents = sd.sample_latents(batch_size)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            sd.pipeline.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
        )
        noisy_latents = sd.pipeline.scheduler.add_noise(latents, noise, timesteps)

        text_embed = sd.encode_text(config.get("prompt", ""))
        text_embed = text_embed.detach()

        pred_noise = sd.pipeline.unet(
            noisy_latents, timesteps, encoder_hidden_states=text_embed
        ).sample
        loss_diff_reg = F.mse_loss(pred_noise, noise)

        decoded_images = sd.decode_latents(latents.detach())
        frozen_outputs = _run_frozen_model(F_model, decoded_images)

        prior_before = _aggregate_model_predictions(pre_models, decoded_images)
        prior_after = _aggregate_model_predictions(post_models, decoded_images)

        loss_semantic = _semantic_alignment_loss(
            frozen_outputs.get("semantic"), targets.get("f_target_semantic"), device
        )
        loss_appearance = _semantic_alignment_loss(
            frozen_outputs.get("appearance"), targets.get("g_target_appearance"), device
        )
        loss_prior = _prior_loss(
            prior_before if prior_after is None else prior_after,
            targets.get("prior", {}),
            level,
            device,
        )
        loss_confidence = _confidence_loss(
            frozen_outputs.get("logits"), targets, level, device
        )

        total_loss = (
            loss_weights["diff_reg"] * loss_diff_reg
            + loss_weights["semantic"] * loss_semantic
            + loss_weights["appearance"] * loss_appearance
            + loss_weights["prior"] * loss_prior
            + loss_weights["confidence"] * loss_confidence
        )

        total_loss.backward()

        optimizer.step()

    sd.pipeline.unet.eval()
    return sd


def stage2_refine_sd_with_pseudo(
    sd: StableDiffusionWrapper,
    F_model: torch.nn.Module,
    pseudo_images: Sequence[torch.Tensor],
    pseudo_labels: Optional[Sequence[torch.Tensor]],
    targets: Mapping[str, Any],
    config: Mapping[str, Any],
) -> StableDiffusionWrapper:
    """Second-stage refinement using high-confidence pseudo samples.

    Args:
        sd: Stable Diffusion wrapper already adapted by ``stage1_finetune_sd``.
        F_model: Frozen model that provides semantic guidance and classification
            logits.
        pseudo_images: Collection of pseudo-labeled images selected from the
            first stage.
        pseudo_labels: Optional labels aligned with ``pseudo_images``. When
            ``None``, classification and semantic alignment terms are skipped.
        targets: Same structure as the first stage, reused for light semantic
            nudging.
        config: Hyperparameters for refinement (smaller learning rate/steps,
            batch size, and weighting).

    Returns:
        The refined Stable Diffusion wrapper after in-place updates.
    """

    if not pseudo_images:
        return sd

    device = sd.device
    num_steps = int(config.get("num_steps", 50))
    batch_size = int(config.get("batch_size", 1))
    learning_rate = float(config.get("learning_rate", 5e-5))
    reg_fraction = float(config.get("regularization_fraction", 0.5))
    level = config.get("level", "class")

    loss_weights = _extract_loss_weights(config)
    optimizer = torch.optim.Adam(_collect_trainable_parameters(sd), lr=learning_rate)
    scaling_factor = getattr(sd.pipeline.vae.config, "scaling_factor", 0.18215)

    sd.pipeline.unet.train()

    for step in range(num_steps):
        optimizer.zero_grad()

        pseudo_batch, label_batch = _sample_pseudo_batch(
            pseudo_images, pseudo_labels, batch_size
        )
        reg_batch_size = max(1, int(batch_size * reg_fraction))
        reg_latents = sd.sample_latents(reg_batch_size)
        reg_noise = torch.randn_like(reg_latents)
        reg_timesteps = torch.randint(
            0,
            sd.pipeline.scheduler.config.num_train_timesteps,
            (reg_batch_size,),
            device=device,
        )
        reg_noisy_latents = sd.pipeline.scheduler.add_noise(
            reg_latents, reg_noise, reg_timesteps
        )
        text_embed = sd.encode_text(config.get("prompt", ""))
        text_embed = text_embed.detach()

        reg_pred_noise = sd.pipeline.unet(
            reg_noisy_latents, reg_timesteps, encoder_hidden_states=text_embed
        ).sample
        loss_diff_reg = F.mse_loss(reg_pred_noise, reg_noise)

        pseudo_images_tensor = pseudo_batch.to(device=device, dtype=sd.dtype)
        encoded = sd.pipeline.vae.encode(pseudo_images_tensor).latent_dist.sample()
        pseudo_latents = encoded * scaling_factor

        pseudo_noise = torch.randn_like(pseudo_latents)
        pseudo_timesteps = torch.randint(
            0,
            sd.pipeline.scheduler.config.num_train_timesteps,
            (pseudo_latents.shape[0],),
            device=device,
        )
        noisy_pseudo = sd.pipeline.scheduler.add_noise(
            pseudo_latents, pseudo_noise, pseudo_timesteps
        )
        pred_noise = sd.pipeline.unet(
            noisy_pseudo, pseudo_timesteps, encoder_hidden_states=text_embed
        ).sample
        loss_reconstruction = F.mse_loss(pred_noise, pseudo_noise)

        frozen_outputs = _run_frozen_model(F_model, pseudo_images_tensor)
        logits = frozen_outputs.get("logits")

        loss_ce = _classification_loss(logits, label_batch, device)
        loss_semantic = _semantic_alignment_loss(
            frozen_outputs.get("semantic"), targets.get("f_target_semantic"), device
        )
        loss_confidence = _confidence_loss(logits, targets, level, device)

        total_loss = (
            loss_weights["diff_reg"] * loss_diff_reg
            + loss_weights["reconstruction"] * loss_reconstruction
            + loss_weights["classification"] * loss_ce
            + loss_weights["semantic"] * loss_semantic
            + loss_weights["confidence"] * loss_confidence
        )

        total_loss.backward()
        optimizer.step()

    sd.pipeline.unet.eval()
    return sd

def _collect_trainable_parameters(sd: StableDiffusionWrapper) -> list[torch.nn.Parameter]:
    """Gather parameters that should be optimized."""

    parameters = [
        p
        for module in [sd.pipeline.unet, sd.pipeline.text_encoder, sd.pipeline.vae]
        for p in module.parameters()
        if p.requires_grad
    ]
    if not parameters:
        parameters = list(sd.pipeline.unet.parameters())
    return parameters


def _run_frozen_model(
    model: torch.nn.Module, images: torch.Tensor
) -> Dict[str, Optional[torch.Tensor]]:
    """Execute the frozen guidance model and normalize its outputs."""

    with torch.no_grad():
        output = model(images)

    semantic = None
    appearance = None
    logits = None

    if isinstance(output, dict):
        semantic = output.get("semantic") or output.get("f")
        appearance = output.get("appearance") or output.get("g")
        logits = output.get("logits") or output.get("pred") or output.get("logit")
    elif isinstance(output, (list, tuple)):
        if len(output) >= 1:
            semantic = output[0]
        if len(output) >= 2:
            appearance = output[1]
        if len(output) >= 3:
            logits = output[2]
    elif torch.is_tensor(output):
        logits = output

    return {"semantic": semantic, "appearance": appearance, "logits": logits}


def _aggregate_model_predictions(
    models: Iterable[torch.nn.Module], images: torch.Tensor
) -> Optional[torch.Tensor]:
    """Aggregate softmax predictions across auxiliary models."""

    preds = []
    for model in models:
        with torch.no_grad():
            logits = model(images)
            if not torch.is_tensor(logits):
                continue
            preds.append(logits.softmax(dim=-1))
    if not preds:
        return None
    return torch.stack(preds, dim=0).mean(dim=0)


def _prior_loss(
    prediction: Optional[torch.Tensor],
    prior_targets: Mapping[str, Any],
    level: str,
    device: torch.device,
) -> torch.Tensor:
    """Compute KL divergence to the provided prior distribution."""

    if prediction is None:
        return torch.zeros((), device=device)
    target_distribution = prior_targets.get(level)
    if target_distribution is None:
        return torch.zeros((), device=device)
    target_distribution = target_distribution.to(device=prediction.device)
    log_pred = prediction.clamp_min(1e-8).log()
    return F.kl_div(log_pred, target_distribution, reduction="batchmean")


def _semantic_alignment_loss(
    features: Optional[torch.Tensor], target: Optional[torch.Tensor], device: torch.device
) -> torch.Tensor:
    """Mean-squared error between pooled features and target anchors."""

    if features is None or target is None:
        return torch.zeros((), device=device)
    pooled = features.mean(dim=0, keepdim=True)
    return F.mse_loss(pooled, target.to(device=pooled.device))


def _confidence_loss(
    logits: Optional[torch.Tensor],
    targets: Mapping[str, Any],
    level: str,
    device: torch.device,
) -> torch.Tensor:
    """Encourage higher confidence for the target entity than non-targets."""

    if logits is None:
        return torch.zeros((), device=device)

    target_indices = targets.get("target_indices") or targets.get(level)
    if target_indices is None:
        return torch.zeros((), device=device)

    probs = logits.softmax(dim=-1)
    target_prob = probs[..., target_indices].mean()

    non_target_mask = torch.ones_like(probs, dtype=torch.bool)
    non_target_mask[..., target_indices] = False
    non_target_probs = probs[non_target_mask]
    if non_target_probs.numel() == 0:
        return torch.zeros((), device=device)
    non_target_prob = non_target_probs.mean()

    mode = targets.get("confidence_mode", "gap")
    if mode == "stable":
        return torch.abs(target_prob - non_target_prob)
    return F.relu(non_target_prob - target_prob)


def _classification_loss(
    logits: Optional[torch.Tensor], labels: Optional[torch.Tensor], device: torch.device
) -> torch.Tensor:
    """Standard cross-entropy classification loss."""

    if logits is None or labels is None:
        return torch.zeros((), device=device)
    return F.cross_entropy(logits, labels.to(device=logits.device))


def _extract_loss_weights(config: Mapping[str, Any]) -> Dict[str, float]:
    """Gather weighted coefficients for all supported losses."""

    guidance = config.get("guidance_config", {})
    return {
        "diff_reg": float(guidance.get("w_diff", config.get("w_diff", 1.0))),
        "prior": float(guidance.get("w_prior", config.get("w_prior", 1.0))),
        "semantic": float(guidance.get("w_semantic", config.get("w_semantic", 1.0))),
        "appearance": float(
            guidance.get("w_appearance", config.get("w_appearance", 0.0))
        ),
        "confidence": float(
            guidance.get("w_confidence", config.get("w_confidence", 1.0))
        ),
        "reconstruction": float(
            guidance.get("w_reconstruction", config.get("w_reconstruction", 1.0))
        ),
        "classification": float(
            guidance.get("w_classification", config.get("w_classification", 1.0))
        ),
    }


def _sample_pseudo_batch(
    images: Sequence[torch.Tensor],
    labels: Optional[Sequence[torch.Tensor]],
    batch_size: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Randomly sample a pseudo batch and align labels if available."""

    indices = torch.randperm(len(images))[:batch_size]
    batch = torch.stack([images[idx] for idx in indices], dim=0)
    if labels is None:
        return batch, None
    label_tensor = torch.stack([labels[idx] for idx in indices], dim=0)
    return batch, label_tensor
