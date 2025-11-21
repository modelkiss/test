"""Evaluation helpers for client-level reconstruction quality."""
from __future__ import annotations

from typing import Iterable, Mapping

import torch


def compute_feature_distribution_distance(
    real_loader: Iterable, reconstructed_images: torch.Tensor, feature_extractor: torch.nn.Module
) -> float:
    """Estimate distribution mismatch in a feature space."""

    feature_extractor.eval()
    with torch.no_grad():
        recon_features = feature_extractor(reconstructed_images).mean(dim=0)
        real_features = []
        for batch in real_loader:
            images, _ = batch
            real_features.append(feature_extractor(images))
        if real_features:
            real_mean = torch.cat(real_features, dim=0).mean(dim=0)
            return torch.norm(real_mean - recon_features).item()
    return 0.0


def summarize_metrics(custom_metrics: Mapping[str, float] | None = None) -> Mapping[str, float]:
    base = {"placeholder": 1.0}
    if custom_metrics:
        base.update(custom_metrics)
    return base
