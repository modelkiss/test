"""Evaluation stubs for class-level attack performance."""
from __future__ import annotations

from typing import Iterable, Mapping

import torch


def evaluate_reconstructed_class(
    real_loader: Iterable,
    reconstructed_images: torch.Tensor,
    feature_extractor: torch.nn.Module,
    classifier: torch.nn.Module,
) -> Mapping[str, float]:
    """Return basic metrics comparing reconstructed images to real class samples."""

    feature_extractor.eval()
    classifier.eval()
    with torch.no_grad():
        feats_recon = feature_extractor(reconstructed_images)
        preds = classifier(reconstructed_images).softmax(dim=1)
        confidence = preds.max(dim=1).values.mean().item()
        feature_norm = feats_recon.norm(dim=1).mean().item()
    return {"avg_confidence": confidence, "feature_norm": feature_norm}
