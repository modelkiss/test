"""Evaluation utilities for single-sample reconstructions."""
from __future__ import annotations

from typing import Iterable, Mapping

import torch
import torch.nn.functional as F


def evaluate_single_sample(
    real_image: torch.Tensor,
    reconstructed_images: Iterable[torch.Tensor],
    feature_extractor: torch.nn.Module,
    classifier: torch.nn.Module | None = None,
) -> Mapping[str, float]:
    """Compute basic similarity metrics against a reference image."""

    feature_extractor.eval()
    feats_real = feature_extractor(real_image.unsqueeze(0))
    sims = []
    with torch.no_grad():
        for recon in reconstructed_images:
            feat_recon = feature_extractor(recon.unsqueeze(0))
            sims.append(F.cosine_similarity(feat_recon, feats_real).item())
    metrics: dict[str, float] = {"avg_feature_cosine": float(sum(sims) / max(len(sims), 1))}

    if classifier is not None:
        classifier.eval()
        with torch.no_grad():
            preds = classifier(torch.stack(list(reconstructed_images)))
            metrics["avg_confidence"] = preds.softmax(dim=1).max(dim=1).values.mean().item()
    return metrics
