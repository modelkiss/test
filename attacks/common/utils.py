"""攻击流程通用工具函数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from attacks.common.diffusion_sd import DiffusionCheckpoint, ensure_local_diffusion_checkpoint
from attacks.common.finetune_sd import (
    FeatureAlignment,
    FinetuneRecipe,
    apply_gradient_projection,
    build_feature_alignment,
    build_finetune_recipes,
)
from attacks.common.grad_diff import GradientSignals, compute_weighted_gradient_signals
from federated.state import TrainingLog, UnlearningLog


@dataclass
class ReconstructionAssets:
    """封装数据重建攻击的关键资产。"""

    signals: GradientSignals
    alignment: FeatureAlignment
    diffusion: DiffusionCheckpoint
    finetune_plan: Sequence[FinetuneRecipe]
    notes: List[str]


def prepare_reconstruction_assets(
    config: Mapping[str, Any],
    training_log: TrainingLog,
    unlearning_log: UnlearningLog,
    *,
    target_classes: Iterable[int],
) -> ReconstructionAssets:
    """按既定方案组装攻击所需的全部信息。"""

    keep_last = int(config.get("keep_last_rounds", 5)) if isinstance(config, Mapping) else 5
    alpha = float(config.get("signal_alpha", 0.7)) if isinstance(config, Mapping) else 0.7
    focus_layers = config.get("signal_layers") if isinstance(config, Mapping) else None

    signals = compute_weighted_gradient_signals(
        training_log, unlearning_log, keep_last=keep_last, alpha=alpha, focus_layers=focus_layers
    )

    alignment = build_feature_alignment(signals.semantic_signal, signals.appearance_signal)

    diffusion_cfg = config.get("diffusion", {}) if isinstance(config, Mapping) else {}
    model_name = diffusion_cfg.get("model_name", "stable-diffusion-pretrained")
    diffusion = ensure_local_diffusion_checkpoint(model_name, target_dir=diffusion_cfg.get("target_dir", "SD"))

    finetune_plan = build_finetune_recipes(target_classes)

    notes = [
        f"Collected {len(signals.used_rounds)} snapshots for gradient diff weighting.",
        f"Semantic layers: {list(signals.semantic_signal.keys())[:5]}",
        f"Appearance layers: {list(signals.appearance_signal.keys())[:5]}",
        diffusion.note or "Diffusion checkpoint ready for finetuning.",
    ]

    return ReconstructionAssets(
        signals=signals,
        alignment=alignment,
        diffusion=diffusion,
        finetune_plan=finetune_plan,
        notes=notes,
    )


__all__ = [
    "ReconstructionAssets",
    "prepare_reconstruction_assets",
    "apply_gradient_projection",
    "build_feature_alignment",
    "build_finetune_recipes",
    "DiffusionCheckpoint",
    "FeatureAlignment",
    "FinetuneRecipe",
    "GradientSignals",
]
