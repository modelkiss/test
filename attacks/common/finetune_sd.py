"""面向数据重建的两阶段扩散模型微调辅助工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import torch

from .grad_diff import ParamDiff


@dataclass
class FeatureAlignment:
    """表示语义 / 外观信号及其合成目标。"""

    semantic: ParamDiff
    appearance: ParamDiff
    fused: ParamDiff


@dataclass
class FinetuneRecipe:
    """描述单次微调所需的关键信息。"""

    class_priors: Sequence[int]
    guidance_scale: float
    max_epochs: int
    notes: List[str]


def build_feature_alignment(
    semantic_signal: ParamDiff, appearance_signal: ParamDiff, *, appearance_scale: float = 0.35
) -> FeatureAlignment:
    """将语义 / 外观梯度差对齐到统一特征空间。"""

    fused: ParamDiff = {}

    for name, tensor in semantic_signal.items():
        fused[name] = tensor.clone()

    for name, tensor in appearance_signal.items():
        if name in fused:
            fused[name] = fused[name] + appearance_scale * tensor
        else:
            fused[name] = appearance_scale * tensor

    return FeatureAlignment(semantic=semantic_signal, appearance=appearance_signal, fused=fused)


def build_finetune_recipes(
    target_classes: Iterable[int],
    *,
    primary_guidance: float = 3.0,
    refinement_guidance: float = 1.5,
    primary_epochs: int = 10,
    refinement_epochs: int = 5,
) -> List[FinetuneRecipe]:
    """生成两阶段的微调计划。"""

    target_list = list(target_classes)
    if not target_list:
        target_list = [0]

    stage1 = FinetuneRecipe(
        class_priors=target_list,
        guidance_scale=primary_guidance,
        max_epochs=primary_epochs,
        notes=[
            "Stage1: enforce class priors and semantic dominance.",
            "Use stronger guidance to引导扩散模型聚焦遗忘类别。",
        ],
    )

    stage2 = FinetuneRecipe(
        class_priors=target_list,
        guidance_scale=refinement_guidance,
        max_epochs=refinement_epochs,
        notes=[
            "Stage2: fine-grained refinement with gradient-aligned samples.",
            "Reduce guidance strength to preserve appearance cues while keeping semantics.",
        ],
    )

    return [stage1, stage2]


def apply_gradient_projection(
    images: torch.Tensor, fused_signal: Mapping[str, torch.Tensor], scale: float = 0.1
) -> torch.Tensor:
    """将梯度差信号投影到生成图像上，用于精细化微调。"""

    if images.numel() == 0 or not fused_signal:
        return images

    normed = []
    for img in images:
        adjusted = img
        for tensor in fused_signal.values():
            adjusted = adjusted + scale * tensor.mean()
        normed.append(adjusted)
    return torch.stack(normed)


__all__ = [
    "FeatureAlignment",
    "FinetuneRecipe",
    "build_feature_alignment",
    "build_finetune_recipes",
    "apply_gradient_projection",
]
