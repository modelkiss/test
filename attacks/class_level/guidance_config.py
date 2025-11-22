"""类级攻击：类别先验、置信度差、语义对齐的权重配置。"""
from __future__ import annotations

from typing import Dict, Mapping


def _merge_weights(defaults: Dict[str, float], overrides: Mapping[str, float]) -> Dict[str, float]:
    """Merge user-provided overrides into default weights."""

    merged = defaults.copy()
    for key, value in overrides.items():
        merged[key] = float(value)
    return merged


def get_class_level_stage1_weights(overrides: Mapping[str, float] | None = None) -> Dict[str, float]:
    """阶段一：类别先验与置信度差主导。"""

    defaults = {
        "w_diff": 1.0,
        "w_prior": 1.5,
        "w_sem": 1.0,
        "w_app": 0.1,
        "w_conf": 1.0,
        "w_grad": 0.5,
    }
    return _merge_weights(defaults, overrides or {})


def get_class_level_stage2_weights(overrides: Mapping[str, float] | None = None) -> Dict[str, float]:
    """阶段二：偏向重构与 CE，保留轻量语义对齐。"""

    defaults = {
        "w_diff": 1.2,
        "w_prior": 1.0,
        "w_sem": 0.3,
        "w_app": 0.1,
        "w_conf": 1.2,
        "w_grad": 0.2,
    }
    return _merge_weights(defaults, overrides or {})
