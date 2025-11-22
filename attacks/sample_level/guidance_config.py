"""样本级攻击：关注单样本梯度匹配或相似度的权重。"""

from __future__ import annotations

from typing import Dict, Mapping


def _merge_weights(defaults: Dict[str, float], overrides: Mapping[str, float]) -> Dict[str, float]:
    """Merge user-provided overrides into default weights."""

    merged = defaults.copy()
    for key, value in overrides.items():
        merged[key] = float(value)
    return merged


def get_sample_level_stage1_weights(overrides: Mapping[str, float] | None = None) -> Dict[str, float]:
    """阶段一：重度依赖单样本梯度匹配与置信度差。"""

    defaults = {
        "w_diff": 1.0,
        "w_prior": 0.5,
        "w_sem": 0.5,
        "w_app": 1.0,
        "w_conf": 1.0,
        "w_grad": 1.5,
    }
    return _merge_weights(defaults, overrides or {})


def get_sample_level_stage2_weights(overrides: Mapping[str, float] | None = None) -> Dict[str, float]:
    """阶段二：突出重构 + 单样本 CE，保留轻量语义 / 外观对齐。"""

    defaults = {
        "w_diff": 1.3,
        "w_prior": 0.3,
        "w_sem": 0.4,
        "w_app": 0.8,
        "w_conf": 1.2,
        "w_grad": 1.0,
    }
    return _merge_weights(defaults, overrides or {})
