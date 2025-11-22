"""客户端攻击的损失权重与调度策略。"""
from __future__ import annotations

from typing import Dict, Mapping


def _merge_weights(defaults: Dict[str, float], overrides: Mapping[str, float]) -> Dict[str, float]:
    """Merge user-provided overrides into default weights."""

    merged = defaults.copy()
    for key, value in overrides.items():
        merged[key] = float(value)
    return merged


def get_client_level_stage1_weights(overrides: Mapping[str, float] | None = None) -> Dict[str, float]:
    """Stage-1 loss weights for client-level attacks."""

    defaults = {
        "w_diff": 1.0,
        "w_prior": 1.0,
        "w_sem": 0.5,
        "w_app": 0.2,
        "w_conf": 0.3,
        "w_grad": 0.5,
    }
    return _merge_weights(defaults, overrides or {})


def get_client_level_stage2_weights(overrides: Mapping[str, float] | None = None) -> Dict[str, float]:
    """Stage-2 loss weights that emphasize reconstruction and priors."""

    defaults = {
        "w_diff": 0.5,
        "w_prior": 1.5,
        "w_sem": 1.0,
        "w_app": 0.5,
        "w_conf": 0.2,
        "w_grad": 0.2,
    }
    return _merge_weights(defaults, overrides or {})
