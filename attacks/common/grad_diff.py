"""多轮参数差 / 梯度差信号准备工具。

该模块为数据重建攻击提供以下功能：

* 自动从训练 / 遗忘日志中收集最近几轮的模型快照；
* 根据权重衰减系数计算加权参数差（可视为梯度差的近似）；
* 支持按层名前缀筛选需要关注的层；
* 将参数差信号拆分为“语义”（深层）与“外观”（浅层）两类，便于后续特征对齐。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

import torch

from federated.grad_diff import (
    build_weighted_param_diff,
    compute_param_diff,
    split_param_diff_into_groups,
)
from federated.state import TrainingLog, UnlearningLog


ParamDiff = Dict[str, torch.Tensor]


@dataclass
class GradientSignals:
    """聚合后的梯度差信号。"""

    weighted_diff: ParamDiff
    semantic_signal: ParamDiff
    appearance_signal: ParamDiff
    used_rounds: Sequence[int]


def collect_recent_snapshots(
    training_log: TrainingLog, unlearning_log: UnlearningLog, keep_last: int = 5
) -> Mapping[int, Mapping[str, torch.Tensor]]:
    """收集用于梯度差计算的模型快照。

    选择策略：
    1) 取训练阶段最后 ``keep_last`` 个全局模型快照，若不足则退化为最终模型；
    2) 取遗忘阶段及其后的快照（同样最多 ``keep_last`` 个）。
    第一项作为 ``t=0`` 的基准，后续快照按时间顺序标号 ``1..n``。
    """

    if keep_last <= 0:
        raise ValueError("keep_last must be positive")

    baseline_candidates = training_log.global_model_snapshots[-keep_last :]
    if not baseline_candidates and training_log.final_model_state is not None:
        baseline_candidates = [training_log.final_model_state]
    if not baseline_candidates:
        raise RuntimeError("Training log does not contain model snapshots.")

    baseline = baseline_candidates[-1]

    post_unlearning = list(unlearning_log.global_model_snapshots_after_unlearning)[
        -keep_last:
    ]
    if unlearning_log.final_model_state is not None:
        post_unlearning = [
            *post_unlearning,
            unlearning_log.final_model_state,
        ][-keep_last:]

    snapshots: MutableMapping[int, Mapping[str, torch.Tensor]] = {0: baseline}
    for idx, snapshot in enumerate(post_unlearning, start=1):
        snapshots[idx] = snapshot

    return snapshots


def filter_layers(param_diff: ParamDiff, allowed_prefixes: Iterable[str] | None) -> ParamDiff:
    """仅保留给定前缀的参数差。"""

    if not allowed_prefixes:
        return param_diff
    prefixes = tuple(allowed_prefixes)
    return {k: v for k, v in param_diff.items() if k.startswith(prefixes)}


def compute_weighted_gradient_signals(
    training_log: TrainingLog,
    unlearning_log: UnlearningLog,
    *,
    keep_last: int = 5,
    alpha: float = 0.7,
    focus_layers: Iterable[str] | None = None,
) -> GradientSignals:
    """基于最近 ``keep_last`` 轮快照计算加权梯度差并拆分信号。"""

    snapshots = collect_recent_snapshots(training_log, unlearning_log, keep_last)
    rounds = list(snapshots.keys())
    weighted_diff = build_weighted_param_diff(snapshots, rounds, alpha)

    # 若未指定关注层，默认保留卷积与分类头
    if focus_layers is None:
        focus_layers = ("conv", "layer", "fc")

    weighted_diff = filter_layers(weighted_diff, focus_layers)
    semantic_signal, appearance_signal = split_param_diff_into_groups(weighted_diff)

    # 如果拆分后为空，退化为全部信号，避免后续步骤无法对齐
    if not semantic_signal and weighted_diff:
        semantic_signal = dict(weighted_diff)
    if not appearance_signal and weighted_diff:
        appearance_signal = {
            k: v for k, v in weighted_diff.items() if k not in semantic_signal
        }

    return GradientSignals(
        weighted_diff=weighted_diff,
        semantic_signal=semantic_signal,
        appearance_signal=appearance_signal,
        used_rounds=rounds,
    )


__all__ = [
    "collect_recent_snapshots",
    "compute_weighted_gradient_signals",
    "filter_layers",
    "GradientSignals",
    "ParamDiff",
    "compute_param_diff",
    "build_weighted_param_diff",
    "split_param_diff_into_groups",
]
