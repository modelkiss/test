"""类级目标信号构造（依赖 common 组件）。"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from federated.grad_diff import build_weighted_param_diff, split_param_diff_into_groups
from federated.unlearning.utils import collect_pre_post_models

from attacks.common.targets import (
    build_appearance_target,
    build_class_prior,
    build_semantic_target,
)


def _as_dict_maybe(obj: Any) -> Dict[str, Any]:
    """Safely convert configs or state containers to a dict-like object."""

    return obj if isinstance(obj, dict) else obj.__dict__ if hasattr(obj, "__dict__") else {}


def _gather_snapshots(
    base_state: Mapping[str, Any],
    pre_snapshots: Sequence[Mapping[str, Any]],
    post_snapshots: Sequence[Mapping[str, Any]],
) -> Dict[int, Mapping[str, Any]]:
    """Arrange snapshots with negative offsets for pre rounds and positive for post."""

    snapshots: Dict[int, Mapping[str, Any]] = {0: base_state}
    for offset, snap in enumerate(reversed(pre_snapshots), start=1):
        snapshots[-offset] = snap
    for offset, snap in enumerate(post_snapshots, start=1):
        snapshots[offset] = snap
    return snapshots


def _resolve_base_state(training_log: Any) -> Mapping[str, Any] | None:
    """Choose the baseline state from training logs."""

    if training_log is None:
        return None
    if getattr(training_log, "final_model_state", None) is not None:
        return training_log.final_model_state
    snapshots = getattr(training_log, "global_model_snapshots", None)
    if snapshots:
        return snapshots[-1]
    if isinstance(training_log, dict):
        snaps = training_log.get("global_model_snapshots")
        if snaps:
            return snaps[-1]
    return None


def _resolve_reference_model(federated_state: Any) -> Any:
    """Locate a reference model used for extracting classifier priors."""

    for key in ("model", "base_model", "global_model"):
        model = getattr(federated_state, key, None)
        if model is not None:
            return model
    mapping = _as_dict_maybe(federated_state)
    for key in ("model", "base_model", "global_model"):
        if key in mapping:
            return mapping[key]
    return None


def build_class_level_targets(
    federated_state: Any, forgotten_classes: List[int], config: Mapping[str, Any]
) -> Dict[int, Dict[str, Any]]:
    """按照类别粒度构造目标信号与先验。

    Args:
        federated_state: 训练 / 遗忘的完整状态或日志容器。
        forgotten_classes: 需要重建的类别 ID 列表。
        config: 用于控制采样轮次与权重的配置，支持的键：
            - ``num_pre_rounds``：选择多少个训练前轮次（默认 3）。
            - ``num_post_rounds``：选择多少个遗忘后轮次（默认 3）。
            - ``alpha``：构造 ``param_diff`` 的衰减系数（默认 0.5）。

    Returns:
        dict: 每个类别的目标集合。
    """

    if not forgotten_classes:
        raise ValueError("No forgotten classes provided for class-level attack.")

    training_log = getattr(federated_state, "training_log", None)
    unlearning_log = getattr(federated_state, "unlearning_log", None)
    training_log = training_log if training_log is not None else _as_dict_maybe(federated_state).get("training_log")
    unlearning_log = unlearning_log if unlearning_log is not None else _as_dict_maybe(federated_state).get("unlearning_log")

    if training_log is None or unlearning_log is None:
        raise ValueError("Both training and unlearning logs are required to build class-level targets.")

    num_pre = int(config.get("num_pre_rounds", 3))
    num_post = int(config.get("num_post_rounds", 3))
    alpha = float(config.get("alpha", 0.5))

    base_state = _resolve_base_state(training_log)
    if base_state is None:
        raise ValueError("Base model state is required to build class-level targets.")

    pre_models, post_models = collect_pre_post_models(training_log, unlearning_log, num_pre, num_post)
    snapshots = _gather_snapshots(base_state, pre_models, post_models)
    param_diff = build_weighted_param_diff(snapshots, snapshots.keys(), alpha)

    semantic_group, appearance_group = split_param_diff_into_groups(param_diff)
    f_target_semantic = build_semantic_target(semantic_group)
    g_target_appearance = build_appearance_target(appearance_group)

    model = _resolve_reference_model(federated_state)
    if model is None:
        raise ValueError("A reference model is required to extract class priors.")
    class_prior = build_class_prior(model, forgotten_classes)

    targets_class: Dict[int, Dict[str, Any]] = {}
    for class_id in forgotten_classes:
        targets_class[class_id] = {
            "param_diff": param_diff,
            "f_target_semantic": f_target_semantic,
            "g_target_appearance": g_target_appearance,
            "class_prior": class_prior.get(class_id, {}),
            "pre_models": pre_models,
            "post_models": post_models,
            "level": "class",
        }

    return targets_class
