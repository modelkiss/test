"""客户端级目标信号构造（依赖 common 组件）。"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from federated.grad_diff import build_weighted_param_diff, split_param_diff_into_groups

from attacks.common.targets import (
    build_appearance_target,
    build_client_prior,
    build_semantic_target,
)


def _as_dict_maybe(obj: Any) -> Dict[str, Any]:
    """Safely convert configs or state containers to a dict-like object."""

    return obj if isinstance(obj, dict) else obj.__dict__ if hasattr(obj, "__dict__") else {}


def _select_round_indices(round_client_ids: Sequence[Sequence[Any]], client_id: Any, k: int) -> List[int]:
    """Pick up to ``k`` rounds that involve the given client (latest first)."""

    indices = [idx for idx, clients in enumerate(round_client_ids) if client_id in clients]
    if k <= 0:
        return indices
    return indices[-k:]


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


def _resolve_base_state(training_log: Any) -> Optional[Mapping[str, Any]]:
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


def _fetch_snapshots_by_idx(snapshots: Sequence[Mapping[str, Any]], indices: Iterable[int]) -> List[Mapping[str, Any]]:
    """Safely gather snapshot states with bounds checking."""

    collected: List[Mapping[str, Any]] = []
    for idx in indices:
        if 0 <= idx < len(snapshots):
            collected.append(snapshots[idx])
    return collected


def build_client_level_targets(
    federated_state: Any, target_clients: List[int], config: Mapping[str, Any]
) -> Dict[int, Dict[str, Any]]:
    """按照客户端粒度构造目标信号与先验。

    Args:
        federated_state: 训练 / 遗忘的完整状态或日志容器。
        target_clients: 需要重建的客户端 ID 列表。
        config: 用于控制采样轮次与权重的配置，支持的键：
            - ``num_pre_rounds``：选择多少个训练前轮次（默认 3）。
            - ``num_post_rounds``：选择多少个遗忘后轮次（默认 3）。
            - ``alpha``：构造 ``param_diff`` 的衰减系数（默认 0.5）。

    Returns:
        dict: 每个客户端的目标集合。
    """

    training_log = getattr(federated_state, "training_log", None)
    unlearning_log = getattr(federated_state, "unlearning_log", None)
    training_log = training_log if training_log is not None else _as_dict_maybe(federated_state).get("training_log")
    unlearning_log = unlearning_log if unlearning_log is not None else _as_dict_maybe(federated_state).get("unlearning_log")

    num_pre = int(config.get("num_pre_rounds", 3))
    num_post = int(config.get("num_post_rounds", 3))
    alpha = float(config.get("alpha", 0.5))

    # Gather participation traces
    round_client_ids: List[List[Any]] = getattr(training_log, "round_client_ids", []) or []
    if isinstance(training_log, dict):
        round_client_ids = training_log.get("round_client_ids", round_client_ids) or []

    training_snapshots: List[Mapping[str, Any]] = getattr(
        training_log, "global_model_snapshots", []
    ) or []
    if isinstance(training_log, dict):
        training_snapshots = training_log.get("global_model_snapshots", training_snapshots) or []

    unlearning_snapshots: List[Mapping[str, Any]] = getattr(
        unlearning_log, "global_model_snapshots_after_unlearning", []
    ) or []
    if isinstance(unlearning_log, dict):
        unlearning_snapshots = unlearning_log.get(
            "global_model_snapshots_after_unlearning", unlearning_snapshots
        ) or []

    base_state = _resolve_base_state(training_log)
    if base_state is None:
        raise ValueError("Base model state is required to build client-level targets.")

    client_prior = build_client_prior(federated_state, target_clients)

    targets_client: Dict[int, Dict[str, Any]] = {}
    for client_id in target_clients:
        pre_indices = _select_round_indices(round_client_ids, client_id, num_pre)
        post_indices = list(range(max(0, len(unlearning_snapshots) - num_post), len(unlearning_snapshots)))

        pre_models = _fetch_snapshots_by_idx(training_snapshots, pre_indices)
        post_models = _fetch_snapshots_by_idx(unlearning_snapshots, post_indices)

        snapshots = _gather_snapshots(base_state, pre_models, post_models)
        param_diff_client = build_weighted_param_diff(snapshots, snapshots.keys(), alpha)

        semantic_group, appearance_group = split_param_diff_into_groups(param_diff_client)
        f_target_semantic = build_semantic_target(semantic_group)
        g_target_appearance = build_appearance_target(appearance_group)

        targets_client[client_id] = {
            "param_diff": param_diff_client,
            "f_target_semantic": f_target_semantic,
            "g_target_appearance": g_target_appearance,
            "client_prior": client_prior.get(client_id, {}),
            "pre_models": pre_models,
            "post_models": post_models,
            "level": "client",
        }

    return targets_client
