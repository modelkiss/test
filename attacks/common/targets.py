"""语义目标、外观目标、类别 / 客户端 / 样本先验定义。"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

import torch


def _l2_normalize(vector: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Apply safe L2 normalization to a 1-D tensor."""

    return vector / (vector.norm() + eps)


def _flatten_and_concat(tensor_group: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten tensors in a dict and concatenate them into one vector."""

    if not tensor_group:
        return torch.tensor([])
    flattened = [param.view(-1) for param in tensor_group.values()]
    return torch.cat(flattened) if flattened else torch.tensor([])


def build_semantic_target(semantic_group: Dict[str, torch.Tensor], eps: float = 1e-12) -> torch.Tensor:
    """把高层语义参数差展平成单一向量并进行 L2 归一化。"""

    vector = _flatten_and_concat(semantic_group)
    return _l2_normalize(vector, eps)


def build_appearance_target(
    appearance_group: Dict[str, torch.Tensor], eps: float = 1e-12, min_norm: float = 1e-8
) -> Optional[torch.Tensor]:
    """构造浅层外观目标，若信号过弱则返回 ``None``。"""

    vector = _flatten_and_concat(appearance_group)
    norm = vector.norm()
    if vector.numel() == 0 or norm < min_norm:
        return None
    return vector / (norm + eps)


def _resolve_classifier_head(model: torch.nn.Module) -> torch.nn.Module:
    """Try to locate a classifier head (fc / classifier / head)."""

    for attr in ("fc", "classifier", "head"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise AttributeError("Model does not expose a recognized classifier head (fc/classifier/head).")


def _maybe_normalize_prototype(tensor: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Clone & normalize prototype tensors to avoid in-place mutation."""

    cloned = tensor.detach().clone()
    norm = cloned.norm()
    if norm == 0:
        return cloned
    return cloned / (norm + eps)


def build_class_prior(
    model: torch.nn.Module, forgotten_classes: List[int], eps: float = 1e-12
) -> Dict[int, Dict[str, torch.Tensor]]:
    """从分类头和可选特征原型中提取类别先验。"""

    head = _resolve_classifier_head(model)
    if not hasattr(head, "weight"):
        raise AttributeError("Classifier head must expose a 'weight' tensor.")

    weights = head.weight.detach().cpu()
    num_classes = weights.size(0)
    class_prior: Dict[int, Dict[str, torch.Tensor]] = {}

    feature_prototypes = getattr(model, "feature_prototypes", None)
    for class_id in forgotten_classes:
        if class_id < 0 or class_id >= num_classes:
            raise IndexError(f"Class id {class_id} is out of range for classifier with {num_classes} classes.")

        weight_vector = _maybe_normalize_prototype(weights[class_id], eps)
        prior: Dict[str, torch.Tensor] = {"weight_vector": weight_vector}

        if isinstance(feature_prototypes, dict) and class_id in feature_prototypes:
            prototype_tensor = feature_prototypes[class_id]
            prior["feature_prototype"] = _maybe_normalize_prototype(
                prototype_tensor.detach().cpu(), eps
            )

        class_prior[class_id] = prior

    return class_prior


def build_client_prior(
    federated_state: Any, target_clients: List[int], eps: float = 1e-12
) -> Dict[int, Dict[str, Any]]:
    """汇总客户端训练日志，生成每个客户端的标签分布和特征原型。"""

    client_prior: Dict[int, Dict[str, Any]] = {}

    # client_updates 结构：{round_idx: {client_id: ClientUpdate}}
    client_updates = getattr(federated_state, "client_updates", {})
    if isinstance(federated_state, dict):
        client_updates = federated_state.get("client_updates", client_updates)

    for client_id in target_clients:
        label_hist = Counter()
        feature_vectors: List[torch.Tensor] = []

        for round_updates in client_updates.values():
            if not isinstance(round_updates, dict) or client_id not in round_updates:
                continue
            update = round_updates[client_id]
            metrics = getattr(update, "metrics", {}) or {}

            # 标签分布可能存储为 dict 或 Counter
            hist = metrics.get("label_hist") if isinstance(metrics, dict) else None
            if hist:
                label_hist.update(hist)

            feature_proto = metrics.get("feature_proto") if isinstance(metrics, dict) else None
            if isinstance(feature_proto, torch.Tensor):
                feature_vectors.append(feature_proto.detach().cpu())

        prior: Dict[str, Any] = {}
        if label_hist:
            total = sum(label_hist.values())
            prior["label_dist"] = {k: v / total for k, v in label_hist.items() if total > 0}

        if feature_vectors:
            stacked = torch.stack(feature_vectors)
            mean_feature = stacked.mean(dim=0)
            prior["feature_proto"] = _maybe_normalize_prototype(mean_feature, eps)

        client_prior[client_id] = prior

    return client_prior


def build_sample_prior(
    training_log: Any, target_samples: List[str], eps: float = 1e-12
) -> Dict[str, Dict[str, torch.Tensor]]:
    """构造样本级先验，例如单样本特征或梯度方向模板。"""

    sample_prior: Dict[str, Dict[str, torch.Tensor]] = {}
    sample_features = getattr(training_log, "sample_features", None)
    if isinstance(training_log, dict):
        sample_features = training_log.get("sample_features", sample_features)

    for sample_id in target_samples:
        prototypes: List[torch.Tensor] = []
        if isinstance(sample_features, dict) and sample_id in sample_features:
            feats = sample_features[sample_id]
            if isinstance(feats, torch.Tensor):
                prototypes.append(feats.detach().cpu())
            elif isinstance(feats, (list, tuple)):
                for f in feats:
                    if isinstance(f, torch.Tensor):
                        prototypes.append(f.detach().cpu())

        if prototypes:
            stacked = torch.stack(prototypes)
            feature_proto = _maybe_normalize_prototype(stacked.mean(dim=0), eps)
            sample_prior[sample_id] = {"feature_proto": feature_proto}

    return sample_prior
