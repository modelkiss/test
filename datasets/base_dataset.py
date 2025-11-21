"""Abstract base class for federated datasets."""
from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from . import partition as partition_utils
from . import transforms as transform_utils


class FederatedDataset(abc.ABC):
    """Base class that standardizes federated dataset operations."""

    name: str
    num_classes: int
    input_shape: Tuple[int, ...]

    def __init__(self, config: Any) -> None:
        self.config = config
        self.root = _get_from_config(config, "dataset.root", default="./data")
        self.download = bool(_get_from_config(config, "dataset.download", default=True))
        self.test_ratio = float(_get_from_config(config, "dataset.test_ratio", default=0.0))
        self.val_ratio = float(_get_from_config(config, "dataset.val_ratio", default=0.1))

        self.train_set: Optional[Dataset] = None
        self.val_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None

        self._client_partitions: Optional[Dict[Any, List[int]]] = None

        self._load_dataset()

    def _load_dataset(self) -> None:
        train, test = self.load_raw_dataset()
        if train is None:
            raise RuntimeError("Train dataset could not be loaded.")
        self.train_set = train
        self.test_set = test
        self._maybe_split_val()

    def _maybe_split_val(self) -> None:
        if self.val_ratio <= 0 or self.train_set is None:
            return
        total_size = len(self.train_set)
        val_size = int(total_size * self.val_ratio)
        indices = np.arange(total_size)
        rng = np.random.default_rng(_get_from_config(self.config, "seed", default=None))
        rng.shuffle(indices)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        self.val_set = Subset(self.train_set, val_indices.tolist())
        self.train_set = Subset(self.train_set, train_indices.tolist())

    @abc.abstractmethod
    def load_raw_dataset(self) -> Tuple[Dataset, Optional[Dataset]]:
        """Load raw training and test datasets."""

    def get_client_dataset(self, client_id: Any, indices: Sequence[int]) -> Dataset:
        if self.train_set is None:
            raise RuntimeError("Train set is not initialized.")
        return Subset(self.train_set, list(indices))

    def partition_federated(self) -> Dict[Any, List[int]]:
        if self._client_partitions is not None:
            return self._client_partitions

        num_clients = int(_get_from_config(self.config, "federated.num_clients", default=1))
        strategy = _get_from_config(self.config, "federated.partition.type", default="iid")
        kwargs = _get_from_config(self.config, "federated.partition.kwargs", default={}) or {}

        if self.train_set is None:
            raise RuntimeError("Train set is required for federated partitioning.")

        labels = _extract_labels(self.train_set)

        if strategy == "iid":
            partitions = partition_utils.iid_partition(len(self.train_set), num_clients)
        elif strategy == "dirichlet":
            alpha = float(kwargs.get("alpha", 0.5))
            partitions = partition_utils.dirichlet_partition(labels, num_clients, alpha)
        elif strategy == "label_skew":
            num_classes_per_client = int(kwargs.get("num_classes_per_client", max(1, self.num_classes // 2)))
            partitions = partition_utils.label_skew_partition(labels, num_clients, num_classes_per_client)
        elif strategy == "quantity_skew":
            min_size = int(kwargs.get("min_size", max(1, len(self.train_set) // (num_clients * 2))))
            partitions = partition_utils.quantity_skew_partition(len(self.train_set), num_clients, min_size)
        elif strategy == "user":
            user_ids = kwargs.get("user_ids")
            if user_ids is None:
                raise ValueError("user partition requires 'user_ids' in kwargs")
            partitions = partition_utils.user_partition(user_ids)
        else:
            raise ValueError(f"Unsupported partition strategy: {strategy}")

        self._client_partitions = partitions
        return partitions

    @property
    def train_transform(self):
        return transform_utils.build_train_transform(self.name, self.config)

    @property
    def eval_transform(self):
        return transform_utils.build_eval_transform(self.name, self.config)


def _extract_labels(dataset: Dataset) -> List[int]:
    # torch Subset does not expose targets/labels, so unwrap and index manually
    if isinstance(dataset, Subset):
        base_labels = _extract_labels(dataset.dataset)
        return [int(base_labels[i]) for i in dataset.indices]

    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "labels"):
        targets = dataset.labels
    else:
        raise ValueError("Dataset must provide 'targets' or 'labels' to determine partitions.")

    if isinstance(targets, list):
        return [int(t) for t in targets]
    if isinstance(targets, torch.Tensor):
        return targets.cpu().numpy().tolist()
    return [int(t) for t in targets]


def _get_from_config(config: Any, dotted_key: str, default: Optional[Any] = None) -> Any:
    keys = dotted_key.split(".")
    current = config
    for key in keys:
        if isinstance(current, dict):
            if key in current:
                current = current[key]
            else:
                return default
        else:
            if hasattr(current, key):
                current = getattr(current, key)
            else:
                return default
    return current

__all__ = ["FederatedDataset"]
