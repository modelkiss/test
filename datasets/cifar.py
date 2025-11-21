"""Federated wrappers for CIFAR datasets."""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torchvision
from torch.utils.data import Dataset

from .base_dataset import FederatedDataset


class _BaseCIFARFederated(FederatedDataset):
    def __init__(self, config: Any, dataset_cls, name: str, num_classes: int) -> None:
        self.dataset_cls = dataset_cls
        self.name = name
        self.num_classes = num_classes
        self.input_shape: Tuple[int, ...] = (3, 32, 32)
        super().__init__(config)

    def load_raw_dataset(self) -> Tuple[Dataset, Optional[Dataset]]:
        train_set = self.dataset_cls(
            root=self.root,
            train=True,
            download=self.download,
            transform=self.train_transform,
        )
        test_set = self.dataset_cls(
            root=self.root,
            train=False,
            download=self.download,
            transform=self.eval_transform,
        )
        return train_set, test_set


class CIFAR10Federated(_BaseCIFARFederated):
    def __init__(self, config: Any) -> None:
        super().__init__(config, torchvision.datasets.CIFAR10, "CIFAR10", 10)


class CIFAR100Federated(_BaseCIFARFederated):
    def __init__(self, config: Any) -> None:
        super().__init__(config, torchvision.datasets.CIFAR100, "CIFAR100", 100)


__all__ = ["CIFAR10Federated", "CIFAR100Federated"]
