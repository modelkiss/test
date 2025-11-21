"""FEMNIST federated dataset wrapper built from EMNIST-balanced."""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torchvision
from torch.utils.data import Dataset

from .base_dataset import FederatedDataset
from .partition import user_partition


class FEMNISTFederated(FederatedDataset):
    name = "FEMNIST"
    num_classes = 47
    input_shape: Tuple[int, ...] = (1, 28, 28)

    def __init__(self, config: Any) -> None:
        self.expand_channels = bool(_get_from_config(config, "dataset.expand_channels", default=False))
        super().__init__(config)

    def load_raw_dataset(self) -> Tuple[Dataset, Optional[Dataset]]:
        train_transform = self.train_transform
        test_transform = self.eval_transform
        if self.expand_channels:
            to_three_channel = torchvision.transforms.Lambda(lambda img: img.expand(3, -1, -1))
            train_transform = torchvision.transforms.Compose([train_transform, to_three_channel])
            test_transform = torchvision.transforms.Compose([test_transform, to_three_channel])
            self.input_shape = (3, 28, 28)

        train_set = torchvision.datasets.EMNIST(
            root=self.root,
            split="balanced",
            train=True,
            download=self.download,
            transform=train_transform,
        )
        test_set = torchvision.datasets.EMNIST(
            root=self.root,
            split="balanced",
            train=False,
            download=self.download,
            transform=test_transform,
        )
        return train_set, test_set

    def partition_federated(self):
        # FEMNIST often partitions by user IDs supplied in the metadata if provided
        user_ids = getattr(self.train_set, "_user_ids", None) if self.train_set else None
        if user_ids is not None:
            self._client_partitions = user_partition(user_ids)
            return self._client_partitions
        return super().partition_federated()


def _get_from_config(config: Any, dotted_key: str, default=None):
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


__all__ = ["FEMNISTFederated"]
