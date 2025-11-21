"""Federated wrapper for the SVHN dataset."""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torchvision
from torch.utils.data import Dataset

from .base_dataset import FederatedDataset


class SVHNFederated(FederatedDataset):
    name = "SVHN"
    num_classes = 10
    input_shape: Tuple[int, ...] = (3, 32, 32)

    def __init__(self, config: Any) -> None:
        super().__init__(config)

    def load_raw_dataset(self) -> Tuple[Dataset, Optional[Dataset]]:
        train_set = torchvision.datasets.SVHN(
            root=self.root,
            split="train",
            download=self.download,
            transform=self.train_transform,
        )
        test_set = torchvision.datasets.SVHN(
            root=self.root,
            split="test",
            download=self.download,
            transform=self.eval_transform,
        )
        return train_set, test_set


__all__ = ["SVHNFederated"]
