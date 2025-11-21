"""Data augmentation and normalization utilities."""
from __future__ import annotations

from typing import Any, Dict

from torchvision import transforms


_MEAN_STD: Dict[str, Dict[str, tuple]] = {
    "CIFAR10": {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
    "CIFAR100": {"mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761)},
    "SVHN": {"mean": (0.4377, 0.4438, 0.4728), "std": (0.1980, 0.2010, 0.1970)},
    "FEMNIST": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
}


def build_train_transform(dataset_name: str, config: Any):
    dataset_name = dataset_name.upper()
    normalize = _normalize(dataset_name)
    augmentation = _augmentation(dataset_name, config)
    preprocess = _preprocess(dataset_name)
    return transforms.Compose(preprocess + augmentation + [transforms.ToTensor(), normalize])


def build_eval_transform(dataset_name: str, config: Any):
    dataset_name = dataset_name.upper()
    normalize = _normalize(dataset_name)
    preprocess = _preprocess(dataset_name)
    return transforms.Compose(preprocess + [transforms.ToTensor(), normalize])


def _augmentation(dataset_name: str, config: Any):
    aug = []
    if dataset_name in {"CIFAR10", "CIFAR100", "SVHN"}:
        aug.append(transforms.RandomCrop(32, padding=4))
        aug.append(transforms.RandomHorizontalFlip())
    # FEMNIST/EMNIST typically no heavy augmentation by default
    return aug


def _preprocess(dataset_name: str):
    if dataset_name == "FEMNIST":
        return [transforms.Resize(32), transforms.Grayscale(num_output_channels=3)]
    return []


def _normalize(dataset_name: str):
    params = _MEAN_STD.get(dataset_name, {"mean": (0.5,), "std": (0.5,)})
    return transforms.Normalize(mean=params["mean"], std=params["std"])


__all__ = ["build_train_transform", "build_eval_transform"]
