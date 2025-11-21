"""Dataset loading and federated partitioning utilities.

This package exposes two main entry points:
1) :func:`get_dataset` for constructing a federated dataset wrapper.
2) :func:`get_federated_dataloaders` for building per-client dataloaders.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .base_dataset import FederatedDataset
from .cifar import CIFAR10Federated, CIFAR100Federated
from .femnist import FEMNISTFederated
from .svhn import SVHNFederated

_DATASET_REGISTRY = {
    "CIFAR10": CIFAR10Federated,
    "CIFAR100": CIFAR100Federated,
    "FEMNIST": FEMNISTFederated,
    "SVHN": SVHNFederated,
}


def get_dataset(config: Any) -> FederatedDataset:
    """Instantiate a federated dataset wrapper based on ``config``.

    Args:
        config: An object or mapping containing at least ``dataset.name``.

    Returns:
        A :class:`FederatedDataset` subclass instance.
    """

    name = _get_from_config(config, "dataset.name")
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset name: {name!r}. Available: {list(_DATASET_REGISTRY)}")

    dataset_cls = _DATASET_REGISTRY[name]
    return dataset_cls(config)


def get_federated_dataloaders(config: Any) -> Tuple[Dict[Any, DataLoader], Dict[str, DataLoader]]:
    """Create dataloaders for each federated client.

    Args:
        config: Configuration with dataset + dataloader parameters. Expected
            fields include ``dataset.name``, ``federated.num_clients`` and
            optionally ``federated.partition``.

    Returns:
        A tuple of (client_train_loaders, eval_loaders). ``eval_loaders`` may
        contain keys like ``val`` and ``test`` for global evaluation.
    """

    dataset = get_dataset(config)
    client_indices = dataset.partition_federated()

    device = str(_get_from_config(config, "logging.device", default="cpu")).lower()
    pin_memory_default = device.startswith("cuda") and torch.cuda.is_available()

    loader_kwargs = {
        "batch_size": _get_from_config(config, "dataloader.batch_size", default=64),
        "shuffle": _get_from_config(config, "dataloader.shuffle", default=True),
        "num_workers": _get_from_config(config, "dataloader.num_workers", default=2),
        "pin_memory": _get_from_config(
            config, "dataloader.pin_memory", default=pin_memory_default
        ),
    }

    client_loaders: Dict[Any, DataLoader] = {}
    for client_id, indices in client_indices.items():
        subset = dataset.get_client_dataset(client_id, indices)
        client_loaders[client_id] = DataLoader(subset, **loader_kwargs)

    eval_loaders: Dict[str, DataLoader] = {}
    if dataset.val_set is not None:
        eval_loaders["val"] = DataLoader(dataset.val_set, batch_size=loader_kwargs["batch_size"], shuffle=False,
                                          num_workers=loader_kwargs["num_workers"], pin_memory=loader_kwargs["pin_memory"])
    if dataset.test_set is not None:
        eval_loaders["test"] = DataLoader(dataset.test_set, batch_size=loader_kwargs["batch_size"], shuffle=False,
                                           num_workers=loader_kwargs["num_workers"], pin_memory=loader_kwargs["pin_memory"])

    return client_loaders, eval_loaders


def _get_from_config(config: Any, dotted_key: str, default: Optional[Any] = None) -> Any:
    """Fetch ``dotted_key`` from ``config`` supporting dict or attribute access."""

    keys: Iterable[str] = dotted_key.split(".")
    current: Any = config
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
        if current is default:
            break
    return current

__all__ = [
    "get_dataset",
    "get_federated_dataloaders",
    "CIFAR10Federated",
    "CIFAR100Federated",
    "FEMNISTFederated",
    "SVHNFederated",
]
