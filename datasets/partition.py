"""Partition strategies for federated learning datasets."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np


def iid_partition(num_samples: int, num_clients: int) -> Dict[int, List[int]]:
    """Evenly split ``num_samples`` indices across ``num_clients``."""

    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return {client: split.tolist() for client, split in enumerate(splits)}


def dirichlet_partition(labels: Sequence[int], num_clients: int, alpha: float) -> Dict[int, List[int]]:
    """Non-iid label partitioning using a Dirichlet distribution."""

    labels = np.array(labels)
    num_classes = labels.max() + 1
    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    class_indices = [np.where(labels == y)[0] for y in range(num_classes)]
    for c in class_indices:
        np.random.shuffle(c)

    class_sizes = [len(c) for c in class_indices]
    proportions = np.random.dirichlet(alpha=np.full(num_clients, alpha), size=num_classes)
    for class_id, (class_ind, prop) in enumerate(zip(class_indices, proportions)):
        prop = (prop / prop.sum())
        split_points = (np.cumsum(prop) * len(class_ind)).astype(int)[:-1]
        shards = np.split(class_ind, split_points)
        for client_id, shard in enumerate(shards):
            client_indices[client_id].extend(shard.tolist())

    return client_indices


def label_skew_partition(labels: Sequence[int], num_clients: int, num_classes_per_client: int) -> Dict[int, List[int]]:
    """Assign each client a subset of classes and sample evenly within them."""

    labels = np.array(labels)
    classes = np.unique(labels)
    rng = np.random.default_rng()
    client_classes = {i: rng.choice(classes, size=num_classes_per_client, replace=False).tolist() for i in range(num_clients)}

    buckets = {cls: np.where(labels == cls)[0].tolist() for cls in classes}
    for cls_indices in buckets.values():
        rng.shuffle(cls_indices)

    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
    for client_id, allowed_classes in client_classes.items():
        for cls in allowed_classes:
            take = math.ceil(len(buckets[cls]) / num_clients)
            cls_samples = [buckets[cls].pop() for _ in range(min(take, len(buckets[cls])))]
            client_indices[client_id].extend(cls_samples)

    # Assign remaining samples in round robin to keep every sample used
    leftover = [idx for cls_samples in buckets.values() for idx in cls_samples]
    rng.shuffle(leftover)
    for i, idx in enumerate(leftover):
        client_indices[i % num_clients].append(idx)

    return client_indices


def quantity_skew_partition(num_samples: int, num_clients: int, min_size: int) -> Dict[int, List[int]]:
    """Impose quantity skew by varying client dataset sizes."""

    if min_size <= 0:
        raise ValueError("min_size must be > 0")

    remaining = num_samples - min_size * num_clients
    if remaining < 0:
        raise ValueError("min_size too large relative to total samples")

    rng = np.random.default_rng()
    random_shares = rng.dirichlet(np.ones(num_clients)) * remaining
    sizes = [min_size + int(round(share)) for share in random_shares]

    # Normalize sizes to exactly num_samples
    size_correction = num_samples - sum(sizes)
    for i in range(abs(size_correction)):
        sizes[i % num_clients] += 1 if size_correction > 0 else -1

    indices = np.arange(num_samples)
    rng.shuffle(indices)
    splits = []
    start = 0
    for size in sizes:
        end = start + size
        splits.append(indices[start:end])
        start = end

    return {client: split.tolist() for client, split in enumerate(splits)}


def user_partition(user_ids: Iterable[int]) -> Dict[int, List[int]]:
    """Partition samples strictly by provided user IDs."""

    mapping: Dict[int, List[int]] = defaultdict(list)
    for idx, user in enumerate(user_ids):
        mapping[int(user)].append(idx)
    return dict(mapping)

__all__ = [
    "iid_partition",
    "dirichlet_partition",
    "label_skew_partition",
    "quantity_skew_partition",
    "user_partition",
]
