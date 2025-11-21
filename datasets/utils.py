"""Helper utilities for dataset handling."""
from __future__ import annotations

import json
import os
import random
from typing import Dict, Iterable, List

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_partition(mapping: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f)


def load_partition(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def label_histogram(labels: Iterable[int], num_classes: int) -> List[int]:
    counts = [0 for _ in range(num_classes)]
    for label in labels:
        counts[int(label)] += 1
    return counts


__all__ = ["set_seed", "save_partition", "load_partition", "label_histogram"]
