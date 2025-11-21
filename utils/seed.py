"""Utility for reproducible random seeds."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int]) -> None:
    """Set seeds for Python, NumPy, and PyTorch."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
