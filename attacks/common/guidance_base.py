"""Abstract guidance interface for diffusion priors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseGuidance(ABC):
    """Callable interface used by :class:`DiffusionPrior`."""

    def __init__(self, model: torch.nn.Module, target_signal: Any, config: Any) -> None:
        self.model = model
        self.target_signal = target_signal
        self.config = config

    @abstractmethod
    def __call__(self, images: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        """Compute a scalar guidance loss for a batch of images."""

    def postprocess(self, images: torch.Tensor, extra_info: Any | None = None) -> None:
        """Optional hook for subclasses to record intermediate artefacts."""

        _ = images, extra_info
