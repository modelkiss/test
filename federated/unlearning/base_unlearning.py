"""Abstract unlearning driver classes."""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Sequence

import torch

from ..state import TrainingLog, UnlearningLog
from .. import utils


class BaseUnlearning(ABC):
    """Base class encapsulating common unlearning behaviour."""

    def __init__(self, model: torch.nn.Module, training_log: TrainingLog, config: Any):
        self.model = copy.deepcopy(model)
        self.training_log = training_log
        self.config = config
        self.forgotten_clients: List[Any] = []
        self.forgotten_classes: List[Any] = []
        self.forgotten_samples: List[Any] = []
        self.metrics = {}

    @abstractmethod
    def select_targets(self) -> None:
        """Populate ``forgotten_*`` fields according to the configuration."""

    @abstractmethod
    def perform_unlearning(self) -> Sequence[Mapping[str, torch.Tensor]]:
        """Run the concrete unlearning routine and return model snapshots."""

    def build_unlearning_log(
        self, snapshots: Sequence[Mapping[str, torch.Tensor]]
    ) -> UnlearningLog:
        return UnlearningLog(
            global_model_snapshots_after_unlearning=list(snapshots),
            forgotten_clients=list(self.forgotten_clients),
            forgotten_classes=list(self.forgotten_classes),
            forgotten_samples=list(self.forgotten_samples),
            unlearning_metrics=self.metrics,
        )

    def run(self) -> UnlearningLog:
        self.select_targets()
        snapshots = self.perform_unlearning()
        return self.build_unlearning_log(snapshots)

    def _load_training_terminal_model(self) -> None:
        if self.training_log.global_model_snapshots:
            last_snapshot = self.training_log.global_model_snapshots[-1]
            self.model.load_state_dict(last_snapshot)

    def _clone_current_state(self) -> Mapping[str, torch.Tensor]:
        return utils.clone_model_state(self.model)
