"""Top-level interfaces exposed to training and attack pipelines."""
from __future__ import annotations

from typing import Any

from .aggregator import Aggregator
from .trainer import FederatedTrainer
from .unlearning.client_level import ClientLevelUnlearning
from .unlearning.class_level import ClassLevelUnlearning
from .unlearning.sample_level import SampleLevelUnlearning
from .state import TrainingLog, UnlearningLog


def federated_train(config: Any) -> TrainingLog:
    """Run a full federated training job and return a ``TrainingLog``."""

    model = _get_config_attr(config, "model", None)
    clients = _get_config_attr(config, "clients", [])
    aggregator = Aggregator(method=_get_config_attr(config, "aggregation_method", "fedavg"))

    trainer = FederatedTrainer(model=model, clients=clients, aggregator=aggregator, config=config)
    return trainer.run()


def apply_unlearning(config: Any, training_log: TrainingLog) -> UnlearningLog:
    """Dispatch unlearning according to configuration."""

    model = _get_config_attr(config, "model", None)
    clients = _get_config_attr(config, "clients", [])
    aggregator = Aggregator(method=_get_config_attr(config, "aggregation_method", "fedavg"))

    level = _get_config_attr(config, "unlearning_level", "client")
    if level == "client":
        unlearning = ClientLevelUnlearning(model, training_log, config, aggregator, clients)
    elif level == "class":
        unlearning = ClassLevelUnlearning(model, training_log, config)
    elif level == "sample":
        unlearning = SampleLevelUnlearning(model, training_log, config, aggregator, clients)
    else:
        raise ValueError(f"Unsupported unlearning level: {level}")

    return unlearning.run()


def _get_config_attr(config: Any, name: str, default: Any) -> Any:
    if hasattr(config, name):
        return getattr(config, name)
    if isinstance(config, dict) and name in config:
        return config[name]
    return default
