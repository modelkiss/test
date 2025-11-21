"""Configuration dataclasses and loaders for experiments."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List

import torch
import yaml


@dataclass
class DatasetConfig:
    name: str
    root: str
    num_clients: int
    partition_type: str
    noniid_alpha: float
    batch_size: int
    num_workers: int
    download: bool = True
    val_ratio: float = 0.1
    test_ratio: float = 0.0


@dataclass
class ModelConfig:
    arch: str
    num_classes: int
    pretrained: bool = False


@dataclass
class TrainingConfig:
    num_rounds: int
    client_fraction: float
    local_epochs: int
    optimizer: str
    lr: float
    weight_decay: float
    record_global_every: int


@dataclass
class UnlearningConfig:
    enabled: bool
    level: str
    method: str
    target_clients: List[int] = field(default_factory=list)
    target_classes: List[int] = field(default_factory=list)
    target_samples: List[str] = field(default_factory=list)
    num_unlearning_rounds: int = 0
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiffusionConfig:
    model_name: str
    image_size: int
    num_inference_steps: int
    step_size: float
    batch_size: int
    guidance_scale: float = 1.0


@dataclass
class AttackConfig:
    enabled: bool
    level: str
    save_images: bool = True
    save_dir: str = "./attacks"
    num_reconstruct_per_target: int = 0
    topk_candidates: int = 0
    loss_weights: Dict[str, float] = field(default_factory=dict)
    diffusion: DiffusionConfig | None = None


@dataclass
class LoggingConfig:
    output_dir: str
    experiment_name: str
    seed: int
    log_interval: int
    device: str


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    unlearning: UnlearningConfig
    attack: AttackConfig
    logging: LoggingConfig
    seed: int = field(init=False)
    federated: Dict[str, Any] = field(init=False)
    dataloader: Dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.seed = self.logging.seed
        device_str = str(self.logging.device).lower()
        pin_memory_default = device_str.startswith("cuda") and torch.cuda.is_available()
        self.federated = {
            "num_clients": self.dataset.num_clients,
            "partition": {
                "type": self.dataset.partition_type,
                "kwargs": {"alpha": self.dataset.noniid_alpha},
            },
        }
        self.dataloader = {
            "batch_size": self.dataset.batch_size,
            "shuffle": True,
            "num_workers": self.dataset.num_workers,
            "pin_memory": pin_memory_default,
        }

    def to_data_config(self) -> Dict[str, Any]:
        return {
            "dataset": {
                "name": self.dataset.name,
                "root": self.dataset.root,
                "download": self.dataset.download,
                "val_ratio": self.dataset.val_ratio,
                "test_ratio": self.dataset.test_ratio,
            },
            "federated": self.federated,
            "dataloader": self.dataloader,
            "seed": self.logging.seed,
        }

    def to_training_namespace(self) -> SimpleNamespace:
        return SimpleNamespace(
            num_rounds=self.training.num_rounds,
            client_fraction=self.training.client_fraction,
            snapshot_interval=self.training.record_global_every,
            track_client_updates=True,
            seed=self.logging.seed,
        )

    def to_unlearning_namespace(self) -> SimpleNamespace:
        return SimpleNamespace(
            unlearning_level=self.unlearning.level,
            unlearning_method=self.unlearning.method,
            forgotten_clients=list(self.unlearning.target_clients),
            forgotten_classes=list(self.unlearning.target_classes),
            forgotten_samples=list(self.unlearning.target_samples),
            unlearning_rounds=self.unlearning.num_unlearning_rounds,
            client_fraction=self.training.client_fraction,
            **self.unlearning.extra_params,
        )

    def to_attack_config(self) -> Dict[str, Any]:
        diffusion = self.attack.diffusion
        diffusion_dict = None
        if diffusion is not None:
            diffusion_dict = {
                "model_name": diffusion.model_name,
                "image_size": diffusion.image_size,
                "num_inference_steps": diffusion.num_inference_steps,
                "step_size": diffusion.step_size,
                "batch_size": diffusion.batch_size,
                "guidance_scale": diffusion.guidance_scale,
            }
        return {
            "enabled": self.attack.enabled,
            "level": self.attack.level,
            "save_images": self.attack.save_images,
            "save_dir": self.attack.save_dir,
            "num_reconstruct_per_target": self.attack.num_reconstruct_per_target,
            "topk_candidates": self.attack.topk_candidates,
            "loss_weights": self.attack.loss_weights,
            "diffusion": diffusion_dict,
        }


def load_experiment_config(exp_yaml_path: str) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from YAML with defaults."""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    defaults = {
        "datasets": _load_yaml(os.path.join(base_dir, "datasets.yaml")),
        "training": _load_yaml(os.path.join(base_dir, "training.yaml")),
        "unlearning": _load_yaml(os.path.join(base_dir, "unlearning.yaml")),
        "attacks": _load_yaml(os.path.join(base_dir, "attacks.yaml")),
    }

    experiment_cfg = _load_yaml(exp_yaml_path)

    dataset_cfg = _build_dataset_config(experiment_cfg.get("dataset"), defaults["datasets"])
    model_cfg = _build_model_config(experiment_cfg.get("model", {}))
    training_cfg = _build_training_config(experiment_cfg.get("training"), defaults["training"])
    unlearning_cfg = _build_unlearning_config(
        experiment_cfg.get("unlearning"), defaults["unlearning"]
    )
    attack_cfg = _build_attack_config(experiment_cfg.get("attack"), defaults["attacks"])
    logging_cfg = _build_logging_config(experiment_cfg.get("logging", {}), experiment_cfg)

    return ExperimentConfig(
        dataset=dataset_cfg,
        model=model_cfg,
        training=training_cfg,
        unlearning=unlearning_cfg,
        attack=attack_cfg,
        logging=logging_cfg,
    )


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_dataset_config(value: Any, defaults: Dict[str, Any]) -> DatasetConfig:
    if isinstance(value, str):
        base = defaults.get(value, {})
        data = dict(base)
    elif isinstance(value, dict):
        base = defaults.get(value.get("name", ""), {}) if value.get("name") else {}
        data = _merge_dict(base, value)
    else:
        raise ValueError("Dataset configuration must be a string or mapping.")

    required = ["name", "root", "num_clients", "partition_type", "noniid_alpha", "batch_size", "num_workers"]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing dataset config field: {key}")

    return DatasetConfig(
        name=data["name"],
        root=data["root"],
        num_clients=int(data["num_clients"]),
        partition_type=str(data["partition_type"]),
        noniid_alpha=float(data["noniid_alpha"]),
        batch_size=int(data["batch_size"]),
        num_workers=int(data["num_workers"]),
        download=bool(data.get("download", True)),
        val_ratio=float(data.get("val_ratio", 0.1)),
        test_ratio=float(data.get("test_ratio", 0.0)),
    )


def _build_model_config(value: Dict[str, Any]) -> ModelConfig:
    required = ["arch", "num_classes"]
    for key in required:
        if key not in value:
            raise ValueError(f"Missing model config field: {key}")
    return ModelConfig(
        arch=str(value["arch"]),
        num_classes=int(value["num_classes"]),
        pretrained=bool(value.get("pretrained", False)),
    )


def _build_training_config(value: Any, defaults: Dict[str, Any]) -> TrainingConfig:
    if isinstance(value, str):
        data = dict(defaults.get(value, {}))
    elif isinstance(value, dict):
        base = defaults.get(value.get("base", ""), {}) if value.get("base") else {}
        data = _merge_dict(base, value)
    else:
        raise ValueError("Training configuration must be a string or mapping.")

    required = [
        "num_rounds",
        "client_fraction",
        "local_epochs",
        "optimizer",
        "lr",
        "weight_decay",
        "record_global_every",
    ]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing training config field: {key}")

    return TrainingConfig(
        num_rounds=int(data["num_rounds"]),
        client_fraction=float(data["client_fraction"]),
        local_epochs=int(data["local_epochs"]),
        optimizer=str(data["optimizer"]),
        lr=float(data["lr"]),
        weight_decay=float(data["weight_decay"]),
        record_global_every=int(data["record_global_every"]),
    )


def _build_unlearning_config(value: Any, defaults: Dict[str, Any]) -> UnlearningConfig:
    if value is None:
        return UnlearningConfig(enabled=False, level="client", method="")

    level = value.get("level")
    method = value.get("method")
    if level is None or method is None:
        raise ValueError("Unlearning configuration requires 'level' and 'method'.")

    default_params = defaults.get(f"{level}_level", {}).get(method, {})
    merged = _merge_dict(default_params, value)

    return UnlearningConfig(
        enabled=bool(merged.get("enabled", True)),
        level=str(level),
        method=str(method),
        target_clients=list(merged.get("target_clients", [])),
        target_classes=list(merged.get("target_classes", [])),
        target_samples=list(merged.get("target_samples", [])),
        num_unlearning_rounds=int(merged.get("num_unlearning_rounds", default_params.get("num_unlearning_rounds", 0))),
        extra_params=dict(merged.get("extra_params", {})),
    )


def _build_diffusion_config(name_or_dict: Any, defaults: Dict[str, Any]) -> DiffusionConfig:
    if isinstance(name_or_dict, str):
        data = dict(defaults.get(name_or_dict, {}))
    elif isinstance(name_or_dict, dict):
        base = defaults.get(name_or_dict.get("base", ""), {}) if name_or_dict.get("base") else {}
        data = _merge_dict(base, name_or_dict)
    else:
        raise ValueError("Diffusion config must be a string or mapping.")

    required = ["model_name", "image_size", "num_inference_steps", "step_size", "batch_size"]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing diffusion config field: {key}")

    return DiffusionConfig(
        model_name=str(data["model_name"]),
        image_size=int(data["image_size"]),
        num_inference_steps=int(data["num_inference_steps"]),
        step_size=float(data["step_size"]),
        batch_size=int(data["batch_size"]),
        guidance_scale=float(data.get("guidance_scale", 1.0)),
    )


def _build_attack_config(value: Any, defaults: Dict[str, Any]) -> AttackConfig:
    if value is None:
        return AttackConfig(enabled=False, level="client")

    level = value.get("level")
    if level is None:
        raise ValueError("Attack configuration requires 'level'.")

    default_attack = defaults.get(f"{level}_level", {})
    data = _merge_dict(default_attack, value)

    diffusion_cfg = data.get("diffusion", defaults.get("default_diffusion"))
    diffusion = _build_diffusion_config(diffusion_cfg, defaults) if diffusion_cfg else None

    return AttackConfig(
        enabled=bool(data.get("enabled", True)),
        level=str(level),
        save_images=bool(data.get("save_images", True)),
        save_dir=str(data.get("save_dir", "./attacks")),
        num_reconstruct_per_target=int(data.get("num_reconstruct_per_target", 0)),
        topk_candidates=int(data.get("topk_candidates", 0)),
        loss_weights=dict(data.get("loss_weights", {})),
        diffusion=diffusion,
    )


def _build_logging_config(value: Dict[str, Any], experiment_cfg: Dict[str, Any]) -> LoggingConfig:
    required = ["output_dir", "seed", "log_interval", "device"]
    for key in required:
        if key not in value:
            raise ValueError(f"Missing logging config field: {key}")

    experiment_name = value.get("experiment_name") or experiment_cfg.get("experiment_name")
    if not experiment_name:
        raise ValueError("Logging config requires 'experiment_name'.")

    return LoggingConfig(
        output_dir=str(value["output_dir"]),
        experiment_name=str(experiment_name),
        seed=int(value["seed"]),
        log_interval=int(value["log_interval"]),
        device=str(value["device"]),
    )


__all__ = [
    "DatasetConfig",
    "ModelConfig",
    "TrainingConfig",
    "UnlearningConfig",
    "DiffusionConfig",
    "AttackConfig",
    "LoggingConfig",
    "ExperimentConfig",
    "load_experiment_config",
]
