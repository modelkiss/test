"""Unified entry point for federated training, unlearning, and attacks."""
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace
from typing import Dict, Tuple

import torch

from attacks.class_level.attack_runner import run_class_level_attack
from attacks.client_level.attack_runner import run_client_level_attack
from attacks.sample_level.attack_runner import run_sample_level_attack
from configs.base_config import ExperimentConfig, load_experiment_config
from datasets import get_federated_dataloaders
from federated.aggregator import Aggregator
from federated.client import FederatedClient
from federated.core import apply_unlearning
from federated.state import TrainingLog, UnlearningLog
from federated.trainer import FederatedTrainer
from utils.model import build_model
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated unlearning benchmark runner")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment yaml, e.g. configs/experiments/cifar10_client_federaser.yaml",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "train_only", "unlearning_only", "attack_only"],
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="Optional path to load a pretrained global model state dict.",
    )
    return parser.parse_args()


def build_clients(
    train_loaders: Dict[int, torch.utils.data.DataLoader],
    eval_loaders: Dict[str, torch.utils.data.DataLoader],
    exp_config: ExperimentConfig,
) -> Tuple[list[FederatedClient], torch.utils.data.DataLoader | None]:
    """Construct :class:`FederatedClient` objects for each client."""

    val_loader = eval_loaders.get("val")
    clients = [
        FederatedClient(
            client_id=client_id,
            train_loader=loader,
            val_loader=val_loader,
            local_epochs=exp_config.training.local_epochs,
            lr=exp_config.training.lr,
            device=exp_config.logging.device,
        )
        for client_id, loader in train_loaders.items()
    ]
    return clients, eval_loaders.get("test")


def build_trainer_config(
    exp_config: ExperimentConfig,
    *,
    override_rounds: int | None = None,
    override_client_fraction: float | None = None,
) -> SimpleNamespace:
    optimizer_cls = (
        torch.optim.SGD
        if exp_config.training.optimizer.lower() == "sgd"
        else torch.optim.Adam
    )
    return SimpleNamespace(
        num_rounds=override_rounds
        if override_rounds is not None
        else exp_config.training.num_rounds,
        client_fraction=override_client_fraction
        if override_client_fraction is not None
        else exp_config.training.client_fraction,
        snapshot_interval=exp_config.training.record_global_every,
        track_client_updates=True,
        seed=exp_config.logging.seed,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer_cls=optimizer_cls,
        optimizer_kwargs={
            "lr": exp_config.training.lr,
            "weight_decay": exp_config.training.weight_decay,
        },
    )


def build_unlearning_config(
    exp_config: ExperimentConfig,
    clients: list[FederatedClient],
    model: torch.nn.Module,
) -> SimpleNamespace:
    config = exp_config.to_unlearning_namespace()
    config.model = model
    config.clients = clients
    config.client_fraction = exp_config.training.client_fraction
    config.device = exp_config.logging.device
    config.retrain_rounds = getattr(config, "retrain_rounds", exp_config.unlearning.num_unlearning_rounds)
    config.recovery_rounds = getattr(config, "recovery_rounds", exp_config.unlearning.extra_params.get("recovery_epochs", 0))
    return config


def dispatch_attack(
    exp_config: ExperimentConfig,
    model: torch.nn.Module,
    training_log: TrainingLog,
    unlearning_log: UnlearningLog,
) -> None:
    attack_level = exp_config.attack.level
    attack_config = {
        "forgotten_clients": exp_config.unlearning.target_clients,
        "forgotten_classes": exp_config.unlearning.target_classes,
        "forgotten_samples": exp_config.unlearning.target_samples,
        "device": exp_config.logging.device,
    }

    if attack_level == "client":
        result = run_client_level_attack(attack_config, model, training_log, unlearning_log)
    elif attack_level == "class":
        result = run_class_level_attack(attack_config, model, training_log, unlearning_log)
    elif attack_level == "sample":
        result = run_sample_level_attack(attack_config, model, training_log, unlearning_log)
    else:
        raise ValueError(f"Unknown attack level: {attack_level}")

    save_dir = os.path.join(exp_config.logging.output_dir, exp_config.logging.experiment_name, "attacks")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(result.metrics, os.path.join(save_dir, "attack_metrics.pt"))
    if exp_config.attack.save_images:
        torch.save(result.reconstructed_images, os.path.join(save_dir, "reconstructions.pt"))


def main() -> None:
    args = parse_args()
    exp_config = load_experiment_config(args.config)

    set_seed(exp_config.logging.seed)
    os.makedirs(exp_config.logging.output_dir, exist_ok=True)

    train_loaders, eval_loaders = get_federated_dataloaders(exp_config)
    model = build_model(exp_config.model).to(exp_config.logging.device)

    if args.resume_path:
        state_dict = torch.load(args.resume_path, map_location=exp_config.logging.device)
        model.load_state_dict(state_dict)

    training_log: TrainingLog | None = None
    unlearning_log: UnlearningLog | None = None
    relearn_log: TrainingLog | None = None

    if args.mode in {"full", "train_only"}:
        clients, _ = build_clients(train_loaders, eval_loaders, exp_config)
        trainer = FederatedTrainer(
            model=model,
            clients=clients,
            aggregator=Aggregator(),
            config=build_trainer_config(exp_config),
        )
        training_log = trainer.run()
        model.load_state_dict(trainer.global_model.state_dict())
        torch.save(
            training_log,
            os.path.join(exp_config.logging.output_dir, f"{exp_config.logging.experiment_name}_training_log.pt"),
        )

    if args.mode in {"full", "unlearning_only", "attack_only"}:
        if training_log is None:
            raise RuntimeError("Training log required for unlearning/attack stages.")
        clients, _ = build_clients(train_loaders, eval_loaders, exp_config)
        unlearning_config = build_unlearning_config(exp_config, clients, model)
        unlearning_log = apply_unlearning(unlearning_config, training_log)
        if unlearning_log.final_model_state is not None:
            model.load_state_dict(unlearning_log.final_model_state)
        torch.save(
            unlearning_log,
            os.path.join(exp_config.logging.output_dir, f"{exp_config.logging.experiment_name}_unlearning_log.pt"),
        )

        relearn_rounds = exp_config.unlearning.extra_params.get("relearn_rounds", 0)
        relearn_fraction = exp_config.unlearning.extra_params.get("relearn_client_fraction")
        if relearn_rounds > 0:
            clients, _ = build_clients(train_loaders, eval_loaders, exp_config)
            relearn_trainer = FederatedTrainer(
                model=model,
                clients=clients,
                aggregator=Aggregator(),
                config=build_trainer_config(
                    exp_config,
                    override_rounds=relearn_rounds,
                    override_client_fraction=relearn_fraction,
                ),
            )
            relearn_log = relearn_trainer.run()
            model.load_state_dict(relearn_trainer.global_model.state_dict())
            torch.save(
                relearn_log,
                os.path.join(
                    exp_config.logging.output_dir,
                    f"{exp_config.logging.experiment_name}_relearn_log.pt",
                ),
            )

    if exp_config.attack.enabled and args.mode in {"full", "attack_only"}:
        if training_log is None or unlearning_log is None:
            raise RuntimeError("Attack stage requires training and unlearning logs.")
        dispatch_attack(exp_config, model, relearn_log or training_log, unlearning_log)

    print("Experiment finished.")


if __name__ == "__main__":
    main()
