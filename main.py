"""Unified entry point for federated training, unlearning, and attacks."""
from __future__ import annotations

import argparse
import math
import os
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Tuple

import torch
from torchvision.utils import make_grid, save_image

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
    dp_config: Mapping[str, Any] | None = None,
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
            dp_config=dp_config,
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
    must_track_updates = exp_config.unlearning.level == "client"
    track_client_updates = exp_config.training.track_client_updates or must_track_updates
    tracked_client_ids = exp_config.unlearning.target_clients if must_track_updates else None
    return SimpleNamespace(
        num_rounds=override_rounds
        if override_rounds is not None
        else exp_config.training.num_rounds,
        client_fraction=override_client_fraction
        if override_client_fraction is not None
        else exp_config.training.client_fraction,
        snapshot_interval=exp_config.training.record_global_every,
        track_client_updates=track_client_updates,
        tracked_client_ids=tracked_client_ids,
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


def configure_local_privacy(exp_config: ExperimentConfig) -> Mapping[str, Any] | None:
    """Resolve client-side differential privacy settings if provided."""

    dp_cfg = getattr(exp_config.training, "local_privacy", None)
    if not dp_cfg or not dp_cfg.get("enabled", False):
        print("Local differential privacy is disabled for clients.")
        return None

    resolved = {
        "enabled": True,
        "clip_norm": float(dp_cfg.get("clip_norm", 1.0)),
        "noise_multiplier": float(dp_cfg.get("noise_multiplier", 0.0)),
    }
    print(
        "Enabling client-side differential privacy with clip_norm=",
        resolved["clip_norm"],
        " noise_multiplier=",
        resolved["noise_multiplier"],
    )
    return resolved


def save_tail_model_snapshots(
    training_log: TrainingLog, exp_config: ExperimentConfig, keep_last: int = 5
) -> list[str]:
    """Persist the final ``keep_last`` global models for later gradient diffing."""

    snapshots = list(training_log.global_model_snapshots[-keep_last:])
    if not snapshots and training_log.final_model_state is not None:
        snapshots = [training_log.final_model_state]

    save_dir = os.path.join(
        exp_config.logging.output_dir, exp_config.logging.experiment_name, "snapshots"
    )
    os.makedirs(save_dir, exist_ok=True)

    saved_paths: list[str] = []
    for idx, state in enumerate(snapshots):
        path = os.path.join(save_dir, f"global_round_tail_{idx}.pt")
        torch.save(state, path)
        saved_paths.append(path)

    print(
        f"Saved {len(saved_paths)} trailing global model snapshots to '{save_dir}' for gradient analysis."
    )
    return saved_paths


def save_initial_unlearned_model(
    unlearning_log: UnlearningLog, exp_config: ExperimentConfig
) -> str | None:
    """Store the first post-unlearning aggregation checkpoint for attack alignment."""

    anchor_state = None
    if unlearning_log.global_model_snapshots_after_unlearning:
        anchor_state = unlearning_log.global_model_snapshots_after_unlearning[0]
    elif unlearning_log.final_model_state is not None:
        anchor_state = unlearning_log.final_model_state

    if anchor_state is None:
        print("No unlearning snapshot available to persist as anchor model.")
        return None

    save_dir = os.path.join(
        exp_config.logging.output_dir, exp_config.logging.experiment_name, "unlearning"
    )
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "anchor_model.pt")
    torch.save(anchor_state, path)
    print(f"Saved unlearning anchor model to '{path}'.")
    return path


def dispatch_attack(
    exp_config: ExperimentConfig,
    model: torch.nn.Module,
    training_log: TrainingLog,
    unlearning_log: UnlearningLog,
) -> None:
    attack_level = exp_config.attack.level
    attack_config = exp_config.to_attack_config()
    attack_config.update(
        {
            "forgotten_clients": exp_config.unlearning.target_clients,
            "forgotten_classes": exp_config.unlearning.target_classes,
            "forgotten_samples": exp_config.unlearning.target_samples,
            "device": exp_config.logging.device,
        }
    )

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
    metrics_payload = (
        {cls: res.metrics for cls, res in result.items()}
        if attack_level == "class"
        else result.metrics
    )
    torch.save(metrics_payload, os.path.join(save_dir, "attack_metrics.pt"))

    if exp_config.attack.save_images:
        reconstructions, preview_images = _extract_reconstructions_for_saving(attack_level, result)
        if reconstructions is not None and preview_images is not None:
            min_required = max(len(unlearning_log.forgotten_samples), 1)
            reconstructions, preview_images = _enforce_minimum_image_count(
                reconstructions, preview_images, min_required
            )
            torch.save(reconstructions, os.path.join(save_dir, "reconstructions.pt"))
            _save_reconstruction_png(preview_images, save_dir, f"reconstructions_{attack_level}.png")


def _extract_reconstructions_for_saving(
    attack_level: str, result: Any
) -> tuple[Any | None, torch.Tensor | None]:
    """Normalize reconstruction outputs so they can be saved uniformly."""

    if attack_level == "client":
        images = _coalesce_tensor_images(
            getattr(result, "reconstructed_images", None),
            getattr(result, "final_images", None),
            getattr(result, "pseudo_images", None),
        )
        return images, images

    if attack_level == "class":
        if isinstance(result, Mapping):
            images_by_class = {
                cls: _coalesce_tensor_images(
                    getattr(res, "reconstructed_images", None),
                    getattr(res, "final_images", None),
                    getattr(res, "pseudo_images", None),
                )
                for cls, res in result.items()
            }
            flattened = [img for img in images_by_class.values() if isinstance(img, torch.Tensor) and img.numel() > 0]
            preview = torch.cat(flattened, dim=0) if flattened else torch.empty(0)
            return images_by_class, preview

        images = _coalesce_tensor_images(
            getattr(result, "reconstructed_images", None),
            getattr(result, "final_images", None),
            getattr(result, "pseudo_images", None),
        )
        return images, images

    if attack_level == "sample":
        candidate_images = [
            cand.image.detach().cpu()
            for cand in getattr(result, "top_candidates", [])
            if hasattr(cand, "image")
        ]
        stacked = torch.stack(candidate_images) if candidate_images else torch.empty(0)
        return stacked, stacked

    return None, None


def _coalesce_tensor_images(*candidates: Any) -> torch.Tensor:
    """Extract the first available image batch from heterogeneous containers."""

    for candidate in candidates:
        if candidate is None:
            continue

        if isinstance(candidate, torch.Tensor):
            return _ensure_batch(candidate.detach().cpu())

        if isinstance(candidate, Mapping):
            stacked = _stack_tensor_collection(candidate.values())
            if stacked is not None:
                return stacked

        if isinstance(candidate, (list, tuple)):
            stacked = _stack_tensor_collection(candidate)
            if stacked is not None:
                return stacked

    return torch.empty(0)


def _stack_tensor_collection(items: Iterable[Any]) -> torch.Tensor | None:
    tensors = []
    for item in items:
        if isinstance(item, torch.Tensor):
            tensors.append(_ensure_batch(item.detach().cpu()))
        elif isinstance(item, (list, tuple)):
            for sub in item:
                if isinstance(sub, torch.Tensor):
                    tensors.append(_ensure_batch(sub.detach().cpu()))

    if not tensors:
        return None

    try:
        return torch.cat(tensors, dim=0)
    except Exception:
        return torch.stack(tensors)


def _ensure_batch(tensor: torch.Tensor) -> torch.Tensor:
    """Guarantee the first dimension represents batch size."""

    return tensor.unsqueeze(0) if tensor.dim() == 3 else tensor


def _enforce_minimum_image_count(
    reconstructions: Any, preview_images: torch.Tensor, min_required: int
) -> tuple[Any, torch.Tensor]:
    """Ensure saved artifacts include at least ``min_required`` images."""

    adjusted_preview = _ensure_minimum_images(preview_images, min_required)

    if isinstance(reconstructions, torch.Tensor):
        adjusted_reconstructions: Any = _ensure_minimum_images(reconstructions, min_required)
    elif isinstance(reconstructions, Mapping):
        adjusted_reconstructions = dict(reconstructions)
        total = sum(
            tensor.shape[0]
            for tensor in adjusted_reconstructions.values()
            if isinstance(tensor, torch.Tensor)
        )
        missing = max(min_required - total, 0)
        if missing > 0:
            for key in reversed(list(adjusted_reconstructions.keys())):
                tensor = adjusted_reconstructions[key]
                if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                    adjusted_reconstructions[key] = _ensure_minimum_images(
                        tensor, tensor.shape[0] + missing
                    )
                    break
    else:
        adjusted_reconstructions = reconstructions

    return adjusted_reconstructions, adjusted_preview


def _ensure_minimum_images(images: torch.Tensor, min_required: int) -> torch.Tensor:
    """Pad the image batch by repeating samples until the minimum count is met."""

    if images is None or images.numel() == 0:
        return images

    batch = images.shape[0]
    if batch >= min_required:
        return images

    repeats = math.ceil(min_required / batch)
    tiled = images.repeat((repeats, 1, 1, 1))
    return tiled[:min_required]


def _save_reconstruction_png(images: torch.Tensor, save_dir: str, filename: str) -> None:
    """Save a grid preview of reconstructions to disk."""

    if images is None or images.numel() == 0:
        print("No reconstructed images available for visualization.")
        return

    grid = make_grid(images, nrow=min(8, images.shape[0]), normalize=True, value_range=(-1, 1))
    save_path = os.path.join(save_dir, filename)
    save_image(grid, save_path)
    print(f"Saved reconstruction preview to {save_path}")


def main() -> None:
    args = parse_args()
    print(f"Loading experiment configuration from {args.config}...")
    exp_config = load_experiment_config(args.config)
    print(
        f"Configuration loaded for experiment '{exp_config.logging.experiment_name}'"
        f" on device '{exp_config.logging.device}'."
    )

    set_seed(exp_config.logging.seed)
    os.makedirs(exp_config.logging.output_dir, exist_ok=True)

    print("[Step 1] Preparing dataset (download/preprocess if needed)...")
    train_loaders, eval_loaders = get_federated_dataloaders(exp_config)
    print(
        f"[Step 2] Allocated dataset to {len(train_loaders)} federated clients"
        f" with evaluation splits: {list(eval_loaders.keys())}"
    )

    dp_config = configure_local_privacy(exp_config)

    print(f"Building model '{exp_config.model.arch}'...")
    model = build_model(exp_config.model).to(exp_config.logging.device)
    print("Model built and moved to device.")

    if args.resume_path:
        state_dict = torch.load(args.resume_path, map_location=exp_config.logging.device)
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {args.resume_path}")

    training_log: TrainingLog | None = None
    unlearning_log: UnlearningLog | None = None
    relearn_log: TrainingLog | None = None

    if args.mode in {"full", "train_only"}:
        print("Starting federated training stage...")
        clients, _ = build_clients(train_loaders, eval_loaders, exp_config, dp_config)
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
        save_tail_model_snapshots(training_log, exp_config, keep_last=5)
        print("Training stage complete. Training log saved.")

    if args.mode in {"full", "unlearning_only", "attack_only"}:
        if training_log is None:
            raise RuntimeError("Training log required for unlearning/attack stages.")
        print("Starting unlearning stage...")
        clients, _ = build_clients(train_loaders, eval_loaders, exp_config, dp_config)
        unlearning_config = build_unlearning_config(exp_config, clients, model)
        unlearning_log = apply_unlearning(unlearning_config, training_log)
        if unlearning_log.final_model_state is not None:
            model.load_state_dict(unlearning_log.final_model_state)
        torch.save(
            unlearning_log,
            os.path.join(exp_config.logging.output_dir, f"{exp_config.logging.experiment_name}_unlearning_log.pt"),
        )
        save_initial_unlearned_model(unlearning_log, exp_config)
        print("Unlearning stage complete. Unlearning log saved.")

        relearn_rounds = exp_config.unlearning.extra_params.get("relearn_rounds", 0)
        relearn_fraction = exp_config.unlearning.extra_params.get("relearn_client_fraction")
        if relearn_rounds > 0:
            print(
                f"Starting relearning for {relearn_rounds} rounds"
                f" with client fraction {relearn_fraction}."
            )
            clients, _ = build_clients(train_loaders, eval_loaders, exp_config, dp_config)
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
            print("Relearning stage complete. Relearning log saved.")
        else:
            print("Skipping relearning stage (no relearn rounds configured).")

    if exp_config.attack.enabled and args.mode in {"full", "attack_only"}:
        if training_log is None or unlearning_log is None:
            raise RuntimeError("Attack stage requires training and unlearning logs.")
        print(f"Starting attack evaluation at '{exp_config.attack.level}' level...")
        dispatch_attack(exp_config, model, relearn_log or training_log, unlearning_log)
        print("Attack evaluation complete. Metrics saved.")

    print("Experiment finished.")


if __name__ == "__main__":
    main()
