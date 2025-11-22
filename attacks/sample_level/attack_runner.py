"""样本级完整攻击流程（调用 common 组件）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from attacks.sample_level.guidance_config import (
    get_sample_level_stage1_weights,
    get_sample_level_stage2_weights,
)
from attacks.sample_level.target_builder import build_sample_level_targets
from attacks.common.utils import ReconstructionAssets, prepare_reconstruction_assets


@dataclass
class AttackResult:
    """Container for reconstructed artifacts and metadata."""

    targets: Dict[str, Dict[str, Any]]
    candidate_pool: Dict[str, List[Any]] = field(default_factory=dict)
    final_images: Dict[str, List[Any]] = field(default_factory=dict)
    scores: Dict[str, List[float]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    preparation: Optional[ReconstructionAssets] = None

    @property
    def metrics(self) -> Dict[str, List[float]]:
        """Alias for ``scores`` to align with downstream consumers."""

        return self.scores


def run_sample_level_attack(
    config: Mapping[str, Any], model_state: Any, training_log: Any, unlearning_log: Any
) -> AttackResult:
    """串联样本级攻击流程，返回重建结果。

    该流程专注于单样本目标构造、候选池维护与两阶段权重配置，
    具体的扩散模型微调 / 采样步骤交由上层实现。
    """

    target_samples: List[str] = list(config.get("target_samples", []))
    if not target_samples:
        raise ValueError("No target samples provided for sample-level attack.")

    target_cfg = config.get("target_builder", {}) if isinstance(config, dict) else {}

    federated_state: Dict[str, Any] = {
        "model": model_state,
        "training_log": training_log,
        "unlearning_log": unlearning_log,
        "client_updates": getattr(training_log, "client_updates", {}),
    }

    targets = build_sample_level_targets(federated_state, target_samples, target_cfg)

    stage1_weights = get_sample_level_stage1_weights(config.get("stage1_weights", {}))
    stage2_weights = get_sample_level_stage2_weights(config.get("stage2_weights", {}))

    prep = prepare_reconstruction_assets(
        config,
        training_log,
        unlearning_log,
        target_classes=config.get("forgotten_classes", []),
    )

    notes = [
        f"Prepared targets for {len(targets)} samples.",
        f"Stage1 weights: {stage1_weights}",
        f"Stage2 weights: {stage2_weights}",
        "Candidate pool enabled for iterative screening.",
        *prep.notes,
    ]

    # 具体的微调、候选生成与精修由调用方实现，这里仅汇总调度信息。
    return AttackResult(targets=targets, notes=notes, preparation=prep)
