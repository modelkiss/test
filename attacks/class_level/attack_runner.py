"""类级完整攻击流程（调用 common 组件）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from attacks.class_level.guidance_config import (
    get_class_level_stage1_weights,
    get_class_level_stage2_weights,
)
from attacks.class_level.target_builder import build_class_level_targets
from attacks.common.utils import ReconstructionAssets, prepare_reconstruction_assets


@dataclass
class AttackResult:
    """Container for reconstructed artifacts and metadata."""

    targets: Dict[int, Dict[str, Any]]
    pseudo_images: Dict[int, List[Any]] = field(default_factory=dict)
    final_images: Dict[int, List[Any]] = field(default_factory=dict)
    scores: Dict[int, List[float]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    preparation: Optional[ReconstructionAssets] = None

    @property
    def metrics(self) -> Dict[int, List[float]]:
        """Alias for ``scores`` to align with downstream consumers."""

        return self.scores


def run_class_level_attack(
    config: Mapping[str, Any], model_state: Any, training_log: Any, unlearning_log: Any
) -> AttackResult:
    """串联类级攻击流程，返回重建结果。

    本实现侧重于数据准备和调度信息记录，具体的扩散模型微调 / 采样步骤
    由上层或后续组件完成。
    """

    forgotten_classes: List[int] = list(config.get("forgotten_classes", []))
    if not forgotten_classes:
        raise ValueError("No forgotten classes provided for class-level attack.")

    target_cfg = config.get("target_builder", {}) if isinstance(config, dict) else {}

    federated_state: Dict[str, Any] = {
        "model": model_state,
        "training_log": training_log,
        "unlearning_log": unlearning_log,
        "client_updates": getattr(training_log, "client_updates", {}),
    }

    targets = build_class_level_targets(federated_state, forgotten_classes, target_cfg)

    stage1_weights = get_class_level_stage1_weights(config.get("stage1_weights", {}))
    stage2_weights = get_class_level_stage2_weights(config.get("stage2_weights", {}))

    prep = prepare_reconstruction_assets(
        config,
        training_log,
        unlearning_log,
        target_classes=forgotten_classes,
    )

    notes = [
        f"Prepared targets for {len(targets)} classes.",
        f"Stage1 weights: {stage1_weights}",
        f"Stage2 weights: {stage2_weights}",
        *prep.notes,
    ]

    # 具体的微调与生成过程由调用方实现，这里返回构建好的调度信息。
    return AttackResult(targets=targets, notes=notes, preparation=prep)
