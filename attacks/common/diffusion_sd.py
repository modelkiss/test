"""预训练 Stable Diffusion 资源管理工具。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DiffusionCheckpoint:
    """简单的扩散模型检查点描述。"""

    model_name: str
    path: str
    downloaded: bool
    note: str = ""


def ensure_local_diffusion_checkpoint(
    model_name: str,
    target_dir: str = "SD",
    filename: Optional[str] = None,
    format_hint: Optional[str] = "ckpt",
) -> DiffusionCheckpoint:
    """确保本地存在可用的扩散模型权重。

    出于环境限制（无网络），本函数不会尝试真实下载，而是：
    1. 创建 ``target_dir`` 目录；
    2. 如果指定文件不存在，则写入一个占位符，记录需要手动放置权重。
    """

    os.makedirs(target_dir, exist_ok=True)

    # 自动从文件名或者格式提示中推导扩展名，便于直接使用 .safetensors 等单文件权重。
    ext_hint = None
    if filename and "." in filename:
        ext_hint = filename.split(".")[-1].lower()
    elif format_hint:
        ext_hint = format_hint.lstrip(".").lower()

    ckpt_name = filename or f"{model_name}.{ext_hint or 'ckpt'}"
    ckpt_path = os.path.join(target_dir, ckpt_name)

    downloaded = os.path.exists(ckpt_path)
    note = ""

    if not downloaded:
        suffix = ext_hint or "ckpt"
        load_hint = (
            "Use StableDiffusionPipeline.from_single_file for safetensors checkpoints."
            if suffix == "safetensors"
            else ""
        )
        note = " ".join(
            filter(
                None,
                [
                    f"Placeholder {suffix} checkpoint created.",
                    "Replace with actual diffusion weights for full reconstruction fidelity.",
                    load_hint,
                ],
            )
        )
        with open(ckpt_path, "w", encoding="utf-8") as f:
            f.write(
                "This is a placeholder for the diffusion checkpoint. "
                "Please place the pretrained weights here.\n"
            )
        downloaded = False

    return DiffusionCheckpoint(
        model_name=model_name,
        path=ckpt_path,
        downloaded=downloaded,
        note=note,
    )


__all__ = ["DiffusionCheckpoint", "ensure_local_diffusion_checkpoint"]
