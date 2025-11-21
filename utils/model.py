"""Model factory helpers."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torchvision.models as tv_models


class SimpleConvNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        return self.classifier(x)


def build_model(model_config: Any) -> nn.Module:
    """Instantiate a model according to ``model_config``."""

    arch = getattr(model_config, "arch", None) or model_config.get("arch")
    num_classes = getattr(model_config, "num_classes", None) or model_config.get("num_classes")
    pretrained = getattr(model_config, "pretrained", False)

    if arch is None or num_classes is None:
        raise ValueError("Model config requires 'arch' and 'num_classes'.")

    if arch.lower() == "resnet18":
        model = tv_models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if arch.lower() == "convnet":
        return SimpleConvNet(num_classes)

    raise ValueError(f"Unsupported model architecture: {arch}")
