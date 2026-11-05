"""Build the model a config names."""

from __future__ import annotations

import torch.nn as nn

from subfl.config.experiment import ModelConfig
from subfl.models.cifar_resnet import CIFARResNet


def build_model(model_config: ModelConfig) -> nn.Module:
    """Construct the model named by ``model_config``."""

    if model_config.model_name == "cifar_resnet20":
        return CIFARResNet(depth=20, num_classes=model_config.num_classes)
    if model_config.model_name == "cifar_resnet56":  # 9 blocks per stage
        return CIFARResNet(depth=56, num_classes=model_config.num_classes)
    raise ValueError(f"Unsupported model: {model_config.model_name}")
