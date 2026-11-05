"""CIFAR ResNets of depth 6n+2 (ResNet-20, ResNet-56) with GroupNorm."""

from __future__ import annotations

import torch
import torch.nn as nn


def _group_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with at most 8 groups (no running statistics)."""

    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class CIFARBasicBlock(nn.Module):
    """Basic residual block used by the CIFAR ResNet family."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = _group_norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = _group_norm(planes)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                _group_norm(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class CIFARResNet(nn.Module):
    """The CIFAR ResNet of He et al. (2016), depth 6n+2, with GroupNorm."""

    def __init__(self, depth: int = 20, num_classes: int = 10) -> None:
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError("CIFARResNet depth must satisfy depth = 6n + 2")
        block_count = (depth - 2) // 6
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = _group_norm(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, block_count, stride=1)
        self.layer2 = self._make_layer(32, block_count, stride=2)
        self.layer3 = self._make_layer(64, block_count, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_out = nn.Linear(64, num_classes)

        self._initialize_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers: list[nn.Module] = []
        for layer_stride in strides:
            layers.append(CIFARBasicBlock(self.in_planes, planes, layer_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc_out(out)
