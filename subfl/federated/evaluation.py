"""Evaluate a classifier on a data loader."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_classifier(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Average cross-entropy loss and accuracy of ``model`` on ``data_loader``."""

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    model_dtype = next(model.parameters()).dtype

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device=device, dtype=model_dtype if model_dtype != torch.float32 else torch.float32)
            targets = targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs.float(), targets).item()
            total_correct += (outputs.argmax(dim=1) == targets).sum().item()
            total_examples += targets.size(0)

    return total_loss / total_examples, total_correct / total_examples
