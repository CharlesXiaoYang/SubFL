"""Resolve and log the training device."""

from __future__ import annotations

import torch

from subfl.utils.logging import log_kv


def resolve_training_device(device_preference: str) -> torch.device:
    """Map "auto", "cuda" or "cpu" to a device; "cuda" fails if no GPU is visible."""

    preference = device_preference.lower()
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' was requested but no CUDA device is available")
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device preference: {device_preference}")


def log_device_info(device: torch.device) -> None:
    """Log the training device."""

    log_kv("Training device", str(device))
    if device.type == "cuda":
        log_kv("CUDA device name", torch.cuda.get_device_name(device))
        log_kv("CUDA device count", torch.cuda.device_count())
