"""Data loaders by dataset name (CIFAR-10 or SVHN)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from subfl.config.experiment import DataConfig
from subfl.data.cifar10 import _normalization_stats as _cifar10_normalization_stats
from subfl.data.cifar10 import build_cifar10_dataloaders, build_cifar10_public_loader
from subfl.data.svhn import _stats as _svhn_normalization_stats
from subfl.data.svhn import build_svhn_dataloaders, build_svhn_public_loader


def num_classes_for_dataset(dataset_name: str) -> int:
    """Number of classes of a dataset."""

    if dataset_name in {"cifar10", "svhn"}:
        return 10
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def build_federated_dataloaders(config: DataConfig) -> tuple[list[DataLoader], DataLoader]:
    """Build private FL client loaders and a global test loader."""

    if config.dataset_name == "cifar10":
        return build_cifar10_dataloaders(config)
    if config.dataset_name == "svhn":
        return build_svhn_dataloaders(config)
    raise ValueError(f"Unsupported dataset_name: {config.dataset_name}")


def build_public_anchor_loader(
    config: DataConfig,
    *,
    batch_size: int | None = None,
    shuffle: bool = True,
) -> DataLoader:
    """Loader over the public pool P with the training transform (the SubFL anchors)."""

    if config.dataset_name == "cifar10":
        return build_cifar10_public_loader(config, batch_size=batch_size, shuffle=shuffle)
    if config.dataset_name == "svhn":
        return build_svhn_public_loader(config, batch_size=batch_size, shuffle=shuffle)
    raise ValueError(f"{config.dataset_name} does not define a public pool")


def build_public_validation_loader(config: DataConfig, *, batch_size: int | None = None) -> DataLoader:
    """The public pool as a validation set: evaluation transform, fixed order, no draws from the global RNG."""

    if config.dataset_name == "cifar10":
        public = build_cifar10_public_loader(config, batch_size=batch_size, shuffle=False, eval_transform=True)
    elif config.dataset_name == "svhn":
        public = build_svhn_public_loader(config, batch_size=batch_size, shuffle=False, eval_transform=True)
    else:
        raise ValueError(f"{config.dataset_name} does not define a public split")
    return DataLoader(public.dataset, batch_size=public.batch_size, shuffle=False, num_workers=public.num_workers,
                      pin_memory=True, generator=torch.Generator().manual_seed(0))


def data_record(config: DataConfig) -> dict:
    """Normalization mean/std and SVHN public split of a run; reads no private data and draws no random numbers."""

    if config.dataset_name == "cifar10":
        mean, std = _cifar10_normalization_stats(config.normalization)
    elif config.dataset_name == "svhn":
        mean, std = _svhn_normalization_stats(config)
    else:
        raise ValueError(f"Unsupported dataset_name: {config.dataset_name}")
    return {"dataset_name": str(config.dataset_name), "normalization": str(config.normalization),
            "normalization_mean": mean, "normalization_std": std,
            "svhn_public_split": str(config.svhn_public_split)}
