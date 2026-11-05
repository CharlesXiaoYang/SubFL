"""CIFAR-10 loaders: the public/private split, the client partition of the private part, and the transforms."""

from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from subfl.config.experiment import DataConfig
from subfl.data.download import open_dataset
from subfl.data.partition import client_subsets_from_indices


def _normalization_stats(mode: str) -> tuple[list[float], list[float]]:
    """CIFAR-10 per-channel mean and std."""

    if mode == "cifar10":
        return [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
    raise ValueError("normalization must be cifar10")


def _build_train_transform(config: DataConfig) -> transforms.Compose:
    """CIFAR-10 training transforms (32x32 images)."""

    mean, std = _normalization_stats(config.normalization)
    return transforms.Compose([
        transforms.RandomCrop((32, 32), padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _build_eval_transform(config: DataConfig) -> transforms.Compose:
    """CIFAR-10 evaluation transforms."""

    mean, std = _normalization_stats(config.normalization)
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])


def _loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """DataLoader with the shared worker settings."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )


def stratified_public_private_indices(
    targets: Sequence[int],
    public_samples_per_class: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split CIFAR-10 train indices into class-balanced public/private subsets."""

    if public_samples_per_class < 0:
        raise ValueError("public_samples_per_class must be non-negative")

    generator = torch.Generator().manual_seed(seed)
    by_class: dict[int, list[int]] = {}
    for index, target in enumerate(targets):
        by_class.setdefault(int(target), []).append(index)

    public_indices: list[int] = []
    private_indices: list[int] = []
    for class_id in sorted(by_class):
        class_indices = by_class[class_id]
        if public_samples_per_class > len(class_indices):
            raise ValueError(
                f"public_samples_per_class={public_samples_per_class} exceeds "
                f"class {class_id} size {len(class_indices)}"
            )
        perm = torch.randperm(len(class_indices), generator=generator).tolist()
        shuffled = [class_indices[i] for i in perm]
        public_indices.extend(shuffled[:public_samples_per_class])
        private_indices.extend(shuffled[public_samples_per_class:])

    public_indices.sort()
    private_indices.sort()
    return public_indices, private_indices


def _split_private_client_subsets(dataset, config: DataConfig) -> list[Subset]:
    """Split the private part of the training set across clients."""

    public_indices, private_indices = stratified_public_private_indices(
        targets=dataset.targets,
        public_samples_per_class=config.public_samples_per_class,
        seed=config.private_split_seed,
    )
    del public_indices
    return client_subsets_from_indices(
        dataset,
        dataset.targets,
        private_indices,
        num_clients=config.num_clients,
        seed=config.private_split_seed,
        dirichlet_alpha=config.dirichlet_alpha,
        partition=config.partition,
    )


def build_cifar10_dataloaders(config: DataConfig) -> tuple[list[DataLoader], DataLoader]:
    """Private CIFAR-10 client loaders and the test loader."""

    train_dataset = open_dataset(
        CIFAR10,
        config,
        train=True,
        transform=_build_train_transform(config),
    )
    test_dataset = open_dataset(
        CIFAR10,
        config,
        train=False,
        transform=_build_eval_transform(config),
    )
    client_loaders = [
        _loader(
            subset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
        for subset in _split_private_client_subsets(train_dataset, config)
    ]
    test_loader = _loader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return client_loaders, test_loader


def build_cifar10_public_loader(
    config: DataConfig,
    *,
    batch_size: int | None = None,
    shuffle: bool = True,
    eval_transform: bool = False,
) -> DataLoader:
    """Loader of the public CIFAR-10 pool P (anchors or validation)."""

    transform = _build_eval_transform(config) if eval_transform else _build_train_transform(config)
    dataset = open_dataset(CIFAR10, config, train=True, transform=transform)
    public_indices, _ = stratified_public_private_indices(
        targets=dataset.targets,
        public_samples_per_class=config.public_samples_per_class,
        seed=config.private_split_seed,
    )
    if not public_indices:
        raise ValueError("CIFAR-10 public loader requires public_samples_per_class > 0")
    return _loader(
        Subset(dataset, public_indices),
        batch_size=batch_size or config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
    )
