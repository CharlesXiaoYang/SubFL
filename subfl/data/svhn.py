"""SVHN loaders: private client subsets, the public pool P and the test set."""

from __future__ import annotations

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import SVHN

from subfl.config.experiment import DataConfig
from subfl.data.download import open_dataset
from subfl.data.cifar10 import stratified_public_private_indices
from subfl.data.partition import client_subsets_from_indices

PUBLIC_SPLITS = ("same", "extra")  # DataConfig.svhn_public_split
# P per (root, split, public_samples_per_class, private_split_seed), read once per process; only P is kept, since
# 'extra' is large
_PUBLIC_POOLS: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _normalization_stats(mode: str) -> tuple[list[float], list[float]]:
    """Return normalization constants for SVHN-style inputs."""

    if mode == "cifar10":
        return [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
    raise ValueError("normalization must be one of: cifar10, public_pool")


def _stats(config: DataConfig) -> tuple[list[float], list[float]]:
    """Normalization mean and std: for public_pool the per-channel statistics of P's raw images, else fixed ones."""

    if config.normalization != "public_pool":
        return _normalization_stats(config.normalization)
    images, _ = _public_pool_arrays(config)
    pixels = images.astype(np.float64) / 255.0  # (N, 3, H, W)
    return pixels.mean(axis=(0, 2, 3)).tolist(), pixels.std(axis=(0, 2, 3)).tolist()


def _build_train_transform(config: DataConfig) -> transforms.Compose:
    """SVHN training transforms (32x32 images, no flip)."""

    mean, std = _stats(config)
    return transforms.Compose([
        transforms.RandomCrop((32, 32), padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _build_eval_transform(config: DataConfig) -> transforms.Compose:
    """SVHN evaluation transforms."""

    mean, std = _stats(config)
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])


def _loader(dataset, *, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    """Create a DataLoader with the repo's usual worker settings."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )


def _train_split(config: DataConfig) -> str:
    """The torchvision SVHN split of the private training data: 'train'."""

    split = config.svhn_train_split.lower()
    if split != "train":
        raise ValueError("svhn_train_split must be 'train'")
    return split


def _public_split(config: DataConfig) -> str:
    """The SVHN split P is drawn from: 'train' (svhn_public_split "same") or 'extra'."""

    mode = str(config.svhn_public_split)
    if mode not in PUBLIC_SPLITS:
        raise ValueError(f"svhn_public_split must be one of {PUBLIC_SPLITS}, got {mode!r}")
    return _train_split(config) if mode == "same" else "extra"


def _public_pool_arrays(config: DataConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return P as read-only arrays (uint8 images (N, 3, H, W), labels): public_samples_per_class per class."""

    split = _public_split(config)
    key = (str(config.root), split, int(config.public_samples_per_class), int(config.private_split_seed))
    if key not in _PUBLIC_POOLS:
        dataset = open_dataset(SVHN, config, split=split)
        public_indices, _ = stratified_public_private_indices(
            targets=dataset.labels.tolist(),
            public_samples_per_class=config.public_samples_per_class,
            seed=config.private_split_seed,
        )
        if not public_indices:
            raise ValueError("SVHN public loader requires public_samples_per_class > 0")
        images = np.ascontiguousarray(dataset.data[public_indices])
        labels = np.ascontiguousarray(dataset.labels[public_indices])
        images.flags.writeable = labels.flags.writeable = False
        _PUBLIC_POOLS[key] = (images, labels)
    return _PUBLIC_POOLS[key]


class _PublicPool(Dataset):
    """P as a dataset whose items equal torchvision's SVHN items of the same records."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform) -> None:
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        img = Image.fromarray(np.transpose(self.images[index], (1, 2, 0)))  # as SVHN.__getitem__
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[index])


def _split_private_client_subsets(dataset: SVHN, config: DataConfig) -> list[Subset]:
    """Build per-client private subsets after removing public data (none when P comes from another split)."""

    if _public_split(config) == _train_split(config):
        _, private_indices = stratified_public_private_indices(
            targets=dataset.labels.tolist(),
            public_samples_per_class=config.public_samples_per_class,
            seed=config.private_split_seed,
        )
    else:  # svhn_public_split "extra": every record of 'train' is private
        private_indices = range(len(dataset))
    return client_subsets_from_indices(
        dataset,
        dataset.labels,
        private_indices,
        num_clients=config.num_clients,
        seed=config.private_split_seed,
        dirichlet_alpha=config.dirichlet_alpha,
        partition=config.partition,
    )


def build_svhn_dataloaders(config: DataConfig) -> tuple[list[DataLoader], DataLoader]:
    """Build private SVHN FL client loaders and the official test loader."""

    train_dataset = open_dataset(
        SVHN,
        config,
        split=_train_split(config),
        transform=_build_train_transform(config),
    )
    test_dataset = open_dataset(
        SVHN,
        config,
        split="test",
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


def build_svhn_public_loader(
    config: DataConfig,
    *,
    batch_size: int | None = None,
    shuffle: bool = True,
    eval_transform: bool = False,
) -> DataLoader:
    """Build the public SVHN subset loader (P, _public_pool_arrays) for the anchors or validation."""

    transform = _build_eval_transform(config) if eval_transform else _build_train_transform(config)
    images, labels = _public_pool_arrays(config)
    return _loader(
        _PublicPool(images, labels, transform),
        batch_size=batch_size or config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
    )
