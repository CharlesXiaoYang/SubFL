"""Split dataset indices across clients: IID or class-wise Dirichlet."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Subset


def _targets_array(targets: Sequence[int] | torch.Tensor | np.ndarray) -> np.ndarray:
    """Targets as a one-dimensional NumPy array."""

    if isinstance(targets, torch.Tensor):
        return targets.detach().cpu().numpy()
    return np.asarray(targets)


def iid_client_subsets(dataset, indices: Sequence[int], num_clients: int, seed: int) -> list[Subset]:
    """Split indices evenly across clients after a seeded shuffle."""

    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    indices_list = [int(index) for index in indices]
    if len(indices_list) < num_clients:
        raise ValueError(
            f"Cannot split {len(indices_list)} examples across {num_clients} clients"
        )

    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(len(indices_list), generator=generator).tolist()
    shuffled = [indices_list[i] for i in permutation]

    base_len = len(shuffled) // num_clients
    lengths = [base_len] * num_clients
    for client_id in range(len(shuffled) - base_len * num_clients):
        lengths[client_id] += 1

    subsets: list[Subset] = []
    cursor = 0
    for length in lengths:
        subsets.append(Subset(dataset, shuffled[cursor : cursor + length]))
        cursor += length
    return subsets


def dirichlet_client_subsets(
    dataset,
    targets: Sequence[int] | torch.Tensor | np.ndarray,
    indices: Sequence[int],
    *,
    num_clients: int,
    alpha: float,
    seed: int,
    min_size: int = 1,
    max_retries: int = 100,
) -> list[Subset]:
    """Split labeled indices into non-IID clients with class-wise Dirichlet draws."""

    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if alpha <= 0:
        raise ValueError("dirichlet_alpha must be positive")

    targets_np = _targets_array(targets)
    indices_np = np.asarray([int(index) for index in indices], dtype=np.int64)
    if len(indices_np) < num_clients:
        raise ValueError(
            f"Cannot split {len(indices_np)} examples across {num_clients} clients"
        )

    class_ids = np.unique(targets_np[indices_np])
    best_split: list[list[int]] | None = None
    best_min_size = -1

    for attempt in range(max_retries):
        rng = np.random.default_rng(int(seed) + attempt)
        client_indices: list[list[int]] = [[] for _ in range(num_clients)]

        for class_id in class_ids:
            class_indices = indices_np[targets_np[indices_np] == class_id].copy()
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(num_clients, float(alpha)))
            split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
            for client_id, split in enumerate(np.split(class_indices, split_points)):
                client_indices[client_id].extend(int(index) for index in split.tolist())

        for split in client_indices:
            rng.shuffle(split)

        smallest = min(len(split) for split in client_indices)
        if smallest > best_min_size:
            best_min_size = smallest
            best_split = [list(split) for split in client_indices]
        if smallest >= min_size:
            return [Subset(dataset, split) for split in client_indices]

    if best_split is None:
        raise RuntimeError("Dirichlet partition failed to create any client split")

    # a small alpha can leave a client below min_size: move examples over from the largest clients
    for client_id, split in enumerate(best_split):
        while len(split) < min_size:
            donor_id = max(range(num_clients), key=lambda i: len(best_split[i]))
            if len(best_split[donor_id]) <= min_size:
                raise RuntimeError("Dirichlet partition repair could not satisfy min_size")
            split.append(best_split[donor_id].pop())

    return [Subset(dataset, split) for split in best_split]


PARTITIONS = ("iid", "dirichlet")


def client_subsets_from_indices(
    dataset,
    targets: Sequence[int] | torch.Tensor | np.ndarray,
    indices: Sequence[int],
    *,
    num_clients: int,
    seed: int,
    dirichlet_alpha: float,
    partition: str,
) -> list[Subset]:
    """IID or Dirichlet(``dirichlet_alpha``) client subsets of ``indices``."""

    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}, got {partition!r}")
    if partition == "iid":
        return iid_client_subsets(dataset, indices, num_clients, seed)
    return dirichlet_client_subsets(
        dataset,
        targets,
        indices,
        num_clients=num_clients,
        alpha=dirichlet_alpha,
        seed=seed,
    )
