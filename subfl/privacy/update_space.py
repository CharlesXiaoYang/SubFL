"""Per-tensor parameter groups, their update subspaces, and the split of the subspace rank across tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ParamGroup:
    """One named group of contiguous parameter coordinates in the flat model vector."""

    group_name: str
    sl: tuple[int, int]
    param_names: list[str]


@dataclass
class UpdateSubspace:
    """One orthonormal update-space basis for a parameter tensor."""

    group_name: str
    sl: tuple[int, int]
    basis: torch.Tensor
    actual_rank: int
    protected: bool = False


def build_tensor_param_groups(model: nn.Module) -> list[ParamGroup]:
    """Group each parameter tensor separately in flattened-vector order."""

    groups: list[ParamGroup] = []
    pointer = 0
    for name, param in model.named_parameters():
        start = pointer
        end = start + param.numel()
        pointer = end
        groups.append(ParamGroup(group_name=name, sl=(start, end), param_names=[name]))
    if not groups:
        raise ValueError("model has no parameters selected for update-space grouping")
    return groups


def _is_bias_group(group: ParamGroup) -> bool:
    return all(name.endswith(".bias") for name in group.param_names)


def _is_classifier_group(group: ParamGroup, classifier_head_names: set[str]) -> bool:
    for name in group.param_names:
        top_level = name.split(".", 1)[0]
        if top_level in classifier_head_names or name in classifier_head_names:
            return True
    return False


def _protected_group(
    group: ParamGroup,
    *,
    classifier_head_names: set[str],
    protect_classifier_head: bool,
    full_rank_bias: bool,
    head_full_rank_threshold: int,
) -> bool:
    width = group.sl[1] - group.sl[0]
    if full_rank_bias and _is_bias_group(group):
        return True
    if protect_classifier_head and _is_classifier_group(group, classifier_head_names):
        return width <= head_full_rank_threshold
    return False


def _allocate_nonprotected_ranks(
    groups: list[ParamGroup],
    basis_dim: int,
    *,
    anchor_matrix: torch.Tensor,
    strategy: str,
    min_rank: int,
) -> list[int]:
    """Split the rank budget across non-protected groups in proportion to their anchor gradient energy."""

    if basis_dim < 0:
        raise ValueError("basis_dim must be non-negative")
    if not groups or basis_dim == 0:
        return [0 for _ in groups]
    if strategy != "energy":
        raise ValueError(f"Unsupported rank allocation strategy: {strategy}")
    energy_weights = []
    for group in groups:
        start, end = group.sl
        group_matrix = anchor_matrix[:, start:end]
        energy_weights.append(max(float(torch.sum(group_matrix * group_matrix).item()), 1e-12))
    weights = np.array(energy_weights, dtype=float)

    weights = weights / weights.sum()
    raw = weights * float(basis_dim)
    ranks = np.floor(raw).astype(int)
    if basis_dim >= min_rank * len(groups):
        ranks = np.maximum(ranks, min_rank)

    while int(ranks.sum()) > basis_dim:
        idx = int(np.argmax(ranks))
        if ranks[idx] <= 0:
            break
        ranks[idx] -= 1

    remainders = raw - ranks
    while int(ranks.sum()) < basis_dim:
        idx = int(np.argmax(remainders))
        ranks[idx] += 1
        remainders[idx] = 0.0

    return [max(0, int(rank)) for rank in ranks.tolist()]
