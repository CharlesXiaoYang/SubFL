"""Per-step GEP primitives: per-example gradients, the public anchor subspace, and the privatized lot."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from subfl.config.experiment import ExperimentConfig
from subfl.privacy.update_space import (
    UpdateSubspace,
    _allocate_nonprotected_ranks,
    _protected_group,
    build_tensor_param_groups,
)


def _disable_inplace_relu(model: nn.Module) -> None:
    """Make every ReLU out-of-place for torch.func (same forward pass)."""

    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


def _per_example_grads(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    param_names: list[str],
    *,
    chunk: int = 128,
) -> torch.Tensor:
    """Vectorized per-example gradients via torch.func -> (B, D), columns in named_parameters order."""

    params = {name: param.detach() for name, param in model.named_parameters()}
    buffers = {name: buf.detach() for name, buf in model.named_buffers()}
    if inputs.shape[0] == 0:  # an empty Poisson lot: no rows
        return torch.zeros(0, sum(params[name].numel() for name in param_names), device=inputs.device)

    def loss_fn(params, buffers, x, y):
        out = functional_call(model, (params, buffers), (x.unsqueeze(0),))
        return F.cross_entropy(out.float(), y.unsqueeze(0))

    grad_fn = grad(loss_fn)
    rows: list[torch.Tensor] = []
    for start in range(0, inputs.shape[0], chunk):
        xb = inputs[start : start + chunk]
        yb = targets[start : start + chunk]
        per_sample = vmap(grad_fn, in_dims=(None, None, 0, 0))(params, buffers, xb, yb)
        rows.append(torch.cat([per_sample[name].reshape(xb.shape[0], -1) for name in param_names], dim=1))
    return torch.cat(rows, dim=0)


def _build_anchor_subspace(
    model: nn.Module,
    anchor: torch.Tensor,
    config: ExperimentConfig,
    head_names: set,
    basis_dim: int,
    basis_gen: torch.Generator | None = None,
) -> list[UpdateSubspace]:
    """Per-tensor subspaces of the anchor gradients (randomized SVD, or random bases from ``basis_gen``)."""

    source = str(config.gep.subspace_source)
    if source not in SUBSPACE_SOURCES:
        raise ValueError(f"gep.subspace_source must be one of {SUBSPACE_SOURCES}, got {source!r}")
    if source == "random" and basis_gen is None:
        raise ValueError("gep.subspace_source=random needs a basis generator")
    if anchor.shape[0] > 0 and not bool(_rows_finite(anchor).all()):
        raise FloatingPointError("public anchor gradients are not finite (the model has diverged)")
    groups = build_tensor_param_groups(model)
    protected_flags = [
        _protected_group(
            group,
            classifier_head_names=head_names,
            protect_classifier_head=config.gep.protect_classifier_head,
            full_rank_bias=config.gep.full_rank_bias,
            head_full_rank_threshold=config.gep.head_full_rank_threshold,
        )
        for group in groups
    ]
    nonprotected = [group for group, prot in zip(groups, protected_flags) if not prot]
    ranks = _allocate_nonprotected_ranks(
        nonprotected,
        basis_dim,
        anchor_matrix=anchor,
        strategy=config.gep.rank_allocation,
        min_rank=config.gep.min_subspace_rank,
    )
    rank_by_name = {group.group_name: rank for group, rank in zip(nonprotected, ranks)}

    subspaces: list[UpdateSubspace] = []
    for group, prot in zip(groups, protected_flags):
        start, end = group.sl
        width = end - start
        if prot:  # protected tensors (biases, a small classifier head) keep the identity basis
            subspaces.append(UpdateSubspace(group.group_name, group.sl, torch.eye(width), width, True))
            continue
        requested = int(rank_by_name.get(group.group_name, 0))
        if requested <= 0:
            continue
        matrix = anchor[:, start:end]
        rank = min(requested, matrix.shape[0], matrix.shape[1])
        if rank <= 0:
            continue
        if source == "random":  # uniformly random subspace of the same rank: QR of a Gaussian matrix
            gauss = torch.randn(width, rank, generator=basis_gen, device=basis_gen.device)
            q_mat, _ = torch.linalg.qr(gauss)
            basis = q_mat.T.contiguous().detach().cpu().float()
        else:
            q = min(rank + 4, matrix.shape[0], matrix.shape[1])
            _, _, v = torch.svd_lowrank(matrix, q=q, niter=2)
            basis = v[:, :rank].T.contiguous().detach().cpu().float()
        subspaces.append(UpdateSubspace(group.group_name, group.sl, basis, rank, False))
    return subspaces


def _apply_flat_step(model: nn.Module, step_vec: torch.Tensor, lr: float) -> None:
    """In-place w <- w - lr * step_vec, scattering the flat vector back to parameters."""

    with torch.no_grad():
        pointer = 0
        for param in model.parameters():
            numel = param.numel()
            param.add_(step_vec[pointer : pointer + numel].view_as(param), alpha=-lr)
            pointer += numel


SUBSPACE_SOURCES = ("anchor_svd", "random")

# Per-example rows zeroed by the non-finite guards; the runner resets it per run and records it in
# run_metadata.json. A health check computed without noise, not part of the DP release.
NONFINITE_ROWS_ZEROED = [0]
_FINITE_CHECK_ELEMENTS = 1 << 24  # elements per isfinite() call: never a full (B, D) bool tensor


def _rows_finite(t: torch.Tensor) -> torch.Tensor:
    """(rows,) bool for a 2-D tensor: True where the row has no NaN or inf, checked in row chunks."""

    if t.shape[0] == 0:
        return torch.ones(0, dtype=torch.bool, device=t.device)
    rows = max(1, _FINITE_CHECK_ELEMENTS // max(1, t.shape[1]))
    return torch.cat([torch.isfinite(t[s : s + rows]).all(dim=1) for s in range(0, t.shape[0], rows)])


def _zero_nonfinite_rows_(grads: torch.Tensor) -> int:
    """Zero in place every row of ``grads`` holding a NaN or inf, so that record adds nothing; return how many."""

    if grads.shape[0] == 0:
        return 0
    ok = _rows_finite(grads)
    if bool(ok.all()):
        return 0
    bad = torch.nonzero(~ok).flatten()
    grads[bad] = 0.0
    NONFINITE_ROWS_ZEROED[0] += int(bad.numel())
    return int(bad.numel())


def _privatize_gep(
    grads: torch.Tensor,
    subspaces,
    *,
    embedding_clip_norm: float,
    residual_clip_norm: float,
    upload_residual: bool,
    z: float,
    gen: torch.Generator,
    divisor: int,
    residual_z: float | None = None,
) -> torch.Tensor:
    """One GEP DP-SGD lot: project, clip, sum and noise embedding and residual, divide by ``divisor``, reconstruct."""

    _zero_nonfinite_rows_(grads)  # first, before anything reads grads
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    device = grads.device
    lot = grads.shape[0]
    dim = grads.shape[1]

    emb_chunks: list[torch.Tensor] = []
    recon_info: list[tuple[int, int, torch.Tensor, bool]] = []
    residual = grads.clone()
    for sub in subspaces:
        start, end = sub.sl
        basis = sub.basis.to(device=device, dtype=torch.float32)  # (k_g, w_g)
        chunk = grads[:, start:end]  # (B, w_g)
        emb = chunk @ basis.T  # (B, k_g)
        residual[:, start:end] = chunk - emb @ basis
        emb_chunks.append(emb)
        recon_info.append((start, end, basis, bool(sub.protected)))

    embeddings = torch.cat(emb_chunks, dim=1) if emb_chunks else torch.zeros(lot, 0, device=device)

    emb_norms = torch.linalg.norm(embeddings, dim=1, keepdim=True)
    emb_scale = torch.clamp(embedding_clip_norm / (emb_norms + 1e-12), max=1.0)
    embeddings = embeddings * emb_scale

    if upload_residual:
        res_norms = torch.linalg.norm(residual, dim=1, keepdim=True)
        res_scale = torch.clamp(residual_clip_norm / (res_norms + 1e-12), max=1.0)
        residual = residual * res_scale

    # a finite row can still overflow in the projection: zero that record's embedding and residual
    overflow = ~_rows_finite(embeddings)
    if upload_residual:
        overflow = overflow | ~_rows_finite(residual)
    if bool(overflow.any()):
        embeddings[overflow] = 0.0
        if upload_residual:
            residual[overflow] = 0.0
        NONFINITE_ROWS_ZEROED[0] += int(overflow.sum())

    emb_sum = embeddings.sum(dim=0)
    emb_noise = torch.randn(emb_sum.shape, generator=gen, device=device) * (z * embedding_clip_norm)
    noisy_emb_mean = (emb_sum + emb_noise) / float(divisor)

    grad_vec = torch.zeros(dim, device=device)
    if upload_residual:
        res_sum = residual.sum(dim=0)
        res_z = z if residual_z is None else residual_z
        res_noise = torch.randn(res_sum.shape, generator=gen, device=device) * (res_z * residual_clip_norm)
        grad_vec = (res_sum + res_noise) / float(divisor)
        # project the noisy residual off each subspace (post-processing; protected tensors get 0)
        for start, end, basis, protected in recon_info:
            if protected:
                grad_vec[start:end] = 0.0
            else:
                seg = grad_vec[start:end]
                grad_vec[start:end] = seg - (seg @ basis.T) @ basis

    offset = 0
    for start, end, basis, _protected in recon_info:
        k_g = basis.shape[0]
        grad_vec[start:end] += noisy_emb_mean[offset : offset + k_g] @ basis
        offset += k_g

    return grad_vec


def _collect_public_examples(loader, num_samples: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect up to ``num_samples`` public (input, target) pairs on ``device`` as gradient anchors."""

    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    have = 0
    for inputs, targets in loader:
        xs.append(inputs)
        ys.append(targets)
        have += inputs.shape[0]
        if have >= num_samples:
            break
    x = torch.cat(xs, dim=0)[:num_samples].to(device)
    y = torch.cat(ys, dim=0)[:num_samples].to(device)
    return x, y
