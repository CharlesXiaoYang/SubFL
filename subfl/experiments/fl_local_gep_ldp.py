"""Federated training with record-level DP at every local step: SubFL (gep_step_keep), dense DP-SGD,
and the non-private reference. Each client clips and noises its own per-example gradients; the server
only averages the client updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from subfl.analysis.manifest import save_experiment_manifest
from subfl.analysis.provenance import update_provenance, write_provenance, write_status
from subfl.analysis.metrics import save_metrics_table
from subfl.config.experiment import ExperimentConfig
from subfl.data.factory import (
    build_federated_dataloaders,
    build_public_anchor_loader,
    build_public_validation_loader,
    data_record,
)
from subfl.data.svhn import PUBLIC_SPLITS as SVHN_PUBLIC_SPLITS
from subfl.experiments.gep_step import (
    NONFINITE_ROWS_ZEROED,
    SUBSPACE_SOURCES,
    _apply_flat_step,
    _build_anchor_subspace,
    _collect_public_examples,
    _disable_inplace_relu,
    _per_example_grads,
    _privatize_gep,
    _zero_nonfinite_rows_,
)
from subfl.federated.evaluation import evaluate_classifier
from subfl.models.factory import build_model
from subfl.privacy.rdp import (
    calibrate_noise_multiplier,
    epsilon_for_noise_multiplier,
    parse_rdp_orders,
    split_noise_multiplier,
)
from subfl.privacy.sampling import (
    LOT_DIVISORS,
    SAMPLERS,
    local_schedule,
    lot_divisor,
    make_poisson_loader,
    poisson_seed,
)
from subfl.utils.device import log_device_info, resolve_training_device
from subfl.utils.io import RunPaths, build_run_paths
from subfl.utils.logging import log_kv
from subfl.utils.seed import set_global_seed

ARMS = ("dense", "gep_step_keep", "nonprivate")
GEP_ACCOUNTINGS = ("joint",)
Z_CALIBRATIONS = ("per_client",)
DIAGNOSTICS = ("off",)
VAL_SOURCES = ("none", "public_pool")
NAN = float("nan")


@dataclass
class ExperimentArtifacts:
    paths: RunPaths
    metrics: list[dict]


# ----------------------------------------------------------------------------------------
# flat-vector + subspace helpers
# ----------------------------------------------------------------------------------------

def _classifier_head_names(raw: str) -> list[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    return names or ["fc_out", "classifier"]


def _flat(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _load_flat(model: nn.Module, vec: torch.Tensor) -> None:
    with torch.no_grad():
        ptr = 0
        for p in model.parameters():
            n = p.numel()
            p.copy_(vec[ptr : ptr + n].view_as(p))
            ptr += n


def _augment_batch(x: torch.Tensor) -> torch.Tensor:
    """Random crop (reflect padding 4) and horizontal flip of each image of a (B,C,H,W) batch, from the global RNG."""

    b, c, h, w = x.shape
    pad = 4
    padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")  # (B,C,H+2p,W+2p)
    dy = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    dx = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    rows = dy.view(b, 1, 1, 1) + torch.arange(h, device=x.device).view(1, 1, h, 1)  # (B,1,H,1)
    cols = dx.view(b, 1, 1, 1) + torch.arange(w, device=x.device).view(1, 1, 1, w)  # (B,1,1,W)
    bidx = torch.arange(b, device=x.device).view(b, 1, 1, 1)
    cidx = torch.arange(c, device=x.device).view(1, c, 1, 1)
    cropped = padded[bidx, cidx, rows, cols]  # (B,C,H,W) per-example crop at offset (dy,dx)
    flip = (torch.rand(b, device=x.device) < 0.5).view(b, 1, 1, 1)
    return torch.where(flip, cropped.flip(-1), cropped)


def _augmented_per_example_grads(model, x, y, param_names, k: int) -> torch.Tensor:
    """(B, D) per-example gradients averaged over ``k`` augmented views before clipping; plain ones for k <= 1."""

    if k <= 1:
        return _per_example_grads(model, x, y, param_names)
    acc = _per_example_grads(model, _augment_batch(x), y, param_names)
    for _ in range(k - 1):
        acc = acc + _per_example_grads(model, _augment_batch(x), y, param_names)
    return acc / float(k)


def _subspace_overlap(cur, prev) -> float:
    """Rank-weighted fraction of the previous learned subspace kept in the current one (1 = same, 0 = orthogonal)."""

    if prev is None:
        return 1.0
    prev_by = {s.group_name: s for s in prev}
    tot_overlap = 0.0
    tot_rank = 0.0
    for s in cur:
        if s.protected or s.basis.numel() == 0:
            continue
        p = prev_by.get(s.group_name)
        if p is None or p.protected or p.basis.numel() == 0:
            continue
        m = s.basis.float() @ p.basis.float().t()  # (k_cur, k_prev); ||.||_F^2 = sum cos^2(angles)
        tot_overlap += float(torch.sum(m * m).item())
        tot_rank += float(p.basis.shape[0])
    return tot_overlap / tot_rank if tot_rank > 0.0 else 1.0


# ----------------------------------------------------------------------------------------
# local trainers (one per arm)
# ----------------------------------------------------------------------------------------

def _scheduled_lr(config, rnd: int) -> float:
    """Learning rate of round ``rnd``: constant, or cosine decay to lr * lr_min_ratio."""

    base = float(config.train.lr)
    mode = str(config.train.lr_schedule)
    if mode == "constant":
        return base
    if mode != "cosine":
        raise ValueError(f"unknown lr_schedule: {mode}")
    total = max(1, int(config.train.num_rounds))
    frac = rnd / max(1, total - 1)  # 0 at the first round -> 1 at the last
    floor = base * float(config.train.lr_min_ratio)
    return floor + (base - floor) * 0.5 * (1.0 + math.cos(math.pi * frac))


def _new_local(global_model, config, device, inplace_relu_off=False):
    if not bool(torch.isfinite(_flat(global_model)).all()):
        raise FloatingPointError("the global model is not finite (training has diverged)")
    local = build_model(config.model).to(device)
    local.load_state_dict(global_model.state_dict())
    if inplace_relu_off:
        _disable_inplace_relu(local)
    return local


def _local_train_free(global_model, loader, config, device, lr=None) -> torch.Tensor:
    """Plain local SGD (the nonprivate arm)."""

    local = _new_local(global_model, config, device)
    local.train()
    opt = torch.optim.SGD(
        local.parameters(), lr=(config.train.lr if lr is None else float(lr)),
        momentum=config.train.momentum,
        weight_decay=config.train.weight_decay,
    )
    crit = nn.CrossEntropyLoss()
    for _ in range(config.train.local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(local(x).float(), y).backward()
            opt.step()
    return (_flat(global_model) - _flat(local)).to(device)


def _local_train_step_keep(global_model, loader, config, device, subspaces, z, gen, param_names,
                           public_x=None, public_y=None, head_names=None, basis_dim=0,
                           residual_z=None, lr=None, basis_gen=None) -> torch.Tensor:
    """SubFL local training: GEP privatization at every step; with ``public_x`` the subspace is refit locally."""

    local = _new_local(global_model, config, device, inplace_relu_off=True)
    local.eval()  # GroupNorm-only -> eval == train for stats; needed for torch.func grads
    params = list(local.parameters())
    dim = sum(p.numel() for p in params)
    velocity = torch.zeros(dim, device=device)
    upload_residual = bool(config.gep.upload_residual)
    local_refresh = max(1, int(config.gep.local_recompute_interval))
    eff_lr = float(config.train.lr if lr is None else lr)
    divisor_mode = str(config.privacy.lot_divisor)
    # this client's divisor min(B, n_c) and step count, as the accountant computes them
    sched = local_schedule(len(loader.dataset), config.data.batch_size, config.train.local_epochs)
    expected_lot = sched.expected_lot
    step_idx = 0
    for _ in range(config.train.local_epochs):
        for x, y in loader:
            if x is not None:  # None: an empty Poisson lot (noise only, still one DP step)
                x, y = x.to(device), y.to(device)
            if public_x is not None and step_idx % local_refresh == 0:
                # every local_recompute_interval steps: refit B on public gradients at the current local weights
                anchor = _per_example_grads(local, public_x, public_y, param_names)
                subspaces = _build_anchor_subspace(local, anchor, config, head_names, basis_dim, basis_gen=basis_gen)
                del anchor
            if x is None:
                grads = torch.zeros(0, dim, device=device)
            else:
                grads = _augmented_per_example_grads(local, x, y, param_names, int(config.train.augmult_k))  # (B, D)
            priv_grad = _privatize_gep(
                grads, subspaces,
                embedding_clip_norm=config.gep.embedding_clip_norm,
                residual_clip_norm=config.gep.residual_clip_norm,
                upload_residual=upload_residual, z=z, gen=gen, residual_z=residual_z,
                divisor=lot_divisor(divisor_mode, expected=expected_lot),
            )
            if config.train.weight_decay > 0.0:
                priv_grad = priv_grad + config.train.weight_decay * _flat(local).to(device)
            velocity = config.train.momentum * velocity + priv_grad
            _apply_flat_step(local, velocity, eff_lr)
            step_idx += 1
    _check_step_count(step_idx, sched.steps)
    return (_flat(global_model) - _flat(local)).to(device)


def _privatize_dense_dpsgd(grads, clip, z, gen, *, divisor):
    """Dense DP-SGD: clip each row, sum, add N(0, (z*clip)^2), divide by ``divisor``."""

    _zero_nonfinite_rows_(grads)
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    norms = torch.linalg.norm(grads, dim=1, keepdim=True)
    scale = torch.clamp(float(clip) / (norms + 1e-12), max=1.0)
    gsum = (grads * scale).sum(dim=0)
    noise = torch.randn(gsum.shape, generator=gen, device=grads.device) * (float(z) * float(clip))
    return (gsum + noise) / float(divisor)


def _local_train_dpsgd(global_model, loader, config, device, z, gen, param_names, lr=None):
    """Dense DP-SGD local training: the same per-step mechanism as SubFL without the projection."""

    local = _new_local(global_model, config, device, inplace_relu_off=True)
    local.eval()
    params = list(local.parameters())
    dim = sum(p.numel() for p in params)
    velocity = torch.zeros(dim, device=device)
    clip = float(config.privacy.clip_norm)
    eff_lr = float(config.train.lr if lr is None else lr)
    divisor_mode = str(config.privacy.lot_divisor)
    # this client's divisor min(B, n_c) and step count, as the accountant computes them
    sched = local_schedule(len(loader.dataset), config.data.batch_size, config.train.local_epochs)
    expected_lot = sched.expected_lot
    steps = 0
    for _ in range(config.train.local_epochs):
        for x, y in loader:
            if x is None:  # an empty Poisson lot: noise only, still one DP step
                grads = torch.zeros(0, dim, device=device)
            else:
                x, y = x.to(device), y.to(device)
                grads = _augmented_per_example_grads(local, x, y, param_names, int(config.train.augmult_k))  # (B, D)
            priv_grad = _privatize_dense_dpsgd(
                grads, clip, z, gen, divisor=lot_divisor(divisor_mode, expected=expected_lot)
            )
            if config.train.weight_decay > 0.0:
                priv_grad = priv_grad + config.train.weight_decay * _flat(local).to(device)
            velocity = config.train.momentum * velocity + priv_grad
            _apply_flat_step(local, velocity, eff_lr)
            steps += 1
    _check_step_count(steps, sched.steps)
    return (_flat(global_model) - _flat(local)).to(device)


def _noise_convention(arm, config) -> str:
    """The metrics' ``noise_convention`` label: which noise a released update carries."""

    if arm == "nonprivate":
        return "none"
    divisor = "min(B,n_c)"
    if arm == "dense":
        return f"per-step DP-SGD: N(0,(z*C)^2) on each lot sum, / {divisor}"
    residual = ("residual N(0,(z_r*C_r)^2), projected off B" if bool(config.gep.upload_residual)
                else "residual dropped")
    return f"per-step GEP: embedding N(0,(z_e*C_e)^2), {residual}; on each lot sum, / {divisor}"


def _check_step_count(ran: int, expected: int) -> None:
    """Under Poisson sampling the number of local DP steps must be exactly the accountant's."""

    if ran != expected:
        raise RuntimeError(f"ran {ran} local DP steps, but the accountant assumes {expected}")


def _effective_sampler(arm, config) -> str:
    """Sampler of the arm's local lots: Poisson for the DP arms, shuffled batches for the nonprivate arm."""

    sampler = str(config.privacy.sampler)
    if sampler not in SAMPLERS:
        raise ValueError(f"privacy.sampler must be one of {SAMPLERS}, got {sampler!r}")
    if arm == "nonprivate":
        return "shuffle"
    return sampler


def _check_data(config) -> None:
    """Refuse inconsistent SVHN options (public split, public_pool normalization, augmentation multiplicity)."""

    data = config.data
    svhn = str(data.dataset_name) == "svhn"
    if str(data.svhn_public_split) not in SVHN_PUBLIC_SPLITS:
        raise ValueError(f"data.svhn_public_split must be one of {SVHN_PUBLIC_SPLITS}, got {data.svhn_public_split!r}")
    if str(data.svhn_public_split) == "extra" and not svhn:
        raise ValueError(f"data.svhn_public_split=extra (P from SVHN 'extra', all of 'train' private) needs "
                         f"dataset_name svhn, got {data.dataset_name}")
    if str(data.normalization) == "public_pool" and not svhn:
        raise ValueError(f"data.normalization=public_pool (the statistics of SVHN's public pool P) needs dataset_name "
                         f"svhn, got {data.dataset_name}")
    if svhn and int(config.train.augmult_k) > 1:
        raise ValueError(f"train.augmult_k={config.train.augmult_k} on svhn: _augment_batch flips the augmented views "
                         f"horizontally, which digits must not be; use augmult_k 1")


# ----------------------------------------------------------------------------------------
# privacy calibration (per arm)
# ----------------------------------------------------------------------------------------

def _calibrate_z(arm, config, client_size):
    """(z_emb, eps, order, total mechanisms, q, mechanisms per round, z_res, z_accounting) of one client."""

    if arm == "nonprivate":
        return 0.0, 0.0, 0, 0, 0.0, 0, 0.0, 0.0
    orders = parse_rdp_orders(config.privacy.rdp_orders)
    accounting = str(config.privacy.gep_accounting)
    if accounting not in GEP_ACCOUNTINGS:
        raise ValueError(f"privacy.gep_accounting must be one of {GEP_ACCOUNTINGS}, got {accounting!r}")
    joint_split = arm == "gep_step_keep" and bool(config.gep.upload_residual)
    # per-local-step DP-SGD: compose over every step, with local-batch subsampling amplification
    sched = local_schedule(client_size, int(config.data.batch_size), int(config.train.local_epochs))
    q = sched.sample_rate
    mech_per_round = sched.steps
    total = mech_per_round * int(config.train.num_rounds)
    ratio = float(config.gep.embedding_epsilon_ratio)
    if joint_split and not (0.0 < ratio < 1.0):
        raise ValueError("gep.embedding_epsilon_ratio must satisfy 0 < ratio < 1")
    z, eps, order = calibrate_noise_multiplier(
        target_epsilon=config.privacy.total_epsilon, target_delta=config.privacy.delta,
        sample_rate=q, steps=int(total), orders=orders,
    )
    if joint_split:  # one mechanism per step with multiplier z; the embedding and residual parts split it
        z_emb, z_res = split_noise_multiplier(z, ratio)
        return z_emb, eps, order, int(total), float(q), int(mech_per_round), z_res, float(z)
    return z, eps, order, int(total), float(q), int(mech_per_round), float(z), float(z)


@dataclass(frozen=True)
class ClientPrivacy:
    """DP parameters of one client (record-level DP, zero-out adjacency, n_c public): a row of client_privacy.csv."""

    client: int
    n: int  # n_c
    q: float  # sample rate the accountant uses: min(1, B / n_c) for the per-step arms
    steps_per_round: int  # local DP steps per round: ceil(n_c / B) * local epochs
    expected_lot: int  # min(B, n_c): the constant lot divisor (lot_divisor="expected")
    mech_per_round: int  # mechanisms the accountant composes per round
    total_mech: int  # T_c = mech_per_round * rounds
    z: float  # multiplier used in training (SubFL: the embedding part)
    z_res: float  # SubFL: the residual part (= z otherwise)
    z_acct: float  # multiplier the accountant composes
    eps: float  # epsilon after all rounds
    order: int  # RDP order attaining eps
    z_calibration: str  # per_client | none (nonprivate)
    sampler: str
    lot_divisor: str
    gep_accounting: str


def _z_calibration(arm, config) -> str:
    """Calibration label: per_client (each client from its own n_c), none for the nonprivate arm."""

    mode = str(config.privacy.dense_z_calibration)
    if mode not in Z_CALIBRATIONS:
        raise ValueError(f"privacy.dense_z_calibration must be one of {Z_CALIBRATIONS}, got {mode!r}")
    return "none" if arm == "nonprivate" else "per_client"


def _calibrate_clients(arm, config, client_sizes, sampler: str) -> list[ClientPrivacy]:
    """Calibrate every client from its own n_c (once per distinct n_c); list index = client id."""

    mode = _z_calibration(arm, config)
    cache: dict[int, tuple] = {}

    def cal(n: int):
        if n not in cache:
            cache[n] = _calibrate_z(arm, config, n)
        return cache[n]

    out = []
    for ci, n in enumerate(client_sizes):
        n = int(n)
        z, eps, order, total, q, mpr, z_res, z_acct = cal(n)
        sched = local_schedule(n, config.data.batch_size, config.train.local_epochs)
        out.append(ClientPrivacy(
            client=ci, n=n, q=float(q), steps_per_round=int(sched.steps), expected_lot=int(sched.expected_lot),
            mech_per_round=int(mpr), total_mech=int(total), z=float(z), z_res=float(z_res),
            z_acct=float(z_acct), eps=float(eps), order=int(order), z_calibration=mode,
            sampler=str(sampler), lot_divisor=str(config.privacy.lot_divisor),
            gep_accounting=str(config.privacy.gep_accounting),
        ))
    return out


def _spent_eps_client(config, cp: ClientPrivacy, rounds_done: int, orders) -> float:
    """Epsilon spent by one client after ``rounds_done`` rounds (``cp.eps`` after the last round)."""

    if cp.mech_per_round <= 0:
        return 0.0
    return float(epsilon_for_noise_multiplier(
        noise_multiplier=cp.z_acct, sample_rate=cp.q, steps=int(cp.mech_per_round * rounds_done),
        orders=orders, delta=config.privacy.delta,
    )[0])


def _record_level_epsilon(arm, client_priv):
    """The run's record-level epsilon, max over clients; None for the nonprivate arm (no guarantee at all)."""

    if arm == "nonprivate":
        return None
    return max(cp.eps for cp in client_priv)


def _privacy_contract(arm, config, client_priv, nonfinite_rows: int, *, final: bool,
                      warm: dict | None = None) -> dict:
    """Whether (max_c eps_c, delta) is a valid record-level guarantee of the run, with every reason it is not."""

    reasons = []
    if arm == "nonprivate":
        reasons.append("nonprivate: no noise, no DP guarantee (the ceiling reference)")
    else:
        if (warm or {}).get("init_pretraining") == "unverified":
            reasons.append("the warm-start checkpoint records neither its pretraining data nor its split, so that it "
                           "saw only public data is unverified")
        target = float(config.privacy.total_epsilon)
        worst = max(cp.eps for cp in client_priv)
        if worst > target:
            reasons.append(f"record-level epsilon max_c eps_c = {worst:.6f} exceeds the target {target:g} "
                           f"(z calibration {_z_calibration(arm, config)})")
    if nonfinite_rows > 0:
        reasons.append(f"the guard zeroed {nonfinite_rows} non-finite per-example gradient rows")
    return {"dp_valid": not reasons, "reasons": reasons, "final": bool(final)}


def _save_model(path: Path, global_model, *, rounds_done: int, arm: str, config) -> None:
    """Save the global model atomically; its "arm" key makes ``_warm_start`` refuse it."""

    tmp = path.with_name(path.name + ".tmp")
    torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in global_model.state_dict().items()},
                "round": int(rounds_done), "arm": arm, "experiment_name": config.experiment_name,
                "seed": int(config.train.seed)}, tmp)
    os.replace(tmp, path)


def _warm_start(global_model, config, device) -> dict:
    """Load the public-pretraining checkpoint ``model.init_checkpoint``; returns what was loaded."""

    path = Path(str(config.model.init_checkpoint))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    want = str(config.model.init_checkpoint_sha256).strip().lower()
    if want and not sha.startswith(want):
        raise ValueError(f"init checkpoint {path}: sha256 {sha} does not start with the expected {want}")
    # weights_only=False: the payload holds config dicts (torch>=2.6 defaults to True)
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    if "arm" in ckpt:  # saved by _save_model: the model of an FL run, trained on private data
        raise ValueError(f"init checkpoint {path} is the global model of an FL run (arm={ckpt.get('arm')}, round "
                         f"{ckpt.get('round')}): warm-starting from it composes that run's privacy cost with this "
                         f"run's, which is not accounted. Warm-start only from public pretraining "
                         f"(pretrain.py).")
    pretraining = _pretraining_source(ckpt, config, path)
    src = ckpt.get("model_state_dict", ckpt)
    model_sd = global_model.state_dict()
    # tensors of matching shape: a 100-class head is skipped, every tensor outside the head must load
    transfer = {k: v for k, v in src.items() if k in model_sd and model_sd[k].shape == v.shape}
    skipped = [k for k in src if k not in transfer]
    head = tuple(name + "." for name in _classifier_head_names(config.gep.classifier_head_names))
    missing = [k for k in model_sd if k not in transfer]
    missing_backbone = [k for k in missing if not k.startswith(head)]
    if missing_backbone:
        raise ValueError(
            f"warm start from {path}: {len(missing_backbone)} of {len(model_sd)} model tensors outside the classifier "
            f"head are missing from the checkpoint or have another shape (e.g. {missing_backbone[:3]}); wrong file?")
    global_model.load_state_dict(transfer, strict=False)
    log_kv("Init checkpoint", str(path))
    log_kv("Init checkpoint sha256", sha)
    log_kv("Init checkpoint epoch/test_acc", f"{ckpt.get('epoch', '?')}/{ckpt.get('test_accuracy', '?')}")
    log_kv("Warm-start transfer", f"{len(transfer)}/{len(src)} tensors (skipped: {skipped if skipped else 'none'}; "
                                  f"model tensors not loaded: {missing if missing else 'none'})")
    log_kv("Warm-start pretraining", pretraining)
    return {"init_checkpoint_sha256": sha, "init_transferred": len(transfer), "init_missing": missing,
            "init_skipped": skipped, "init_pretraining": pretraining}


def _pretraining_source(ckpt, config, path) -> str:
    """What the checkpoint was pretrained on ("external ..." or "unverified"). Refuses one pretrained on this run's
    dataset, whose pretraining data may include private records of this run."""

    model_cfg, data_cfg = ckpt.get("model_config"), ckpt.get("data_config")
    if not isinstance(model_cfg, dict) or not isinstance(data_cfg, dict):
        return "unverified"
    classes = model_cfg.get("num_classes")
    if not isinstance(classes, int) or isinstance(classes, bool):
        return "unverified"
    if classes != int(config.model.num_classes):
        return f"external {classes}-class dataset"
    dataset = data_cfg.get("dataset_name")
    if dataset != str(config.data.dataset_name):
        return f"external dataset {dataset}"
    raise ValueError(f"init checkpoint {path} was pretrained on {dataset} ({classes} classes), the dataset of this run, "
                     f"so its pretraining data may include private records of this run. Warm-start only from "
                     f"pretraining on another dataset (pretrain.py: CIFAR-100).")


def _cost_record(device, round_seconds: list[float], evaluation_seconds: list[float], wall_seconds: float) -> dict:
    """Peak GPU memory (None on the CPU) and the seconds of each round's training, each evaluation and the run."""

    cuda = torch.device(device).type == "cuda"
    return {"device": str(device),
            "gpu_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if cuda else None,
            "gpu_peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if cuda else None,
            "round_seconds": [float(s) for s in round_seconds],
            "evaluation_seconds": [float(s) for s in evaluation_seconds],
            "wall_seconds": float(wall_seconds)}


def _cost_text(cost: dict) -> str:
    """The log's one-line summary of ``_cost_record``."""

    gib = 2 ** 30
    allocated, reserved = cost["gpu_peak_memory_allocated_bytes"], cost["gpu_peak_memory_reserved_bytes"]
    memory = ("no GPU" if allocated is None else
              f"peak GPU memory {allocated / gib:.2f} GiB allocated, {reserved / gib:.2f} GiB reserved")

    def seconds(what, values):
        if not values:
            return f"no {what}"
        return f"{what} {values[-1]:.1f} s the last ({sum(values) / len(values):.1f} s mean of {len(values)})"

    return (f"{memory}; {seconds('training per round', cost['round_seconds'])}; "
            f"{seconds('evaluation', cost['evaluation_seconds'])}; the run {cost['wall_seconds'] / 3600:.3f} h")


# ----------------------------------------------------------------------------------------
# top-level runner
# ----------------------------------------------------------------------------------------

def run_fl_local_gep_ldp(config: ExperimentConfig, arm: str) -> ExperimentArtifacts:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm}")
    if arm == "dense" and str(config.privacy.dense_noise_mode) != "dpsgd":
        raise ValueError(f"privacy.dense_noise_mode must be dpsgd, got {config.privacy.dense_noise_mode!r}")
    if config.privacy.accountant != "rdp" and arm != "nonprivate":
        raise ValueError("private arms require privacy.accountant=rdp")
    if str(config.privacy.lot_divisor) not in LOT_DIVISORS:
        raise ValueError(f"privacy.lot_divisor must be one of {LOT_DIVISORS}, got {config.privacy.lot_divisor!r}")
    sampler_mode = _effective_sampler(arm, config)
    if str(config.privacy.gep_accounting) not in GEP_ACCOUNTINGS:
        raise ValueError(f"privacy.gep_accounting must be one of {GEP_ACCOUNTINGS}, got {config.privacy.gep_accounting!r}")
    _z_calibration(arm, config)  # validates privacy.dense_z_calibration
    if str(config.data.val_source) not in VAL_SOURCES:
        raise ValueError(f"data.val_source must be one of {VAL_SOURCES}, got {config.data.val_source!r}")
    if int(config.output.save_model_every) < 0:
        raise ValueError(f"output.save_model_every must be >= 0 (0 = never), got {config.output.save_model_every}")
    if str(config.gep.subspace_source) not in SUBSPACE_SOURCES:
        raise ValueError(f"gep.subspace_source must be one of {SUBSPACE_SOURCES}, got {config.gep.subspace_source!r}")
    if str(config.train.diagnostics) not in DIAGNOSTICS:
        raise ValueError(f"train.diagnostics must be one of {DIAGNOSTICS}, got {config.train.diagnostics!r}")
    if float(config.privacy.client_sample_rate) != 1.0:
        raise ValueError(f"privacy.client_sample_rate must be 1.0 (every client in every round), got "
                         f"{config.privacy.client_sample_rate}")
    _check_data(config)
    NONFINITE_ROWS_ZEROED[0] = 0  # per run: per-example rows zeroed by the non-finite guard

    set_global_seed(config.train.seed)
    device = resolve_training_device(config.train.device)
    torch.backends.cudnn.benchmark = True
    paths = build_run_paths(config)
    # before any work: the run's status, code and config
    write_status(paths.run_dir, "started")
    try:
        write_provenance(paths.run_dir, arm=arm, experiment_name=config.experiment_name)
        if config.output.save_manifest_json:
            save_experiment_manifest(config, paths.manifest_json)
        artifacts = _train(config, arm, device, paths, sampler_mode)
    except BaseException as exc:  # also KeyboardInterrupt and SIGTERM: a stopped run must not look complete
        write_status(paths.run_dir, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    write_status(paths.run_dir, "completed")
    return artifacts


def _train(config: ExperimentConfig, arm: str, device, paths: RunPaths, sampler_mode: str) -> ExperimentArtifacts:
    """One run after validation and seeding: data, calibration, the rounds and the outputs."""

    started = time.perf_counter()  # for _cost_record
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    client_loaders, test_loader = build_federated_dataloaders(config.data)
    num_clients = len(client_loaders)
    client_sizes = [len(ld.dataset) for ld in client_loaders]  # n_c (public under zero-out adjacency)
    private_size = sum(client_sizes)
    avg_client_size = max(1, private_size // num_clients)
    train_loaders = client_loaders  # nonprivate: the shuffled loaders as built
    if sampler_mode == "poisson":  # same datasets, lots drawn by each client's own Poisson schedule
        train_loaders = []
        for ci, ld in enumerate(client_loaders):
            if len(ld.dataset) < 1:
                raise ValueError(f"client {ci} has no data; Poisson DP-SGD needs every client to hold a record")
            sched_c = local_schedule(len(ld.dataset), config.data.batch_size, config.train.local_epochs)
            train_loaders.append(make_poisson_loader(ld.dataset, sched_c, seed=poisson_seed(config.train.seed, ci),
                                                     num_workers=int(config.data.num_workers)))

    global_model = build_model(config.model).to(device)
    _disable_inplace_relu(global_model)
    warm = {}
    if str(config.model.init_checkpoint):  # warm start from public pretraining ("" = random init)
        warm = _warm_start(global_model, config, device)
    val_loader = None
    if str(config.data.val_source) == "public_pool":
        if warm.get("init_pretraining") == "unverified":
            raise ValueError("val_source=public_pool: the warm-start checkpoint may have been pretrained on the public "
                             "pool (unverified), so it cannot validate this run")
        val_loader = build_public_validation_loader(config.data, batch_size=config.data.batch_size)
    param_names = [name for name, _ in global_model.named_parameters()]
    head_names = _classifier_head_names(config.gep.classifier_head_names)
    basis_dim = int(config.gep.subspace_dim)
    uses_subspace = arm == "gep_step_keep"

    public_loader = public_x = public_y = None
    if uses_subspace:
        public_loader = build_public_anchor_loader(config.data, batch_size=config.data.batch_size, shuffle=False)
        public_x, public_y = _collect_public_examples(public_loader, int(config.gep.anchor_samples), device)
        if config.data.auxiliary_random_labels:
            lg = torch.Generator().manual_seed(config.train.seed + 7)
            public_y = torch.randint(0, int(config.model.num_classes), (public_y.shape[0],), generator=lg).to(device)
    # the data as used: normalization statistics and the sizes of the private data, of P and of the anchors
    pool = val_loader if val_loader is not None else public_loader
    data = {**data_record(config.data), "private_size": private_size,
            "public_pool_size": None if pool is None else len(pool.dataset),
            "anchor_count": None if public_x is None else int(public_x.shape[0])}

    # the run's guarantee is max_c eps_c; the scalar z / mechanisms / epsilon columns describe the client attaining it
    client_priv = _calibrate_clients(arm, config, client_sizes, sampler_mode)
    save_metrics_table([asdict(cp) for cp in client_priv], paths.run_dir / "client_privacy.csv")
    bind = max(client_priv, key=lambda cp: (cp.eps, -cp.client))
    z, achieved_eps, res_z, z_acct = bind.z, bind.eps, bind.z_res, bind.z_acct
    total_mech, mech_per_round = bind.total_mech, bind.mech_per_round
    eps_client_min = min(cp.eps for cp in client_priv)
    z_client_min, z_client_max = min(cp.z for cp in client_priv), max(cp.z for cp in client_priv)
    z_calibration = _z_calibration(arm, config)
    record_eps = _record_level_epsilon(arm, client_priv)  # None for nonprivate
    contract = _privacy_contract(arm, config, client_priv, 0, final=False, warm=warm)  # before training
    update_provenance(paths.run_dir, privacy_contract=contract, data=data)  # a diverged run keeps both
    acct_orders = parse_rdp_orders(config.privacy.rdp_orders)
    clip = float(config.privacy.clip_norm)
    refresh = max(1, int(config.gep.recompute_interval))
    noise_gen = torch.Generator(device=device).manual_seed(config.train.seed + 4242)
    round_basis_gen = step_basis_gen = None  # random-subspace control only (anchor_svd draws nothing extra)
    if uses_subspace and str(config.gep.subspace_source) == "random":
        round_basis_gen = torch.Generator(device=device).manual_seed(int(config.train.seed) + 5152)
        step_basis_gen = torch.Generator(device=device).manual_seed(int(config.train.seed) + 5151)

    method_label = (
        f"fl_local_gep-arm={arm}-model={config.model.model_name}-k={basis_dim}"
        f"-clients={num_clients}-LE={config.train.local_epochs}-alpha={config.data.dirichlet_alpha:g}"
        f"-eps={config.privacy.total_epsilon:g}-res={config.gep.upload_residual}"
        + ("-src=random" if uses_subspace and str(config.gep.subspace_source) == "random" else "")
    )

    log_device_info(device)
    log_kv("Arm", arm)
    log_kv("Model", config.model.model_name)
    log_kv("Clients", num_clients)
    log_kv("Avg client size", avg_client_size)
    log_kv("Local epochs", config.train.local_epochs)
    log_kv("Partition", str(config.data.partition))
    log_kv("Dirichlet alpha", config.data.dirichlet_alpha)
    log_kv("Rounds", config.train.num_rounds)
    log_kv("Subspace dim k", basis_dim if uses_subspace else 0)
    if uses_subspace:
        log_kv("Subspace source", str(config.gep.subspace_source))
    log_kv("Target epsilon", config.privacy.total_epsilon)
    log_kv("Client sizes", client_sizes)
    log_kv("Data", f"{data['dataset_name']}: P {data['public_pool_size']}, anchors {data['anchor_count']}, "
                   f"normalization {data['normalization']} mean {data['normalization_mean']} std "
                   f"{data['normalization_std']}"
                   + (f", P from {data['svhn_public_split']}" if data["dataset_name"] == "svhn" else ""))
    log_kv("z calibration", z_calibration)
    for cp in client_priv:
        log_kv(f"  client {cp.client}",
               f"n={cp.n} q={cp.q:.6f} T={cp.total_mech} z={cp.z:.6f}"
               + (f" z_res={cp.z_res:.6f}" if cp.z_res != cp.z else "")
               + (f" z_acct={cp.z_acct:.6f}" if cp.z_acct != cp.z else "")
               + f" eps={cp.eps:.6f}")
    log_kv("Record-level epsilon", "none (no DP guarantee)" if record_eps is None else
           f"max_c eps_c = {achieved_eps:.6f} (client {bind.client}); min_c eps_c = {eps_client_min:.6f}")
    log_kv("Noise multiplier z", f"{z:.6f}")
    if arm == "gep_step_keep":
        log_kv("Residual noise multiplier", f"{res_z:.6f}")
        log_kv("GEP accounting", str(config.privacy.gep_accounting))
    if total_mech > 0:
        log_kv("Accounting z", f"{z_acct:.6f} (x {total_mech} subsampled Gaussian mechanisms)")
    log_kv("Total mechanisms", total_mech)
    log_kv("Achieved epsilon", f"{achieved_eps:.6f}")
    log_kv("Lot divisor", str(config.privacy.lot_divisor))
    log_kv("Noise convention", _noise_convention(arm, config))
    log_kv("Sampler", sampler_mode)
    log_kv("Diagnostics", str(config.train.diagnostics))
    log_kv("DP contract", "valid (paper settings)" if contract["dp_valid"] else
           "NOT valid for the paper: " + "; ".join(contract["reasons"]))

    subspaces = None
    last_overlap = 1.0
    rows: list[dict] = []
    eval_interval = max(1, int(config.train.num_rounds) // 50) if config.train.num_rounds > 50 else 1
    # EMA(s) of the global weights, for evaluation only (post-processing, no privacy cost)
    ema_decays = [d for d in (float(config.train.ema_decay), float(config.train.ema_decay2)) if d > 0.0]
    ema_flats = [None] * len(ema_decays)
    round_seconds: list[float] = []  # for _cost_record
    evaluation_seconds: list[float] = []

    for rnd in range(int(config.train.num_rounds)):
        round_start = time.perf_counter()
        cur_lr = _scheduled_lr(config, rnd)
        if uses_subspace and (subspaces is None or rnd % refresh == 0):
            prev_subspaces = subspaces
            anchor = _per_example_grads(global_model, public_x, public_y, param_names)
            subspaces = _build_anchor_subspace(global_model, anchor, config, head_names, basis_dim,
                                               basis_gen=round_basis_gen)
            last_overlap = _subspace_overlap(subspaces, prev_subspaces)  # B_t vs B_{t-1} stability (public)
            del anchor

        deltas: list[torch.Tensor] = []
        raw_norms: list[float] = []          # ||raw update|| (nonprivate)
        uploaded_norms: list[float] = []     # ||final uploaded vector|| (after clip+noise)
        for ci in range(num_clients):  # every client takes part in every round
            loader = train_loaders[ci]
            cp = client_priv[ci]  # this client's own multipliers
            if arm == "nonprivate":
                delta = _local_train_free(global_model, loader, config, device, lr=cur_lr)
                raw_norms.append(float(torch.linalg.norm(delta).item()))
            elif arm == "dense":
                delta = _local_train_dpsgd(global_model, loader, config, device, cp.z, noise_gen, param_names,
                                           lr=cur_lr)
            else:  # gep_step_keep
                delta = _local_train_step_keep(
                    global_model, loader, config, device, subspaces, cp.z, noise_gen, param_names,
                    public_x=public_x, public_y=public_y, head_names=head_names, basis_dim=basis_dim,
                    residual_z=cp.z_res, lr=cur_lr, basis_gen=step_basis_gen,
                )
            uploaded_norms.append(float(torch.linalg.norm(delta).item()))
            deltas.append(delta)

        avg_raw_norm = float(np.mean(raw_norms)) if raw_norms else 0.0
        avg_uploaded_norm = float(np.mean(uploaded_norms)) if uploaded_norms else 0.0
        learned_rank = int(sum(s.actual_rank for s in subspaces if not s.protected)) if uses_subspace else 0
        avg_delta = torch.stack(deltas, dim=0).mean(dim=0)
        if not bool(torch.isfinite(avg_delta).all()):
            raise FloatingPointError(f"round {rnd + 1}: the aggregated update is not finite (training has diverged)")
        _load_flat(global_model, _flat(global_model).to(device) - avg_delta)
        round_seconds.append(time.perf_counter() - round_start)

        if ema_decays:  # EMA(s) of the post-update global weights (eval-only post-processing)
            cur = _flat(global_model).to(device)
            for i, d in enumerate(ema_decays):
                ema_flats[i] = cur.clone() if ema_flats[i] is None else d * ema_flats[i] + (1.0 - d) * cur

        save_every = int(config.output.save_model_every)
        if save_every > 0 and ((rnd + 1) % save_every == 0 or rnd == int(config.train.num_rounds) - 1):
            _save_model(paths.run_dir / f"ckpt_r{rnd + 1:03d}.pt", global_model, rounds_done=rnd + 1, arm=arm,
                        config=config)

        if rnd % eval_interval == 0 or rnd == int(config.train.num_rounds) - 1:
            evaluation_start = time.perf_counter()
            # epsilon spent so far by each client (clients with the same parameters share one curve); the
            # run's record-level guarantee is the maximum over clients
            spent_cache: dict[tuple, float] = {}
            for cp in client_priv:
                key = (cp.q, cp.mech_per_round, cp.z, cp.z_res, cp.z_acct)
                if key not in spent_cache:
                    spent_cache[key] = _spent_eps_client(config, cp, rnd + 1, acct_orders)
            spent_by_client = [spent_cache[(cp.q, cp.mech_per_round, cp.z, cp.z_res, cp.z_acct)] for cp in client_priv]
            spent_eps, spent_eps_min = max(spent_by_client), min(spent_by_client)
            test_loss, test_acc = evaluate_classifier(global_model, test_loader, device)
            val_loss = val_acc = NAN  # the global model on the public pool (tuning selects on this only)
            if val_loader is not None:
                val_loss, val_acc = evaluate_classifier(global_model, val_loader, device)
            ema_accs = [float("nan"), float("nan")]
            if any(f is not None for f in ema_flats):  # eval EMA model(s), save/restore raw weights around it
                _saved = _flat(global_model).to(device)
                for i, f in enumerate(ema_flats):
                    if f is not None:
                        _load_flat(global_model, f)
                        _, ema_accs[i] = evaluate_classifier(global_model, test_loader, device)
                _load_flat(global_model, _saved)
            noise_conv = _noise_convention(arm, config)
            # n/a: not computed for this arm (the per-step arms never form a clean update); the metrics keep 0.0
            raw_txt = f"{avg_raw_norm:.3f}" if raw_norms else "n/a"
            # a tuning run hides the test accuracy in the log (it is still computed and saved)
            acc_txt = "hidden(tuning)" if val_loader is not None else f"{test_acc:.4f}"
            log_kv(
                f"[{arm}] r{rnd + 1}/{config.train.num_rounds}",
                f"acc={acc_txt} spent_eps={spent_eps:.3f}/{config.privacy.total_epsilon:g}  "
                f"raw||Δ||={raw_txt} up||Δ||={avg_uploaded_norm:.3f}  Boverlap={last_overlap:.3f}"
                + (f"  val={val_acc:.4f}" if val_loader is not None else "")
                + (f"  ema={ema_accs[0]:.4f}" if ema_flats and ema_flats[0] is not None and val_loader is None else "")
                + (f"  ema2={ema_accs[1]:.4f}" if len(ema_flats) > 1 and ema_flats[1] is not None
                   and val_loader is None else "")
                + (f"  lr={cur_lr:.4f}" if str(config.train.lr_schedule) != "constant" else ""),
            )
            rows.append({
                "method": method_label, "arm": arm, "round": rnd + 1,
                "test_loss": float(test_loss), "test_accuracy": float(test_acc),
                "val_loss": float(val_loss), "val_accuracy": float(val_acc),
                "ema_test_accuracy": float(ema_accs[0]), "ema2_test_accuracy": float(ema_accs[1]),
                "learning_rate": float(cur_lr),
                "spent_epsilon": spent_eps,
                # NAN: statistics of private gradients and of the noise, not recorded (diagnostics off)
                "raw_update_norm": avg_raw_norm, "uploaded_norm": avg_uploaded_norm,
                "noise_norm": NAN, "clip": float(clip),
                "uploaded_learned_energy": 0.0, "uploaded_protected_energy": 0.0,
                "aggregate_learned_energy": 0.0, "aggregate_protected_energy": 0.0,
                "free_learned_energy": NAN, "free_protected_energy": NAN,
                "grad_learned_energy": NAN, "grad_protected_energy": NAN,
                "step_gradient_energy_ratio": NAN,
                "subspace_overlap": last_overlap,
                "embedding_norm": NAN, "residual_norm": NAN,
                "embedding_clip_norm": float(config.gep.embedding_clip_norm),
                "residual_clip_norm": float(config.gep.residual_clip_norm),
                "noise_convention": noise_conv,
                # the binding client's (max eps) values; *_min / *_max over clients
                "noise_multiplier": float(z), "achieved_epsilon": float(achieved_eps),
                "total_mechanisms": int(total_mech), "mechanisms_per_round": int(mech_per_round),
                "binding_client": int(bind.client), "z_calibration": z_calibration,
                "spent_epsilon_min": float(spent_eps_min), "achieved_epsilon_min": float(eps_client_min),
                "noise_multiplier_min": float(z_client_min), "noise_multiplier_max": float(z_client_max),
                "subspace_dim": basis_dim if uses_subspace else 0, "learned_rank": learned_rank,
                "num_clients": num_clients, "participating": num_clients,
            })
            save_metrics_table(rows, paths.metrics_csv)  # on disk after every evaluation
            evaluation_seconds.append(time.perf_counter() - evaluation_start)

    log_kv("Non-finite rows zeroed", int(NONFINITE_ROWS_ZEROED[0]))
    contract = _privacy_contract(arm, config, client_priv, int(NONFINITE_ROWS_ZEROED[0]), final=True, warm=warm)
    update_provenance(paths.run_dir, privacy_contract=contract)
    log_kv("DP contract", "valid (paper settings)" if contract["dp_valid"] else
           "NOT valid for the paper: " + "; ".join(contract["reasons"]))
    save_metrics_table(rows, paths.metrics_csv)
    cost = _cost_record(device, round_seconds, evaluation_seconds, time.perf_counter() - started)
    log_kv("Cost", _cost_text(cost))
    if config.output.save_manifest_json:  # manifest.json was written when the run started
        (paths.run_dir / "run_metadata.json").write_text(json.dumps({
            "method": config.method, "experiment_name": config.experiment_name, "arm": arm,
            "dp_valid": contract["dp_valid"], "privacy_contract": contract,
            "num_clients": num_clients, "avg_client_size": avg_client_size,
            # z / mechanisms / epsilon: the binding client (the one attaining max_c eps_c); every client in "clients"
            "dense_z_calibration": str(config.privacy.dense_z_calibration), "z_calibration": z_calibration,
            # max_c eps_c; None (no guarantee) for nonprivate
            "record_level_epsilon": record_eps,
            "epsilon_client_min": float(eps_client_min),
            "binding_client": int(bind.client),
            "noise_multiplier": float(z), "residual_noise_multiplier": float(res_z),
            "accounting_noise_multiplier": None if arm == "nonprivate" else float(z_acct),
            "gep_accounting": str(config.privacy.gep_accounting),
            "subspace_source": str(config.gep.subspace_source) if uses_subspace else None,
            "project_residual": bool(arm == "gep_step_keep" and bool(config.gep.upload_residual)),
            "init_checkpoint": str(config.model.init_checkpoint),
            "init_checkpoint_sha256": warm.get("init_checkpoint_sha256"), "init_transferred": warm.get("init_transferred"),
            "init_missing": warm.get("init_missing"), "init_skipped": warm.get("init_skipped"),
            "init_pretraining": warm.get("init_pretraining"),
            "embedding_epsilon_ratio": float(config.gep.embedding_epsilon_ratio),
            "total_mechanisms": int(total_mech),
            "achieved_epsilon": float(achieved_eps), "subspace_dim": basis_dim if uses_subspace else 0,
            "sampler": sampler_mode, "lot_divisor": str(config.privacy.lot_divisor),
            "noise_convention": _noise_convention(arm, config),
            "nonfinite_rows_zeroed": int(NONFINITE_ROWS_ZEROED[0]),
            "diagnostics": str(config.train.diagnostics), "partition": str(config.data.partition),
            "val_source": str(config.data.val_source),
            "data": data,
            "cost": cost,  # peak GPU memory, time per round and per evaluation
            "clients": [asdict(cp) for cp in client_priv],
        }, indent=2, default=str), encoding="utf-8")
    return ExperimentArtifacts(paths=paths, metrics=rows)
