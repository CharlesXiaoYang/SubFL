"""PrivateFL (Yang et al., USENIX Security 2023) on the dense DP-SGD arm: each client keeps a local transformation,
trained with the model under the same per-example clip and noise; only the model update is uploaded. model=full trains
the whole model (input_norm), model=probe only fc_out on the frozen warm-start backbone (feature_norm)."""

from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from subfl.experiments import fl_local_gep_ldp as fl
from subfl.models.cifar_resnet import CIFARResNet

MODELS = ("full", "probe")
TRANSFORMS = {"full": ("input_norm", "none"), "probe": ("feature_norm", "none")}
IMAGE_SHAPE = (3, 32, 32)
FEATURES = 64  # CIFARResNet's feature width (fc_out's input)
HEAD = "fc_out"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAN = float("nan")


def code_files() -> dict[str, str]:
    """{path: sha256} of baselines/*.py and train_privatefl.py."""

    files = sorted(PROJECT_ROOT.glob("baselines/*.py")) + [PROJECT_ROOT / "train_privatefl.py"]
    return {p.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files if p.is_file()}


# ---- transformations and models ----

class InputNorm(nn.Module):
    """PrivateFL's input transformation x -> gamma[c] * x + beta[c, h, w], the identity at initialisation."""

    def __init__(self, channels: int, height: int, width: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels, height, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma[:, None, None] + self.beta


class FeatureNorm(nn.Module):
    """PrivateFL's feature transformation f -> gamma * f + beta[j] (scalar gamma), the identity at initialisation."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1, features))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return f * self.gamma + self.beta


def features(model: CIFARResNet, x: torch.Tensor) -> torch.Tensor:
    """CIFARResNet.forward up to the input of fc_out (the 64-d features)."""

    out = model.relu(model.bn1(model.conv1(x)))
    out = model.layer3(model.layer2(model.layer1(out)))
    return torch.flatten(model.avgpool(out), 1)


class CIFARResNetPFL(CIFARResNet):
    """CIFARResNet with a client's transformation (input_norm in front, or feature_norm before fc_out)."""

    def __init__(self, depth: int, num_classes: int, transform: str) -> None:
        super().__init__(depth=depth, num_classes=num_classes)
        self.transform = transform
        # registered after the backbone: same random init as build_model, and its parameters come last
        if transform == "input_norm":
            self.input_norm = InputNorm(*IMAGE_SHAPE)
        elif transform == "feature_norm":
            self.feature_norm = FeatureNorm(FEATURES)
        else:
            raise ValueError(f"transform must be input_norm or feature_norm, got {transform!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform == "input_norm":
            return super().forward(self.input_norm(x))
        return self.fc_out(self.feature_norm(features(self, x)))

    def tf(self) -> nn.Module:
        return self.input_norm if self.transform == "input_norm" else self.feature_norm


class StandardizedLinear(nn.Module):
    """The probe's fc_out on standardised features (f - mean) / sqrt(var + 1e-5), 0 where var is 0."""

    def __init__(self, linear: nn.Linear, mean: torch.Tensor, var: torch.Tensor) -> None:
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        # same parameter names as the nn.Linear it replaces; mean and var are fixed public statistics
        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = nn.Parameter(linear.bias.detach().clone())
        self.register_buffer("mean", mean.detach().clone().reshape(-1))
        self.register_buffer("var", var.detach().clone().reshape(-1))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        x = (f - self.mean) / torch.sqrt(self.var + 1e-5) * (self.var != 0)
        return F.linear(x, self.weight, self.bias)


def probe_feature_stats(global_model, config, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-feature mean and population variance of the frozen backbone's features over the public pool."""

    with torch.random.fork_rng(devices=[]):  # building and reading a loader draws from the CPU RNG
        loader = fl.build_public_validation_loader(config.data, batch_size=config.data.batch_size)
        with torch.no_grad():
            feats = torch.cat([features(global_model, x.to(device)) for x, _ in loader])
    return feats.mean(dim=0), feats.var(dim=0, unbiased=False)


class ProbeHead(nn.Module):
    """The probe's trainable part on backbone features: the client's FeatureNorm (if any), then fc_out."""

    def __init__(self, fc: nn.Linear, transform: str) -> None:
        super().__init__()
        self.fc = fc
        if transform == "feature_norm":
            self.feature_norm = FeatureNorm(fc.in_features)
        self.transform = transform

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.fc(self.feature_norm(f) if self.transform == "feature_norm" else f)


def identity_state(transform: str, device) -> dict[str, torch.Tensor]:
    """A client's transformation at the start: the identity."""

    if transform == "input_norm":
        return {"gamma": torch.ones(IMAGE_SHAPE[0], device=device), "beta": torch.zeros(*IMAGE_SHAPE, device=device)}
    return {"gamma": torch.ones(1, device=device), "beta": torch.zeros(1, FEATURES, device=device)}


def _set_state(module: nn.Module, state: dict) -> None:
    with torch.no_grad():
        module.gamma.copy_(state["gamma"])
        module.beta.copy_(state["beta"])


def _get_state(module: nn.Module) -> dict:
    return {"gamma": module.gamma.detach().clone(), "beta": module.beta.detach().clone()}


def _distance(state: dict) -> tuple[float, float]:
    """(RMS of gamma - 1, RMS of beta): how far a transformation is from the identity."""

    return (float(torch.sqrt(torch.mean((state["gamma"] - 1.0) ** 2)).item()),
            float(torch.sqrt(torch.mean(state["beta"] ** 2)).item()))


def _local_model(global_model, config, device, transform: str, state: dict) -> CIFARResNetPFL:
    """The global model behind a client's transformation."""

    if not bool(torch.isfinite(fl._flat(global_model)).all()):
        raise FloatingPointError("the global model is not finite (training has diverged)")
    local = CIFARResNetPFL(depth=20, num_classes=int(config.model.num_classes), transform=transform).to(device)
    _load_global(local, global_model)
    _set_state(local.tf(), state)
    fl._disable_inplace_relu(local)
    return local


def _load_global(local: CIFARResNetPFL, global_model) -> None:
    missing, unexpected = local.load_state_dict(global_model.state_dict(), strict=False)
    want = sorted(f"{local.transform}.{p}" for p in ("gamma", "beta"))
    if sorted(missing) != want or unexpected:
        raise RuntimeError(f"the PrivateFL model does not match the global model: missing {missing}, "
                           f"unexpected {unexpected}")


# ---- local DP-SGD with the transformation ----

def local_train_full(global_model, loader, config, device, z, gen, state, lr):
    """DP-SGD on the global model behind the client's InputNorm; returns (model delta, new state)."""

    local = _local_model(global_model, config, device, "input_norm", state)
    local.eval()
    names = [name for name, _ in local.named_parameters()]
    dim = sum(p.numel() for p in local.parameters())
    dim_model = sum(p.numel() for p in global_model.parameters())
    if names[-2:] != ["input_norm.gamma", "input_norm.beta"] or dim - dim_model != 3 + 3 * 32 * 32:
        raise RuntimeError(f"unexpected PrivateFL parameter layout: {names[-3:]}, {dim} vs {dim_model}")
    decayed = torch.ones(dim, device=device)
    decayed[dim_model:] = 0.0  # no weight decay on the transformation
    return _dpsgd_steps(local, names, dim, decayed, loader, config, device, z, gen, lr,
                        lambda x: x, local.input_norm,
                        lambda: fl._flat(global_model).to(device) - fl._flat(local).to(device)[:dim_model])


def local_train_probe(global_model, loader, config, device, z, gen, state, transform, lr):
    """DP-SGD on fc_out (and the client's FeatureNorm) over frozen backbone features; returns as local_train_full."""

    if not bool(torch.isfinite(fl._flat(global_model)).all()):
        raise FloatingPointError("the global model is not finite (training has diverged)")
    fc = copy.deepcopy(global_model.fc_out)  # copying draws no random numbers
    head = ProbeHead(fc, transform).to(device)
    if transform == "feature_norm":
        _set_state(head.feature_norm, state)
    names = [name for name, _ in head.named_parameters()]
    dim = sum(p.numel() for p in head.parameters())
    dim_head = fc.weight.numel() + fc.bias.numel()
    want = ["fc.weight", "fc.bias"] + (["feature_norm.gamma", "feature_norm.beta"] if transform == "feature_norm" else [])
    if names != want:
        raise RuntimeError(f"unexpected probe parameter layout: {names}")
    decayed = torch.ones(dim, device=device)
    decayed[dim_head:] = 0.0  # no weight decay on the transformation
    lo, hi = _head_slice(global_model)
    backbone = global_model  # frozen

    def embed(x):
        with torch.no_grad():
            return features(backbone, x)

    def delta():
        out = torch.zeros(sum(p.numel() for p in global_model.parameters()), device=device)
        out[lo:hi] = fl._flat(global_model.fc_out).to(device) - fl._flat(head.fc).to(device)
        return out

    return _dpsgd_steps(head, names, dim, decayed, loader, config, device, z, gen, lr, embed,
                        head.feature_norm if transform == "feature_norm" else None, delta)


def _head_slice(global_model) -> tuple[int, int]:
    """The coordinates of fc_out in the global model's flat vector (the last ones of a CIFARResNet)."""

    pointer, lo, hi = 0, None, None
    for name, p in global_model.named_parameters():
        if name.startswith(HEAD + ".") and lo is None:
            lo = pointer
        pointer += p.numel()
        if name.startswith(HEAD + "."):
            hi = pointer
    if lo is None or hi != pointer:
        raise RuntimeError("fc_out is not the last module of the global model")
    return lo, hi


def _dpsgd_steps(model, names, dim, decayed, loader, config, device, z, gen, lr, embed, tf, delta):
    """The step loop of fl._local_train_dpsgd on ``model``, inputs passed through ``embed`` first."""

    velocity = torch.zeros(dim, device=device)
    clip = float(config.privacy.clip_norm)
    eff_lr = float(config.train.lr if lr is None else lr)
    divisor_mode = str(config.privacy.lot_divisor)
    sched = fl.local_schedule(len(loader.dataset), config.data.batch_size, config.train.local_epochs)
    expected_lot = sched.expected_lot
    steps = 0
    for _ in range(config.train.local_epochs):
        for x, y in loader:
            if x is None:  # an empty Poisson lot: noise only, still one DP step
                grads = torch.zeros(0, dim, device=device)
            else:
                x, y = x.to(device), y.to(device)
                grads = fl._augmented_per_example_grads(model, embed(x), y, names, int(config.train.augmult_k))
            # one clip and one noise draw over model and transformation together: one Gaussian mechanism per step
            priv_grad = fl._privatize_dense_dpsgd(
                grads, clip, z, gen, divisor=fl.lot_divisor(divisor_mode, expected=expected_lot)
            )
            if config.train.weight_decay > 0.0:
                priv_grad = priv_grad + config.train.weight_decay * (fl._flat(model).to(device) * decayed)
            velocity = config.train.momentum * velocity + priv_grad
            fl._apply_flat_step(model, velocity, eff_lr)
            steps += 1
    fl._check_step_count(steps, sched.steps)
    return delta(), (None if tf is None else _get_state(tf))


# ---- evaluation of each client's deployed model ----

def cache_batches(loader, device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """A loader's batches on the device, read once."""

    with torch.random.fork_rng(devices=[]):  # starting a DataLoader iterator draws from the CPU RNG
        return [(x.to(device), y.to(device)) for x, y in loader]


def evaluate_batches(model: nn.Module, batches) -> tuple[float, float]:
    """(mean loss, accuracy) on cached batches, as evaluate_classifier."""

    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for x, y in batches:
            out = model(x)
            total_loss += F.cross_entropy(out.float(), y, reduction="sum").item()
            correct += (out.argmax(dim=1) == y).sum().item()
            count += y.size(0)
    return total_loss / count, correct / count


def deployed(eval_model: CIFARResNetPFL, global_model, states: list[dict], batches) -> list[tuple[float, float]]:
    """(loss, accuracy) of every client's deployed model: the global model behind its own transformation."""

    _load_global(eval_model, global_model)
    out = []
    for state in states:
        _set_state(eval_model.tf(), state)
        out.append(evaluate_batches(eval_model, batches))
    return out


# ---- the run ----

def check_config(config, arm: str, model: str, transform: str) -> None:
    """Raise ValueError unless the configuration is one the port supports."""

    if model not in MODELS:
        raise ValueError(f"privatefl model must be one of {MODELS}, got {model!r}")
    if transform not in TRANSFORMS[model]:
        raise ValueError(f"privatefl transform for model={model} must be one of {TRANSFORMS[model]}, got {transform!r}")
    if arm != "dense" or str(config.privacy.dense_noise_mode) != "dpsgd":
        raise ValueError(f"the PrivateFL port is a variant of the dense DP-SGD arm only (got arm={arm})")
    if str(config.model.model_name) != "cifar_resnet20" or str(config.data.dataset_name) != "cifar10":
        raise ValueError("the PrivateFL port supports cifar_resnet20 on cifar10 (3x32x32 inputs, 64-d features)")
    if model == "probe" and not str(config.model.init_checkpoint):
        raise ValueError("model=probe needs the warm-start checkpoint (--init_checkpoint)")
    if model == "probe" and int(config.train.augmult_k) > 1:
        raise ValueError("model=probe: augmentation multiplicity is not supported (augmult_k must be 1)")


def settings(model: str, transform: str) -> dict:
    """What run_metadata.json and provenance.json record about the port."""

    per_client = {"input_norm": 3 + 3 * 32 * 32, "feature_norm": 1 + FEATURES, "none": 0}[transform]
    return {
        "method": "PrivateFL (Yang et al., USENIX Security 2023), ported", "upstream_commit": "e82bb3e",
        "model": model, "transform": transform, "local_parameters_per_client": per_client,
        "trained": "fc_out (backbone frozen)" if model == "probe" else "every model parameter",
        "transform_clip": "jointly with the model (one flat per-example clip)",
        "transform_lr": "the model's", "transform_momentum": "the model's", "transform_weight_decay": 0.0,
        "uploaded": "fc_out delta" if model == "probe" else "model delta (the transformation stays local)",
        "metrics": ("test/val_accuracy = mean over clients of the deployed model (global model + own "
                    "transformation); *_global = the global model alone") if transform != "none" else "the runner's",
        "code_sha256": code_files(),
    }


def make_train(model: str, transform: str):
    """The replacement for fl._train that train_privatefl.py installs."""

    def train(config, arm, device, paths, sampler_mode):
        return _train(config, arm, device, paths, sampler_mode, model=model, transform=transform)

    return train


def _train(config, arm, device, paths, sampler_mode, *, model: str, transform: str):
    """fl._train for the dense arm, with each client's local transformation."""

    check_config(config, arm, model, transform)
    pfl = transform != "none"
    pfl_settings = settings(model, transform)
    fl.update_provenance(paths.run_dir, privatefl=pfl_settings)  # before training, so a failed run keeps it

    client_loaders, test_loader = fl.build_federated_dataloaders(config.data)
    num_clients = len(client_loaders)
    client_sizes = [len(ld.dataset) for ld in client_loaders]  # public under zero-out adjacency
    private_size = sum(client_sizes)
    avg_client_size = max(1, private_size // num_clients)
    train_loaders = []  # Poisson lots, one schedule per client
    for ci, ld in enumerate(client_loaders):
        if len(ld.dataset) < 1:
            raise ValueError(f"client {ci} has no data; Poisson DP-SGD needs every client to hold a record")
        sched_c = fl.local_schedule(len(ld.dataset), config.data.batch_size, config.train.local_epochs)
        train_loaders.append(fl.make_poisson_loader(ld.dataset, sched_c, seed=fl.poisson_seed(config.train.seed, ci),
                                                    num_workers=int(config.data.num_workers)))

    global_model = fl.build_model(config.model).to(device)
    fl._disable_inplace_relu(global_model)
    warm = {}
    if str(config.model.init_checkpoint):  # warm start from public pretraining
        warm = fl._warm_start(global_model, config, device)
    val_loader = None
    if str(config.data.val_source) == "public_pool":
        if warm.get("init_pretraining") == "unverified":
            raise ValueError("val_source=public_pool: the warm-start checkpoint may have been pretrained on the public "
                             "pool (unverified), so it cannot validate this run")
        val_loader = fl.build_public_validation_loader(config.data, batch_size=config.data.batch_size)
    feature_stats = None
    if model == "probe":  # PrivateFL's feature standardisation, fixed statistics from the public pool
        feature_stats = probe_feature_stats(global_model, config, device)
        global_model.fc_out = StandardizedLinear(global_model.fc_out, *feature_stats).to(device)
        pfl_settings["feature_standardisation"] = {
            "formula": "(f - mean) / sqrt(var + 1e-5), 0 where var == 0 (PrivateFL's standardize)",
            "statistics": "P (the public pool, evaluation transform) at the frozen warm-start backbone",
            "mean": [float(v) for v in feature_stats[0]], "var": [float(v) for v in feature_stats[1]]}
        fl.update_provenance(paths.run_dir, privatefl=pfl_settings)
    param_names = [name for name, _ in global_model.named_parameters()]

    client_priv = fl._calibrate_clients(arm, config, client_sizes, sampler_mode)
    fl.save_metrics_table([asdict(cp) for cp in client_priv], paths.run_dir / "client_privacy.csv")
    bind = max(client_priv, key=lambda cp: (cp.eps, -cp.client))
    z, achieved_eps, res_z, z_acct = bind.z, bind.eps, bind.z_res, bind.z_acct
    total_mech, mech_per_round = bind.total_mech, bind.mech_per_round
    eps_client_min = min(cp.eps for cp in client_priv)
    z_client_min, z_client_max = min(cp.z for cp in client_priv), max(cp.z for cp in client_priv)
    z_calibration = fl._z_calibration(arm, config)
    record_eps = fl._record_level_epsilon(arm, client_priv)
    contract = fl._privacy_contract(arm, config, client_priv, 0, final=False, warm=warm)
    fl.update_provenance(paths.run_dir, privacy_contract=contract)
    acct_orders = fl.parse_rdp_orders(config.privacy.rdp_orders)
    clip = float(config.privacy.clip_norm)
    noise_gen = torch.Generator(device=device).manual_seed(config.train.seed + 4242)

    method_label = (
        f"privatefl-{model}-{transform}-"
        f"fl_local_gep-arm={arm}-model={config.model.model_name}-k={int(config.gep.subspace_dim)}"
        f"-clients={num_clients}-LE={config.train.local_epochs}-alpha={config.data.dirichlet_alpha:g}"
        f"-eps={config.privacy.total_epsilon:g}-res={config.gep.upload_residual}"
    )
    # each client's transformation, kept across rounds
    states = [identity_state(transform, device) for _ in range(num_clients)] if pfl else []
    eval_model = test_batches = val_batches = None
    if pfl:
        with torch.random.fork_rng(devices=[]):  # building a model draws from the RNG
            eval_model = CIFARResNetPFL(depth=20, num_classes=int(config.model.num_classes), transform=transform)
        if feature_stats is not None:  # same head as the global model; its values are loaded at each evaluation
            eval_model.fc_out = StandardizedLinear(eval_model.fc_out, *feature_stats)
        eval_model = eval_model.to(device)
        test_batches = cache_batches(test_loader, device)
        val_batches = cache_batches(val_loader, device) if val_loader is not None else None

    fl.log_device_info(device)
    fl.log_kv("Arm", arm)
    fl.log_kv("PrivateFL", f"model={model} transform={transform} ({pfl_settings['local_parameters_per_client']} "
                           f"local parameters per client; {pfl_settings['trained']})")
    fl.log_kv("Model", config.model.model_name)
    fl.log_kv("Clients", num_clients)
    fl.log_kv("Avg client size", avg_client_size)
    fl.log_kv("Local epochs", config.train.local_epochs)
    fl.log_kv("Partition", str(config.data.partition))
    fl.log_kv("Dirichlet alpha", config.data.dirichlet_alpha)
    fl.log_kv("Rounds", config.train.num_rounds)
    fl.log_kv("Target epsilon", config.privacy.total_epsilon)
    fl.log_kv("Client sizes", client_sizes)
    fl.log_kv("z calibration", z_calibration)
    for cp in client_priv:
        fl.log_kv(f"  client {cp.client}", f"n={cp.n} q={cp.q:.6f} T={cp.total_mech} z={cp.z:.6f} eps={cp.eps:.6f}")
    fl.log_kv("Record-level epsilon", f"max_c eps_c = {achieved_eps:.6f} (client {bind.client}); "
                                      f"min_c eps_c = {eps_client_min:.6f}")
    fl.log_kv("Noise multiplier z", f"{z:.6f}")
    if total_mech > 0:
        fl.log_kv("Accounting z", f"{z_acct:.6f} (x {total_mech} subsampled Gaussian mechanisms)")
    fl.log_kv("Total mechanisms", total_mech)
    fl.log_kv("Achieved epsilon", f"{achieved_eps:.6f}")
    fl.log_kv("Lot divisor", str(config.privacy.lot_divisor))
    fl.log_kv("Noise convention", fl._noise_convention(arm, config))
    fl.log_kv("Sampler", sampler_mode)
    fl.log_kv("Diagnostics", str(config.train.diagnostics))
    fl.log_kv("DP contract", "valid (paper settings)" if contract["dp_valid"] else
              "NOT valid for the paper: " + "; ".join(contract["reasons"]))

    rows: list[dict] = []
    eval_interval = max(1, int(config.train.num_rounds) // 50) if config.train.num_rounds > 50 else 1
    ema_decays = [d for d in (float(config.train.ema_decay), float(config.train.ema_decay2)) if d > 0.0]
    ema_flats = [None] * len(ema_decays)

    for rnd in range(int(config.train.num_rounds)):
        cur_lr = fl._scheduled_lr(config, rnd)
        deltas: list[torch.Tensor] = []
        uploaded_norms: list[float] = []
        for ci in range(num_clients):  # every client takes part in every round
            loader = train_loaders[ci]
            cp = client_priv[ci]
            if model == "probe":
                delta, state = local_train_probe(global_model, loader, config, device, cp.z, noise_gen,
                                                 states[ci] if pfl else None, transform, lr=cur_lr)
            elif pfl:
                delta, state = local_train_full(global_model, loader, config, device, cp.z, noise_gen, states[ci],
                                                lr=cur_lr)
            else:
                delta = fl._local_train_dpsgd(global_model, loader, config, device, cp.z, noise_gen, param_names,
                                              lr=cur_lr)
                state = None
            if pfl:  # the transformation stays on the client
                states[ci] = state
            uploaded_norms.append(float(torch.linalg.norm(delta).item()))
            deltas.append(delta)

        avg_uploaded_norm = float(np.mean(uploaded_norms)) if uploaded_norms else 0.0
        avg_delta = torch.stack(deltas, dim=0).mean(dim=0)
        if not bool(torch.isfinite(avg_delta).all()):
            raise FloatingPointError(f"round {rnd + 1}: the aggregated update is not finite (training has diverged)")
        fl._load_flat(global_model, fl._flat(global_model).to(device) - avg_delta)

        if ema_decays:  # EMA of the global weights, for evaluation only
            cur = fl._flat(global_model).to(device)
            for i, d in enumerate(ema_decays):
                ema_flats[i] = cur.clone() if ema_flats[i] is None else d * ema_flats[i] + (1.0 - d) * cur

        save_every = int(config.output.save_model_every)
        if save_every > 0 and ((rnd + 1) % save_every == 0 or rnd == int(config.train.num_rounds) - 1):
            fl._save_model(paths.run_dir / f"ckpt_r{rnd + 1:03d}.pt", global_model, rounds_done=rnd + 1,
                           arm=f"{arm}+privatefl_{model}_{transform}", config=config)
            if pfl:  # the clients' transformations too (never uploaded)
                tf_path = paths.run_dir / f"ckpt_r{rnd + 1:03d}_tf.pt"
                tmp = tf_path.with_name(tf_path.name + ".tmp")
                torch.save({"arm": f"{arm}+privatefl_{model}_{transform}", "round": int(rnd + 1),
                            "transform": transform,
                            "states": [{k: v.detach().cpu() for k, v in s.items()} for s in states]}, tmp)
                os.replace(tmp, tf_path)

        if rnd % eval_interval == 0 or rnd == int(config.train.num_rounds) - 1:
            spent_cache: dict[tuple, float] = {}
            for cp in client_priv:
                key = (cp.q, cp.mech_per_round, cp.z, cp.z_res, cp.z_acct)
                if key not in spent_cache:
                    spent_cache[key] = fl._spent_eps_client(config, cp, rnd + 1, acct_orders)
            spent_by_client = [spent_cache[(cp.q, cp.mech_per_round, cp.z, cp.z_res, cp.z_acct)] for cp in client_priv]
            spent_eps, spent_eps_min = max(spent_by_client), min(spent_by_client)
            test_loss, test_acc = fl.evaluate_classifier(global_model, test_loader, device)
            val_loss = val_acc = NAN
            if val_loader is not None:
                val_loss, val_acc = fl.evaluate_classifier(global_model, val_loader, device)
            ema_accs = [float("nan"), float("nan")]
            if any(f is not None for f in ema_flats):
                _saved = fl._flat(global_model).to(device)
                for i, f in enumerate(ema_flats):
                    if f is not None:
                        fl._load_flat(global_model, f)
                        _, ema_accs[i] = fl.evaluate_classifier(global_model, test_loader, device)
                fl._load_flat(global_model, _saved)
            extra: dict = {}
            if pfl:  # test/val: mean over clients of their deployed models; *_global: the global model alone
                tests = deployed(eval_model, global_model, states, test_batches)
                vals = deployed(eval_model, global_model, states, val_batches) if val_batches is not None else None
                extra = {"test_loss_global": float(test_loss), "test_accuracy_global": float(test_acc),
                         "val_loss_global": float(val_loss), "val_accuracy_global": float(val_acc)}
                for k, (_, acc) in enumerate(tests):
                    extra[f"test_accuracy_c{k}"] = float(acc)
                for k in range(num_clients):
                    extra[f"val_accuracy_c{k}"] = float(vals[k][1]) if vals is not None else NAN
                    extra[f"tf_gamma_dev_c{k}"], extra[f"tf_beta_rms_c{k}"] = _distance(states[k])
                test_loss = float(np.mean([loss for loss, _ in tests]))
                test_acc = float(np.mean([acc for _, acc in tests]))
                if vals is not None:
                    val_loss = float(np.mean([loss for loss, _ in vals]))
                    val_acc = float(np.mean([acc for _, acc in vals]))
            noise_conv = fl._noise_convention(arm, config)
            acc_txt = "hidden(tuning)" if val_loader is not None else f"{test_acc:.4f}"
            fl.log_kv(
                f"[{arm}{'+pfl' if pfl else ''}] r{rnd + 1}/{config.train.num_rounds}",
                f"acc={acc_txt} spent_eps={spent_eps:.3f}/{config.privacy.total_epsilon:g}  "
                f"up||Δ||={avg_uploaded_norm:.3f}"
                + (f"  val={val_acc:.4f}" if val_loader is not None else "")
                + (f"  val_global={extra['val_accuracy_global']:.4f}" if pfl and val_loader is not None else "")
                + (f"  ema={ema_accs[0]:.4f}" if ema_flats and ema_flats[0] is not None and val_loader is None else "")
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
                "raw_update_norm": 0.0, "uploaded_norm": avg_uploaded_norm,
                "noise_norm": NAN, "clip": float(clip),
                "uploaded_learned_energy": 0.0, "uploaded_protected_energy": 0.0,
                "aggregate_learned_energy": 0.0, "aggregate_protected_energy": 0.0,
                "free_learned_energy": NAN, "free_protected_energy": NAN,
                "grad_learned_energy": NAN, "grad_protected_energy": NAN,
                "step_gradient_energy_ratio": NAN,
                "subspace_overlap": 1.0,
                "embedding_norm": NAN, "residual_norm": NAN,
                "embedding_clip_norm": float(config.gep.embedding_clip_norm),
                "residual_clip_norm": float(config.gep.residual_clip_norm),
                "noise_convention": noise_conv,
                "noise_multiplier": float(z), "achieved_epsilon": float(achieved_eps),
                "total_mechanisms": int(total_mech), "mechanisms_per_round": int(mech_per_round),
                "binding_client": int(bind.client), "z_calibration": z_calibration,
                "spent_epsilon_min": float(spent_eps_min), "achieved_epsilon_min": float(eps_client_min),
                "noise_multiplier_min": float(z_client_min), "noise_multiplier_max": float(z_client_max),
                "subspace_dim": 0, "learned_rank": 0,
                "num_clients": num_clients, "participating": num_clients,
                **extra,
            })
            fl.save_metrics_table(rows, paths.metrics_csv)  # written after every evaluation

    fl.log_kv("Non-finite rows zeroed", int(fl.NONFINITE_ROWS_ZEROED[0]))
    contract = fl._privacy_contract(arm, config, client_priv, int(fl.NONFINITE_ROWS_ZEROED[0]), final=True,
                                    warm=warm)
    fl.update_provenance(paths.run_dir, privacy_contract=contract)
    fl.log_kv("DP contract", "valid (paper settings)" if contract["dp_valid"] else
              "NOT valid for the paper: " + "; ".join(contract["reasons"]))
    fl.save_metrics_table(rows, paths.metrics_csv)
    if config.output.save_manifest_json:  # manifest.json was written at the start
        (paths.run_dir / "run_metadata.json").write_text(json.dumps({
            "method": config.method, "experiment_name": config.experiment_name, "arm": arm,
            "dp_valid": contract["dp_valid"], "privacy_contract": contract,
            "num_clients": num_clients, "avg_client_size": avg_client_size,
            "dense_z_calibration": str(config.privacy.dense_z_calibration), "z_calibration": z_calibration,
            "record_level_epsilon": record_eps,
            "epsilon_client_min": float(eps_client_min),
            "binding_client": int(bind.client),
            "noise_multiplier": float(z), "residual_noise_multiplier": float(res_z),
            "accounting_noise_multiplier": float(z_acct),
            "gep_accounting": str(config.privacy.gep_accounting),
            "subspace_source": None,
            "project_residual": False,
            "init_checkpoint": str(config.model.init_checkpoint),
            "init_checkpoint_sha256": warm.get("init_checkpoint_sha256"), "init_transferred": warm.get("init_transferred"),
            "init_missing": warm.get("init_missing"), "init_skipped": warm.get("init_skipped"),
            "init_pretraining": warm.get("init_pretraining"),
            "embedding_epsilon_ratio": float(config.gep.embedding_epsilon_ratio),
            "total_mechanisms": int(total_mech),
            "achieved_epsilon": float(achieved_eps), "subspace_dim": 0,
            "sampler": sampler_mode, "lot_divisor": str(config.privacy.lot_divisor),
            "noise_convention": fl._noise_convention(arm, config),
            "nonfinite_rows_zeroed": int(fl.NONFINITE_ROWS_ZEROED[0]),
            "diagnostics": str(config.train.diagnostics), "partition": str(config.data.partition),
            "val_source": str(config.data.val_source),
            "clients": [asdict(cp) for cp in client_priv],
            "privatefl": pfl_settings,
        }, indent=2, default=str), encoding="utf-8")
    return fl.ExperimentArtifacts(paths=paths, metrics=rows)
