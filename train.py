"""Federated training in which every client trains with record-level DP: dense DP-SGD, SubFL, or no noise.

One call is one run; experiments/*.txt holds the commands of the paper's runs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import signal
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IID_ALPHA = 1000.0  # dirichlet_alpha recorded for IID runs


def _safe(value: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in value.strip()).strip("_")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Federated training with record-level DP on every client.")
    p.add_argument("--arm", type=str, required=True, choices=["dense", "gep_step_keep", "nonprivate"],
                   help="dense = per-step DP-SGD | gep_step_keep = SubFL | nonprivate = no noise")
    p.add_argument("--dataset_name", type=str, default="cifar10", choices=["cifar10", "svhn"])
    p.add_argument("--data_root", type=Path, default=Path("./data/cifar10"))
    p.add_argument("--download_data", action=argparse.BooleanOptionalAction, default=False,
                   help="download the dataset into --data_root if it is missing (default: off)")
    p.add_argument("--output_root", type=Path, default=Path("runs/train"))
    p.add_argument("--run_tag", type=str, default="")
    p.add_argument("--save_model_every", type=int, default=0,
                   help="save the global model as ckpt_rNNN.pt in the run directory every N rounds and after the "
                        "last round; 0 = never")
    p.add_argument("--model_name", type=str, default="cifar_resnet20",
                   choices=["cifar_resnet20", "cifar_resnet56"])
    p.add_argument("--init_checkpoint", type=str, default="",
                   help="warm-start the global model from a pretrain.py checkpoint; '' = random init")
    p.add_argument("--init_checkpoint_sha256", type=str, default="",
                   help="required sha256 prefix (8 to 64 hex characters) of --init_checkpoint")

    p.add_argument("--num_clients", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--partition", type=str, required=True, choices=["iid", "dirichlet"],
                   help="how the private data is split across clients: iid = even random split (recorded as "
                        "--dirichlet_alpha 1000.0) | dirichlet = class-wise Dirichlet(alpha) split")
    p.add_argument("--dirichlet_alpha", type=float, default=None,
                   help=f"Dirichlet concentration, required by --partition dirichlet; --partition iid records "
                        f"{IID_ALPHA}")
    p.add_argument("--public_samples_per_class", type=int, default=200)
    p.add_argument("--val_source", type=str, default="none", choices=["none", "public_pool"],
                   help="validation set: public_pool = the public pool P, evaluated with the test set "
                        "(val_accuracy column) | none")
    p.add_argument("--normalization", type=str, default="cifar10", choices=["cifar10", "public_pool"],
                   help="input normalization: cifar10 = the CIFAR-10 constants | public_pool = per-channel mean and "
                        "std of the public pool P (SVHN only)")
    p.add_argument("--svhn_public_split", type=str, default="same", choices=["same", "extra"],
                   help="source of SVHN's public pool P: same = --public_samples_per_class per class of 'train', the "
                        "rest private | extra = SVHN 'extra', all of 'train' private")
    p.add_argument("--private_split_seed", type=int, default=1234)

    p.add_argument("--num_rounds", type=int, default=100)
    p.add_argument("--local_epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--augmult_k", type=int, default=1,
                   help="augmentation multiplicity: average each per-example gradient over K augmented views before "
                        "clipping (no extra privacy cost); 1 = off")
    p.add_argument("--ema_decay", type=float, default=0.0,
                   help="decay of an EMA of the global weights, used for evaluation only (ema_test_accuracy "
                        "column); 0 = off")
    p.add_argument("--ema_decay2", type=float, default=0.0,
                   help="decay of a second EMA (ema2_test_accuracy column); 0 = off")
    p.add_argument("--lr_schedule", type=str, default="constant", choices=["constant", "cosine"],
                   help="learning-rate schedule over rounds")
    p.add_argument("--lr_min_ratio", type=float, default=0.0,
                   help="final learning rate as a fraction of --lr for cosine")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--diagnostics", type=str, default="off", choices=["off"],
                   help="off = no statistics of private data beyond the DP release (their metrics columns are NaN)")
    p.add_argument("--device", type=str, required=True, choices=["cuda", "cpu"],
                   help="cuda (choose the GPU with CUDA_VISIBLE_DEVICES) or cpu")

    p.add_argument("--total_epsilon", type=float, required=True,
                   help="record-level epsilon of each client over the whole run")
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--clip_norm", type=float, default=1.0,
                   help="per-example clip norm of dense (gep_step_keep uses --embedding_clip_norm and "
                        "--residual_clip_norm)")
    p.add_argument("--client_sample_rate", type=float, default=1.0, choices=[1.0],
                   help="fraction of clients per round: 1.0 = every client in every round")
    p.add_argument("--rdp_orders", type=str, default="2-64")
    p.add_argument("--dense_noise_mode", type=str, default="dpsgd", choices=["dpsgd"],
                   help="dpsgd = per-example clipping and Gaussian noise at every local step")
    p.add_argument("--lot_divisor", type=str, default="expected", choices=["expected"],
                   help="divisor of each noisy lot sum: expected = the constant min(B, n_c)")
    p.add_argument("--sampler", type=str, required=True, choices=["poisson"],
                   help="how local lots are drawn: poisson = each record joins independently with probability "
                        "min(1, B/n_c), ceil(n_c/B) lots per local epoch (the nonprivate arm uses shuffled batches)")
    p.add_argument("--gep_accounting", type=str, default="joint", choices=["joint"],
                   help="joint = each step's (embedding, residual) pair is one Gaussian mechanism with multiplier z, "
                        "calibrated as for dense; the parts get z/sqrt(r) and z/sqrt(1-r), r = "
                        "--embedding_epsilon_ratio")
    p.add_argument("--dense_z_calibration", type=str, default="per_client", choices=["per_client"],
                   help="per_client = each client calibrates its noise multiplier from its own n_c")

    p.add_argument("--subspace_dim", type=int, default=1000)
    p.add_argument("--anchor_samples", type=int, default=1024)
    p.add_argument("--recompute_interval", type=int, default=1, help="refit the subspace every N rounds")
    p.add_argument("--local_recompute_interval", type=int, default=1,
                   help="gep_step_keep: refit the subspace every N local steps")
    p.add_argument("--rank_allocation", type=str, default="energy", choices=["energy"],
                   help="energy = split the subspace rank across tensors by their share of the anchor gradient energy")
    p.add_argument("--embedding_clip_norm", type=float, default=1.0)
    p.add_argument("--residual_clip_norm", type=float, default=0.5)
    p.add_argument("--embedding_epsilon_ratio", type=float, default=0.5,
                   help="gep_step_keep: share r of the privacy budget given to the embedding (0.5 = equal split)")
    p.add_argument("--upload_residual", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--subspace_source", type=str, default="anchor_svd", choices=["anchor_svd", "random"],
                   help="per-tensor basis: anchor_svd = top singular vectors of the public anchor gradients | "
                        "random = uniformly random subspace of the same ranks")
    p.add_argument("--anchor_random_labels", action=argparse.BooleanOptionalAction, required=True,
                   help="random labels for the public anchors (only the random-label control); ignored by dense and "
                        "nonprivate")
    p.add_argument("--protect_classifier_head", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--classifier_head_names", type=str, default="fc_out,classifier")
    p.add_argument("--full_rank_bias", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--head_full_rank_threshold", type=int, default=8192)
    p.add_argument("--min_subspace_rank", type=int, default=1)
    return p


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Check constraints between flags that argparse cannot express."""

    if args.save_model_every < 0:
        parser.error("--save_model_every must be >= 0 (0 = never)")
    if args.init_checkpoint_sha256:
        if not args.init_checkpoint:
            parser.error("--init_checkpoint_sha256 needs --init_checkpoint")
        if not re.fullmatch(r"[0-9a-fA-F]{8,64}", args.init_checkpoint_sha256):
            parser.error("--init_checkpoint_sha256 must be 8 to 64 hex characters (a prefix of the file's sha256)")
    if args.partition == "iid":
        if args.dirichlet_alpha is None:
            args.dirichlet_alpha = IID_ALPHA
        elif args.dirichlet_alpha != IID_ALPHA:
            parser.error(f"--dirichlet_alpha is only for --partition dirichlet (--partition iid records {IID_ALPHA})")
    elif args.dirichlet_alpha is None or not args.dirichlet_alpha > 0.0:
        parser.error("--partition dirichlet needs --dirichlet_alpha > 0")
    elif not (math.isfinite(args.dirichlet_alpha) and args.dirichlet_alpha < 100.0):
        parser.error("--partition dirichlet needs --dirichlet_alpha < 100 (alpha >= 100 is the IID split); "
                     "use --partition iid")
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse ``argv`` (``None``: ``sys.argv[1:]``) and check it."""

    parser = build_parser()
    return _validate(parser.parse_args(argv), parser)


def build_config(args: argparse.Namespace):
    """Map parsed arguments to an ExperimentConfig."""

    from subfl.config.experiment import (
        DataConfig, ExperimentConfig, GEPConfig, ModelConfig, OutputConfig, PrivacyConfig, TrainConfig,
    )
    from subfl.data.factory import num_classes_for_dataset

    num_classes = num_classes_for_dataset(args.dataset_name)
    tag = _safe(args.run_tag)
    experiment_name = f"fl_local_gep_{args.arm}" + (f"_{tag}" if tag else "")

    config = ExperimentConfig(
        experiment_name=experiment_name,
        method="fl_local_gep_ldp",
        data=DataConfig(
            dataset_name=args.dataset_name, root=args.data_root, num_clients=args.num_clients,
            batch_size=args.batch_size, num_workers=args.num_workers,
            public_samples_per_class=args.public_samples_per_class, private_split_seed=args.private_split_seed,
            dirichlet_alpha=args.dirichlet_alpha, partition=args.partition, download=args.download_data,
            val_source=args.val_source, normalization=args.normalization, svhn_public_split=args.svhn_public_split,
            auxiliary_random_labels=args.anchor_random_labels,
        ),
        model=ModelConfig(model_name=args.model_name, num_classes=num_classes,
                          init_checkpoint=args.init_checkpoint, init_checkpoint_sha256=args.init_checkpoint_sha256),
        train=TrainConfig(
            num_rounds=args.num_rounds, local_epochs=args.local_epochs, lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, seed=args.seed, device=args.device,
            augmult_k=args.augmult_k, ema_decay=args.ema_decay, ema_decay2=args.ema_decay2,
            lr_schedule=args.lr_schedule, lr_min_ratio=args.lr_min_ratio, diagnostics=args.diagnostics,
        ),
        privacy=PrivacyConfig(
            total_epsilon=args.total_epsilon, delta=args.delta, clip_norm=args.clip_norm,
            accountant="rdp", rdp_orders=args.rdp_orders, client_sample_rate=args.client_sample_rate,
            dense_noise_mode=args.dense_noise_mode, lot_divisor=args.lot_divisor, sampler=args.sampler,
            gep_accounting=args.gep_accounting, dense_z_calibration=args.dense_z_calibration,
        ),
        gep=GEPConfig(
            enabled=True, method_variant="pfa", subspace_dim=args.subspace_dim, anchor_samples=args.anchor_samples,
            recompute_interval=args.recompute_interval, local_recompute_interval=args.local_recompute_interval,
            rank_allocation=args.rank_allocation,
            embedding_clip_norm=args.embedding_clip_norm, residual_clip_norm=args.residual_clip_norm,
            embedding_epsilon_ratio=args.embedding_epsilon_ratio,
            upload_residual=args.upload_residual, subspace_source=args.subspace_source,
            protect_classifier_head=args.protect_classifier_head,
            classifier_head_names=args.classifier_head_names, full_rank_bias=args.full_rank_bias,
            head_full_rank_threshold=args.head_full_rank_threshold, min_subspace_rank=args.min_subspace_rank,
        ),
        output=OutputConfig(output_root=args.output_root, save_model_every=args.save_model_every),
    )
    return config


def _exit_on_sigterm(signum, frame):  # SIGTERM (docker stop, timeout): unwind so the run is marked failed
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(args)
    from subfl.experiments.fl_local_gep_ldp import run_fl_local_gep_ldp

    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    run_fl_local_gep_ldp(config, args.arm)


if __name__ == "__main__":
    main()
