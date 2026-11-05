"""Run the PrivateFL baseline: the dense DP-SGD arm of train.py with a local transformation on each client.

Usage: python train_privatefl.py --pfl_model full --pfl_transform input_norm <flags of train.py>
"""

from __future__ import annotations

from pathlib import Path
import signal
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train  # noqa: E402

TRANSFORMS = {"full": ("input_norm", "none"), "probe": ("feature_norm", "none")}  # same as baselines.privatefl


def build_parser():
    p = train.build_parser()
    p.description = "PrivateFL baseline: the dense DP-SGD arm with a local transformation on each client."
    p.set_defaults(output_root=Path("runs/privatefl"))
    g = p.add_argument_group("PrivateFL")
    g.add_argument("--pfl_model", type=str, required=True, choices=sorted(TRANSFORMS),
                   help="full: train every model parameter; probe: freeze the warm-start backbone and train fc_out "
                        "only (needs --init_checkpoint)")
    g.add_argument("--pfl_transform", type=str, required=True,
                   choices=sorted({t for ts in TRANSFORMS.values() for t in ts}),
                   help="local transformation of each client: input_norm or none for full, feature_norm or none "
                        "for probe")
    return p


def parse_args(argv: list[str] | None = None):
    parser = build_parser()
    args = train._validate(parser.parse_args(argv), parser)
    if args.arm != "dense":
        parser.error("PrivateFL runs on the dense DP-SGD arm: use --arm dense")
    if args.pfl_transform not in TRANSFORMS[args.pfl_model]:
        parser.error(f"--pfl_model {args.pfl_model} takes --pfl_transform {' or '.join(TRANSFORMS[args.pfl_model])}")
    if args.pfl_model == "probe" and not args.init_checkpoint:
        parser.error("--pfl_model probe needs the warm-start checkpoint (--init_checkpoint)")
    return args


def build_config(args):
    config = train.build_config(args)
    tag = train._safe(args.run_tag)
    config.experiment_name = f"privatefl_{args.pfl_model}_{args.pfl_transform}" + (f"_{tag}" if tag else "")
    config.method = "privatefl_ldp"
    return config


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(args)
    from baselines import privatefl as pfl
    from subfl.experiments import fl_local_gep_ldp as fl

    pfl.check_config(config, args.arm, args.pfl_model, args.pfl_transform)  # before any output is written
    signal.signal(signal.SIGTERM, train._exit_on_sigterm)
    saved = fl._train
    fl._train = pfl.make_train(args.pfl_model, args.pfl_transform)  # run_fl_local_gep_ldp calls fl._train
    try:
        fl.run_fl_local_gep_ldp(config, args.arm)
    finally:
        fl._train = saved


if __name__ == "__main__":
    main()
