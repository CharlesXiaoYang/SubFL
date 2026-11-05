"""Pretrain the warm-start model on public data: the CIFAR-100 training set (the paper's checkpoint).

The checkpoint holds the model and its config, which the warm start checks.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the warm-start model on public data")
    parser.add_argument("--output_root", type=Path, default=Path("runs/pretrain"))
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--normalization", type=str, choices=["cifar10"], default="cifar10")
    parser.add_argument("--pretrain_dataset", type=str, choices=["cifar100"], default="cifar100",
                        help="cifar100 = the CIFAR-100 training set (100 classes)")
    parser.add_argument("--pretrain_data_root", type=Path, default=Path("./data/cifar100"),
                        help="CIFAR-100 directory (downloaded if missing)")
    parser.add_argument("--lr_schedule", type=str, choices=["constant", "cosine"], default="constant",
                        help="learning-rate schedule over epochs")
    parser.add_argument("--model_name", type=str, choices=["cifar_resnet20"],
                        default="cifar_resnet20")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], required=True,
                        help="training device")
    return parser.parse_args()


def _cifar100_loaders(root, batch_size, num_workers, normalization):
    """CIFAR-100 train and test loaders, normalized like the CIFAR-10 runs."""

    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import CIFAR100

    from subfl.data.cifar10 import _normalization_stats

    mean, std = _normalization_stats(normalization)
    train_steps = [
        transforms.RandomCrop((32, 32), padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    eval_steps = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]

    train = CIFAR100(root=str(root), train=True, transform=transforms.Compose(train_steps), download=True)
    test = CIFAR100(root=str(root), train=False, transform=transforms.Compose(eval_steps), download=True)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    predictions = logits.argmax(dim=1)
    return int((predictions == targets).sum().item()), int(targets.numel())


def _train_one_epoch(model: nn.Module, loader, device, optimizer) -> tuple[float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.float(), targets)
        loss.backward()
        optimizer.step()
        correct, count = _accuracy(outputs, targets)
        total_loss += float(loss.item()) * count
        total_correct += correct
        total_examples += count
    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def _evaluate(model: nn.Module, loader, device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs.float(), targets)
        correct, count = _accuracy(outputs, targets)
        total_loss += float(loss.item()) * count
        total_correct += correct
        total_examples += count
    return total_loss / total_examples, total_correct / total_examples


def _pretrain(args, device, run_dir: Path, data_config, model_config) -> None:
    """Train and evaluate every epoch, keeping the best and the last checkpoint; writes metrics.csv and
    manifest.json."""

    from subfl.analysis.metrics import save_metrics_table
    from subfl.models.factory import build_model
    from subfl.utils.device import log_device_info
    from subfl.utils.logging import log_header, log_kv

    best_checkpoint = run_dir / "best_checkpoint.pt"
    last_checkpoint = run_dir / "last_checkpoint.pt"
    train_loader, test_loader = _cifar100_loaders(
        args.pretrain_data_root, args.batch_size, args.num_workers, args.normalization)
    model = build_model(model_config).to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
    )

    log_device_info(device)
    log_kv("Pretrain dataset", args.pretrain_dataset)
    log_kv("Num classes", model_config.num_classes)
    log_kv("Train size", len(train_loader.dataset))
    log_kv("Model", args.model_name)
    log_kv("Epochs / lr_schedule", f"{args.epochs} / {args.lr_schedule}")
    log_kv("Output directory", str(run_dir))

    rows: list[dict] = []
    best_accuracy = -1.0
    for epoch in range(args.epochs):
        if args.lr_schedule == "cosine":
            cur_lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, args.epochs)))
            for group in optimizer.param_groups:
                group["lr"] = cur_lr
        train_loss, train_accuracy = _train_one_epoch(model, train_loader, device, optimizer)
        test_loss, test_accuracy = _evaluate(model, test_loader, device)
        is_best = test_accuracy > best_accuracy
        payload = {
            "model_state_dict": model.state_dict(),
            # Path objects become strings, so the checkpoint loads with weights_only=True
            "model_config": json.loads(json.dumps(asdict(model_config), default=str)),
            "data_config": json.loads(json.dumps(asdict(data_config), default=str)),
            "epoch": epoch + 1,
            "test_accuracy": test_accuracy,
        }
        if is_best:
            best_accuracy = test_accuracy
            torch.save(payload, best_checkpoint)
        torch.save(payload, last_checkpoint)
        log_kv(f"epoch {epoch + 1}/{args.epochs}",
               f"train_acc={train_accuracy:.4f} test_acc={test_accuracy:.4f} best={best_accuracy:.4f}")
        rows.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss), "train_accuracy": float(train_accuracy),
            "test_loss": float(test_loss), "test_accuracy": float(test_accuracy),
            "best_accuracy": float(best_accuracy), "is_best": bool(is_best),
        })
        save_metrics_table(rows, run_dir / "metrics.csv")

    (run_dir / "manifest.json").write_text(json.dumps({
        "data": asdict(data_config), "model": asdict(model_config),
        "train": {"epochs": args.epochs, "lr": args.lr, "momentum": args.momentum,
                  "weight_decay": args.weight_decay, "seed": args.seed},
        "best_checkpoint": str(best_checkpoint), "last_checkpoint": str(last_checkpoint),
        "best_accuracy": best_accuracy,
    }, indent=2, default=str), encoding="utf-8")
    log_header("Public pretraining complete")
    log_kv("Best test accuracy", f"{best_accuracy:.4f}")
    log_kv("Best checkpoint", str(best_checkpoint))


def main() -> None:
    args = parse_args()

    from subfl.analysis.provenance import write_status
    from subfl.config.experiment import DataConfig, ModelConfig
    from subfl.utils.device import resolve_training_device
    from subfl.utils.seed import set_global_seed

    set_global_seed(args.seed)
    device = resolve_training_device(args.device)
    torch.backends.cudnn.benchmark = True

    # recorded in the checkpoint and manifest.json, as in the paper's checkpoint; the warm start reads only the
    # model's class count from a CIFAR-100 checkpoint
    data_config = DataConfig(
        dataset_name="cifar10",
        root=Path("./data/cifar10"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalization=args.normalization,
        public_samples_per_class=200,
        private_split_seed=1234,
    )
    model_config = ModelConfig(model_name=args.model_name, num_classes=100)

    tag = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in args.run_tag.strip()).strip("_")
    run_name = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_public_pretrain_{args.model_name}"
        + (f"_{tag}" if tag else "")
    )
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # run_status.json as for train.py runs; run.sh skips a run whose status is completed
    write_status(run_dir, "started")
    try:
        _pretrain(args, device, run_dir, data_config, model_config)
    except BaseException as exc:  # also KeyboardInterrupt: a stopped run must not look complete
        write_status(run_dir, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    write_status(run_dir, "completed")


if __name__ == "__main__":
    main()
