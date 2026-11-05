"""Per-client local sampling: the local schedule (steps, sample rate, lot divisor) that the accountant, the
lot sampler and the trainer all read, and the Poisson lot sampler."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataloader import default_collate


@dataclass(frozen=True)
class LocalSchedule:
    """Local DP-SGD schedule of one client for one round."""

    num_examples: int  # n_c (treated as public)
    batch_size: int  # configured batch size B (the expected lot is q * n_c = min(B, n_c))
    epochs: int  # local epochs E per round
    steps_per_epoch: int  # max(1, ceil(n_c / B))
    steps: int  # local DP steps per round = steps_per_epoch * E
    sample_rate: float  # q = min(1, B / max(1, n_c))
    expected_lot: int  # q * n_c = min(B, n_c): the constant DP-SGD divisor (lot_divisor="expected")


LOT_DIVISORS = ("expected",)


def lot_divisor(mode: str, *, expected: int) -> int:
    """Divisor of each noisy lot sum: the constant expected lot size q * n_c = min(B, n_c)."""

    if mode != "expected":
        raise ValueError(f"lot_divisor must be one of {LOT_DIVISORS}, got {mode!r}")
    divisor = int(expected)
    if divisor <= 0:
        raise ValueError(f"lot divisor must be positive, got {divisor} ({mode}; a client without data?)")
    return divisor


def local_schedule(num_examples: int, batch_size: int, epochs: int) -> LocalSchedule:
    """Local schedule of a client with n records: max(1, ceil(n / B)) steps per epoch, q = min(1, B / max(1, n))."""

    n, batch, local_epochs = _as_int(num_examples, "num_examples"), _as_int(batch_size, "batch_size"), _as_int(
        epochs, "epochs"
    )
    if n < 0:
        raise ValueError(f"num_examples must be non-negative, got {num_examples}")
    if batch <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if local_epochs < 0:
        raise ValueError(f"epochs must be non-negative, got {epochs}")
    steps_per_epoch = max(1, math.ceil(n / batch))
    sample_rate = min(1.0, batch / max(1, n))
    return LocalSchedule(
        num_examples=n,
        batch_size=batch,
        epochs=local_epochs,
        steps_per_epoch=steps_per_epoch,
        steps=steps_per_epoch * local_epochs,
        sample_rate=sample_rate,
        expected_lot=min(batch, n),
    )


def _as_int(value, name: str) -> int:
    """int(value); refuses a non-integral value."""

    as_int = int(value)
    if as_int != value:
        raise TypeError(f"{name} must be an integer, got {value!r}")
    return as_int


SAMPLERS = ("poisson",)  # the DP arms; the nonprivate arm uses shuffled batches


class PoissonBatchSampler(Sampler):
    """Poisson lots: each record joins each lot with probability q; a pass yields exactly steps_per_epoch lots."""

    def __init__(self, num_examples: int, sample_rate: float, steps_per_epoch: int, generator: torch.Generator):
        if int(num_examples) < 1:
            raise ValueError(f"Poisson sampling needs at least one record, got {num_examples}")
        if not 0.0 < float(sample_rate) <= 1.0:
            raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
        if int(steps_per_epoch) < 1:
            raise ValueError(f"steps_per_epoch must be positive, got {steps_per_epoch}")
        self.num_examples = int(num_examples)
        self.sample_rate = float(sample_rate)
        self.steps_per_epoch = int(steps_per_epoch)
        self.generator = generator

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            # float64 uniforms, so q is not rounded to float32
            mask = torch.rand(self.num_examples, generator=self.generator, dtype=torch.float64) < self.sample_rate
            yield torch.nonzero(mask).flatten().tolist()


def poisson_seed(seed: int, client_index: int) -> int:
    """Seed of a client's mask generator, distinct from the run's other generators (seed plus a small offset)."""

    return (int(seed) * 1_000_003 + 7_919 * (int(client_index) + 1) + 0x5EED) % (2**63 - 1)


def collate_allow_empty(batch):
    """``default_collate``; an empty lot becomes ``(None, None)`` (a noise-only DP step)."""

    if len(batch) == 0:
        return None, None
    return default_collate(batch)


def make_poisson_loader(dataset, schedule: LocalSchedule, *, seed: int, num_workers: int = 0) -> DataLoader:
    """DataLoader over ``dataset`` whose lots follow ``schedule``."""

    if len(dataset) != schedule.num_examples:
        raise ValueError(f"dataset has {len(dataset)} records, the schedule assumes {schedule.num_examples}")
    # the generator goes to the sampler only: as DataLoader(generator=) every iterator would draw from it
    sampler = PoissonBatchSampler(schedule.num_examples, schedule.sample_rate, schedule.steps_per_epoch,
                                  torch.Generator().manual_seed(int(seed)))
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_allow_empty,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
