"""RDP accounting of the Poisson-subsampled Gaussian mechanism and noise calibration."""

from __future__ import annotations

import math
from typing import Iterable


def parse_rdp_orders(spec: str) -> list[int]:
    """Parse an integer RDP order list like ``2-64`` or ``2,4,8,16``."""

    orders: set[int] = set()
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if start > end:
                raise ValueError("RDP order range start must be <= end")
            orders.update(range(start, end + 1))
        else:
            orders.add(int(item))
    result = sorted(order for order in orders if order > 1)
    if not result:
        raise ValueError("At least one RDP order > 1 is required")
    return result


def _log_add(values: Iterable[float]) -> float:
    """Stable log-sum-exp for a short sequence."""

    vals = list(values)
    max_val = max(vals)
    if max_val == -math.inf:
        return -math.inf
    return max_val + math.log(sum(math.exp(v - max_val) for v in vals))


def _log_comb(n: int, k: int) -> float:
    """Return log binomial coefficient."""

    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def subsampled_gaussian_rdp_one_step(q: float, noise_multiplier: float, order: int) -> float:
    """RDP upper bound of one Poisson-subsampled Gaussian step at an integer order; alpha / (2 z^2) when q = 1."""

    if not (0.0 < q <= 1.0):
        raise ValueError("sample rate q must satisfy 0 < q <= 1")
    if noise_multiplier <= 0.0:
        raise ValueError("noise_multiplier must be positive")
    if order <= 1:
        raise ValueError("RDP order must be > 1")
    if q == 1.0:
        return float(order / (2.0 * noise_multiplier**2))

    log_q = math.log(q)
    log_one_minus_q = math.log1p(-q)
    terms = []
    for j in range(order + 1):
        log_coef = _log_comb(order, j)
        log_prob = j * log_q + (order - j) * log_one_minus_q
        log_moment = (j * j - j) / (2.0 * noise_multiplier**2)
        terms.append(log_coef + log_prob + log_moment)
    return float(_log_add(terms) / (order - 1))


def compute_subsampled_gaussian_rdp(
    *,
    sample_rate: float,
    noise_multiplier: float,
    steps: int,
    orders: list[int],
) -> list[float]:
    """Compute composed RDP values for repeated subsampled Gaussian mechanisms."""

    if steps < 0:
        raise ValueError("steps must be non-negative")
    return [
        steps * subsampled_gaussian_rdp_one_step(sample_rate, noise_multiplier, order)
        for order in orders
    ]


def privacy_spent_from_rdp(orders: list[int], rdp_values: list[float], delta: float) -> tuple[float, int]:
    """Convert RDP values to the best epsilon at target delta."""

    if len(orders) != len(rdp_values):
        raise ValueError("orders and rdp_values must have the same length")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must satisfy 0 < delta < 1")

    eps_candidates = [
        rdp + math.log(1.0 / delta) / (order - 1)
        for order, rdp in zip(orders, rdp_values)
    ]
    best_index = min(range(len(eps_candidates)), key=lambda idx: eps_candidates[idx])
    return float(eps_candidates[best_index]), int(orders[best_index])


def epsilon_for_noise_multiplier(
    *,
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    orders: list[int],
    delta: float,
) -> tuple[float, int]:
    """Return epsilon spent by a repeated subsampled Gaussian mechanism."""

    rdp_values = compute_subsampled_gaussian_rdp(
        sample_rate=sample_rate,
        noise_multiplier=noise_multiplier,
        steps=steps,
        orders=orders,
    )
    return privacy_spent_from_rdp(orders, rdp_values, delta)


def calibrate_noise_multiplier(
    *,
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    steps: int,
    orders: list[int],
    tolerance: float = 1e-4,
) -> tuple[float, float, int]:
    """Binary-search a noise multiplier that meets the target RDP epsilon."""

    if target_epsilon <= 0.0:
        raise ValueError("target_epsilon must be positive")
    if not (0.0 < target_delta < 1.0):
        raise ValueError("target_delta must satisfy 0 < delta < 1")
    if not (0.0 < sample_rate <= 1.0):
        raise ValueError("sample_rate must satisfy 0 < q <= 1")
    if steps <= 0:
        raise ValueError("steps must be positive")

    lo = 1e-3
    hi = 1.0
    eps, order = epsilon_for_noise_multiplier(
        noise_multiplier=hi,
        sample_rate=sample_rate,
        steps=steps,
        orders=orders,
        delta=target_delta,
    )
    while eps > target_epsilon:
        hi *= 2.0
        eps, order = epsilon_for_noise_multiplier(
            noise_multiplier=hi,
            sample_rate=sample_rate,
            steps=steps,
            orders=orders,
            delta=target_delta,
        )
        if hi > 1e6:
            raise RuntimeError("could not calibrate RDP noise multiplier")

    for _ in range(80):
        mid = (lo + hi) / 2.0
        eps, order = epsilon_for_noise_multiplier(
            noise_multiplier=mid,
            sample_rate=sample_rate,
            steps=steps,
            orders=orders,
            delta=target_delta,
        )
        if eps > target_epsilon:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tolerance * max(1.0, hi):
            break

    final_eps, final_order = epsilon_for_noise_multiplier(
        noise_multiplier=hi,
        sample_rate=sample_rate,
        steps=steps,
        orders=orders,
        delta=target_delta,
    )
    return float(hi), float(final_eps), int(final_order)


def split_noise_multiplier(effective: float, embedding_ratio: float) -> tuple[float, float]:
    """(z / sqrt(r), z / sqrt(1 - r)) for z = effective, r = embedding_ratio: together one Gaussian mechanism with z."""

    if not (math.isfinite(effective) and effective > 0.0):
        raise ValueError(f"effective noise multiplier must be positive and finite, got {effective}")
    if not 0.0 < embedding_ratio < 1.0:
        raise ValueError(f"embedding_ratio must satisfy 0 < r < 1, got {embedding_ratio}")
    return float(effective / math.sqrt(embedding_ratio)), float(effective / math.sqrt(1.0 - embedding_ratio))
