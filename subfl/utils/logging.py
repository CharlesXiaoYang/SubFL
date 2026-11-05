"""Console log helpers."""

from __future__ import annotations


def log_header(title: str) -> None:
    """Print a section header."""

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def log_kv(key: str, value: object) -> None:
    """Print one aligned ``key: value`` line."""

    print(f"{key:25s}: {value}")
