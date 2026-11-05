"""Write metrics tables as CSV."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd


def save_metrics_table(rows: Iterable[dict], output_path: Path) -> None:
    """Write ``rows`` to ``output_path`` as CSV."""

    output_path = Path(output_path)
    tmp = output_path.with_name(output_path.name + ".tmp")
    pd.DataFrame(list(rows)).to_csv(tmp, index=False)
    os.replace(tmp, output_path)  # atomic: never a half-written file
