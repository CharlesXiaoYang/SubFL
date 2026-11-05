"""Output directory and file paths of a run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from subfl.config.experiment import ExperimentConfig


@dataclass
class RunPaths:
    """File locations of one run."""

    run_dir: Path
    metrics_csv: Path
    manifest_json: Path


MAX_RUN_DIR_SUFFIX = 1000


def build_run_paths(config: ExperimentConfig) -> RunPaths:
    """Create a new run directory ``<time>_<name>`` under the output root (``_1``, ``_2``, ... if taken)."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{timestamp}_{config.experiment_name}"
    output_root = Path(config.output.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for attempt in range(MAX_RUN_DIR_SUFFIX):
        run_dir = output_root / (base_name if attempt == 0 else f"{base_name}_{attempt}")
        try:
            run_dir.mkdir(exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError(f"could not create a new run directory for {base_name} under {output_root}")
    return RunPaths(
        run_dir=run_dir,
        metrics_csv=run_dir / "metrics.csv",
        manifest_json=run_dir / "manifest.json",
    )
