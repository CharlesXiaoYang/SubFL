"""Save a run's resolved configuration (manifest.json)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

from subfl.config.experiment import ExperimentConfig


def save_experiment_manifest(config: ExperimentConfig, output_path: Path) -> None:
    """Write the resolved experiment configuration as JSON."""

    output_path.write_text(json.dumps(asdict(config), indent=2, default=str), encoding="utf-8")
