"""Project-level metadata and output configuration."""

from pathlib import Path

from phase_classification.config.base import StrictModel


class ProjectConfig(StrictModel):
    name: str = "phase-classification"
    output_dir: Path = Path("outputs")
    save_checkpoints: bool = True
