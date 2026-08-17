"""MLflow tracking configuration."""

from pathlib import Path

from pydantic import Field

from phase_classification.config.base import StrictModel


class MLflowConfig(StrictModel):
    tracking_uri: str | None = None
    experiment_name: str = "phase_classification"
    run_name: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    log_artifacts_to: Path | None = None
