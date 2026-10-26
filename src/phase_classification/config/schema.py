"""Top-level experiment configuration aggregating all sub-configs."""

from pydantic import Field

from phase_classification.config.base import StrictModel
from phase_classification.config.dataloader import DataLoaderConfig
from phase_classification.config.dataset import DatasetConfig
from phase_classification.config.mlflow import MLflowConfig
from phase_classification.config.model import ModelConfig
from phase_classification.config.project import ProjectConfig
from phase_classification.config.runtime import RuntimeConfig
from phase_classification.config.training import TrainingConfig
from phase_classification.config.validation import ValidationConfig


class ExperimentConfig(StrictModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    dataset: DatasetConfig = Field(...)
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    def to_mlflow_params(self) -> dict[str, str]:
        """Flatten the config to a ``{section.param: value}`` dict for MLflow.

        Secret fields are excluded so credentials are never recorded as params.
        """

        flat: dict[str, str] = {}
        for name, model in self.model_dump().items():
            if isinstance(model, dict):
                for k, v in model.items():
                    if _is_sensitive(k):
                        continue
                    flat[f"{name}.{k}"] = _stringify(v)
            else:
                flat[name] = _stringify(model)
        return flat

    def to_public_dict(self) -> dict[str, object]:
        """Return a JSON-compatible config representation with secrets masked."""
        return _redact_sensitive(self.model_dump(mode="json"))


def _is_sensitive(key: str) -> bool:
    normalized = key.lower()
    return (
        "secret" in normalized
        or normalized.endswith(("_password", "_token"))
        or normalized in {"password", "token", "aws_access_key_id"}
    )


def _redact_sensitive(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if _is_sensitive(key) else _redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return value


def _stringify(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    if isinstance(value, dict):
        return ";".join(f"{k}:{v}" for k, v in sorted(value.items()))
    return str(value)
