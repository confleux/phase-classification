"""Configuration loading and validation.

Loading is split into two stages:

1. ``load_config`` -- parse YAML and validate against the pydantic schema. This is
   always safe and needs no data present on disk.
2. ``check_paths`` -- optional environment checks against the (possibly mounted)
   dataset root. Use this on the remote run box when data is available.
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from phase_classification.config import ExperimentConfig


class ConfigLoadError(Exception):
    """Raised when a config file cannot be parsed or fails schema validation."""


_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_environment(value: Any) -> Any:
    """Expand ``${NAME}`` and ``${NAME:-default}`` in parsed YAML strings."""
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.groups()
        environment_value = os.environ.get(name)
        if environment_value is not None and (environment_value or default is None):
            return environment_value
        if default is not None:
            return default
        raise ConfigLoadError(
            f"Config references unset environment variable {name!r}."
        )

    return _ENV_VAR_PATTERN.sub(replace, value)


def load_config(path: Path | str) -> ExperimentConfig:
    """Load a YAML config file and validate it against the experiment schema.

    Raises ConfigLoadError on missing/invalid YAML or schema violations.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigLoadError(f"Config file not found: {path}")

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"Invalid YAML in {path}: {exc}") from exc

    if data is None:
        data = {}

    try:
        return ExperimentConfig.model_validate(_expand_environment(data))
    except ConfigLoadError:
        raise
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        raise ConfigLoadError(
            f"Config validation failed for {path}: {details}"
        ) from exc


def check_paths(cfg: ExperimentConfig, dataset_root: Path | None = None) -> list[str]:
    """Verify data layout exists on disk.

    Returns a list of missing-path messages (empty if everything is present).
    This is an environment check and is not part of schema validation.
    """
    missing: list[str] = []
    root = dataset_root if dataset_root is not None else cfg.dataset.root_dir
    if not root.is_dir():
        return [f"dataset root not found: {root}"]

    videos_dir = root / "videos"
    labels_dir = root / "labels"
    if not videos_dir.is_dir():
        missing.append(f"missing 'videos' dir under {root}")
    if not labels_dir.is_dir():
        missing.append(f"missing 'labels' dir under {root}")
    if cfg.dataset.feature_cache is not None and not cfg.dataset.feature_cache.is_file():
        missing.append(f"feature cache not found: {cfg.dataset.feature_cache}")
    return missing
