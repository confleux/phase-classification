"""Runnable workflows that sit alongside the core training pipeline.

These modules provide focused evaluation, cross-dataset inference, and
foundation-model experiments. They are available through
``python -m phase_classification.workflows.<module>``.
"""

from pathlib import Path


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir():
            return parent
    return Path.cwd()


PROJECT_ROOT = _project_root()
