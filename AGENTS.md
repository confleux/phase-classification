# AGENTS.md

## Status

Real torchal package for surgical video phase classification (CholecT50), beyond the original `uv init` scaffold: has a CLI, pydantic config schema, and a vendored dataset loader. No training/experiment code yet — only config, data, and CLI plumbing. No CI.

## Toolchain

- Managed with **uv**, Python **3.14** (`/Users/.../phase-classification/pyproject.toml`, `.python-version`).
- Default branch is `dev` (not `main`).
- `src/` layout (`src/phase_classification`, hatchling wheel). `pyproject.toml` defines `[project.scripts] phase-cls`, dev deps (`pytest`, `ruff`), and `uv.lock` is committed — keep `pyproject.toml`/`uv.lock` in sync via `uv add`/`uv remove`.

## Entrypoints

- CLI: `uv run phase-cls <subcommand> <config.yaml>`. Subcommands: `validate-config`, `inspect`, `build-dataloaders` (`src/phase_classification/cli.py`).
- Module: `uv run python -m phase_classification`.
- `main.py` is leftover `uv init` boilerplate — ignore it; it is not the real entrypoint.

## Config system

- YAML config parsed + schema-validated in `src/phase_classification/loader.py`. The pydantic base (`config/base.py`) is strict with `extra="forbid"` — unknown fields are errors.
- `load_config()` validates the schema; `check_paths()` is a separate call that checks on-disk paths. `build_dataloaders()` raises `DatasetUnavailableError` if `videos/` and `labels/` dirs are missing.
- `configs/default.yaml` is the reference example: only override fields you need; everything else falls back to schema defaults.

## Data

- Only dataset: CholecT50, via the vendored reference loader `data/cholec_t50.py` (Apache-2.0, from CAMMA — do not modify).
- `data/builders.py` dispatches on `dataset_variant` prefix `cholect`; unknown variants raise `DatasetRegistryError`.
- Requires a local dataset dir with `videos/` and `labels/` (config `dataset.root_dir`, e.g. `/data/cholect50`); downstream won't work without the data on disk.

## opencode.json

Edit permission is restricted to `src/**`, `tests/**`, and `*.md`; edits anywhere else are denied and bash commands prompt. `watcher.useGitignore: true`.

## Workflow rules

- Run Python via `uv run <...>`; never raw `python` or manual venv activation.
- Add/remove deps with `uv add`/`uv remove`; never `pip install`.
- Test with `uv run pytest`; lint with `uv run ruff check .`.
- Work in micro-steps: modify one logical module/class/function at a time, verify, and ask for feedback before touching a second file.
