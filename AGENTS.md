# AGENTS.md

## Project

- Python package for surgical-video phase/action classification; Python `>=3.14`, managed with **uv** (`.python-version`, `uv.lock`). The `src/phase_classification` package is built with Hatchling; CLI entry point is `phase-cls`.
- Main flow: `cli.py` loads a strict Pydantic YAML config (`config/`, `loader.py`); `data/` builds datasets/loaders; `models/registry.py` constructs registered models; `engine.py` trains/evaluates; `run.py` orchestrates train → per-epoch validation → held-out test and MLflow/checkpoint logging.
- Head names/order and default CholecT50 label counts are centralized in `heads.py`. Do not duplicate these across config, data, losses, metrics, or CLI code.

## Current capabilities

- Dataset adapters: CholecT50 (vendored reference loader) and HeiChole; optional SurgCLIP cached-feature loading is in `data/surgclip_features.py`. Preserve `data/cholec_t50.py` byte-for-byte; its local Ruff config excludes it.
- Registry models: ResNet18, ConvNeXt-MS-TCN, Swin-MS-TCN, cached-feature SurgCLIP-MS-TCN, and Swin classifier. The two `surgclip-ms-tcn*-diffusion` names are custom run modes dispatched by `run.py`, not `models/registry.py` entries: they train a coarse cached-feature temporal model and then a diffusion refiner (MS-TCN++ variant included).
- Config is strict (`extra="forbid"`); `docs/CONFIG.md` is the field/default/constraint reference and `docs/METRICS.md` defines metric semantics and MLflow names. `configs/default.yaml` is an example, but its dataset path `/data/cholect50` must be changed to a real local dataset.
- Feature-cache runs require a compatible CholecT50 cache, `training.heads: [phase]`, and `dataloader.batch_size: 1`; see `docs/CONFIG.md` for cache formats and diffusion/model constraints.

## Commands and constraints

- `uv run phase-cls validate-config configs/default.yaml` checks schema only; add `--check-paths` to check local dataset files. `inspect`, `build-dataloaders`, and `train` are the other CLI subcommands. Dataset commands and training require data/cache files on disk.
- Focused checks: `uv run pytest tests/test_metrics.py`; full suite: `uv run pytest`; lint: `uv run ruff check .`. Tests use small synthetic fixtures; they do not need the full dataset or an MLflow server.
- Use `uv add`/`uv remove` for dependency changes so `pyproject.toml` and committed `uv.lock` stay synchronized. Run Python through `uv run`.
- Training logs to the configured MLflow tracking URI (local file store if unset). For the bundled PostgreSQL/MinIO server, run `docker compose up -d` from `mlflow/`; the training client also needs its own S3/MinIO endpoint and credentials in config or environment to upload model artifacts.
