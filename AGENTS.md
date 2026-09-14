# AGENTS.md

## Status

Real torchal package for surgical video phase classification (CholecT50): CLI, pydantic config schema, vendored dataset loader, and a full train/validate/test loop with MLflow tracking. No CI.

Spec references: `docs/CONFIG.md` (every config field, type, default, allowed values), `docs/METRICS.md` (metric semantics + MLflow naming).

## Toolchain

- Managed with **uv**, Python **3.14** (`/Users/.../phase-classification/pyproject.toml`, `.python-version`).
- Default branch is `dev` (not `main`).
- `src/` layout (`src/phase_classification`, hatchling wheel). `pyproject.toml` defines `[project.scripts] phase-cls`, dev deps (`pytest`, `ruff`), and `uv.lock` is committed — keep `pyproject.toml`/`uv.lock` in sync via `uv add`/`uv remove`.

## Entrypoints

- CLI: `uv run phase-cls <subcommand> <config.yaml>`. Subcommands: `validate-config`, `inspect`, `build-dataloaders`, `train` (`src/phase_classification/cli.py`).
- Module: `uv run python -m phase_classification`.

## Config system

- YAML config parsed + schema-validated in `src/phase_classification/loader.py`. The pydantic base (`config/base.py`) is strict with `extra="forbid"` — unknown fields are errors.
- `load_config()` validates the schema; `check_paths()` is a separate call that checks on-disk paths. `build_dataloaders()` raises `DatasetUnavailableError` if `videos/` and `labels/` dirs are missing.
- `configs/default.yaml` is the reference example: only override fields you need; everything else falls back to schema defaults.

## Data

- Only dataset: CholecT50, via the vendored reference loader `data/cholec_t50.py` (Apache-2.0, from CAMMA — do not modify).
- `data/builders.py` dispatches on `dataset_variant` prefix `cholect`; unknown variants raise `DatasetRegistryError`.
- Requires a local dataset dir with `videos/` and `labels/` (config `dataset.root_dir`, e.g. `/data/cholect50`); downstream won't work without the data on disk.

## Running an experiment

- `uv run phase-cls train configs/default.yaml` runs the whole lifecycle in one run (`src/phase_classification/run.py`): train → per-epoch validation → test on the held-out test split.
- Model architecture is selected by `model.name` through the registry `models/registry.py` (`resnet18` → `models/resnet18.py` `ResNet18Classifier` with `pretrained`/`freeze_backbone` params; `convnext-ms-tcn`/`ms-tcn` → `ConvNeXtMSTCN` = ConvNeXt frame encoder + MS-TCN temporal head; `swin-ms-tcn` → `SwinMSTCN` = Swin frame encoder + MS-TCN temporal head; both require `dataset.sequence_length >= 2`, trained with `MSTCNLoss`); `model.params` carries architecture knobs.
- Engine `src/phase_classification/engine.py`: optimizer (`sgd`/`adam`/`adamw`), scheduler (`none`/`step`/`cosine`/`plateau`, stepped once per epoch), fp16 AMP, grad clipping. Losses in `losses.py` (CE for single-label heads, BCE for multi-label, focal); metrics in `metrics.py` (jaccard, f1_macro, f1_per_class, mean_accuracy, accuracy_per_class, edit_score).
- Head label-space facts are centralized in `src/phase_classification/heads.py` (head order, `head_index`, single-label heads, default `num_classes`); losses/metrics/CLI/config all import from there.
- MLflow logging per run:
  - Config params via `cfg.to_mlflow_params()`; tag `gitcommithash` = current git HEAD (set in `run.py:_start_mlflow`).
  - Per-epoch metrics logged at `step=epoch`: `train_loss` + `train_*` + `val_*` (train/val compute the same metric set as test).
  - Test metrics logged at the end (`test_*`).
  - Best model (by primary val metric) saved via `mlflow.pytorch.log_model` and to `outputs/checkpoints/best.pt`.
  - Run status is set explicitly: `FINISHED` on success, `FAILED` on exception (`run.py:_end_mlflow_run`).

## MLflow stack (docker)

- `mlflow/docker-compose.yaml` + `mlflow/.env` run MLflow with PostgreSQL (backend store) and MinIO (S3-compatible artifact store); `.env` overrides all defaults (credentials, ports, image). Start from `mlflow/` with modern Compose v2+: `docker compose up -d`; UI at `http://localhost:5000`.
- The stack auto-creates the S3 bucket via a one-shot `createbucket` service (`mlflow/docker-compose.yaml`), on which `mlflow` depends (`service_completed_successfully`). Requires modern Compose v2+ (legacy `docker-compose` v1 fails with `'ContainerConfig'`).
- Point config `mlflow.tracking_uri` (or `MLFLOW_TRACKING_URI` env) at the server for experiments to log there.
- The training client also needs S3/MinIO credentials to upload the model artifact — set `mlflow.s3_endpoint_url`/`aws_access_key_id`/`aws_secret_access_key`/`aws_region` in the config (or the corresponding env vars). Credentials are excluded from MLflow params (`run.py:_set_s3_env`).

## Spec track / what's been done

- **Model registry** (`models/registry.py` → `models/resnet18.py`, `models/ms_tcn.py`, `models/swin_classifier.py`): `resnet18` (`ResNet18Classifier`), `convnext-ms-tcn`/`ms-tcn` (`ConvNeXtMSTCN`), `swin-ms-tcn` (`SwinMSTCN`), `swin-cls`/`swin-classifier` (`SwinClassifier`). `model.params` rejects unknown keys (spec-driven, one helper).
- **Losses** (`losses.py`): CE for single-label heads (`phase`, `verb`), BCE for multi-label heads (`tool`, `target`, `triplet`), optional focal for single-label. `MSTCNLoss` (per-stage CE/BCE + truncated-MSE smoothing) for the MS-TCN models.
- **Metrics** (`metrics.py`): jaccard, f1_macro, f1_per_class, mean_accuracy, accuracy_per_class, edit_score; accumulated per head across each split. Test `edit_score` = mean over per-video sequences (`MetricAccumulator.merged`); Levenshtein via `rapidfuzz`.
- **Training loop** (`engine.py` + `run.py`): per-epoch train (with live train metrics) + validation, final test, MLflow params/tags/metrics/model, explicit `FINISHED`/`FAILED` run status.
- **DataLoader** worker seeding is module-level (`data/dataloaders.py:_seed_worker`) so `num_workers>0` works on Linux forkserver. `DataLoaders.test` is the list of per-video datasets; `build_test_loaders()` wraps them with the same config knobs.
- **Dependency**: `mlflow` and `rapidfuzz` added via `uv add` (`pyproject.toml`/`uv.lock`).
- **Tests**: `tests/` covers config schema, loader, CLI, losses, metrics (hand-computed values + Levenshtein reference DP), registry, engine smoke tests, and run.py helpers; run with `uv run pytest` (if the console script is missing, `uv sync --reinstall-package pytest`).
- Docs: `docs/CONFIG.md`, `docs/METRICS.md` (specs kept in sync with code).

## opencode.json

Edit permission is restricted to `src/**`, `tests/**`, and `*.md`; edits anywhere else are denied and bash commands prompt. `watcher.useGitignore: true`.

## Workflow rules

- Run Python via `uv run <...>`; never raw `python` or manual venv activation.
- Add/remove deps with `uv add`/`uv remove`; never `pip install`.
- Test with `uv run pytest`; lint with `uv run ruff check .`.
- Work in micro-steps: modify one logical module/class/function at a time, verify, and ask for feedback before touching a second file.
