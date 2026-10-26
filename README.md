# Phase Classification

Python package and CLI for surgical-video phase/action classification. The core
pipeline supports CholecT50 and HeiChole datasets, image and temporal models,
and optional cached SurgCLIP feature workflows.

## Setup

Install the project and development tools with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Optional evaluation and foundation-model workflows can be installed with:

```bash
uv sync --extra workflows
```

## Configure and run

Edit a config under `configs/` for local dataset/cache paths, then validate it:

```bash
uv run phase-cls validate-config configs/default.yaml
uv run phase-cls validate-config configs/default.yaml --check-paths
uv run phase-cls train configs/default.yaml
```

The default examples target a local MinIO artifact store. Start the bundled
MLflow/PostgreSQL/MinIO services from `mlflow/` with `docker compose up -d`.
The example configs use local development credentials; set
`MLFLOW_S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and
`AWS_DEFAULT_REGION` in the environment to connect to another S3-compatible
store. The config loader expands `${NAME}` and `${NAME:-default}` values.

See [`docs/CONFIG.md`](docs/CONFIG.md) for all fields and data requirements,
[`docs/METRICS.md`](docs/METRICS.md) for metric definitions, and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for package boundaries and the
incremental refactor plan. Current model/workflow coverage is summarized in
[`docs/MODELS_AND_EXPERIMENTS.md`](docs/MODELS_AND_EXPERIMENTS.md).
