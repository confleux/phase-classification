# Architecture and Refactor Plan

## Goal

Keep the project a small, testable Python package with explicit boundaries—not
turn it into a framework. Configuration describes an experiment; data adapters
own dataset-specific parsing; models own forward computation; the engine owns
generic optimization/evaluation; application workflows connect those pieces
and handle external side effects.

## Current structure

```text
src/phase_classification/
  config/       strict Pydantic configuration models
  data/         CholecT50 / HeiChole adapters and loader construction
  models/       model implementations and the standard-model registry
  engine.py     reusable training and evaluation mechanics
  run.py        main train → validate → test orchestration
  tracking.py   MLflow run lifecycle and S3 client environment
  workflows/    optional evaluation and foundation-model workflows
  *diffusion.py custom SurgCLIP refinement implementations
```

The main boundaries are useful and should be retained. The main architectural
friction is that `run.py` both orchestrates standard training and dispatches
custom diffusion run modes, while metric/artifact logging and
experiment-specific validation remain mixed into the lifecycle. The MLflow run
and S3 client setup/finalization now live in `tracking.py`. There are also
multiple CLI entry points for workflows, so their dependency and shared utility
boundaries should stay explicit.

## Refactor plan

### 1. Make configuration the single experiment contract

- Keep the strict schema and the centralized label/head definitions.
- Make environment-specific values (dataset/cache paths, MLflow endpoint, and
  object-store credentials) overrideable without changing checked-in examples.
- Never print or log secret-valued fields. Keep config parsing free from data
  access and training side effects.
- Keep `docs/CONFIG.md` synchronized with schema defaults and constraints.

**Implemented in this pass:** YAML `${NAME}` / `${NAME:-default}` expansion,
consistent S3 settings in each runnable config, and secret redaction in `inspect`
output. The checked-in S3 defaults are explicitly local MinIO development
credentials; production credentials belong in environment variables.

### 2. Separate application orchestration from run modes

- Keep `engine.py` concerned with reusable train/evaluate mechanics.
- Move top-level execution into a small application layer: a standard trainer
  and named feature/diffusion workflows selected through one explicit dispatch
  table (rather than adding more model-name conditionals to `run.py`).
- Give each workflow the same input contract (`ExperimentConfig`, built data,
  device) and a consistent result contract (metrics and selected artifacts).
- Centralize run start/end, failure status, and common metadata around workflow
  execution so a workflow cannot accidentally leave an MLflow run active.

Do this incrementally: first characterize current dispatch and run lifecycle
with tests, then extract one custom mode at a time. Do not combine a behavioral
rewrite of the diffusion algorithms with this structural change.

### 3. Make data/model contracts explicit

- Keep dataset-specific format and split logic inside `data/` adapters; avoid
  branching on dataset names in the engine.
- Keep model construction in the registry and model-specific validation near
  model configuration/constructors. The custom diffusion run modes are
  workflows, not ordinary model registry entries.
- Standardize batch/output expectations where possible and validate incompatible
  combinations before training starts (for example cached-feature mode, phase
  head, sequence length, and batch size).
- Preserve the vendored CholecT50 reference loader unchanged; wrap it rather than
  editing its upstream behavior.

### 4. Isolate external side effects and artifact handling

- Keep MLflow/S3 setup in one tracking integration boundary, separate from model
  and optimizer code.
- Keep run start/finalization and S3 environment setup there; move metric and
  artifact logging behind the same boundary in a later, separately tested step.
- Have checkpoint/model saving return or record an explicit artifact location;
  avoid hidden assumptions about the current working directory.
- Treat local checkpoints and remote MLflow artifact uploads as distinct sinks
  with independently testable failure handling.

### 5. Keep optional workflows maintainable

- Keep research scripts importable as package workflows and expose stable CLI
  entry points only for supported workflows.
- Group workflow-only dependencies in the optional `workflows` extra; keep the
  core install sufficient for config validation and standard training.
- Prefer shared, tested helpers for device selection, metrics, and dataset
  loading over copying core logic into scripts.
- Add a small CLI smoke test and synthetic-data test for each supported workflow
  before expanding its public interface.

## Suggested target layout

```text
phase_classification/
  application/
    training.py       standard train/validate/test use case
    dispatch.py       named workflow selection
  tracking/
    mlflow.py         run lifecycle and parameters/metrics
    artifacts.py      local and remote model artifacts
  config/             schema and loading remain pure
  data/               dataset adapters and dataloader builders
  models/             implementations and standard-model registry
  workflows/          diffusion, evaluation, and optional experiments
  engine.py           generic optimization/evaluation mechanics
  cli.py              argument parsing and user-facing errors
```

This is a destination, not a mandate to move every file. Only create a package
when it removes a real dependency or responsibility tangle. Each extraction
should preserve the public CLI and be followed by focused tests, the full test
suite, and Ruff.

## Completion criteria

- `run.py` is a thin, readable application entry point with no growing set of
  architecture-specific branches.
- Every supported run mode is discoverable and has one explicit configuration
  contract.
- Schema validation and config inspection do not need dataset files or an
  MLflow server; inspection never reveals credentials.
- Core training and dataset tests continue to run on synthetic fixtures without
  external services.
- Experiment setup and metric semantics remain documented in
  `docs/CONFIG.md`, `docs/METRICS.md`, and `docs/MODELS_AND_EXPERIMENTS.md`.
