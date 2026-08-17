# Configuration Specification

Every configuration is a **YAML file** (see `configs/default.yaml`). It is parsed
by `src/phase_classification/loader.py`, schema-validated against the pydantic
models in `src/phase_classification/config/*.py`, and passed to the CLI.

**Schema is strict** (`extra="forbid"`): adding a field that isn't defined below
raises a validation error.

## How to use a config

```bash
# Validate schema only (no disk access)
uv run phase-cls validate-config configs/my.yaml

# Validate schema + verify dataset paths on disk
uv run phase-cls validate-config configs/my.yaml --check-paths

# Inspect the resolved config, optionally in a different format
uv run phase-cls inspect configs/my.yaml [--format yaml|json|mlflow]

# Print dataset split sizes + a first batch shape (needs real data)
uv run phase-cls build-dataloaders configs/my.yaml

# Train + validate + test in one run, logging to MLflow (needs real data)
uv run phase-cls train configs/my.yaml
```

You only need to write the fields you want to override. Everything else falls
back to the schema defaults below.

> **Wired vs. schema-only.** Several fields are defined in the schema for future
> use but are **not yet read** by the engine/runner. They are marked
> *(not wired)* below — setting them has no effect (except `validation.batch_size`,
> which raises `RunError`).

---

## Top-level sections

The config has 8 top-level sections. All are optional except `dataset`.

| Section | Purpose |
|---|---|
| `project` | run metadata & local outputs |
| `dataset` | data source & preprocessing |
| `dataloader` | torch DataLoader knobs |
| `training` | optimizer / loss / heads / epochs |
| `model` | architecture name + knobs + label space |
| `validation` | which metrics to compute |
| `mlflow` | tracking server & run identity |
| `runtime` | device / precision / seeding |

---

## `project`

| Field | Type | Default | Allowed / notes |
|---|---|---|---|
| `name` | str | `phase-classification` | free-form |
| `seed` | int | `42` | *(not wired)* global seed |
| `output_dir` | Path | `outputs` | best checkpoint → `outputs/checkpoints/best.pt` |
| `save_checkpoints` | bool | `true` | if `false`, skips local checkpoint + `mlflow.pytorch.log_model` |

---

## `dataset` (only required section)

| Field | Type | Default | Allowed values / notes |
|---|---|---|---|
| `dataset_variant` | str | `cholect50-crossval` | `cholect45-crossval`, `cholect50-crossval`, `cholect50-challenge`, `cholect50`, `cholect45`, `cholect50-subset`, `cholect50-small` |
| `root_dir` | Path | — (**required**) | dir containing `videos/` and `labels/`, e.g. `/data/cholect50` |
| `test_fold` | int | `1` | `1..5` (only meaningful for `*-crossval`) |
| `sequence_length` | int | `1` | `1` = single frame (`T50`); `>=2` = temporal windows (`T50Sequence`) |
| `augmentations` | list[str] | `["original"]` | `original`, `vflip`, `hflip`, `contrast`, `rot90`, `brightness`, `random_contrast` |
| `image_size` | [h, w] | `[256, 448]` | *(not wired)* the vendored loader hard-codes `256x448` |
| `normalize` | bool | `true` | ImageNet normalization |
| `video_ids` | list[int] \| null | `null` | *(not wired)* optional explicit video override |

`dataset_variant` must start with `cholect`, otherwise `DatasetRegistryError`.
Crossval variants map `test_fold` to the held-out fold; non-crossval variants use
the built-in train/val/test split from the loader.

---

## `dataloader`

| Field | Type | Default | Notes |
|---|---|---|---|
| `batch_size` | int | `32` | **the effective mini-batch size** |
| `num_workers` | int | `4` | DataLoader workers |
| `shuffle` | bool | `true` | train only; val/test forced to `false` |
| `pin_memory` | bool | `true` | |
| `prefetch_factor` | int \| null | `2` | ignored when `num_workers == 0` |
| `persistent_workers` | bool | `true` | effective only when `num_workers > 0` |
| `drop_last` | bool | `false` | |

---

## `training`

| Field | Type | Default | Allowed values / notes |
|---|---|---|---|
| `epochs` | int | `50` | `>= 1` |
| `batch_size` | int | `32` | *(not wired)* use `dataloader.batch_size` instead |
| `optimizer` | str | `adamw` | `sgd`, `adam`, `adamw` |
| `learning_rate` | float | `0.001` | `> 0` |
| `weight_decay` | float | `0.0` | `>= 0` |
| `momentum` | float | `0.9` | `0..1` (used by `sgd`) |
| `scheduler` | str | `none` | `none`, `step`, `cosine`, `plateau` |
| `step_size` | int | `10` | `step` LR drop interval / `plateau` patience |
| `gamma` | float | `0.1` | factor for `step`/`plateau` |
| `loss` | str | `cross_entropy` | `cross_entropy`, `focal` |
| `heads` | list[str] | `["phase"]` | subset of `phase`, `tool`, `verb`, `target`, `triplet` |
| `grad_clip_norm` | float \| null | `null` | max grad norm; disabled when `null` |

Loss semantics (`losses.py`):
- single-label heads (`phase`, `verb`) → `CE` (or `focal`)
- multi-label heads (`tool`, `target`, `triplet`) → always `BCEWithLogits`

Schedulers are stepped once per epoch; `plateau` tracks the primary val metric
and uses `mode=max` for accuracy-like metrics.

---

## `model`

| Field | Type | Default | Allowed values / notes |
|---|---|---|---|
| `name` | str | `resnet18` | `resnet18` |
| `params` | dict | `{}` | see table below — unknown keys → `ModelRegistryError` |
| `num_classes` | dict[str,int] | see below | keys must be in `phase, tool, verb, target, triplet` |

Default `num_classes` (CholecT50 label space):
```yaml
model:
  num_classes:
    phase: 7
    tool: 6
    verb: 10
    target: 15
    triplet: 100
```

The set of heads that actually get an output layer is `training.heads`. The
`name` is modular through `models/registry.py`.

### `model.params` per architecture

`resnet18` (ResNet18 backbone + classifier heads in `models/net.py`):

| Param | Type | Default | Notes |
|---|---|---|---|
| `in_channels` | int | `3` | replaced `conv1` stem if not 3 |
| `pretrained` | bool | `false` | ImageNet weights (`torchvision.models.ResNet18_Weights.DEFAULT`) |
| `freeze_backbone` | bool | `false` | freeze all backbone weights; only `project` + classifier heads train |
| `features` | int | `512` | feature dim before the classifier heads |

Example (frozen pretrained backbone, phase + tool heads):
```yaml
model:
  name: resnet18
  params: { pretrained: true, freeze_backbone: true }
  num_classes: { phase: 7, tool: 6 }
training:
  heads: [phase, tool]
```

---

## `validation`

| Field | Type | Default | Notes |
|---|---|---|---|
| `metrics` | list[str] | `["accuracy", "accuracy_per_phase"]` | `accuracy`, `accuracy_per_phase`, `f1_macro`, `f1_micro`, `jaccard`, `edit_score` |
| `checkpoint_path` | Path \| null | `null` | *(not wired)* |
| `top_k` | int | `1` | *(not wired)* |
| `batch_size` | int \| null | `null` | **raises `RunError` if set** (not wired) |

Each metric is computed per head and reported as `<head>_<metric>`
(e.g. `phase_accuracy`). Exactly one metric should be listed first to serve as the
"primary val metric" used for best-checkpoint selection.

---

## `mlflow`

| Field | Type | Default | Notes |
|---|---|---|---|
| `tracking_uri` | str \| null | `null` | e.g. `http://localhost:5000`; falls back to local file store |
| `experiment_name` | str | `phase_classification` | |
| `run_name` | str \| null | `null` | auto-generated if not set |
| `tags` | dict[str,str] | `{}` | set as MLflow run tags |
| `log_artifacts_to` | Path \| null | `null` | also write `model.pt` here + log as artifact |

Every run automatically logs:
- all config params (`<section>.<field>` → value),
- the tag `gitcommithash` = current git HEAD,
- per-epoch metrics (`train_loss`, `train_*`, `val_*`),
- test metrics (`test_*`),
- the best model via `mlflow.pytorch.log_model`.

---

## `runtime`

| Field | Type | Default | Allowed values / notes |
|---|---|---|---|
| `device` | str | `auto` | `auto`, `cpu`, `cuda` |
| `cuda_device` | int \| null | `null` | requires `device` in `auto`/`cuda` |
| `mixed_precision` | str | `fp32` | `fp32`, `fp16` (AMP) |
| `seed` | int | `42` | DataLoader worker seeding |
| `num_workers` | int | `4` | *(not wired)* use `dataloader.num_workers` |

---

## Example: minimal

Only `dataset` is required; everything else defaults:

```yaml
dataset:
  dataset_variant: cholect50-small
  root_dir: /data/cholect50
```

## Example: phase classification (matches `configs/default.yaml`)

```yaml
project:
  output_dir: outputs
dataset:
  dataset_variant: cholect50-crossval
  root_dir: /data/cholect50
  test_fold: 1
  sequence_length: 1
  augmentations: [original]
model:
  name: resnet18
  params: {}
  num_classes: { phase: 7, tool: 6, verb: 10, target: 15, triplet: 100 }
training:
  epochs: 50
  optimizer: adamw
  learning_rate: 0.001
  loss: cross_entropy
  heads: [phase]
validation:
  metrics: [accuracy, accuracy_per_phase]
mlflow:
  tracking_uri: http://localhost:5000
  experiment_name: phase_classification
runtime:
  device: auto
  mixed_precision: fp32
```

## Example: multi-head (phase + tool + verb, frozen pretrained backbone) with fp16

```yaml
dataset:
  dataset_variant: cholect50-crossval
  root_dir: /data/cholect50
model:
  name: resnet18
  params: { pretrained: true, freeze_backbone: true, features: 512 }
  num_classes: { phase: 7, tool: 6, verb: 10 }
training:
  epochs: 80
  optimizer: adamw
  learning_rate: 0.0005
  heads: [phase, tool, verb]
dataloader:
  batch_size: 64
validation:
  metrics: [accuracy, f1_macro, f1_micro]
runtime:
  device: cuda
  mixed_precision: fp16
```
