# Metrics Specification

Metrics are computed by `src/phase_classification/metrics.py` and logged to
MLflow on every run. Which metrics are computed is set by `validation.metrics`
in the config (see `CONFIG.md`). Exactly what is reported depends on split
(training / validation / test), head, and the metric itself.

## How predictions and targets are derived

For every head in `training.heads`, each sample is reduced to a **single class
index**:

- **prediction** = `argmax(logits, dim=1)` — the highest-scoring class.
- **target** = `argmax(one-hot label, dim=1)` — the active class.

So all metrics below are single-label, argmax-based. A per-head confusion matrix
over the head's `num_classes` is accumulated across the whole split, then reduced.

> **Caveats.**
> - Multi-label heads (`tool`, `target`, `triplet`) carry more than one active
>   class per frame; argmax keeps only the first, so per-class metrics on those
>   heads underestimate the true value.
> - A class that never appears in a split contributes **0** to
>   `accuracy_per_phase` / `f1_macro` / `jaccard` (its TP/FN/FPs are all 0).
>   This lower-bounds these metrics when a class is absent from a small split.

## Available metrics

Referenced by name in `validation.metrics` (any order; the first listed is the
"primary val metric" used for best-checkpoint selection).

| Config name | Definition | Range | Notes |
|---|---|---|---|
| `accuracy` | `correct / total` | 0–1 | fraction of samples where `argmax(pred) == argmax(target)` |
| `accuracy_per_phase` | mean of per-class recall `tp/(tp+fn)` over all `n` classes | 0–1 | *balanced accuracy*; absent class → 0 |
| `f1_macro` | mean of per-class F1 `2·P·R/(P+R)` | 0–1 | each class weighted equally; absent class → 0 |
| `f1_micro` | `Σtp / (Σtp + 0.5·(Σfp + Σfn))` | 0–1 | **equals `accuracy`** for single-label per-sample targets |
| `jaccard` | mean of per-class IoU `tp/(tp+fn+fp)` | 0–1 | *mean intersection-over-union*; absent class → 0 |

Where `tp` = diagonal of the confusion matrix (correctly predicted per class),
`fp` = `col_sums − tp`, `fn` = `row_sums − tp`, all summed over classes when
averaging.

## How metrics are reported

### Naming

Each metric is reported **per head** and prefixed by the split:

```
<split>_<head>_<metric>

train_loss                          # training loss (epoch mean)
train_<head>_accuracy               # e.g. train_phase_accuracy
val_<head>_accuracy                 # e.g. val_phase_accuracy
test_<head>_accuracy                # e.g. test_phase_accuracy
```

For the default single-head phase model, SVG examples of keys you'll see in
MLflow:

- `train_loss`
- `train_phase_accuracy`, `train_phase_accuracy_per_phase`
- `val_phase_accuracy`, `val_phase_accuracy_per_phase`
- `test_phase_accuracy`, `test_phase_accuracy_per_phase`

### Where each split is logged (from `src/phase_classification/run.py`)

| Split | When | MLflow step |
|---|---|---|
| `train_*` | computed during each training epoch (argmax from the live training forward pass) | `step = epoch` |
| `val_*` | evaluated once per epoch on the validation split | `step = epoch` |
| `test_*` | evaluated once after training on the held-out test split (per-video datasets merged) | end of run |
| `train_loss` | mean of the aggregated per-head loss over the training epoch | `step = epoch` |

A single epoch therefore logs: `train_loss`, all `train_*`, and all `val_*`
metrics at that epoch's step; `test_*` metrics are logged once at run end.

## Example

Config:
```yaml
training:
  heads: [phase]
validation:
  metrics: [accuracy, accuracy_per_phase, f1_macro, f1_micro, jaccard]
```

Produced MLflow metrics (per epoch): `train_loss`,
`train_phase_{accuracy,accuracy_per_phase,f1_macro,f1_micro,jaccard}`,
`val_phase_{...}` ; plus `test_phase_{...}` at run end.