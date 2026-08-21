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

> **Temporal models** (`convnext-ms-tcn`): logits are per-frame `(B, L, K)`. The
> accumulator flattens `(B, L, K) -> (B·L, K)` so every metric is computed **per
> frame**, and `edit_score` runs over the full ordered per-frame sequence — the
> standard action-segmentation evaluation.

> **Caveats.**
> - Multi-label heads (`tool`, `target`, `triplet`) carry more than one active
>   class per frame; argmax keeps only the first, so per-class metrics on those
>   heads underestimate the true value.
> - A class that never appears in a split contributes **0** to every aggregate
>   and per-class metric below (its TP/FN/FPs are all 0). This lower-bounds the
>   aggregates when a class is absent from a small split.

## Available metrics

Referenced by name in `validation.metrics` (any order; the first listed is the
"primary val metric" used for best-checkpoint selection).

Where `tp` = diagonal of the confusion matrix (correctly predicted per class),
`fp` = `col_sums − tp`, `fn` = `row_sums − tp`.

### Aggregate metrics

| Config name | Definition | Range | Notes |
|---|---|---|---|
| `jaccard` | mean of per-class IoU `tp/(tp+fn+fp)` | 0–1 | mean intersection-over-union; absent class → 0 |
| `f1_macro` | mean of per-class F1 `2·P·R/(P+R)` | 0–1 | F1@Macro; each class weighted equally; absent class → 0 |
| `mean_accuracy` | mean of per-class accuracy `tp/(tp+fn)` over all `n` classes | 0–1 | *balanced accuracy*; absent class → 0 |
| `edit_score` | `1 − Levenshtein(pred, target) / max(len(pred), len(target))` | 0–1 | **sequence-level**; higher = closer order/transitions (see below) |

### Per-class metrics

| Config name | Definition | Emits (per head) |
|---|---|---|
| `f1_per_class` | per-class F1 `2·P·R/(P+R)` | `<head>_f1_class_<i>` for `i` in `0..num_classes-1` |
| `accuracy_per_class` | per-class accuracy `tp/(tp+fn)` | `<head>_accuracy_class_<i>` for `i` in `0..num_classes-1` |

Per-class metrics are useful for inspecting model behaviour on each phase/tool
individually; each class contributes `0` when it is absent from the split.

### `edit_score` (sequence-level)

Unlike the confusion-matrix metrics, `edit_score` compares the **ordered
sequence** of predicted class indices against the ordered ground-truth sequence
using the Levenshtein distance (minimum insertions + deletions + substitutions
to turn one sequence into the other). It is normalised by the longer sequence and
inverted, giving `1` for an exact order match and `0` for a complete mismatch —
a higher score means the model better captures the correct order and transitions
of action segments.

Implementation (`metrics.py:_levenshtein`, Wagner–Fischer DP):
- Per-sample predicted and target class indices are accumulated across batches
  **in order** and concatenated into one sequence per head per split.
- The DP is `O(n·m)` in the sequence lengths, which may be costly on very long
  concatenated splits.
- **Caveats.**
  - `edit_score` is only temporally meaningful when samples arrive in sequence
    order. `val`/`test` loaders use `shuffle=False`, so it holds there. The
    `train` split shuffles, so `train_*_edit_score` is *not* a meaningful
    temporal metric (still computed/reported).

## How metrics are reported

### Naming

Each metric is reported **per head** and prefixed by the split:

```
<split>_<head>_<metric>

train_loss                          # training loss (epoch mean)
train_<head>_jaccard                # e.g. train_phase_jaccard
val_<head>_f1_macro                 # e.g. val_phase_f1_macro
val_<head>_f1_class_0               # per-class F1 for class 0 of that head
test_<head>_accuracy_class_1        # per-class accuracy for class 1
```

For the default single-head phase model (7 phases), full examples of keys you'll
see in MLflow:

- `train_loss`
- `train_phase_{jaccard,f1_macro,mean_accuracy,edit_score}`
- `train_phase_f1_class_0..6`, `train_phase_accuracy_class_0..6`
- `val_phase_{jaccard,f1_macro,mean_accuracy,edit_score}` + per-class variants
- `test_phase_{jaccard,f1_macro,mean_accuracy,edit_score}` + per-class variants

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
  metrics: [jaccard, f1_macro, f1_per_class, mean_accuracy, accuracy_per_class, edit_score]
```

Produced MLflow metrics (per epoch): `train_loss`,
`train_phase_{jaccard,f1_macro,mean_accuracy,edit_score}` plus
`train_phase_f1_class_<i>` and `train_phase_accuracy_class_<i>` for each class,
and the `val_phase_*` equivalents; `test_phase_*` variants at run end.