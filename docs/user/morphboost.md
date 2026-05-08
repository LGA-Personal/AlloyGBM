# MorphBoost (Adaptive Split Criterion)

MorphBoost is an opt-in training mode in AlloyGBM that augments the standard
gradient-gain split criterion with a normalized information-theoretic term,
plus several round-aware leaf adjustments. It is implemented in pure Rust
inside the existing engine; enabling it does not change the artifact format
in a backward-incompatible way.

The implementation follows the formulation in
[Kriuk (2025), *MorphBoost*](https://arxiv.org/pdf/2511.13234), with a few
deliberate corrections vs. the paper's reference code (the warmup blend
actually engages, leaf magnitudes are clamped consistently, etc.).

## When To Use It

MorphBoost tends to help most on:

- Tabular problems with low signal-to-noise ratio (financial residuals,
  Numerai-style returns, etc.) where the standard gain criterion can
  overfit to spurious local-best splits
- Workloads where you want the model to find structure that a
  pure-gradient-gain learner misses early in training
- Production setups where you want the optimizer to be relatively robust
  to hyperparameter choice — MorphBoost's defaults are reasonable across
  a wide range of `n_estimators` and `max_depth`

Empirically, AlloyGBM's MorphBoost mode delivers higher MMC (Meta-Model
Contribution) on Numerai validation than the auto mode and the peer
libraries (LightGBM, XGBoost, CatBoost) — see
[benchmarks.md](benchmarks.md).

It is not necessarily faster or more accurate on every dataset; on very
clean tabular data the standard mode can match it. Treat MorphBoost as a
configuration to A/B against `training_mode="auto"` rather than a strict
upgrade.

## How It Works (At A Glance)

For every candidate split, the gain is

```
gradient_score = standard XGBoost-style gradient gain
info_score     = normalized information-gain term over the partition
morph_weight   = tanh(iteration / 20)            # ramps in over training

gain = (1 - info_score_weight) * gradient_score
     +  info_score_weight * info_score * morph_weight
     +  optional balance penalty
```

In addition:

- A per-class EMA over gradient statistics tracks recent training dynamics
  and shapes split selection during evaluation.
- Leaf values are scaled by a depth-based penalty
  (`depth_penalty_base ** (depth / 3)`) and a per-iteration shrinkage
  (`1 - morph_rate * progress`) so deeper / later trees contribute less.
- An optional balance penalty discourages highly imbalanced splits
  (e.g. 99/1 partitions).

A learning-rate schedule can also be activated independently of MorphBoost
(see `lr_schedule` below).

## Enabling It

Pass `training_mode="morph"` to any AlloyGBM estimator. The same parameter
exists on `GBMRegressor`, `GBMClassifier`, and `GBMRanker`.

```python
from alloygbm import GBMRegressor

model = GBMRegressor(
    n_estimators=1200,
    max_depth=6,
    learning_rate=0.05,
    training_mode="morph",
    seed=7,
)
model.fit(X_train, y_train)
```

`training_mode` accepts `"auto"` (default), `"manual"`, or `"morph"`.

## Parameters

All MorphBoost-related parameters are exposed as top-level keyword arguments
on the estimator; the table below notes any mode-specific behavior.

| Parameter | Default | Description |
|---|---|---|
| `morph_rate` | `0.1` | Per-iteration leaf shrinkage rate. Larger values shrink late-round leaves more aggressively. Range `[0.0, 1.0]`. |
| `evolution_pressure` | `0.2` | Strength of EMA-driven gain shaping. Range `[0.0, 1.0]`. |
| `morph_warmup_iters` | `5` | Number of initial rounds for which the morph blend collapses to the pure gradient gain. Helps stabilize early training. |
| `info_score_weight` | `0.3` | Mixing weight between the gradient-gain term and the information-theoretic term in the post-warmup blend. Range `[0.0, 1.0]`. Setting to `0.0` disables the info term entirely (gradient-gain only). |
| `depth_penalty_base` | `0.9` | Base of the depth penalty applied to leaf magnitudes (`depth_penalty_base ** (child_depth / 3.0)`). Range `(0.0, 1.0]`. `1.0` disables the penalty. |
| `balance_penalty` | `True` | Whether to apply a small penalty to highly imbalanced splits. |
| `lr_schedule` | `"constant"` | Per-iteration LR schedule. One of `"constant"` or `"warmup_cosine"`. Independent of MorphBoost — usable on its own. |
| `lr_warmup_frac` | `0.1` | Fraction of `n_estimators` to spend in the linear-warmup phase when `lr_schedule="warmup_cosine"`. Range `[0.0, 1.0]`. Must be left at the default `0.1` when `lr_schedule="constant"`; non-default values with a constant schedule raise `ValueError`. |

The defaults match the values recommended by the MorphBoost paper.

## Learning-Rate Schedules

`lr_schedule` is a separate, independent feature. It applies regardless of
`training_mode`. Two schedules are currently supported:

- `"constant"` (default) — single fixed `learning_rate` for all rounds.
- `"warmup_cosine"` — linear warmup from a small fraction of `learning_rate`
  up to `learning_rate` over the first `lr_warmup_frac * n_estimators`
  rounds, then half-cosine decay to a floor of `0.01 * learning_rate` over
  the remainder.

The warmup-cosine schedule is most useful at very low `learning_rate` and
high `n_estimators` (e.g. `n_estimators=5000`, `learning_rate=0.01`). At
ordinary settings (e.g. `n_estimators=300`, `learning_rate=0.05`), the
constant schedule is usually fine.

```python
model = GBMRegressor(
    n_estimators=5000,
    learning_rate=0.01,
    training_mode="morph",
    lr_schedule="warmup_cosine",
    lr_warmup_frac=0.1,
)
```

When a non-constant LR schedule is active, AlloyGBM's auto-stopping logic
becomes schedule-aware:

- The auto-tuned `min_loss_improvement` threshold is scaled by
  `(current_lr / max_lr)` so warmup rounds aren't classified as "stalled".
- Empty or slightly-negative-improvement rounds during the explicit warmup
  phase do not terminate training; they are skipped.
- After warmup, normal early-stopping logic resumes (loss going up still
  hard-fails, etc.).

This means the schedule is safe to use even with `n_estimators=5000` —
training won't terminate after a handful of warmup rounds.

## Tuning Notes

- Start from the defaults. They are not aggressive.
- If you suspect the info term is dominating, reduce `info_score_weight` to
  `0.1` or `0.2`.
- If your trees are growing too aggressively at depth, lower
  `depth_penalty_base` from `0.9` to `0.85`.
- If the morph blend is engaging too quickly on a short run
  (`n_estimators < 100`), increase `morph_warmup_iters`.
- For long, low-LR runs (`n_estimators >= 1500`, `learning_rate <= 0.02`),
  try `lr_schedule="warmup_cosine"` — empirically improves stability.

## Combining With Piecewise-Linear Leaves

`training_mode="morph"` composes with `leaf_model="linear"`. The MorphBoost
gain criterion is used for split selection, and each resulting leaf still
fits a closed-form linear model via the ridge solve. Pair with
`lambda_l2 >= 0.01` for weight stability. See
[GBMRegressor — Piecewise-Linear Leaves](gbmregressor.md#piecewise-linear-leaves).

## Persistence

Models trained with `training_mode="morph"` save and load identically to
auto-mode models — pickle, `save_model` / `load_model`, and raw artifact
export all work without any extra steps. The morph configuration is
embedded as an optional artifact section so loaded models predict
consistently.
