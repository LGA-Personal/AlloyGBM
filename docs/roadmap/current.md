# Current Roadmap

## Direction

AlloyGBM is a Rust-first gradient boosting system with Python bindings, supporting regression, binary and multi-class classification, and learning-to-rank. It is aimed at strong practical performance on structured tabular workloads, with particular strength on financial and time-aware problems.

The `0.7.2` release is documentation, supply-chain, and repo-hygiene
only — no user-facing Python API changes.  It aligns the docs with
the v0.7.1 surface that actually shipped, hardens CI (full pytest
suite gated on every PR, `cargo-audit` + `cargo-deny` weekly), adds an
`examples/` library, and rewrites `docs/reference/release_checklist.md`
as a top-to-bottom operating manual.

The `0.7.1` release built on the v0.7.0 factor-neutral boosting surface
with five additions: SHAP support for piecewise-linear leaves, per-round
training diagnostics on every estimator, neutralized warm-start (with a
matching-exposures contract), LightGBM-compatible feature interaction
constraints, and `MultiLabelGBMRanker` for multi-output ranking.

The `0.7.0` release introduced factor-neutral boosting, with fit-time factor
exposures, pre-target residualization, per-round gradient projection, and an
optional split exposure penalty. The `0.6.0` release introduced
`leaf_solver="dro"`, a conservative DRO-style scalar leaf solver that penalizes
within-leaf gradient uncertainty while preserving standard prediction-time
artifacts. The `0.5.0` release introduced piecewise-linear (PL) tree leaves via
`leaf_model="linear"` on all three estimators. The `0.4.0` release introduced
the opt-in MorphBoost adaptive split criterion, per-iteration learning-rate
schedules, and SIMD-accelerated histogram and EMA kernels.

## What Shipped In 0.7.2

Documentation, supply-chain, and repo-hygiene release.  No user-facing
Python API surface changes.

- **Doc accuracy.**  Multiple docs that still claimed warm-start was
  rejected, SHAP required `leaf_model="constant"`, interaction
  constraints did not exist, or rankers were single-label only — even
  though v0.7.1 shipped all four — are now consistent with the
  actual API.  Touches README, `docs/user/*.md`, the Sphinx mirror
  under `docs/site/source/*.rst`, `docs/roadmap/current.md`,
  `CLAUDE.md`, `AGENTS.md`, and `benchmarks/README.md`.
- **Release operating manual.**
  `docs/reference/release_checklist.md` is now the authoritative
  inventory of version-pin files, content updates, audit `git grep`
  queries, verification matrix, tag/publish commands, and
  post-release bookkeeping.
- **Runnable examples.**  New `examples/` directory with 8 end-to-end
  scripts covering every public estimator and feature.
- **CI now runs the full pytest suite.**  v0.7.1 built the wheel and
  ran 7 smoke snippets but never invoked
  `pytest bindings/python/tests/` — meaning the 455-test Python suite
  was not enforced on merge.
- **Cargo.lock tracked**, `maturin` pinned in `publish.yml`,
  `cargo-audit` + `cargo-deny` weekly + on every Cargo-manifest PR,
  coverage reporting via Codecov, `publish = false` on every workspace
  crate.
- **Repo metadata.**  `CONTRIBUTING.md`, `SECURITY.md`, GitHub issue /
  PR / CODEOWNERS / Dependabot configs, `.editorconfig`,
  `requirements-dev.txt`, README badges.

## What Shipped In 0.7.1

- **SHAP for piecewise-linear leaves** — `shap_values()` accepts
  `leaf_model="linear"` artifacts and returns an interventional
  decomposition (path-attributed leaf "constant part" plus per-leaf
  row deviations); global feature means are persisted in a new
  `FeatureBaseline` artifact section so SHAP is self-contained at
  explain time.
- **Per-round training diagnostics** — every estimator exposes
  `diagnostics_per_round_`: gradient L2 norm / variance, hessian L2
  norm, sampling counts, and (when factor neutralization is active)
  the `neutralization_effectiveness` score in `[0, 1]`.
- **Neutralized warm-start** — `init_model` / `warm_start=True` works
  across `pre_target`, `per_round_gradient`, and `split_penalty`
  provided the caller supplies the same `factor_exposures` matrix used
  for the initial fit; mode + lambda + (where applicable) penalty must
  match.
- **Feature interaction constraints** — LightGBM-compatible
  `interaction_constraints=[[…]]` on every estimator, up to 64 groups
  per fit, enforced in both level-wise and leaf-wise tree builders.
- **`MultiLabelGBMRanker`** — unified multi-output ranking estimator:
  `y` shaped `(n_rows, n_labels)`, `predict` returns the same shape.
  Trains one independent `GBMRanker` per label sharing `group` and
  `factor_exposures`; supports per-label `ranking_objective` lists.

## What Shipped In 0.7.0

- **Factor-neutral boosting** via `neutralization` on `GBMRegressor`,
  `GBMClassifier`, and `GBMRanker`, with row-aligned fit-time
  `factor_exposures`.
- **Per-round gradient projection** via `neutralization="per_round_gradient"`,
  projecting objective gradients away from user-supplied factors before each
  boosting round. Multiclass classification projects each class-gradient column
  independently.
- **Pre-target residualization** via `neutralization="pre_target"` for built-in
  squared-error `GBMRegressor` training. Classification, ranking, custom
  objectives, and validation sets are rejected for this mode in 0.7.0.
- **Split exposure penalty** via `neutralization="split_penalty"` and
  `factor_penalty`, compatible with constant leaves, DRO leaves, and
  MorphBoost. Piecewise-linear leaves are rejected for split-penalty mode in
  0.7.0.
- **Benchmark coverage**: `alloygbm_factor_neutral` and
  `alloygbm_factor_neutral_dro` arms were added to the comparative benchmark
  runner. Synthetic benchmark factors are smoke/stability checks unless callers
  provide domain factor exposures explicitly.

## What Shipped In 0.6.0

- **DRO-style scalar leaves** via `leaf_solver="dro"` on `GBMRegressor`,
  `GBMClassifier`, and `GBMRanker`. This is a fast closed-form robust Newton
  update over leaf gradient uncertainty, exposed with `dro_radius` and
  `dro_metric="wasserstein"`.
- **Conservative contract**: default `leaf_solver="standard"` preserves existing
  behavior; `dro_radius=0.0` preserves standard predictions while recording
  optional DRO metadata; the DRO solver does not claim full raw-distribution
  Wasserstein DRO guarantees.
- **Interactions**: `leaf_solver="dro"` composes with `training_mode="morph"`
  and requires `leaf_model="constant"` for this release.
- **Benchmark support**: `alloygbm_dro` was added to the comparative benchmark
  runner, with temporal/panel stability reporting focused on mean, worst, and
  standard deviation of task-normalized score.

## What Shipped In 0.5.0

- **Piecewise-linear (PL) tree leaves** via `leaf_model="linear"` on `GBMRegressor`,
  `GBMClassifier`, and `GBMRanker`. Each leaf stores a small linear model
  `f_s(x) = b_s + Σ α_j x_j` whose weights are solved in closed form via the
  ridge regression `α* = -(XᵀHX + λI)⁻¹ Xᵀg`, using the same L2 regularizer
  (`lambda_l2`) as the split criterion. Benchmarks show:
  - ~10× faster convergence on linearly-structured datasets (fewer rounds to reach
    the same RMSE)
  - +3.5% RMSE improvement on California Housing vs constant leaves
  - +1.75pp accuracy improvement on Breast Cancer classification
  - 2–8× training time overhead (Cholesky solve per node)
- **New artifact section** (`ModelSectionKind::LinearLeafCoefficients`) stores
  per-stump linear leaf data; backward-compatible with v0.4.0 artifacts
- **`alloygbm_linear` benchmark arm** in `run_model_comparison.py`; new
  `benchmarks/pl_trees_benchmark.py` script with convergence-curve and λ-sweep
  analysis; report at `docs/benchmarks/pl_trees_v1.md`
- Categorical-native splits continue to use constant leaves when
  `max_cat_threshold > 0`; descendant leaves below a categorical root node use
  linear leaves on all remaining numeric regressors

## What Shipped In 0.4.0

- **MorphBoost adaptive training mode** (`training_mode="morph"`) on `GBMRegressor`, `GBMClassifier`, and `GBMRanker`. Implements the criterion from [Kriuk (2025)](https://arxiv.org/pdf/2511.13234) with EMA-driven gain shaping, depth/iteration leaf penalties, balance penalty, and an information-theoretic blend term ramped in via `tanh(iter/20)` warmup
- **Per-iteration learning-rate schedules** via the new `lr_schedule` parameter (`"constant"` default or `"warmup_cosine"`); schedule-aware auto early-stopping logic so warmup-phase rounds aren't classified as stalled
- **MorphBoost configuration persisted in artifacts** as an optional section so loaded models predict consistently
- **SIMD acceleration** via the `wide` crate (safe API, AVX2/NEON internally, scalar fallback): histogram bin-scan and EMA mean+variance pass are vectorized
- **Tile-size auto-tuning** for histogram parallelism on high-feature workloads (~2 tiles per thread, clamped to `[16, 64]`)
- **`alloygbm_morph` / `alloygbm_morph_cosine` benchmark arms** in `run_model_comparison.py`; new `--models` filter; new `morph_report.py`, `morph_ablation.py`, and updated `numerai_benchmark.py` harnesses (with build-freshness self-check at startup)
- **Dedicated MorphBoost user guide** at `docs/user/morphboost.md` (and Sphinx mirror) plus cross-references across all estimator docs and READMEs

## What Shipped In 0.3.2

- Fixed GBMRanker silent zero-tree training: the auto training policy's density-based `min_split_gain` floor and `min_loss_improvement` floor were being applied to ranking objectives, which have gradient magnitudes an order of magnitude smaller than regression/classification — no split cleared the floor and training exited on round 1. The auto policy is now objective-aware and skips those floors for all ranking objectives.
- Fixed training loop loss-regression early break firing on ranking objectives where round-to-round loss oscillation is expected and benign
- Fixed `inspect.signature(GBMRanker.__init__)` returning only 3 parameters (`self`, `ranking_objective`, `**kwargs`) — parameter-building tools (sklearn clone, benchmarks, IDEs) using signature introspection silently trained with `n_estimators=6` default; now exposes the full parameter set
- Added `stop_reason_` and `rounds_completed_` attributes on all estimators (`GBMRegressor`, `GBMClassifier`, `GBMRanker`) for training diagnostics
- Added `california_ranking` benchmark scenario: California Housing reframed as learning-to-rank with geographic grid cells as queries and median house value bucketed into 5 graded relevance levels (~44 queries × 468 docs)

## What Shipped In 0.3.1

- Fixed multiclass predictor threshold conversion: `class_trees` are now converted in all three threshold-conversion paths (linear, quantile, pre-binned); continuous-feature multiclass models now produce correct predictions
- Fixed multiclass benchmark argmax label mapping: `model.classes_` is now used so accuracy is correct for non-zero-indexed labels
- Added real-dataset benchmark scenarios: `wine_multiclass`, `digits_multiclass`, `adult_income`, `abalone_regression`
- Added `news_ranking` placeholder scenario with dataset selection instructions
- Activated `synthetic_multiclass` and `synthetic_categorical` benchmark scenarios
- Rewrote `benchmarks/README.md` with scenario table, feature coverage matrix, timing reference, and usage examples

## What Shipped In 0.3.0

- Native categorical splits with Fisher-sort algorithm and bitset-based O(1) prediction (`max_cat_threshold`)
- Multi-class classification (`GBMClassifier` with softmax/multinomial for K > 2 classes)
- Custom objective functions (`objective=callable`) with fast numpy I/O
- Custom evaluation metric callbacks (`eval_metric=callable`) with early stopping support
- Synthetic categorical and custom objective benchmark scenarios

## What Shipped In 0.2.0

- Binary classification (`GBMClassifier`) with log-loss objective
- Learning-to-rank (`GBMRanker`) with 5 objectives (RankNet, LambdaMART, XE-NDCG, QueryRMSE, YetiRank)
- NaN / missing value support across all crates
- Sample weight and group ID support from Python
- Model persistence (pickle, save/load, artifact export)
- Feature name capture and sklearn compatibility (`BaseEstimator`, `RegressorMixin`, `ClassifierMixin`)
- TreeSHAP (polynomial-time, replaces the old 25-feature-capped brute-force method)
- Monotone constraints and feature importance weighting
- Leaf-wise (best-first) tree growth strategy
- Warm-starting / incremental training
- Up to 65,535 bins per feature (up from 256)
- Multiple categorical column support
- Histogram buffer reuse
- Objective-aware training metric tracking
- Expanded benchmark suite (regression + classification + ranking)

## Current Priorities

1. Close remaining performance gaps on broad tabular datasets.
2. Explore GPU/accelerator backend after the CPU baseline is solid enough to serve as reference.
3. Continue expanding the benchmark suite with real-world classification and ranking datasets.

## Longer-Term Themes

- Joint shared-tree multi-label ranking (one ensemble updating all label
  predictions simultaneously) — the v0.7.1 `MultiLabelGBMRanker` is a
  K-independent-rankers wrapper; a shared-tree engine is a v0.7.2+ follow-up.
- Path-walk alignment between SHAP and the predictor for piecewise-linear
  leaves (so strict additivity holds on continuous-feature artifacts).
- MorphBoost EMA snapshot persisted in the warm-start artifact so resumed
  training does not restart the EMA cold.
- Dart / GOSS boosting modes.
- GPU backend.

## Planning Style

The project no longer uses the old version-layer planning hierarchy as the active documentation model.

Going forward:

- current intent lives in `docs/roadmap/`
- research notes live in `docs/ideas/`
- benchmark framing lives in `docs/benchmarks/` and `benchmarks/`
- implementation plans from the 0.1.x cycle are archived in `docs/archive/v0.1_plans/`
