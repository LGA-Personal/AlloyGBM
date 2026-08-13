# PR #136 Top-k PL Histogram Construction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

| Date | Author | Production base | Status |
|---|---|---|---|
| 2026-08-12 | OpenAI Codex | `ea4df36` | Implemented; default promotion rejected by A/B evidence |

**Goal:** Add public `pl_split_candidates` control and use a standard-gain feature shortlist to
perform bounded, exhaustive PL split rescoring with reusable one-feature matrix-histogram storage.

**Architecture:** Standard scalar histograms produce the best overall split and an ordered numeric
feature shortlist. For ordinary linear-leaf training, each shortlisted feature is evaluated
sequentially with its own split-path regressor set in thread-local PL scratch; the winning feature's
linear leaves are solved from the same statistics. `pl_split_candidates=0`, MorphBoost, constant
leaves, and standard categorical winners retain their existing paths.

**Tech Stack:** Rust 1.92, edition 2024, Rayon, `wide` 0.7.33, PyO3 0.29, Python 3.11-3.13,
NumPy, scikit-learn, pytest, maturin, Sphinx, subprocess RSS measurement.

## Global Constraints

- Follow the approved design in
  `docs/reviews/2026-08-12-pr-136-top-k-pl-histogram-design.md`.
- Preserve `pl_split_candidates=0` artifacts byte for byte against production base `ea4df36`.
- Default `pl_split_candidates` to `0` on regressor, classifier, ranker, and `TrainParams` after
  the proposed `8` default failed the predeclared quality and fixed-round cost gates.
- Reject booleans, negative values, non-integral values, and native integer overflow.
- Do not add or change artifact sections, metadata fields, prediction code, objective formulas, PL
  solve formulas, model defaults other than the approved new parameter, or dependencies.
- Keep MorphBoost split selection and native categorical winner behavior unchanged.
- Evaluate every valid threshold and both missing directions for every shortlisted numeric feature.
- Use candidate-specific split-path regressors capped at `MAX_PL_REGRESSORS = 8`.
- Keep at most one PL histogram live per worker; reuse its allocation across shortlist candidates.
- Preserve deterministic winner and tie ordering independently of Rayon scheduling.
- Use no `unsafe`; workspace policy remains `unsafe_code = "forbid"`.
- Commit one logical change per task and retain rejected performance prototypes in machine-readable
  evidence.

---

## Implementation Outcome

The architecture was completed, including both tree growers, deterministic
parallel candidate evaluation, reusable one-feature scratch, public estimator
plumbing, and exact `k=0` compatibility. The proposed default of `8` was not
accepted: the five-seed matrix measured an 8.09x median fixed-round cost and
material regressions on multiple fixtures. A joint intercept/slope gain variant
also failed. With explicit user approval, `0` is therefore the production
default and positive values are documented experimental opt-ins. The wide-data
purpose of the shortlist remains demonstrated: `k=8` took 13.96% to 20.66% of
exhaustive `k=all` time across the 15 wide records.

---

### Task 1: Establish the benchmark contract and production baseline

**Files:**
- Create: `benchmarks/pl_topk_performance.py`
- Create: `benchmarks/tests/test_pl_topk_performance.py`
- Modify: `crates/backend_cpu/benches/histogram_kernels.rs`
- Create: `benchmarks/results/pr136_pl_topk_baseline.json`
- Create: `benchmarks/results/pr136_pl_histogram_baseline.txt`

**Interfaces:**
- Produces `PLTopKRecord`, `PLTopKComparison`, `full_specs()`, `run_record_subprocess()`,
  `compare_results()`, deterministic JSON readers/writers, and `run`/`compare` CLI commands.
- Production-base records use arm `legacy`; candidate records use `k0`, `k1`, `k8`, and `all`.
- Record identity is `(arm, dataset, task_family, shape, seed, rounds)`.

- [ ] **Step 1: Add failing benchmark-contract tests**

Create tests that import `benchmarks/pl_topk_performance.py` by path and pin:

```python
def test_full_specs_cover_required_shapes_and_tasks():
    specs = MODULE.full_specs()
    coverage = {(spec.shape, spec.task_family) for spec in specs}
    assert {
        ("small-narrow", "regression"),
        ("small-wide", "regression"),
        ("tall-narrow", "regression"),
        ("tall-wide", "regression"),
        ("medium", "binary"),
        ("small-wide", "multiclass"),
        ("tall-narrow", "ranking"),
    } <= coverage


def test_comparison_requires_exact_k0_production_digests():
    baseline, candidate = MODULE.synthetic_result_pair()
    candidate[0] = dataclasses.replace(candidate[0], artifact_sha256="different")
    summary = MODULE.compare_results(baseline, candidate)
    assert not summary.passed
    assert any("k0 artifact parity" in reason for reason in summary.reasons)


def test_default_quality_gate_rejects_one_percent_regression():
    baseline, candidate = MODULE.synthetic_result_pair(k8_quality_ratio=1.011)
    summary = MODULE.compare_results(baseline, candidate)
    assert not summary.passed
    assert any("quality" in reason for reason in summary.reasons)
```

Also pin duplicate/missing keys, non-finite values, metric direction, deterministic ordering,
subprocess RSS normalization, convergence, fixed-round cost, wide-feature shortlist benefit, and
the rejected-trial schema.

- [ ] **Step 2: Verify the contract tests fail**

Run:

```bash
/Users/lashby/Projects/AlloyGBM/.venv/bin/python -m pytest \
  benchmarks/tests/test_pl_topk_performance.py -q
```

Expected: fail because `benchmarks/pl_topk_performance.py` does not exist.

- [ ] **Step 3: Implement the benchmark harness without changing production routing**

Define fixed full fixtures for local-linear, raw-scale, nonlinear/noisy, binary, multiclass, and
ranking workloads. Each record runs in a fresh child process and records:

```python
@dataclass(frozen=True)
class PLTopKRecord:
    arm: str
    dataset: str
    task_family: str
    shape: str
    seed: int
    rounds: int
    primary_metric: str
    primary_value: float
    higher_is_better: bool
    secondary_metrics: dict[str, float]
    fit_seconds: float
    peak_rss_bytes: int
    rounds_completed: int
    prediction_sha256: str
    artifact_sha256: str
```

For `legacy`, omit `pl_split_candidates` so the harness runs on `ea4df36`. Candidate arms pass
`0`, `1`, `8`, or the fixture feature count. Use fit-only timing, deterministic `n_jobs=1`, five
seeds `0..4`, and checkpoint rounds `5, 10, 20, 40` for the two local-linear fixtures.

- [ ] **Step 4: Add parseable PL histogram storage cases to the Rust benchmark**

Add 8-, 32-, and 128-feature cases using the existing all-feature
`build_linear_histograms_cpu`. Print production base `ea4df36`, feature count, bin count, and
retained bundle bytes. Do not add shortlist production code yet.

- [ ] **Step 5: Run benchmark-contract tests**

Run:

```bash
/Users/lashby/Projects/AlloyGBM/.venv/bin/python -m pytest \
  benchmarks/tests/test_pl_topk_performance.py -q
cargo bench -p alloygbm-backend-cpu --bench histogram_kernels --no-run
```

Expected: pass.

- [ ] **Step 6: Capture immutable production evidence**

Create detached worktree `/tmp/alloygbm-pr136-pl-baseline` at `ea4df36`; copy only the benchmark
harness, its test, and benchmark fixture. Verify the production-source diff is empty, build the
release extension there, then run:

```bash
for run in 1 2 3 4 5 6 7; do
  cargo bench -p alloygbm-backend-cpu --bench histogram_kernels
done | tee benchmarks/results/pr136_pl_histogram_baseline.txt

/Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/pl_topk_performance.py run \
  --arms legacy --seeds 0 1 2 3 4 \
  --output /Users/lashby/Projects/AlloyGBM/.worktrees/pl-top-k-histograms/benchmarks/results/pr136_pl_topk_baseline.json
```

The JSON must record git head `ea4df36`, exact arguments, platform/package versions, and positive
fit/RSS values.

- [ ] **Step 7: Commit the benchmark contract and baseline**

```bash
git add -f benchmarks/pl_topk_performance.py \
  benchmarks/tests/test_pl_topk_performance.py \
  crates/backend_cpu/benches/histogram_kernels.rs \
  benchmarks/results/pr136_pl_topk_baseline.json \
  benchmarks/results/pr136_pl_histogram_baseline.txt
git commit -m "bench: establish top-k PL acceptance baseline"
```

### Task 2: Add the public `pl_split_candidates` contract

**Files:**
- Modify: `crates/core/src/config.rs`
- Modify: `crates/core/src/validation.rs`
- Modify: `crates/core/src/tests/main.rs`
- Modify: `bindings/python/src/params.rs`
- Modify: `bindings/python/src/train.rs`
- Modify: `bindings/python/alloygbm/_regressor/_core.py`
- Modify: `bindings/python/alloygbm/classifier.py`
- Modify: `bindings/python/alloygbm/ranker.py`
- Modify: `bindings/python/tests/test_pl_trees.py`
- Modify: `bindings/python/tests/test_sklearn_conformance.py`

**Interfaces:**
- Produces `TrainParams::pl_split_candidates: usize` with default `0`.
- Produces Python estimator attribute and constructor parameter `pl_split_candidates: int = 0`.

- [ ] **Step 1: Add failing Rust default and validation tests**

Pin `TrainParams::default().pl_split_candidates == 0` and a validation failure for an internal
value above `u32::MAX as usize`, the native bridge limit used for stable cross-platform conversion.

- [ ] **Step 2: Add failing Python estimator-contract tests**

For all three estimators, assert default `0`, explicit positive values, clone, `get_params`, `set_params`,
`repr`, and pickle retention. Assert `True`, `False`, `-1`, `1.5`, and `2**32` raise `ValueError`
mentioning `pl_split_candidates`.

- [ ] **Step 3: Verify focused failures**

```bash
cargo test -p alloygbm-core pl_split_candidates -- --nocapture
/Users/lashby/Projects/AlloyGBM/.venv/bin/python -m pytest \
  bindings/python/tests/test_pl_trees.py -k pl_split_candidates -q
```

Expected: fail because the field and constructor parameter do not exist.

- [ ] **Step 4: Implement Rust and Python parameter plumbing**

Append the field to `TrainParams`, add it to `build_train_params`, and thread it through every
single-output PyO3 training function. In Python use one validator:

```python
def _validate_pl_split_candidates(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError("pl_split_candidates must be an integer between 0 and 4294967295")
    result = int(value)
    if result < 0 or result > 0xFFFF_FFFF:
        raise ValueError("pl_split_candidates must be an integer between 0 and 4294967295")
    return result
```

Update constructor signatures, assignment, `_params_order`, `get_params`, `set_params`, `repr`,
fit bridge kwargs, and state restoration. Keep constant-leaf use valid and inert.

- [ ] **Step 5: Run focused and conformance tests**

```bash
cargo test -p alloygbm-core pl_split_candidates -- --nocapture
maturin develop --release
/Users/lashby/Projects/AlloyGBM/.venv/bin/python -m pytest \
  bindings/python/tests/test_pl_trees.py \
  bindings/python/tests/test_sklearn_conformance.py -q
```

Expected: pass.

- [ ] **Step 6: Commit the API contract**

```bash
git add crates/core/src/config.rs crates/core/src/validation.rs crates/core/src/tests/main.rs \
  bindings/python/src/params.rs bindings/python/src/train.rs \
  bindings/python/alloygbm/_regressor/_core.py bindings/python/alloygbm/classifier.py \
  bindings/python/alloygbm/ranker.py bindings/python/tests/test_pl_trees.py \
  bindings/python/tests/test_sklearn_conformance.py
git commit -m "feat: add PL split shortlist parameter"
```

### Task 3: Add deterministic standard split shortlisting

**Files:**
- Modify: `crates/engine/src/split_options.rs`
- Modify: `crates/engine/src/traits.rs`
- Modify: `crates/engine/src/lib.rs`
- Modify: `crates/backend_cpu/src/backend_ops.rs`
- Modify: `crates/backend_cpu/src/split_helpers.rs`
- Modify: `crates/backend_cpu/src/tests/main.rs`

**Interfaces:**
- Produces `SplitShortlist { best_overall, numeric_candidates }`.
- Produces `BackendOps::shortlist_standard_splits(..., max_numeric_features)`.
- Produces shared `feature_weighted_gain(&SplitCandidate, &[f32]) -> f32`.

- [ ] **Step 1: Add failing shortlist tests**

Build fixed histograms containing numeric and categorical features and assert:

```rust
let shortlist = backend.shortlist_standard_splits(
    &histograms,
    options,
    &feature_weights,
    &categorical_features,
    2,
)?;
assert_eq!(shortlist.numeric_candidates.len(), 2);
assert_eq!(shortlist.numeric_candidates[0].feature_index, expected_first);
assert_eq!(shortlist.best_overall, backend.best_split_with_options(
    &histograms, options, &feature_weights, &categorical_features
)?);
```

Cover `k=0`, `k>feature_count`, weighted ordering, material ties, missing directions, native
categorical overall winner, sequential/parallel feature thresholds, and stable repeated calls.

- [ ] **Step 2: Verify focused failures**

```bash
cargo test -p alloygbm-backend-cpu shortlist_standard -- --nocapture
```

Expected: fail because the trait method and result type do not exist.

- [ ] **Step 3: Implement shortlist types and CPU selection**

Add:

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct SplitShortlist {
    pub best_overall: Option<SplitCandidate>,
    pub numeric_candidates: Vec<SplitCandidate>,
}
```

Scan each feature exactly once with the existing numeric or categorical scanner. Preserve feature
iteration order, derive `best_overall` with `gain_materially_exceeds`, and select numeric entries
with a deterministic `O(F * k)` repeated-best extraction rather than a non-transitive tolerant
sort. Reuse a shared `feature_weighted_gain` helper for shortlist and final selection.

- [ ] **Step 4: Run backend and workspace tests**

```bash
cargo test -p alloygbm-backend-cpu shortlist_standard -- --nocapture
cargo test --workspace -q
```

Expected: pass with the existing best-split contract unchanged.

- [ ] **Step 5: Commit standard shortlisting**

```bash
git add crates/engine/src/split_options.rs crates/engine/src/traits.rs crates/engine/src/lib.rs \
  crates/backend_cpu/src/backend_ops.rs crates/backend_cpu/src/split_helpers.rs \
  crates/backend_cpu/src/tests/main.rs
git commit -m "feat: shortlist numeric features by standard gain"
```

### Task 4: Evaluate one shortlisted PL feature in reusable scratch

**Files:**
- Modify: `crates/engine/src/split_options.rs`
- Modify: `crates/engine/src/traits.rs`
- Modify: `crates/backend_cpu/src/pl_histogram.rs`
- Modify: `crates/backend_cpu/src/pl.rs`
- Modify: `crates/backend_cpu/src/backend_ops.rs`
- Modify: `crates/backend_cpu/src/tests/main.rs`

**Interfaces:**
- Produces `PreparedLinearSplit { split, left_leaf, right_leaf }`.
- Produces `BackendOps::evaluate_shortlisted_linear_feature(...)`.
- Produces CPU-local thread-local `LinearHistogramScratch` with one reusable bin vector.
- Produces slice-based PL scan/stat helpers wrapped by existing owned-histogram APIs.

- [ ] **Step 1: Add failing scratch and parity tests**

Tests must prove:

- scratch capacity grows to one feature histogram and does not grow with shortlist length;
- allocation is reused and isolated across Rayon workers;
- scratch recovers after a callback panic;
- one-feature evaluation matches `best_split_linear_for_feature` for fixed and randomized 2-255
  bin histograms, radii-free standard options, L1/L2, missing-left/right, tails, and invalid edges;
- solved leaves match `compute_linear_leaf_pair` from the owned oracle bundle;
- candidate-specific regressor identities and the eight-feature cap are preserved;
- ill-conditioned systems return no prepared PL candidate or finite scalar-correction fallback.

- [ ] **Step 2: Verify focused failures**

```bash
cargo test -p alloygbm-backend-cpu shortlisted_linear -- --nocapture
cargo test -p alloygbm-backend-cpu linear_histogram_scratch -- --nocapture
```

Expected: fail because the scratch and evaluation method do not exist.

- [ ] **Step 3: Refactor the PL scanner to accept borrowed bins**

Introduce an internal function with the exact behavior of the owned wrapper:

```rust
pub(crate) fn best_split_linear_for_bins(
    feature_index: u32,
    bins: &[LinearHistogramBin],
    node_id: u32,
    options: SplitSelectionOptions,
    ctx: &LinearContext,
) -> Option<SplitCandidate>;
```

Keep `best_split_linear_for_feature` as a wrapper and preserve its operation order, f64 Cholesky
guards, gain tolerance, and child statistics.

- [ ] **Step 4: Implement reusable one-feature accumulation and prepared leaves**

Add thread-local scratch, reset only its active bin slice, accumulate candidate-specific scaled
regressors, scan PL gain, derive the winning child statistics, and call `solve_pl_leaf` for the two
absolute leaves. Return owned leaves and split only; return the histogram storage to scratch before
the method exits.

- [ ] **Step 5: Run focused, mutation, and workspace tests**

```bash
cargo test -p alloygbm-backend-cpu shortlisted_linear -- --nocapture
cargo test -p alloygbm-backend-cpu linear_histogram_scratch -- --nocapture
cargo test -p alloygbm-backend-cpu pl::tests -- --nocapture
cargo test --workspace -q
```

Temporarily perturb one PL parent-gain subtraction in the borrowed scanner and verify an oracle test
fails; restore it and rerun green.

- [ ] **Step 6: Commit one-feature PL evaluation**

```bash
git add crates/engine/src/split_options.rs crates/engine/src/traits.rs \
  crates/backend_cpu/src/pl_histogram.rs crates/backend_cpu/src/pl.rs \
  crates/backend_cpu/src/backend_ops.rs crates/backend_cpu/src/tests/main.rs
git commit -m "perf: evaluate shortlisted PL features in reusable scratch"
```

### Task 5: Integrate top-k PL selection into both tree growers

**Files:**
- Modify: `crates/engine/src/trainer/tree_build.rs`
- Modify: `crates/engine/src/tests/main.rs`

**Interfaces:**
- Produces internal `SelectedNodeSplit { split, prepared_linear_leaf_pair }`.
- Produces one shared selection helper used by level-wise proposals and leaf-wise pending splits.

- [ ] **Step 1: Add failing engine integration tests**

Add deterministic fixtures proving:

- `k=0` matches a captured production artifact byte vector for level and leaf growth;
- opt-in `k=8` can select a different feature or threshold when PL gain favors it;
- `k>=eligible` matches a scalar exhaustive candidate-specific oracle;
- prepared leaves are used without invoking selected-partition PL accumulation again;
- categorical standard winner and MorphBoost bypass PL rescoring;
- interaction constraints, column sampling, feature weights, missing values, and path-regressor caps
  survive both growth modes;
- fallback to the standard winner occurs when all PL candidates are invalid;
- repeated `n_jobs=1` and `n_jobs=4` deterministic fits preserve their expected artifacts.

- [ ] **Step 2: Verify focused failures**

```bash
cargo test -p alloygbm-engine pl_shortlist -- --nocapture
```

Expected: fail because training still uses the legacy split dispatcher.

- [ ] **Step 3: Implement shared node selection**

Use the legacy dispatcher when leaves are constant, `k=0`, or Morph is active. Otherwise request
the standard shortlist. Return the standard winner immediately when absent or categorical. For each
numeric shortlist entry, compute path regressors, evaluate one PL feature, compare weighted gain in
shortlist order, and retain the winning prepared absolute leaf pair. Fall back to the standard
winner when no PL candidate survives.

- [ ] **Step 4: Thread prepared leaves through level-wise growth**

Use the prepared absolute pair after partition validation, apply the existing intercept clamp and
parent-relative weight subtraction, and skip `compute_linear_leaf_pair_from_partitions`. Legacy and
fallback selections keep the current direct solve unchanged.

- [ ] **Step 5: Thread prepared leaves through leaf-wise growth**

Extend `PendingSplit` to carry the optional prepared pair created when the root or child candidate
is queued. Consume it at split commit with the same delta conversion as level-wise growth. Do not
retain a matrix histogram in the heap.

- [ ] **Step 6: Run engine and workspace tests**

```bash
cargo test -p alloygbm-engine pl_shortlist -- --nocapture
cargo test --workspace -q
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Expected: pass.

- [ ] **Step 7: Commit engine integration**

```bash
git add crates/engine/src/trainer/tree_build.rs crates/engine/src/tests/main.rs
git commit -m "feat: rescore top-k PL split features"
```

### Task 6: Close Python behavior and compatibility coverage

**Files:**
- Modify: `bindings/python/tests/test_pl_trees.py`
- Modify: `bindings/python/tests/test_classifier_and_metrics.py`
- Modify: `bindings/python/tests/test_multiclass.py`
- Modify: `bindings/python/tests/test_ranker.py`
- Modify: `bindings/python/tests/test_quantile_objective.py`
- Modify: `bindings/python/tests/test_native_categorical_splits.py`
- Modify: `bindings/python/tests/test_morph.py`

**Interfaces:**
- Verifies the public behavior introduced in Tasks 2-5 through installed Python estimators.

- [ ] **Step 1: Add end-to-end Python cases**

Pin deterministic default/explicit `k=0`, opt-in `k=8`, and exhaustive values across regression, binary,
multiclass, ranking, quantile, raw-scale, missing-value, native categorical, Morph, level-wise, and
leaf-wise fixtures. Assert finite predictions, estimator state, persistence, and exact compatibility
digests where required.

- [ ] **Step 2: Build and run focused Python tests**

```bash
maturin develop --release
/Users/lashby/Projects/AlloyGBM/.venv/bin/python -m pytest \
  bindings/python/tests/test_pl_trees.py \
  bindings/python/tests/test_classifier_and_metrics.py \
  bindings/python/tests/test_multiclass.py \
  bindings/python/tests/test_ranker.py \
  bindings/python/tests/test_quantile_objective.py \
  bindings/python/tests/test_native_categorical_splits.py \
  bindings/python/tests/test_morph.py -q
```

Expected: pass.

- [ ] **Step 3: Commit Python integration coverage**

```bash
git add bindings/python/tests/test_pl_trees.py \
  bindings/python/tests/test_classifier_and_metrics.py bindings/python/tests/test_multiclass.py \
  bindings/python/tests/test_ranker.py bindings/python/tests/test_quantile_objective.py \
  bindings/python/tests/test_native_categorical_splits.py \
  bindings/python/tests/test_morph.py
git commit -m "test: cover top-k PL estimator behavior"
```

### Task 7: Capture candidate evidence and enforce acceptance gates

**Files:**
- Modify: `benchmarks/pl_topk_performance.py`
- Modify: `benchmarks/tests/test_pl_topk_performance.py`
- Modify: `crates/backend_cpu/benches/histogram_kernels.rs`
- Create: `benchmarks/results/pr136_pl_topk_candidate.json`
- Create: `benchmarks/results/pr136_pl_topk_comparison.json`
- Create: `benchmarks/results/pr136_pl_histogram_candidate.txt`

**Interfaces:**
- Produces machine-readable pass/fail for all eight design gates.

- [ ] **Step 1: Add candidate storage benchmark rows**

Print one-feature scratch capacity and bytes for the same 8/32/128-feature fixtures. Seven-run
median parsing must compare one-feature live bytes against the all-feature baseline bundle.

- [ ] **Step 2: Capture seven candidate native repetitions**

```bash
for run in 1 2 3 4 5 6 7; do
  cargo bench -p alloygbm-backend-cpu --bench histogram_kernels
done | tee benchmarks/results/pr136_pl_histogram_candidate.txt
```

- [ ] **Step 3: Capture the full candidate matrix**

```bash
/Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/pl_topk_performance.py run \
  --arms k0 k1 k8 all --seeds 0 1 2 3 4 \
  --output benchmarks/results/pr136_pl_topk_candidate.json
```

- [ ] **Step 4: Run the predeclared comparison**

```bash
/Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/pl_topk_performance.py compare \
  benchmarks/results/pr136_pl_topk_baseline.json \
  benchmarks/results/pr136_pl_topk_candidate.json \
  --output benchmarks/results/pr136_pl_topk_comparison.json
```

Require `passed: true`, exact default and k0 digests, memory/RSS pass, wide-feature shortlist
benefit, and deterministic evidence. Preserve opt-in quality and cost failures as observations.

- [ ] **Step 5: Optimize behavior-preservingly if a performance gate fails**

Try at most, in order:

1. hoist scaled path-regressor loads shared within one feature evaluation;
2. skip zero-padded matrix rows/columns above active `d` while preserving active f32 operation order;
3. reuse child/node scratch capacity more aggressively without retaining more than one feature;
4. avoid solving a prepared leaf pair until its PL gain becomes the current winner.

Do not change formulas, thresholds, or fixture budgets merely to pass a gate. Record each rejected
prototype under `rejected_trials` with commit, timings, memory, and rejection reason. The design's
explicit default-reconsideration clause permits retaining `0` when promotion evidence fails.

- [ ] **Step 6: Commit accepted evidence**

```bash
git add -f benchmarks/pl_topk_performance.py benchmarks/tests/test_pl_topk_performance.py \
  crates/backend_cpu/benches/histogram_kernels.rs \
  benchmarks/results/pr136_pl_topk_candidate.json \
  benchmarks/results/pr136_pl_topk_comparison.json \
  benchmarks/results/pr136_pl_histogram_candidate.txt
git commit -m "bench: verify top-k PL acceptance gates"
```

### Task 8: Document the feature and close the review finding

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/user/gbmregressor.md`
- Modify: `docs/site/source/estimator.rst`
- Modify: `docs/user/benchmarks.md`
- Modify: `docs/site/source/benchmarks.rst`
- Create: `docs/benchmarks/pl_topk_pr136.md`
- Modify: `docs/reviews/2026-07-02-v0.12.10-special-modes-resolutions.md`
- Modify: `docs/reviews/2026-08-12-pr-136-top-k-pl-histogram-design.md`
- Modify: `docs/reviews/2026-08-12-pr-136-top-k-pl-histogram-implementation-plan.md`

**Interfaces:**
- User docs state default `0`, positive opt-in behavior, clipping, Morph/categorical fallbacks, and the
  measured quality/memory/time tradeoff.
- Resolution marks only the top-k PL histogram finding fixed.

- [ ] **Step 1: Write the evidence report**

Document production/candidate commits, host details, versions, fixture definitions, all gates,
per-case metrics, convergence checkpoints, memory/RSS, fixed-round cost, exhaustive comparison,
rejected trials, and exact reproduction commands. State plainly that current `main` was already a
top-1 standard-split/direct-solve path and that this PR trades bounded extra training work for
PL-aware split structure.

- [ ] **Step 2: Update public and review documentation**

Add the parameter to README and estimator docs, mirror Markdown/RST behavior, link the evidence from
both benchmark indexes, update the unreleased changelog, and mark the special-modes PL memory item
fixed only if comparison JSON passes.

- [ ] **Step 3: Mark design and plan implemented**

Set both document status rows to `Implemented; gates passed; ready for review` and check every
completed plan step except PR merge.

- [ ] **Step 4: Commit documentation closure**

```bash
git add README.md CHANGELOG.md docs/user/gbmregressor.md \
  docs/site/source/estimator.rst docs/user/benchmarks.md docs/site/source/benchmarks.rst \
  docs/benchmarks/pl_topk_pr136.md \
  docs/reviews/2026-07-02-v0.12.10-special-modes-resolutions.md \
  docs/reviews/2026-08-12-pr-136-top-k-pl-histogram-design.md \
  docs/reviews/2026-08-12-pr-136-top-k-pl-histogram-implementation-plan.md
git commit -m "docs: close top-k PL histogram finding"
```

### Task 9: Run final verification and open draft PR #136

**Files:**
- Verify only; modify prior task files only if a failure reveals a scoped defect.

**Interfaces:**
- Produces a clean pushed branch and draft PR; does not merge.

- [ ] **Step 1: Run complete verification**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
maturin develop --release
/Users/lashby/Projects/AlloyGBM/.venv/bin/python -m pytest \
  bindings/python/tests/ benchmarks/tests/ -q
/Users/lashby/Projects/AlloyGBM/.venv/bin/sphinx-build -W -b html \
  docs/site/source /tmp/alloygbm-pr136-sphinx
/Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/pl_topk_performance.py compare \
  benchmarks/results/pr136_pl_topk_baseline.json \
  benchmarks/results/pr136_pl_topk_candidate.json \
  --output /tmp/pr136_pl_topk_comparison.json
git diff --check ea4df36...HEAD
git status --short
```

Expected: all commands pass, comparison reports `passed: true`, and the worktree is clean.

- [ ] **Step 2: Inspect compatibility and scope**

```bash
git diff ea4df36...HEAD -- crates/core/src/artifact_format.rs crates/core/src/leaf.rs \
  crates/predictor/src
rg -n 'unsafe|std::env|experimental|approximate' \
  crates/backend_cpu/src/pl.rs crates/backend_cpu/src/pl_histogram.rs \
  crates/engine/src/trainer/tree_build.rs
git diff --stat ea4df36...HEAD
```

Expected: no artifact/predictor diff, no runtime switch or approximation, and changes remain within
the approved architecture.

- [ ] **Step 3: Push and create draft PR #136**

Push `codex/pl-top-k-histograms` and create a draft PR against `main`. The body must include API
semantics, compatibility mode, shortlist and scratch architecture, categorical/Morph fallbacks,
all acceptance results, full verification counts, and links to design, plan, evidence, and
resolution documents. Preserve the worktree and stop before merge.
