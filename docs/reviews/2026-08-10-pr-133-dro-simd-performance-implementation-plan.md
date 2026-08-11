# PR #133 DRO SIMD Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

| Date | Reviewer | Version reviewed | Commit | Status |
|---|---|---|---|---|
| 2026-08-10 | OpenAI Codex | `main` after PR #132 | `2b2e3ef` | Implemented; ready for review |

**Goal:** Close special-modes review section 3.1 by making active scalar-path DRO numeric split
selection materially faster without changing its formula, defaults, public API, artifact format, or
exhaustive search semantics.

**Architecture:** Add a dedicated numeric DRO scanner beside `morph_scan.rs`. It reuses the
thread-local cumulative split scratch, adds cumulative gradient-square scratch, and evaluates every
valid threshold in safe four-lane f64 SIMD so the variance and square-root calculation follows
`leaf_effective_gradient`'s f64 contract. The existing scalar scanner remains the correctness oracle
and fallback for native categorical, factor-penalized, joint-output, and Morph+DRO combinations.

**Tech Stack:** Rust 1.92, edition 2024, `wide` 0.7.33 safe SIMD, Rayon, PyO3, Python 3.11-3.13,
NumPy, scikit-learn metrics/datasets, pytest, maturin, and Sphinx.

## Global Constraints

- Preserve exhaustive evaluation of every valid numeric threshold and both missing directions.
- Preserve the exact DRO formula and operation boundaries in `leaf_effective_gradient`: variance is
  computed in f64, clamped to zero, the radius contribution is cast to f32, then added to the
  non-negative f32 L1 threshold before soft thresholding.
- Use no `unsafe`; workspace policy remains `unsafe_code = "forbid"`.
- Add no dependency, public parameter, environment switch, histogram plane, artifact section, or
  model metadata field.
- Reuse the existing gradient-square histogram data and thread-local scratch. Do not allocate per
  feature, threshold, or missing direction.
- Keep active DRO native categorical splits scalar because Fisher sorting and bitset construction are
  not the numeric scanner reviewed in section 3.1.
- Keep factor-penalized DRO scalar because factor prefix state is not represented by the DRO SIMD
  scanner.
- Keep Morph+DRO scalar in this PR. It blends additional Morph terms and was explicitly retained as
  a fallback by PR #132.
- Keep joint multi-output DRO leaf-only; do not add gradient-square planes to shared histograms.
- Treat `dro_config=None` and `dro_config.radius == 0.0` as standard-gain fast-path cases.
- Compare SIMD gain with the scalar oracle using
  `abs_error <= max(1e-5, 1e-5 * abs(scalar_gain))`; require the same winner, missing direction, and
  child statistics unless scalar leaders tie under `gain_materially_exceeds`.
- Require at least 1.5x median scanner speedup at 64 and 255 bins, or at least 15% median end-to-end
  DRO fit-time improvement if either microbenchmark target is missed. Reject any measured shape
  regression above 5% and any standard-arm regression above 3%.
- Use seven release-mode scanner repetitions and five fixed end-to-end seeds (`0,1,2,3,4`).
- Do not change `dro_radius=0.05`, the Wasserstein metric, or any quality formula. Benchmark quality
  is an equivalence guard, not a default-calibration exercise.
- Commit one logical change at a time and keep benchmark evidence from rejected prototypes.

---

### Task 1: Establish DRO scanner and fit baselines

**Files:**
- Modify: `crates/backend_cpu/benches/histogram_kernels.rs`
- Create: `benchmarks/dro_performance.py`
- Create: `benchmarks/tests/test_dro_performance.py`
- Create: `benchmarks/results/pr133_dro_split_baseline.txt`
- Create: `benchmarks/results/pr133_dro_fit_baseline.json`

**Interfaces:**
- `dro_performance.py` produces `DroPerfRecord`, `ComparisonSummary`, `run_matrix`,
  `compare_results`, `write_results`, and a CLI with `run` and `compare` subcommands.
- Result JSON schema version 1 records `git_head`, `production_base="2b2e3ef"`, platform and package
  versions, arguments, and records.
- Each record key is `(arm, dataset, task_family, shape, seed, primary_metric)`.
- Benchmark arms are `standard` and `dro`; DRO always uses `leaf_solver="dro"`,
  `dro_radius=0.05`, and `dro_metric="wasserstein"`.

- [x] **Step 1: Add failing benchmark-contract tests**

Create tests that import the benchmark module by path and pin the stable contracts:

```python
def test_full_specs_cover_shape_and_task_matrix():
    specs = MODULE.full_specs()
    assert {(s.shape, s.task_family) for s in specs} >= {
        ("small-narrow", "regression"),
        ("small-wide", "regression"),
        ("tall-narrow", "regression"),
        ("tall-wide", "regression"),
        ("medium", "binary"),
        ("small-wide", "multiclass"),
        ("tall-narrow", "ranking"),
    }
    assert {s.variant for s in specs if s.task_family == "ranking"} == {
        "small-query",
        "large-query",
    }


def test_compare_results_rejects_quality_drift():
    baseline = MODULE.synthetic_paired_records(metric_delta=0.0, time_ratio=1.0)
    candidate = MODULE.synthetic_paired_records(metric_delta=1e-3, time_ratio=0.7)
    with pytest.raises(ValueError, match="quality equivalence"):
        MODULE.compare_results(baseline, candidate)


def test_compare_results_enforces_shape_regression_limit():
    baseline = MODULE.synthetic_paired_records(metric_delta=0.0, time_ratio=1.0)
    candidate = MODULE.synthetic_paired_records(metric_delta=0.0, time_ratio=1.051)
    summary = MODULE.compare_results(baseline, candidate)
    assert not summary.passed
    assert any("shape regression" in reason for reason in summary.reasons)
```

Also test duplicate/missing keys, non-finite values, metric direction, deterministic JSON ordering,
the 15% median fit-time fallback, and the 3% standard-arm sentinel.

- [x] **Step 2: Run the benchmark tests and verify failure**

Run:

```bash
/Users/lashby/Projects/AlloyGBM/.venv/bin/python -m pytest \
  benchmarks/tests/test_dro_performance.py -q
```

Expected: fail because `benchmarks/dro_performance.py` does not exist.

- [x] **Step 3: Implement the deterministic end-to-end matrix**

Define these full cases, with quick mode capping rows at 768, features at 48, and rounds at 12
without removing a task family:

```python
def full_specs() -> tuple[FixtureSpec, ...]:
    return (
        FixtureSpec("reg-small-narrow", "small-narrow", "regression", 640, 8, 80, "linear"),
        FixtureSpec("reg-small-wide", "small-wide", "regression", 640, 128, 80, "sparse"),
        FixtureSpec("reg-tall-narrow", "tall-narrow", "regression", 8192, 16, 80, "linear"),
        FixtureSpec("reg-tall-wide", "tall-wide", "regression", 8192, 128, 60, "sparse"),
        FixtureSpec("reg-noisy", "medium", "regression", 2048, 32, 100, "noisy"),
        FixtureSpec("binary-imbalanced", "medium", "binary", 4096, 32, 100, "imbalanced"),
        FixtureSpec("multiclass-wide", "small-wide", "multiclass", 2048, 96, 80, "multiclass"),
        FixtureSpec("rank-small-query", "tall-narrow", "ranking", 2400, 24, 60, "small-query", 120),
        FixtureSpec("rank-large-query", "tall-narrow", "ranking", 4096, 24, 60, "large-query", 8),
    )
```

Use RMSE/MAE for regression, log loss/accuracy for classification, and mean per-query NDCG@10 for
ranking. Rotate arm order by `(case_index + seed) % 2`, use the same generated train/test data for
both arms, and set deterministic estimator parameters including `n_jobs=1`.

`compare_results` requires primary and secondary metrics to match with
`abs_error <= max(1e-7, 1e-7 * abs(baseline))`. It reports per-shape DRO median time ratios,
aggregate DRO median ratio, standard-arm median ratio, worst ratios, quality equivalence, and gate
reasons. Timing fields are excluded from JSON identity comparisons.

- [x] **Step 4: Add parseable scalar DRO scanner cases**

Build 16-, 64-, and 255-bin histograms with `build_histograms_with_grad_sq(..., true)`. Use:

```rust
let dro_options = SplitSelectionOptions {
    l1_alpha: 0.1,
    l2_lambda: 1.0,
    dro_config: Some(DroConfig {
        radius: 0.05,
        metric: DroMetric::Wasserstein,
    }),
    ..SplitSelectionOptions::default()
};
```

Print `best_split_dro_16`, `best_split_dro_64`, and `best_split_dro_255` with `run_case`. Update the
benchmark header to `production_base: 2b2e3ef`. Do not change production scanner routing yet.

- [x] **Step 5: Capture immutable baseline evidence**

Run:

```bash
for run in 1 2 3 4 5 6 7; do
  cargo bench -p alloygbm-backend-cpu --bench histogram_kernels
done | tee benchmarks/results/pr133_dro_split_baseline.txt

maturin develop --release
/Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/dro_performance.py run \
  --arms standard dro --seeds 0 1 2 3 4 \
  --output benchmarks/results/pr133_dro_fit_baseline.json
```

Verify both result files identify `production_base=2b2e3ef`, the JSON identifies the actual
benchmark-only HEAD, and:

```bash
git diff 2b2e3ef -- crates/backend_cpu/src crates/engine/src crates/core/src \
  bindings/python/alloygbm
```

prints no production change.

- [x] **Step 6: Commit the benchmark contract and baseline**

```bash
git add -f crates/backend_cpu/benches/histogram_kernels.rs \
  benchmarks/dro_performance.py benchmarks/tests/test_dro_performance.py \
  benchmarks/results/pr133_dro_split_baseline.txt \
  benchmarks/results/pr133_dro_fit_baseline.json
git commit -m "bench: establish DRO SIMD acceptance baseline"
```

### Task 2: Implement exhaustive f64 DRO SIMD scanning

**Files:**
- Modify: `crates/backend_cpu/src/split_scan.rs`
- Create: `crates/backend_cpu/src/dro_scan.rs`
- Modify: `crates/backend_cpu/src/lib.rs`
- Modify: `crates/backend_cpu/src/tests/main.rs`

**Interfaces:**
- `with_dro_split_scan_scratch(len, f)` supplies cumulative gradient, Hessian, gradient-square, and
  count slices from thread-local reusable buffers.
- `best_split_dro_numeric_simd(feature_histogram, node_id, options)` returns the same
  `Option<SplitCandidate>` contract as the scalar standard strategy with active DRO.
- `dro_effective_gradient_f64x4` mirrors the scalar f64 variance/radius computation for four lanes.

- [x] **Step 1: Add failing scalar/SIMD parity and routing tests**

Add a helper that evaluates the same `HistogramFeatureView` through:

```rust
let scalar = CpuBackend::best_split_for_feature_inner(
    view,
    0,
    options,
    GainStrategy::Standard,
    None,
);
let simd = crate::dro_scan::best_split_dro_numeric_simd(view, 0, options);
```

For fixed and deterministic-random histograms at 16, 64, and 255 bins, require matching presence,
feature, threshold, missing direction, and all child `NodeStats`; compare gain using the global
tolerance. Cover:

- missing mass routed left and right;
- radii `0.001`, `0.05`, and `0.5`;
- L1 values `0.0`, `0.1`, and `1.0` plus nonzero L2;
- minimum rows, minimum Hessian, and minimum leaf magnitude;
- zero empirical variance and cancellation-sensitive `grad_sq / n - mean^2`;
- tail lengths 1 through 3 after four-lane chunks;
- non-finite candidate gain masking;
- no valid split and final edge-threshold rejection;
- nested Rayon calls from multiple workers to prove scratch isolation.

Add routing tests proving active numeric DRO selects the new scanner, radius-zero uses standard SIMD,
and factor-penalized, categorical, and Morph+DRO paths retain scalar routing.

- [x] **Step 2: Run focused tests and verify intended failures**

Run:

```bash
cargo test -p alloygbm-backend-cpu dro_simd -- --nocapture
cargo test -p alloygbm-backend-cpu split_scan -- --nocapture
```

Expected: fail because the DRO scratch and scanner do not exist.

- [x] **Step 3: Extend reusable split scratch**

Extend the existing thread-local scratch tuple with `Vec<f32>` for cumulative gradient squares.
Keep `with_split_scan_scratch`'s current callback signature unchanged. Add:

```rust
pub(super) fn with_dro_split_scan_scratch<R>(
    len: usize,
    f: impl FnOnce(&mut [f32], &mut [f32], &mut [f32], &mut [u32]) -> R,
) -> R;
```

Resize all four vectors before creating slices and retain the current `RefCell` unwind behavior.
Add nested/sequential reuse tests that prove the gradient-square slice is cleared or fully
overwritten before reading.

- [x] **Step 4: Implement the dedicated DRO scanner**

The scanner must:

1. validate at least two bins and require `grad_sq_sums()` for active DRO;
2. extract missing-bin gradient, Hessian, gradient square, and count;
3. compute full parent statistics and the scalar parent `leaf_effective_gradient` once;
4. build cumulative non-missing gradient, Hessian, gradient-square, and count arrays;
5. evaluate thresholds in four-lane chunks for missing-left then missing-right, preserving scalar
   tie order;
6. compute the radius term in `wide::f64x4`, extract it, cast each lane to f32 at the same boundary
   as the scalar helper, add f32 L1, and perform f32 soft thresholding;
7. compute f32 Newton gain and the existing validity/leaf-magnitude masks;
8. mask padded, edge, and non-finite lanes;
9. reduce with `gain_materially_exceeds`;
10. reconstruct exactly one winner with gradient-square child statistics.

Use these semantics for each side lane:

```rust
let n = row_count.max(1) as f64;
let mean = f64::from(grad_sum) / n;
let variance = (f64::from(grad_sq_sum) / n - mean * mean).max(0.0);
let radius_term = (f64::from(radius) * n.sqrt() * variance.sqrt()) as f32;
let threshold = l1_alpha.max(0.0) + radius_term;
let effective = if grad_sum > threshold {
    grad_sum - threshold
} else if grad_sum < -threshold {
    grad_sum + threshold
} else {
    0.0
};
```

Do not silently fall back to zero gradient-square data when active DRO is requested. If a private
direct call receives no gradient-square plane, return `None`; production training already builds
the plane whenever DRO is active, and a test must pin that invariant.

- [x] **Step 5: Route eligible active DRO scans**

In `best_split_for_feature`:

- if `factor_context.is_some()`, retain the scalar scaffold;
- else if `options.dro_active()`, call `best_split_dro_numeric_simd`;
- else call the existing standard SIMD scanner, including when radius is zero.

Do not change categorical dispatch or `best_split_morph_numeric_feature`; both retain their existing
active-DRO scalar behavior.

- [x] **Step 6: Run focused correctness suites**

```bash
cargo fmt --all -- --check
cargo test -p alloygbm-backend-cpu dro -- --nocapture
cargo test -p alloygbm-backend-cpu dro_simd -- --nocapture
cargo test -p alloygbm-backend-cpu split_scan -- --nocapture
cargo test -p alloygbm-engine dro -- --nocapture
```

Expected: all pass with no ignored correctness tests.

- [x] **Step 7: Commit scanner implementation**

```bash
git add crates/backend_cpu/src/split_scan.rs crates/backend_cpu/src/dro_scan.rs \
  crates/backend_cpu/src/lib.rs crates/backend_cpu/src/tests/main.rs
git commit -m "perf: vectorize numeric DRO split scanning"
```

### Task 3: Verify performance and end-to-end equivalence

**Files:**
- Create: `benchmarks/results/pr133_dro_split_simd.txt`
- Create: `benchmarks/results/pr133_dro_fit_simd.json`
- Create: `benchmarks/results/pr133_dro_comparison.json`

**Interfaces:**
- Candidate scanner output uses the exact benchmark row names established in Task 1.
- `dro_performance.py compare BASELINE CANDIDATE --output COMPARISON` emits a machine-readable gate
  and exits nonzero when quality, shape, standard-sentinel, or aggregate performance gates fail.

- [x] **Step 1: Capture seven candidate scanner repetitions**

```bash
for run in 1 2 3 4 5 6 7; do
  cargo bench -p alloygbm-backend-cpu --bench histogram_kernels
done | tee benchmarks/results/pr133_dro_split_simd.txt
```

Parse medians for all three DRO cases. Require 1.5x at 64 and 255 bins or defer acceptance to the
end-to-end fallback gate; require no 16-bin regression above 5%.

- [x] **Step 2: Run candidate matrix and compare with baseline**

```bash
maturin develop --release
/Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/dro_performance.py run \
  --arms standard dro --seeds 0 1 2 3 4 \
  --output benchmarks/results/pr133_dro_fit_simd.json
/Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/dro_performance.py compare \
  benchmarks/results/pr133_dro_fit_baseline.json \
  benchmarks/results/pr133_dro_fit_simd.json \
  --output benchmarks/results/pr133_dro_comparison.json
```

The comparison must report exact key coverage and quality equivalence. Accept performance only when
the scanner gate passes or median DRO fit time improves at least 15%, with no shape above 5% and the
standard sentinel within 3%.

- [x] **Step 3: Reject or retain the implementation based on predeclared gates**

If performance fails, profile only the DRO scanner and try at most these behavior-preserving
adjustments in order:

1. hoist f64 broadcasts and parent terms outside direction/chunk loops;
2. evaluate both missing directions from one set of loaded prefix vectors;
3. combine two `f64x4` groups per eight-bin prefix load while preserving scalar reduction order.

Do not switch variance to f32, add a top-k approximation, change the radius formula, or weaken
parity/performance thresholds. Record rejected prototype timing in
`benchmarks/results/pr133_dro_comparison.json` under `rejected_trials`.

- [x] **Step 4: Run robustness and compatibility sentinels**

```bash
/Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/dro_robustness.py \
  --seeds 7,13 --quick --output /tmp/pr133_dro_robustness.md
/Users/lashby/Projects/AlloyGBM/.venv/bin/python -m pytest \
  bindings/python/tests/test_dro_leaf_solver.py \
  benchmarks/tests/test_dro_robustness.py \
  benchmarks/tests/test_dro_performance.py -q
```

Require the existing robustness report's quality values to remain unchanged within its printed
precision. This is a regression sentinel, not evidence to alter defaults.

- [x] **Step 5: Commit accepted performance evidence**

```bash
git add -f benchmarks/results/pr133_dro_split_simd.txt \
  benchmarks/results/pr133_dro_fit_simd.json \
  benchmarks/results/pr133_dro_comparison.json
git commit -m "bench: verify DRO SIMD performance"
```

### Task 4: Close documentation and run final verification

**Files:**
- Modify: `CHANGELOG.md`
- Create: `docs/benchmarks/dro_simd_pr133.md`
- Modify: `docs/user/gbmregressor.md`
- Modify: `docs/site/source/estimator.rst`
- Modify: `docs/user/benchmarks.md`
- Modify: `docs/site/source/benchmarks.rst`
- Modify: `docs/reviews/2026-07-02-v0.12.10-special-modes-resolutions.md`
- Modify: `docs/reviews/2026-08-10-pr-133-dro-simd-performance-implementation-plan.md`

**Interfaces:**
- The benchmark report distinguishes scanner, native fit, and total fit timing and includes exact
  reproduction commands.
- The resolution marks only section 3.1 fixed. Existing caveats about default-radius robustness and
  joint leaf-only DRO remain unchanged.

- [x] **Step 1: Write the evidence report**

Document:

- source base/candidate commits, hardware, OS, architecture, Rust/Python/package versions;
- 16/64/255-bin seven-run medians and speedups;
- all nine matrix cases, shapes, rounds, five seeds, per-shape and aggregate fit ratios;
- quality-equivalence tolerance and outcome;
- scalar fallback combinations;
- existing robustness sentinel outcome;
- rejected prototypes and why they were rejected;
- commands required to reproduce every result.

Do not claim DRO improves predictive quality or always beats standard leaves. State that this PR
reduces the cost of an opt-in robust leaf solver whose radius remains workload-dependent.

- [x] **Step 2: Update user, Sphinx, changelog, and resolution docs**

Explain that active scalar-model numeric DRO split selection is exhaustive and safe-SIMD, while
native categorical, factor-penalized, Morph+DRO, and joint-output split selection retain their
documented scalar or standard-gain behavior. Link the report from user/Sphinx benchmark indexes.
Mark section 3.1 fixed in the special-modes resolution without changing sections 3.2 or 3.3.

- [x] **Step 3: Run complete verification**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
maturin develop --release
/Users/lashby/Projects/AlloyGBM/.venv/bin/python -m pytest \
  bindings/python/tests/ benchmarks/tests/ -q
/Users/lashby/Projects/AlloyGBM/.venv/bin/sphinx-build -W -b html \
  docs/site/source /tmp/alloygbm-pr133-sphinx
git diff --check
```

Expected: every command passes. Existing artifact-load and radius-zero equivalence tests in the full
suites provide compatibility coverage.

- [x] **Step 4: Inspect final scope and compatibility**

```bash
git diff --stat 2b2e3ef...HEAD
git diff 2b2e3ef...HEAD -- crates/core/src/artifact_format.rs crates/core/src/lib.rs
rg -n 'top.?k|std::env|experimental' crates/backend_cpu/src/dro_scan.rs
git status --short
```

Expected: no artifact-schema diff, no approximation or runtime switch, and no uncommitted files
after the closure commit. Confirm native categorical, factor, Morph+DRO, and joint code has no new
SIMD routing.

- [x] **Step 5: Mark this plan implemented and commit closure**

Set the plan status to `Implemented; ready for review`, check every completed step, and commit:

```bash
git add CHANGELOG.md docs/benchmarks/dro_simd_pr133.md \
  docs/user/gbmregressor.md docs/site/source/estimator.rst \
  docs/user/benchmarks.md docs/site/source/benchmarks.rst \
  docs/reviews/2026-07-02-v0.12.10-special-modes-resolutions.md \
  docs/reviews/2026-08-10-pr-133-dro-simd-performance-implementation-plan.md
git commit -m "docs: close DRO SIMD review finding"
```

- [ ] **Step 6: Push and prepare draft PR #133 without merging**

Push `codex/dro-simd-performance` and open a draft PR against `main`. The PR description must include
the scanner architecture, exact fallback scope, microbenchmark and fit-time gates, equivalence
results, full verification, and links to the evidence and review-resolution documents. Preserve the
worktree for reviewer changes and stop before merge.
