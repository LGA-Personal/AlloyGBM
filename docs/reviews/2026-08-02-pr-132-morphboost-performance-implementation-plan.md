# PR #132 MorphBoost Performance And Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

| Date | Reviewer | Version reviewed | Commit | Status |
|---|---|---|---|---|
| 2026-08-02 | OpenAI Codex | `main` after PR #131 | `77dbf6d` | Approved for implementation |

**Goal:** Close the MorphBoost scalar-scanner review finding with an exhaustive SIMD scanner,
repair its quality benchmarks and warmup semantics, and promote formula/default changes only when
predeclared paired A/B gates show a reliable improvement.

**Architecture:** Keep the scalar Morph formula and scanner scaffold as correctness oracles while a
focused `morph_scan` module evaluates ordinary numeric candidates eight lanes at a time. Build a
reproducible benchmark ladder before optimization, then compare each behavioral candidate against a
frozen optimized control. DRO, factor-penalized, and unpromoted experimental paths remain scalar.

**Tech Stack:** Rust 1.92, edition 2024, `wide` 0.7 safe SIMD, Rayon, PyO3, Python 3.11-3.13,
NumPy, scikit-learn metrics/datasets, pytest, maturin, Sphinx.

## Global Constraints

- Preserve exhaustive evaluation of every valid numeric threshold and both missing directions.
- Keep `unsafe_code = "forbid"`; use only the existing safe `wide` SIMD dependency.
- Do not introduce per-feature or per-candidate allocations in the optimized scanner.
- Keep the artifact schema and existing-artifact loading behavior unchanged.
- Keep scalar fallbacks for Morph+DRO and factor-penalized Morph in this PR.
- Do not add a public experimental parameter or hidden runtime environment switch.
- Use the scalar formula as the gain oracle with
  `abs_error <= max(1e-5, 1e-5 * abs(scalar_gain))`.
- Require identical scalar/SIMD winners unless the scalar leaders tie under
  `gain_materially_exceeds`.
- Require at least 1.5x median scanner speedup at 64 and 255 bins, or at least 15% end-to-end
  improvement if the microbenchmark target is missed; reject any shape regression above 5%.
- Formula/default trials use five fixed seeds, equal dataset weighting, fixed-seed 10,000-resample
  paired bootstrap, and the promotion thresholds from the approved design.
- Treat RMSE, classifier log loss, and NDCG@10 as primary metrics; MAE and accuracy are veto metrics.
- Use calibration seeds `0,1,2`; use confirmation seeds `3,4` and public datasets only after a
  candidate survives calibration.
- Keep learning-rate schedules, morph rate, depth penalty, evolution pressure, tree growth, and
  estimator auto-policy heuristics fixed during formula/default trials.
- Do not alter the unrelated main-worktree `CLAUDE.md` change.

---

### Task 1: Repair Morph benchmarks and freeze the baseline

**Files:**
- Modify: `benchmarks/morph_ablation.py`
- Create: `benchmarks/morph_acceptance.py`
- Create: `benchmarks/tests/test_morph_acceptance.py`
- Modify: `bindings/python/tests/test_morph_helpers.py`
- Modify: `crates/backend_cpu/benches/histogram_kernels.rs`
- Create: `benchmarks/results/pr132_morph_baseline.json`
- Create: `benchmarks/results/pr132_morph_split_baseline.txt`

**Interfaces:**
- Produces `MorphBenchmarkRecord`, `CandidateGate`, `evaluate_candidate`, and JSON read/write helpers
  in `benchmarks/morph_acceptance.py`.
- Produces `relabel_arm(records, old, new)` and `merge_record_sets(*record_sets)`; the latter rejects
  duplicate `(arm, dataset, task_family, shape, seed, primary_metric)` keys.
- Produces fixed dataset case names and shape metadata reused by every later A/B.
- Produces parseable `best_split_morph_{16,64,255}` Rust benchmark rows.
- Freezes quality and timing evidence at source commit `77dbf6d` before production changes.

- [x] **Step 1: Add failing ranking and gate-contract tests**

Add tests that require ranking groups to be positive sizes summing to the row count, require NDCG@10
instead of RMSE, and pin the promotion math:

```python
def test_ranking_fixture_returns_query_sizes():
    X_tr, y_tr, group_tr, X_te, y_te, group_te = _ranking_dataset(
        n=240, n_features=8, n_groups=12, seed=7
    )
    assert all(size > 0 for size in group_tr + group_te)
    assert sum(group_tr) == len(X_tr) == len(y_tr)
    assert sum(group_te) == len(X_te) == len(y_te)


def test_candidate_gate_rejects_bad_worst_case():
    # `_paired_records` builds matched control/candidate records from normalized
    # candidate changes while keeping dataset and seed keys identical.
    records = _paired_records([0.01, 0.02, -0.031, 0.01, 0.02])
    gate = evaluate_candidate(records, control_arm="morph_current", candidate_arm="candidate")
    assert not gate.passed
    assert any("worst paired change" in reason for reason in gate.reasons)
```

Also test error/higher-is-better normalization, the 0.1% practical-tie boundary, equal dataset
weighting, task-family veto, and deterministic bootstrap output.

- [x] **Step 2: Run the focused tests and observe the intended failures**

Run:

```bash
.venv/bin/python -m pytest benchmarks/tests/test_morph_acceptance.py \
  bindings/python/tests/test_morph_helpers.py -q
```

Expected: imports or assertions fail because the acceptance harness does not exist and the ranking
fixture returns a truncated ID vector.

- [x] **Step 3: Implement the benchmark record and gate layer**

Create these stable data contracts:

```python
@dataclass(frozen=True)
class MorphBenchmarkRecord:
    arm: str
    dataset: str
    task_family: str
    shape: str
    seed: int
    primary_metric: str
    primary_value: float
    secondary_metrics: dict[str, float]
    fit_seconds: float


@dataclass(frozen=True)
class CandidateGate:
    candidate_arm: str
    passed: bool
    mean_improvement: float
    median_improvement: float
    win_or_tie_fraction: float
    bootstrap_low: float
    worst_paired_change: float
    family_means: dict[str, float]
    reasons: tuple[str, ...]
```

Implement `evaluate_candidate(records, *, control_arm, candidate_arm, bootstrap_seed=132)` with
the exact thresholds in the design. Serialize records as a versioned JSON object containing the
actual Git HEAD, explicit `production_base="77dbf6d"`, platform, Python/package versions, CLI
arguments, and records. Reject missing pairs, duplicate pair keys, non-finite values, or mismatched
primary metrics before aggregation.

- [x] **Step 4: Repair and strengthen `morph_ablation.py`**

Return query-size lists from `_ranking_dataset`, split only at a query boundary, and compute NDCG@10
with `alloygbm.evaluation.ndcg_score` per query before averaging. Add a multiclass fixture and use
log loss as the classifier primary metric while retaining accuracy in output. Replace the old 35%
RMSE/8-point-accuracy gate with a call to the shared acceptance code for paired records.

- [x] **Step 5: Implement the full acceptance matrix**

Define predeclared cases for small/narrow, small/wide, tall/narrow, tall/wide, noisy nonlinear,
sparse-signal high-dimensional, imbalanced binary/multiclass, and small/large-query ranking.
`run_matrix(arms, seeds, quick)` must construct every dataset from its case name and seed, run arms
in rotating order, force deterministic estimator settings, and retain the same train/test split for
all arms. Expose:

```bash
.venv/bin/python benchmarks/morph_acceptance.py \
  --arms auto morph_current \
  --seeds 0 1 2 3 4 \
  --output benchmarks/results/pr132_morph_baseline.json
```

`--quick` uses reduced row/column/round counts but does not remove a task family. `--profile
regularized` selects the same task families with `lambda_l1` values `0.1` and `0.5` and includes
Morph+DRO compatibility arms; this profile is used only for Task 5's gradient-source decision.

- [x] **Step 6: Add parseable Morph scanner benchmark cases**

Extend `histogram_kernels.rs` with fixtures whose maximum bins are 15, 63, and 255 and construct a
post-warmup `MorphContext` using `morph_warmup_iters=0`. Call `BackendOps::best_split_morph` with
empty feature weights/categories and print `best_split_morph_16`, `best_split_morph_64`, and
`best_split_morph_255` through the existing `run_case` format. Keep the existing standard cases.

- [x] **Step 7: Run tests and capture immutable baseline evidence**

Run the Rust benchmark seven times so baseline and candidate aggregation are symmetric:

```bash
.venv/bin/python -m pytest benchmarks/tests/test_morph_acceptance.py \
  bindings/python/tests/test_morph_helpers.py -q
for run in 1 2 3 4 5 6 7; do
  cargo bench -p alloygbm-backend-cpu --bench histogram_kernels
done | tee benchmarks/results/pr132_morph_split_baseline.txt
maturin develop --release
.venv/bin/python benchmarks/morph_acceptance.py \
  --arms auto morph_current --seeds 0 1 2 3 4 \
  --output benchmarks/results/pr132_morph_baseline.json
```

Verify the JSON identifies the actual branch HEAD and `production_base=77dbf6d`. Add
`production_base=77dbf6d` explicitly to the text benchmark header. Confirm
`git diff 77dbf6d -- crates bindings/python/alloygbm` is empty so the baseline contains no
production implementation change.

- [x] **Step 8: Commit the benchmark contract**

```bash
git add benchmarks/morph_ablation.py benchmarks/morph_acceptance.py \
  benchmarks/tests/test_morph_acceptance.py bindings/python/tests/test_morph_helpers.py \
  crates/backend_cpu/benches/histogram_kernels.rs \
  benchmarks/results/pr132_morph_baseline.json \
  benchmarks/results/pr132_morph_split_baseline.txt
git commit -m "bench: establish MorphBoost acceptance matrix"
```

### Task 2: Repair warmup and parent-gain semantics

**Files:**
- Modify: `crates/backend_cpu/src/morph.rs`
- Modify: `crates/backend_cpu/src/lib.rs`
- Modify: `crates/backend_cpu/src/backend_ops.rs`
- Modify: `crates/backend_cpu/src/tests/main.rs`
- Modify: `crates/engine/src/shared_histogram.rs`

**Interfaces:**
- Changes `SplitSideStats` to carry `gain_gradient_sum` and `info_gradient_sum` separately.
- Changes `MorphGainInputs` to carry explicit `parent`, `left`, and `right` side statistics.
- Produces a backend-private `morph_uses_standard_gain_only(&MorphContext) -> bool` helper with the
  same semantics: warmup, or negligible information weight with balance disabled.
- Leaves current post-warmup information behavior unchanged by initially setting both gradient
  channels to the effective gradient.

- [x] **Step 1: Add failing parent and warmup tests**

Add a direct formula test where L1 thresholding the parent differs from summing thresholded children:

```rust
#[test]
fn morph_gradient_gain_uses_explicit_parent_signal() {
    let inputs = MorphGainInputs {
        parent: side(7.0, 7.0, 8.0, 8),
        left: side(4.0, 4.0, 4.0, 4),
        right: side(1.0, 1.0, 4.0, 4),
        iteration: 10,
        total_iterations: 100,
        grad_mean: 0.0,
        grad_std: 1.0,
        lambda_l2: 1.0,
    };
    let expected = 4.0_f32.powi(2) / 5.0 + 1.0_f32.powi(2) / 5.0
        - 7.0_f32.powi(2) / 9.0;
    assert!((gradient_gain(&inputs) - expected).abs() < 1e-6);
}
```

Strengthen backend warmup tests to compare gain, feature, threshold, missing direction, and both
child `NodeStats` under nonzero L1/L2, missing mass, and `min_leaf_magnitude`. Add categorical
warmup parity. Add a test proving Morph+DRO and factor-penalized Morph do not enter the ordinary
standard SIMD shortcut.

- [x] **Step 2: Run the focused Rust tests and observe failure**

Run:

```bash
cargo test -p alloygbm-backend-cpu morph -- --nocapture
cargo test -p alloygbm-engine multi_output_morph -- --nocapture
```

Expected: the new explicit-parent test does not compile and at least one strengthened L1 fixture
shows the current parent reconstruction contract.

- [x] **Step 3: Separate gain and information signals in the scalar oracle**

Use these field names consistently:

```rust
pub struct SplitSideStats {
    pub gain_gradient_sum: f32,
    pub info_gradient_sum: f32,
    pub hessian_sum: f32,
    pub count: u32,
}

pub struct MorphGainInputs {
    pub parent: SplitSideStats,
    pub left: SplitSideStats,
    pub right: SplitSideStats,
    pub iteration: u32,
    pub total_iterations: u32,
    pub grad_mean: f32,
    pub grad_std: f32,
    pub lambda_l2: f32,
}
```

Define the test helper as
`fn side(gain: f32, info: f32, hessian: f32, count: u32) -> SplitSideStats` with direct field
assignment so formula fixtures remain compact without hiding transformations.

`gradient_gain` consumes `gain_gradient_sum`; `info_gain` consumes `info_gradient_sum`; gradient
normalization consumes `parent.hessian_sum`. In current-control call sites, assign each side's
effective L1/DRO gradient to both channels. Compute the parent effective gradient once from total
raw gradient, gradient-square sum, and count through `leaf_effective_gradient`.

- [x] **Step 4: Route pure-standard Morph rounds through standard scanners**

In `best_split_morph_with_factor_context`, select standard numeric/categorical functions only when
the Morph context is pure-standard and both `dro_config` and `factor_context` are absent. Preserve
Morph leaf scheduling in the engine; only split selection is redirected. Keep direct numeric helper
tests aligned with this routing contract.

- [x] **Step 5: Align joint-output formula inputs**

Update `morph_gain_per_output` to construct and consume explicit parent/child signals even though
joint training currently has no L1/DRO distinction. Do not change its Hessian-derived count proxy
in this task.

- [x] **Step 6: Run focused and workspace tests**

```bash
cargo test -p alloygbm-backend-cpu morph -- --nocapture
cargo test -p alloygbm-engine morph -- --nocapture
cargo test --workspace
```

Expected: all pass, including exact warmup winner/stat parity.

- [x] **Step 7: Commit the semantic repair**

```bash
git add crates/backend_cpu/src/morph.rs crates/backend_cpu/src/lib.rs \
  crates/backend_cpu/src/backend_ops.rs crates/backend_cpu/src/tests/main.rs \
  crates/engine/src/shared_histogram.rs
git commit -m "fix: restore MorphBoost warmup gain semantics"
```

### Task 3: Implement the exhaustive SIMD Morph scanner

**Files:**
- Create: `crates/backend_cpu/src/morph_scan.rs`
- Modify: `crates/backend_cpu/src/lib.rs`
- Modify: `crates/backend_cpu/src/tests/main.rs`

**Interfaces:**
- Produces `best_split_morph_numeric_simd(feature_histogram, node_id, options, morph)` returning
  `Option<SplitCandidate>` for ordinary numeric Morph scans.
- Consumes thread-local prefix buffers through `with_split_scan_scratch`.
- Keeps `best_split_for_feature_inner(..., GainStrategy::Morph(...), ...)` as scalar oracle/fallback.

- [x] **Step 1: Add failing deterministic and randomized parity tests**

Create a fixed-seed local linear-congruential generator in the Rust test module so no dependency is
added. Generate valid histograms across 16/64/255 bins, missing mass, L1 values, row/Hessian floors,
leaf-magnitude filters, warmup positions, and balance settings. For each fixture:

```rust
let scalar = CpuBackend::best_split_for_feature_inner(
    view, node_id, options, GainStrategy::Morph(&morph), None,
);
let simd = morph_scan::best_split_morph_numeric_simd(view, node_id, options, &morph);
assert_morph_split_parity(scalar, simd);
```

`assert_morph_split_parity` must compare presence, feature, threshold, direction, child statistics,
and gain tolerance. If winners differ, recompute scalar gains for both and allow the difference only
when neither materially exceeds the other. Add explicit tests for tail lanes, all-invalid lanes,
non-finite candidate masking, and a balance ratio immediately below/at 0.1.

- [x] **Step 2: Run the new scanner tests and observe the missing symbol**

```bash
cargo test -p alloygbm-backend-cpu morph_simd -- --nocapture
```

Expected: compilation fails because `morph_scan` and its SIMD function do not exist.

- [x] **Step 3: Create the focused SIMD module and prefix preparation**

Declare `mod morph_scan;` in `lib.rs`. In the new module, sum feature totals exactly as the scalar
path does, extract missing stats, fill the three existing prefix slices, and broadcast totals and
per-round constants. Return `None` under the same short-feature, low-parent-Hessian, and empty-scan
conditions as the scalar oracle.

- [x] **Step 4: Implement eight-lane candidate evaluation**

For each missing direction, load padded arrays into `f32x8`; derive left/right statistics; apply
`l1_threshold_f32x8`; compute parent/child Newton terms, curvature normalization, standardized
information terms with `f32x8::ln`, and balance adjustment with masked `f32x8::exp`. Combine row,
Hessian, leaf-magnitude, finite, edge, and tail masks before extracting gains. Use
`gain_materially_exceeds` in scalar lane order so deterministic tie preference matches the oracle.

- [x] **Step 5: Reconstruct one winning candidate**

Store only `(gain, threshold_bin, default_left)` during scanning. Reconstruct raw child
`NodeStats` from prefix/missing statistics once after reduction. Do not retain lane-sized candidate
vectors or allocate a shortlist.

- [x] **Step 6: Route eligible post-warmup scans to SIMD**

In `best_split_morph_numeric_feature`, use SIMD only when `dro_config.is_none()` and
`factor_context.is_none()`. Keep pure-standard routing from Task 2 ahead of this branch. Leave
categorical, DRO, and factor paths on their existing scalar scaffolds.

- [x] **Step 7: Run scanner and backend tests**

```bash
cargo fmt --all -- --check
cargo test -p alloygbm-backend-cpu morph_simd -- --nocapture
cargo test -p alloygbm-backend-cpu morph -- --nocapture
cargo clippy -p alloygbm-backend-cpu --all-targets -- -D warnings
```

- [x] **Step 8: Commit exhaustive SIMD scanning**

```bash
git add crates/backend_cpu/src/morph_scan.rs crates/backend_cpu/src/lib.rs \
  crates/backend_cpu/src/tests/main.rs
git commit -m "perf: vectorize exhaustive MorphBoost split scanning"
```

### Task 4: Verify performance and select the SIMD implementation

**Files:**
- Create: `benchmarks/morph_perf_gate.py`
- Create: `benchmarks/tests/test_morph_perf_gate.py`
- Create: `benchmarks/results/pr132_morph_simd.txt`
- Create: `benchmarks/results/pr132_morph_optimized_control.json`
- Modify conditionally: `crates/backend_cpu/src/morph_scan.rs`

**Interfaces:**
- Produces a parser for existing `run_case` output and an explicit baseline/candidate gate report.
- Freezes the optimized current-formula control used by Tasks 5-7.

- [x] **Step 1: Add failing performance-gate parser tests**

Cover 1.5x pass/fail, 5% regression rejection, missing benchmark names, malformed numbers, median
aggregation across repeated names, and the allowed end-to-end fallback:

```python
def test_perf_gate_accepts_required_scanner_speedups():
    baseline = {"best_split_morph_64": 150.0, "best_split_morph_255": 400.0}
    candidate = {"best_split_morph_64": 90.0, "best_split_morph_255": 240.0}
    assert evaluate_perf_gate(baseline, candidate, end_to_end_improvement=0.18).passed
```

- [x] **Step 2: Run the parser tests and observe failure**

```bash
.venv/bin/python -m pytest benchmarks/tests/test_morph_perf_gate.py -q
```

- [x] **Step 3: Implement the parser and gate report**

Parse `name: total_ms=... iterations=... ns_per_iter=...`, group repeated names as independent
samples, take their medians, compare the required scanner cases, and emit JSON containing per-case
speedup, worst regression, end-to-end changes, pass/fail, and reasons. Require all named cases and
at least five samples per required Morph case in both files.

- [x] **Step 4: Capture repeated candidate measurements**

Run the Rust benchmark seven times and retain the median `ns_per_iter` per case:

```bash
for run in 1 2 3 4 5 6 7; do
  cargo bench -p alloygbm-backend-cpu --bench histogram_kernels
done | tee benchmarks/results/pr132_morph_simd.txt
```

Run quick and full end-to-end controls after rebuilding the extension:

```bash
maturin develop --release
.venv/bin/python benchmarks/morph_acceptance.py \
  --arms auto morph_current --seeds 0 1 2 3 4 \
  --output benchmarks/results/pr132_morph_optimized_control.json
.venv/bin/python benchmarks/morph_perf_gate.py \
  --baseline benchmarks/results/pr132_morph_split_baseline.txt \
  --candidate benchmarks/results/pr132_morph_simd.txt \
  --baseline-fit benchmarks/results/pr132_morph_baseline.json \
  --candidate-fit benchmarks/results/pr132_morph_optimized_control.json
```

- [x] **Step 5: Apply the conditional transcendental decision**

Result: retain lane-wise transcendentals. The seven-run medians were 1.92x faster at 255 bins,
1.08x at 64 bins, and 1.03x at 16 bins; paired end-to-end Morph fit time improved 30.4%, clearing
the declared 15% fallback with no shape regression.

If the scanner reaches 1.5x at 64/255 bins, retain lane-wise `ln`/`exp`. If it misses 1.5x and the
end-to-end gain is below 15%, benchmark a second implementation that extracts standardized values,
uses scalar `(1.0 + x).ln()`/`exp()` only for valid lanes, and vectorizes the surrounding arithmetic.
Retain the faster implementation only when it passes the same oracle tests. Do not add threshold
shortlisting.

- [x] **Step 6: Run the optimized-control quality gate**

Result: the optimized control had median quality change `0.0`, 97.8% practical wins/ties, worst
paired change `-0.18%`, and no family veto. Its mean change was `+0.017%`; as expected for a
behavior-preserving implementation, it does not meet the separate `+0.25%` formula-promotion bar.

Relabel optimized `morph_current` records to `morph_simd`, merge them with baseline
`morph_current` records, and evaluate `morph_simd` against that control. Floating-order differences
must satisfy the formula/default quality vetoes even though no intentional calibration occurred. If
they do not, tighten lane reduction/order or use scalar transcendentals until the control passes.

- [x] **Step 7: Commit performance evidence and selected implementation**

```bash
git add benchmarks/morph_perf_gate.py benchmarks/tests/test_morph_perf_gate.py \
  benchmarks/results/pr132_morph_simd.txt \
  benchmarks/results/pr132_morph_optimized_control.json \
  crates/backend_cpu/src/morph_scan.rs
git commit -m "bench: verify MorphBoost SIMD performance"
```

### Task 5: Evaluate raw-gradient information statistics

**Files:**
- Modify conditionally: `crates/backend_cpu/src/lib.rs`
- Modify conditionally: `crates/backend_cpu/src/morph_scan.rs`
- Modify conditionally: `crates/backend_cpu/src/tests/main.rs`
- Create: `benchmarks/results/pr132_morph_raw_info.json`
- Create: `benchmarks/results/pr132_morph_formula_trials.md`

**Interfaces:**
- Uses the split `gain_gradient_sum`/`info_gradient_sum` contract from Task 2.
- Produces an accepted production change or a documented rejection with production behavior restored.

- [ ] **Step 1: Add a failing separation test**

Construct a post-warmup L1 fixture and require the information channel to receive raw side/parent
gradient sums while Newton gain receives effective sums. Add a Morph+DRO compatibility case with
the same separation. Keep ordinary zero-L1 Morph identical because raw and effective sums coincide.

- [ ] **Step 2: Run focused tests before changing call sites**

```bash
cargo test -p alloygbm-backend-cpu information_gradient -- --nocapture
```

Expected: the new assertion fails because current-control call sites assign effective gradients to
both channels.

- [ ] **Step 3: Implement the candidate consistently**

In scalar numeric/categorical call sites and `morph_scan`, assign raw histogram gradient sums to
`info_gradient_sum`, including raw parent totals. Keep `gain_gradient_sum` L1/DRO-adjusted. Do not
change EMA statistics, leaf solving, or standard gain.

- [ ] **Step 4: Run targeted regularization calibration**

Run calibration seeds on cases with `lambda_l1` in `{0.1, 0.5}`, plus Morph+DRO compatibility
cases, and write the candidate output:

```bash
maturin develop --release
.venv/bin/python benchmarks/morph_acceptance.py \
  --arms auto morph_current --profile regularized \
  --seeds 0 1 2 \
  --output benchmarks/results/pr132_morph_raw_info.json
```

Call `relabel_arm(candidate_records, "morph_current", "morph_raw_info")`, merge that result with
the frozen optimized-control records through `merge_record_sets`, then call `evaluate_candidate`
with `control_arm="morph_current"` and `candidate_arm="morph_raw_info"`.

- [ ] **Step 5: Apply the promotion decision without leaving an experiment switch**

If calibration passes, run confirmation seeds `3,4` and retain the candidate only if the combined
gate passes. If rejected, change every `info_gradient_sum` assignment back to its corresponding
effective gradient, remove the separation-specific production test, rerun scalar/SIMD parity, and
confirm `git diff` contains no production formula change from this task. In both cases, record
mean/median/worst/family/bootstrap results and the decision in
`pr132_morph_formula_trials.md`.

- [ ] **Step 6: Commit the decision and evidence**

If accepted:

```bash
git add crates/backend_cpu/src/lib.rs crates/backend_cpu/src/morph_scan.rs \
  crates/backend_cpu/src/tests/main.rs benchmarks/results/pr132_morph_raw_info.json \
  benchmarks/results/pr132_morph_formula_trials.md
git commit -m "perf: separate MorphBoost gain and information signals"
```

If rejected:

```bash
git add benchmarks/results/pr132_morph_raw_info.json \
  benchmarks/results/pr132_morph_formula_trials.md
git commit -m "bench: reject raw-gradient MorphBoost information trial"
```

### Task 6: Recalibrate balance and information weight sequentially

**Files:**
- Modify: `benchmarks/morph_acceptance.py`
- Modify: `benchmarks/tests/test_morph_acceptance.py`
- Modify conditionally: `crates/backend_cpu/src/morph.rs`
- Modify conditionally: `crates/engine/src/shared_histogram.rs`
- Modify conditionally: `crates/core/src/training_mode.rs`
- Modify conditionally: `bindings/python/alloygbm/_regressor/_core.py`
- Modify conditionally: `bindings/python/alloygbm/_morph.py`
- Modify conditionally: estimator contract tests containing Morph defaults
- Create: `benchmarks/results/pr132_morph_calibration.json`
- Modify: `benchmarks/results/pr132_morph_formula_trials.md`

**Interfaces:**
- Produces named arms `morph_current`, `morph_no_balance`, `morph_info_005`,
  `morph_info_0075`, `morph_info_010`, and `morph_info_015`.
- Produces `select_calibration_candidate(records, candidate_arms) -> str | None`, which reads only
  seeds `0,1,2`, filters arms through `evaluate_candidate`, and returns the passing arm with the
  highest mean improvement using arm name as the deterministic final tie-break.
- Produces at most one new default combination; rejected arms remain benchmark-only.

- [ ] **Step 1: Add failing arm-order and held-out-selection tests**

Require rotating arm order by dataset/seed, exact arm kwargs, calibration-only selection, and a
separate confirmation gate. Assert that confirmation records cannot influence
`select_calibration_candidate`.

- [ ] **Step 2: Implement named calibration arms**

Map arms to public estimator kwargs. `morph_current` uses no overrides,
`morph_no_balance` sets `balance_penalty=False`, and information arms set only
`info_score_weight`. Keep the optimized current formula as control.

- [ ] **Step 3: Evaluate balance enablement first**

```bash
.venv/bin/python benchmarks/morph_acceptance.py \
  --arms morph_current morph_no_balance --seeds 0 1 2 \
  --output /tmp/pr132_balance_calibration.json
```

If disabled balance fails, retain the current behavior. If it passes, run seeds `3,4` plus
`morph_report.py`; change the default only if confirmation passes. Do not evaluate lower-strength
source constants unless both on/off variants show opposing task-family effects and profiling shows
the balance exponential is material. If that condition holds, test exactly one midpoint coefficient
`-0.25` against the current `-0.5` using the same calibration/confirmation split.

- [ ] **Step 4: Evaluate information weight around the surviving balance control**

```bash
.venv/bin/python benchmarks/morph_acceptance.py \
  --arms morph_info_005 morph_info_0075 morph_info_010 morph_info_015 \
  --seeds 0 1 2 --output /tmp/pr132_info_calibration.json
```

Select the highest calibration aggregate that passes all calibration vetoes, then evaluate only
that arm on seeds `3,4` and public `morph_report.py` datasets. Promote it only if the combined
five-seed gate passes. This selection order prevents confirmation-data tuning.

- [ ] **Step 5: Apply an accepted default consistently**

If a balance or information default changes, update Rust `MorphConfig::default`, Python constructor
defaults/signatures, `build_morph_config_dict`, repr/get/set parameter expectations, and focused
tests together. If no candidate passes, leave every production default unchanged.

- [ ] **Step 6: Save one merged result and record rejected arms**

Write all calibration and confirmation records plus gate summaries to
`pr132_morph_calibration.json`. Append each arm's gate statistics and final decision to the formula
trial report. Include public-report results only for the finalist.

- [ ] **Step 7: Run focused defaults and Morph suites**

```bash
cargo test -p alloygbm-core morph -- --nocapture
cargo test -p alloygbm-backend-cpu morph -- --nocapture
.venv/bin/python -m pytest bindings/python/tests/test_morph.py \
  bindings/python/tests/test_morph_helpers.py \
  benchmarks/tests/test_morph_acceptance.py -q
```

- [ ] **Step 8: Commit the calibration result**

```bash
git add benchmarks/morph_acceptance.py benchmarks/tests/test_morph_acceptance.py \
  benchmarks/results/pr132_morph_calibration.json \
  benchmarks/results/pr132_morph_formula_trials.md \
  crates/backend_cpu/src/morph.rs crates/engine/src/shared_histogram.rs \
  crates/core/src/training_mode.rs bindings/python/alloygbm/_regressor/_core.py \
  bindings/python/alloygbm/_morph.py bindings/python/tests/test_morph.py \
  bindings/python/tests/test_morph_helpers.py \
  bindings/python/tests/test_regressor_contract.py
git commit -m "bench: recalibrate MorphBoost defaults"
```

Before committing, unstage unchanged conditional paths so the commit contains only promoted code and
evidence.

### Task 7: Profile and conditionally close secondary Morph costs

**Files:**
- Modify: `crates/engine/src/morph_state.rs`
- Modify: `crates/engine/src/types.rs`
- Modify: `crates/engine/src/tests/morph_state.rs`
- Modify conditionally: `crates/backend_cpu/src/lib.rs`
- Modify conditionally: `crates/engine/src/shared_histogram.rs`
- Modify conditionally: `crates/engine/src/joint/fit.rs`
- Create: `benchmarks/results/pr132_morph_secondary.md`

**Interfaces:**
- Produces measured keep/defer decisions for EMA preparation, categorical scan, and joint counts.
- May produce one implementation only after its design threshold is crossed.

- [ ] **Step 1: Measure EMA, categorical, and joint cases independently**

Add release-mode ignored Rust timing tests named `benchmark_morph_ema_preparation`,
`benchmark_morph_categorical_scan`, and `benchmark_joint_morph_counts`. Each test performs warmup,
then seven measured batches and prints batch medians in the same parseable `name: ns_per_iter=`
style as the CPU benchmark. Run:

```bash
cargo test -p alloygbm-engine --release benchmark_morph_ema_preparation \
  -- --ignored --nocapture
cargo test -p alloygbm-backend-cpu --release benchmark_morph_categorical_scan \
  -- --ignored --nocapture
cargo test -p alloygbm-engine --release benchmark_joint_morph_counts \
  -- --ignored --nocapture
```

Record medians in `pr132_morph_secondary.md` and report each phase as a fraction of its
corresponding end-to-end fit.

- [ ] **Step 2: Apply the EMA threshold**

If EMA preparation is below 3% of native fit time and a direct-pair prototype improves end-to-end
time by less than 3%, leave it unchanged. Otherwise introduce a private `GradientMoments` value
from the existing diagnostics pass containing finite count, mean, and population standard
deviation; update `MorphState` from those moments without copying gradients. Pin f32/f64 behavior
with before/after EMA trajectory tests and run the full quality gate before retaining it.

- [ ] **Step 3: Apply the categorical threshold**

If categorical Morph scanning is below 10% of representative categorical fit time, record it as
deferred. If it exceeds 10%, vectorize only prefix-candidate arithmetic in a focused module, retain
Fisher ordering and bitset construction, and require scalar/SIMD winner parity plus the same 5%
end-to-end regression cap.

- [ ] **Step 4: Apply the joint-count threshold**

Construct joint-output fixtures where fractional Hessians cause proxy counts to differ from exact
row counts. If candidate ordering and five-seed quality remain neutral, record exact counts as
deferred. If a repeatable quality gap appears, add one shared `u32` count plane per feature/bin to
`MultiOutputHistogram`, fill it once per row independent of output, and use it in numeric and
categorical Morph scoring. Retain only if the quality gate passes, histogram time stays within 5%,
and representative joint peak memory stays within 10%.

- [ ] **Step 5: Run focused and full tests for retained work**

```bash
cargo test -p alloygbm-engine morph -- --nocapture
cargo test -p alloygbm-backend-cpu morph -- --nocapture
.venv/bin/python -m pytest bindings/python/tests/test_morph.py \
  bindings/python/tests/test_multi_label_ranker.py -q
```

- [ ] **Step 6: Commit only retained secondary changes and all decisions**

```bash
git add benchmarks/results/pr132_morph_secondary.md \
  crates/engine/src/morph_state.rs crates/engine/src/types.rs \
  crates/engine/src/tests/morph_state.rs crates/backend_cpu/src/lib.rs \
  crates/engine/src/shared_histogram.rs crates/engine/src/joint/fit.rs
git commit -m "perf: close measured MorphBoost secondary costs"
```

Before committing, unstage and restore by explicit patch any conditional implementation that missed
its threshold; the evidence document remains.

### Task 8: Final verification, documentation, and review closure

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `docs/user/morphboost.md`
- Modify: `docs/site/source/morphboost.rst`
- Modify: `docs/user/benchmarks.md`
- Modify: `docs/reviews/2026-07-02-v0.12.10-special-modes-resolutions.md`
- Create: `docs/benchmarks/morphboost_pr132.md`
- Modify: `docs/reviews/2026-08-02-pr-132-morphboost-performance-design.md`
- Modify: `docs/reviews/2026-08-02-pr-132-morphboost-performance-implementation-plan.md`

**Interfaces:**
- Produces the final reproducible performance/quality report and closes special-modes review §2.3.
- Records exact promoted and rejected behavior changes without overstating cross-platform speed.

- [ ] **Step 1: Run the final candidate benchmark ladder**

Rebuild release mode, rerun the seven-repeat scanner benchmark, full five-seed acceptance matrix,
and public Morph report. Run `perf_at_scale.py --scale medium` for auto/Morph/Morph-cosine. Run the
large scale and Numerai only when local memory/data permit; record an explicit omission reason
otherwise.

- [ ] **Step 2: Write the evidence report**

`docs/benchmarks/morphboost_pr132.md` must identify hardware, OS, source commits, commands, dataset
shapes, seeds, medians, scanner speedups, end-to-end changes, quality gate statistics, promoted
formula/defaults, rejected trials, and conditional secondary decisions. Distinguish scanner,
native-fit, and total-fit timing.

- [ ] **Step 3: Update user and review documentation**

Update formulas/default tables only for promoted changes. Explain that numeric split search remains
exhaustive and SIMD-vectorized, list scalar fallback combinations, and avoid claiming Morph always
beats auto. In the special-modes resolution, mark §2.3 fixed with PR #132 evidence and preserve links
to PR #111's calibration closure.

- [ ] **Step 4: Run all formatting, lint, test, and documentation gates**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
maturin develop --release
.venv/bin/python -m pytest bindings/python/tests/ benchmarks/tests/ -q
.venv/bin/python benchmarks/morph_ablation.py --gate
.venv/bin/python benchmarks/morph_acceptance.py \
  --arms auto morph_current --seeds 0 1 2 3 4 \
  --output /tmp/pr132_morph_final_verification.json
.venv/bin/sphinx-build -W -b html docs/site/source /tmp/alloygbm-pr132-sphinx
```

Expected: every command passes. Compare the final verification JSON to the committed final-candidate
records and require the same gate decision.

- [ ] **Step 5: Inspect the final diff and artifact compatibility**

```bash
git diff --check main...HEAD
git status --short
git diff --stat main...HEAD
git diff main...HEAD -- crates/core/src/artifact_format.rs crates/core/src/lib.rs
```

Expected: no uncommitted files, no artifact-schema changes, no experimental runtime switch, and no
top-k scanner. Confirm existing model-load tests passed in the full suite.

- [ ] **Step 6: Mark planning documents complete and commit closure**

Set both PR #132 planning-document statuses to `Implemented; ready for review`, then commit:

```bash
git add CHANGELOG.md docs/user/morphboost.md docs/site/source/morphboost.rst \
  docs/user/benchmarks.md docs/benchmarks/morphboost_pr132.md \
  docs/reviews/2026-07-02-v0.12.10-special-modes-resolutions.md \
  docs/reviews/2026-08-02-pr-132-morphboost-performance-design.md \
  docs/reviews/2026-08-02-pr-132-morphboost-performance-implementation-plan.md
git commit -m "docs: close MorphBoost scanner review finding"
```

- [ ] **Step 7: Prepare the draft PR without merging**

Push `codex/morphboost-performance` and open a draft PR summarizing scanner architecture,
warmup correction, measured speedups, quality gates, promoted/rejected trials, compatibility, and
full verification. Stop before requesting merge so the user's reviewers can inspect it.
