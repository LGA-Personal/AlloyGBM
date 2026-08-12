# PR #136 Top-k PL Histogram Construction Design

| Date | Author | Base | Status |
|---|---|---|---|
| 2026-08-12 | OpenAI Codex | `main` after PR #135 (`ea4df36`) | Approved architecture; ready for implementation planning |

## Objective

Add an explicit `pl_split_candidates` estimator parameter and use it to restore
piecewise-linear (PL) split-gain selection without restoring the historical
all-feature matrix-histogram cost. Standard scalar histograms first identify a
small deterministic feature shortlist. Only those numeric features receive the
expensive `(X^T g, X^T H X)` statistics needed for exhaustive PL rescoring.

The default is `pl_split_candidates=8`. Setting it to `0` preserves the current
v0.12.10+ behavior exactly: select the split with the existing standard or Morph
criterion, partition the rows, and fit linear child models only for that selected
split.

## Current-State Constraint

Current `main` does not build full PL histograms during ordinary training. PRs
leading to v0.12.10 replaced the historical all-feature PL leaf solve with a
direct selected-partition solve. Consequently, this work is not merely a memory
optimization of an active all-feature path. It deliberately reintroduces
PL-aware split selection behind a bounded shortlist and must prove that the
accuracy and convergence benefit justifies its cost relative to
`pl_split_candidates=0`.

## Public API

`GBMRegressor`, `GBMClassifier`, and `GBMRanker` gain:

```python
pl_split_candidates: int = 8
```

The parameter contract is:

- non-boolean integer greater than or equal to zero;
- `0` selects the compatibility path and must reproduce current `main` artifacts
  byte for byte under deterministic training;
- a positive value is a maximum, clipped to the number of eligible numeric
  features at each node;
- values larger than the feature count are valid and mean "rescore every eligible
  numeric feature";
- constant-leaf fits accept but ignore the parameter;
- the parameter is training-only and does not add an artifact section or metadata
  field;
- estimator construction, `get_params`, `set_params`, `repr`, cloning, and pickle
  state all retain the value;
- all native single-output training entry points receive the value through
  `TrainParams`;
- joint multi-output training does not support linear leaves and therefore does
  not gain a separate behavior path.

`TrainParams.pl_split_candidates` is appended to the struct, defaults to `8`, and
is validated centrally. Python validation rejects `True`/`False`, negative
values, non-integral values, and integers that cannot be represented by the
native parameter type.

## Split Selection Architecture

### Stage 1: Standard shortlist

The CPU backend evaluates the existing per-feature split scanner over the
already-built scalar `HistogramBundle`. It returns:

1. the best overall standard candidate, including native categorical features;
2. up to `k` best numeric feature candidates, with at most one candidate per
   feature.

Ranking uses feature-weighted gain. Ties preserve deterministic feature order and
the existing `gain_materially_exceeds` tolerance. Interaction constraints and
column sampling have already filtered the histogram bundle, so ineligible
features cannot enter the shortlist.

The stage-1 candidate's threshold is used only to rank its feature. Stage 2
rescans every valid threshold and both missing directions for that feature under
the PL criterion.

### Stage 2: Candidate-specific PL rescoring

Each shortlisted feature is evaluated sequentially. Its regressor set is
computed with the existing `linear_regressor_path_features` rule using the
current path plus that candidate feature. This is required because different
candidate features produce different leaf models; a shared regressor set would
silently change PL semantics.

For one feature at a time, the backend:

1. accumulates scalar and matrix statistics into reusable per-thread scratch;
2. exhaustively calls the existing f64-guarded PL gain scanner;
3. applies the same row, Hessian, missing-direction, leaf-magnitude, and feature
   weight rules as the standard path;
4. solves the winning feature's linear child pair from the same statistics;
5. returns only the split candidate, regressor identity, and solved leaf pair.

The scratch histogram is then reused for the next shortlisted feature. No
`k`-sized matrix bundle survives the loop. When a later candidate wins, its small
prepared result replaces the prior result.

This avoids both historical failure modes: building matrix histograms for every
feature and rebuilding matrix statistics after the split has already been
selected.

### Selection and fallback rules

- `leaf_model="constant"` always uses the existing split dispatcher.
- `leaf_model="linear", pl_split_candidates=0` always uses the existing split
  dispatcher and selected-partition linear solve.
- MorphBoost retains its Morph split formula and selected-partition linear solve;
  `pl_split_candidates` is intentionally inactive for `training_mode="morph"`.
- If the best standard candidate is native categorical, it remains the winner
  and its immediate children remain constant leaves. This avoids comparing the
  scalar categorical score with a different PL numeric objective and preserves
  the documented categorical contract.
- If the best standard candidate is numeric, the numeric shortlist is PL
  rescored. The highest finite positive feature-weighted PL candidate wins.
- If histogram construction, factorization, or candidate validation produces no
  usable PL candidate, the best overall standard candidate wins and the current
  selected-partition solve/fallback runs.
- `neutralization="split_penalty"`, active monotone constraints, and
  `leaf_solver="dro"` already reject linear leaves and need no new combination.

The final candidate remains subject to the trainer's existing `min_split_gain`,
minimum-row, minimum-Hessian, and leaf-magnitude rejection behavior.

## Internal Boundaries

The implementation should introduce focused interfaces rather than moving the
selection policy into the CPU backend wholesale:

- an engine-level shortlist result containing the best overall candidate and the
  ordered numeric feature candidates;
- a `BackendOps` method for deterministic standard shortlisting;
- a `BackendOps` method that evaluates one shortlisted numeric feature with its
  candidate-specific regressors and returns a prepared PL split result;
- CPU-local reusable linear-bin scratch and a slice-based PL scanner so the
  histogram does not need to escape as an owned `Vec`;
- one engine helper that chooses between legacy, Morph, categorical, and PL
  shortlist paths for both level-wise and leaf-wise growth.

The existing public `build_linear_histograms_cpu`, `best_split_linear`, and
`compute_linear_leaf_pair` helpers remain available for compatibility and oracle
tests. Their implementation may share the new single-feature accumulation
primitive, but this PR does not remove those APIs or change artifact encoding.

## Error Handling

Invalid public parameter values fail before native training begins. Internal
shortlist and PL-rescore failures are handled conservatively:

- malformed standard histograms remain errors under the existing backend
  contract;
- an unavailable or invalid PL matrix candidate is skipped;
- if all PL candidates are skipped, training falls back to the standard winner;
- non-finite gain or leaf coefficients never enter a model;
- scratch is thread-local, works inside the fit's private Rayon pool, and remains
  reusable after callback errors or panics in tests;
- deterministic ordering is independent of Rayon scheduling.

## Verification Matrix

### Unit and integration correctness

Tests must cover:

- parameter default, validation, `get_params`, `set_params`, `repr`, clone, and
  pickle behavior on all three estimators;
- `k=0` deterministic artifact-byte and prediction parity with the production
  base for level-wise and leaf-wise growth;
- shortlist ordering, feature weighting, material-gain ties, and clipping when
  `k` exceeds eligible feature count;
- `k >= eligible_features` parity with a scalar exhaustive reference that uses
  the same candidate-specific path-regressor rule for every eligible feature;
- candidate-specific path regressors, including duplicate path features and the
  eight-regressor cap;
- exhaustive threshold and missing-direction parity against the existing PL
  feature scanner;
- standard categorical winner, MorphBoost, interaction-constraint, column
  sampling, and invalid/ill-conditioned PL fallbacks;
- deterministic artifacts across repeated fits and `n_jobs` values;
- both level-wise and leaf-wise growth;
- regression, binary classification, multiclass classification, ranking,
  quantile regression, raw-scale inputs, and missing regressor values.

### Benchmark pack

Create a machine-readable `benchmarks/pl_topk_performance.py` harness. Each fit
runs in a fresh subprocess so peak resident memory includes native Rust
allocations rather than Python `tracemalloc` only. The fixed matrix compares
`k = 0, 1, 8, all` with five seeds over:

- small/narrow and small/wide local-linear regression;
- tall/narrow and tall/wide regression;
- mixed-scale/raw-scale regression;
- nonlinear/noise-heavy regression;
- binary and multiclass classification;
- small-query and large-query ranking.

The harness records fit time, subprocess peak RSS, rounds completed, primary and
secondary quality metrics, prediction digest, artifact digest, and convergence
checkpoints. It also records a deterministic native allocation model based on
live `LinearHistogramBin` capacities.

Baseline evidence is captured from production `ea4df36` plus benchmark-only
files in a detached worktree. Candidate evidence is captured on the PR branch.

## Predeclared Acceptance Gates

The implementation is accepted only when all gates pass:

1. **Compatibility:** candidate `k=0` matches the production-base prediction and
   artifact digests for every deterministic record.
2. **Memory:** live native matrix-histogram storage is bounded to one feature per
   worker and is no more than `1 / F` of the existing all-feature bundle on each
   `F`-feature oracle fixture. For default `k=8`, subprocess peak RSS may exceed
   `k=0` by no more than the larger of 15% or 32 MiB on any repeated case.
3. **Quality:** the five-seed median default-`k=8` primary metric is within 1%
   relative of `k=0` on every case and improves at least one predefined
   PL-friendly case by at least 5%. Secondary metrics may not regress by more
   than 1% relative.
4. **Convergence:** on both predefined local-linear fixtures, median `k=8`
   reaches the `k=0` final-checkpoint quality in at most 75% of the rounds.
5. **Bounded fixed-round cost:** the median `k=8 / k=0` fit-time ratio across
   shapes is at most 3.0, and no case median exceeds 5.0.
6. **Shortlist benefit:** on fixtures with at least 32 eligible numeric features,
   median `k=8` fit time is at most 50% of exhaustive `k=all`, while remaining
   within 1% relative quality.
7. **Determinism:** repeated candidate runs have identical predictions and
   artifacts for the same seed and thread count; single-thread and multi-thread
   fits satisfy the existing deterministic contract.
8. **Repository health:** formatting, strict Clippy, all Rust tests, release
   extension build, all Python and benchmark tests, sklearn conformance, Sphinx
   with warnings as errors, and `git diff --check` pass.

Gate thresholds are fixed by this design. Failed gates require a
behavior-preserving optimization, a default reconsideration with explicit user
approval, or rejection of the implementation; they must not be weakened after
candidate measurements are known.

## Documentation and Review Closure

Update the README, estimator documentation and Sphinx mirror, benchmark index,
changelog, and the 2026-07-02 special-modes resolution. Documentation must state:

- default `8`, `0` compatibility behavior, and positive-value clipping;
- standard-gain feature shortlist followed by exhaustive PL threshold rescoring;
- MorphBoost and native categorical fallback behavior;
- the parameter controls a training-time accuracy/memory/time tradeoff and does
  not affect prediction cost or artifact compatibility;
- measured memory, quality, convergence, and runtime evidence, including any
  case where `k=0` remains preferable.

The resolution may mark the top-k PL histogram finding fixed only after all
predeclared gates pass. PR #136 remains a draft until independent review and must
not be merged by the implementation agent.
