# PR #137 DART Expected-Drop Calibration Design

| Date | Author | Base | Status |
|---|---|---|---|
| 2026-08-14 | OpenAI Codex | `main` after PR #136 (`8a76ccb`) | Implemented locally; PR publication intentionally omitted |

## Objective

Calibrate the public `dart_max_drop` default using broad, predeclared A/B
evidence. The selected default must materially reduce long-fit DART dropout
work without materially degrading held-out quality across regression, binary
classification, multiclass classification, or ranking.

`dart_max_drop` is already public on AlloyGBM estimators. This PR will not add a
second expected-drop parameter. Explicit values retain their existing meaning:
after deterministic dropout sampling, at most `dart_max_drop` existing trees are
kept in the dropped set. For a uniform arm, expected selected work is
approximately:

```text
min(dart_max_drop, max(1, dart_drop_rate * existing_tree_count))
```

The forced-one rule applies when a positive drop rate samples no tree. Weighted
sampling has the same target expected count before per-tree probability
clipping, followed by the same realized cap.

## Current-State Constraint

PR #124's aggregate-contribution optimization already removed the second tree
walk for selected dropouts. It intentionally preserved `dart_max_drop=50` and
left calibration open. Existing evidence covers one regression fixture and
shows `max_drop=5` was both faster and slightly better than 20 or 50 at 100
rounds, but that matrix is too narrow to justify a public default change.

This PR changes policy only if a broader same-host matrix passes fixed gates.
If no candidate passes, the default remains 50 and the review finding closes
with negative evidence rather than a forced behavior change.

## Considered Approaches

### Selected: recalibrate the existing public default

Evaluate a fixed cap grid and change only the default value of
`dart_max_drop`. This keeps the API compact, preserves explicit behavior, and
directly addresses the review's excessive expected-work finding.

### Rejected: add `dart_expected_drop_cap`

A second cap would overlap `dart_max_drop`, require precedence rules, and make
it unclear whether users should tune expected or realized drops. The existing
parameter already bounds the selected set and therefore bounds expected work.

### Rejected: hidden horizon-dependent policy

Silently rewriting an explicitly supplied `dart_drop_rate` or `dart_max_drop`
would make model behavior harder to reproduce and explain. The configured
values continue to govern every round directly.

## Benchmark Contract

Create `benchmarks/dart_policy_calibration.py` with deterministic JSON inputs
and comparison output. Each timed fit runs in an isolated subprocess. Capture a
production baseline from `8a76ccb` before changing defaults and a candidate run
from the implementation branch.

### Candidate caps

Evaluate `2`, `5`, `10`, and `20` against incumbent `50`. The grid is fixed
before measurements. All arms pass an explicit `dart_max_drop`, so benchmark
results do not depend on which constructor default is currently installed.

### Coverage

Use five seeds and held-out deterministic synthetic data covering:

- small/narrow, small/wide, tall/narrow, and tall/wide regression;
- binary and four-class classification;
- grouped ranking evaluated with NDCG@10;
- 50, 100, 200, and 300-round horizons;
- default `drop_rate=0.10` plus a `0.20` long-horizon stress case;
- uniform/tree, weighted/tree, and uniform/forest DART policies;
- level-wise and leaf-wise growth; and
- warm-start continuation as a deterministic compatibility sentinel.

The benchmark records fit time, peak RSS, completed rounds, primary and
secondary held-out metrics, prediction and artifact digests, and configured
dropout pressure. Multiclass pressure uses the actual class-tree pool size, not
only the logical round count.

### Compatibility evidence

Explicit `dart_max_drop=50` on the candidate branch must match production
prediction and artifact digests for every deterministic compatibility record.
The selected explicit candidate must be deterministic across repeated fits and
`n_jobs` values. Uninterrupted and warm-start continuation must retain the
existing DART equivalence tolerance.

## Predeclared Selection Rule

For every candidate cap, orient the primary metric so a ratio above 1 is worse:
RMSE/log loss use `candidate / incumbent`; NDCG uses
`incumbent / candidate`.

A candidate is eligible only if all of the following hold:

1. every fit is finite and completes the requested rounds;
2. every fixture's five-seed median primary-quality ratio is at most `1.02`;
3. no individual-seed primary-quality ratio exceeds `1.10`;
4. binary/multiclass accuracy median does not fall by more than `0.02`
   absolute and ranking NDCG@10 median does not fall by more than `0.01`
   absolute;
5. median configured dropout pressure across 200/300-round stress fixtures is
   at most `50%` of incumbent 50;
6. median fit time across those stress fixtures is at most `85%` of incumbent;
7. peak RSS does not exceed incumbent by more than the larger of `15%` or
   `32 MiB`; and
8. determinism, explicit-50 production parity, and warm-start gates pass.

Select the **largest** eligible cap. This minimizes behavioral departure while
meeting the work-reduction target. Do not alter thresholds after observing
results. If no cap is eligible, retain 50.

Timing is a same-host policy-selection gate, not a universal speed claim. The
report must preserve all losing candidates and reasons.

## Implementation Scope

If a candidate is selected:

- change the default `dart_max_drop` value in all Python estimator constructor
  surfaces and test fixtures that intentionally pin defaults;
- keep validation (`>= 1`), explicit-value forwarding, native `BoostingMode`,
  dropout RNG, forced-one behavior, truncation order, and normalization exactly
  unchanged;
- add tests proving the new default and explicit `50` override;
- document the selected value, expected-work interpretation, benchmark tradeoff,
  and migration path (`dart_max_drop=50` restores the prior default); and
- make no artifact-format or prediction-path change.

Do not dynamically adjust the cap by horizon, class count, tree growth, or data
shape. The benchmark chooses one understandable public default.

## Verification

Required before draft PR creation:

- benchmark contract tests and full five-seed comparison;
- focused DART, multiclass DART, ranker, warm-start, estimator-contract, and
  sklearn-conformance tests;
- `cargo fmt --all -- --check`;
- `cargo clippy --workspace --all-targets -- -D warnings`;
- `cargo test --workspace`;
- release extension build;
- full Python and benchmark test suites;
- Sphinx with warnings as errors; and
- `git diff --check`.

Update the July special-modes resolution, benchmark indexes, README, estimator
documentation and Sphinx mirror, changelog, and roadmap. PR #137 remains draft
for independent review and is not merged by the implementation agent.
