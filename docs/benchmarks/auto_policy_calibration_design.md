# Auto-Policy Calibration and Diagnostics Design

Date: 2026-07-25

Status: approved for implementation

Review target:
[`docs/reviews/2026-07-02-v0.12.10-core.md`](../reviews/2026-07-02-v0.12.10-core.md),
section 1.3

## Objective

Re-evaluate AlloyGBM's `training_policy="auto"` after the split-search
feasibility and default quantile-binning fixes, then either recalibrate the
remaining heuristics or close the review finding with evidence that the
current policy is preferable to the tested alternatives.

The calibration must be robust across row and column shapes. Results from a
single real-world dataset, an aggregate median, or one objective cannot justify
a global default. Small/tall, small/wide, large/narrow, and large/wide strata
must be visible and independently protected.

This work keeps `training_policy="auto"` as the public default, preserves
explicit manual parameters, and does not change the artifact format.

## Current Policy Surface

For sufficiently large datasets, auto policy can change:

- the effective round cap for noisy small-wide data;
- `min_rows_per_leaf` at row-count thresholds;
- a density- and size-dependent `min_split_gain` floor;
- `row_subsample` at 2,048 and 16,384 rows;
- `col_subsample` at 32, 128, and 256 features; and
- an automatic split-L2 value for noisy small-wide data.

Ranking objectives already bypass the regression-tuned split-gain floor.
Training-loss stopping is opt-in and is not part of this calibration.

## Considered Approaches

### Evidence-backed calibration and diagnostics

This is the selected approach. Measure the current policy against manual and
targeted candidate policies, expose the effective controls, and modify only a
heuristic supported across the shape and objective matrix.

Advantages:

- distinguishes stale review evidence from current behavior;
- identifies which decision point causes any regression;
- gives users visibility into the values auto policy selected; and
- avoids replacing one uncalibrated global rule with another.

### Force a quality-first auto policy

Remove implicit sampling and split-gain floors immediately. A three-seed probe
on California Housing was mixed: current auto beat manual on every seed,
removing the gain floor was neutral, and removing sampling helped one seed but
hurt two. This approach is not supported without broader evidence.

### Make manual policy the default

This was a conditional suggestion in the review. The current probe no longer
shows the original auto-policy deficit after prior fixes, so a default switch
would be a compatibility change without supporting evidence.

## Resolved Policy Diagnostics

Introduce an engine-level `ResolvedTrainingPolicy` snapshot containing:

- requested mode (`auto` or `manual`);
- requested rounds and effective round cap;
- effective `min_rows_per_leaf`;
- effective `min_split_gain`;
- effective `row_subsample`;
- effective `col_subsample`; and
- whether the small-wide automatic split-L2 rule activated, plus its effective
  L2 value.

The snapshot is created from the exact controls and split-selection options
used by training. Scalar and multiclass iteration summaries carry it to the
PyO3 `NativeTrainingSummary`.

After a successful fit, Python estimators expose
`resolved_training_policy_` as a plain dictionary with stable snake-case keys.
The attribute is diagnostic only: it is not a constructor parameter, does not
participate in `get_params`, and is not serialized into model artifacts.
Manual fits report their effective values too, allowing direct inspection of
user overrides.

Older or mocked native summaries without the new field remain accepted by the
Python assignment path and set `resolved_training_policy_` to `None`.

## Calibration Matrix

The offline harness `benchmarks/auto_policy_benchmark.py` generates
deterministic held-out fixtures with independent row-count and feature-count
axes.

### Shape strata

The full matrix includes:

| Stratum | Representative shapes | Policy boundaries exercised |
|---|---|---|
| small-narrow | 512 x 8, 1,023 x 16 | small-data passthrough |
| small-wide | 512 x 128, 1,023 x 256 | small-wide round cap and split L2 |
| medium-narrow | 2,048 x 16, 8,192 x 16 | leaf-row and row-sampling steps |
| medium-wide | 2,048 x 128, 8,192 x 256 | row and column sampling together |
| large-narrow | 16,384 x 16 | large-row sampling step |
| large-wide | 16,384 x 256 | maximum row/column sampling pressure |

Boundary-focused unit tests additionally evaluate one row below and exactly at
1,024, 2,048, 8,192, and 16,384 rows, and one feature below and exactly at 32,
128, and 256 features. Unit tests inspect resolved controls rather than train
large models. Small-wide calibration fits request more than 256 rounds so the
conditional 96-round cap is observable; the other full-matrix fits use a
smaller fixed round count appropriate for repeated held-out evaluation.

### Objective strata

The full harness covers:

- nonlinear heteroscedastic regression;
- sparse wide regression;
- binary classification;
- multiclass classification; and
- grouped LambdaMART ranking.

Each generated target includes learnable signal, nuisance features, and a
held-out split. Wide fixtures mix informative and irrelevant columns so column
sampling is meaningfully exercised. Ranking fixtures use multiple train and
held-out queries and report NDCG.

The full run uses seeds `7`, `13`, and `29`. California Housing is retained as
an optional historical-reference arm when locally available, but it is not
part of candidate selection or CI.

### Policy arms

Each fixture compares:

- `current_auto`: the production auto policy;
- `manual_default`: explicit manual mode with public defaults;
- `no_gain_floor`: current resolved auto controls with only the implicit
  split-gain floor removed; and
- `quality_first`: current resolved auto controls with the implicit gain floor
  removed and implicit row/column sampling restored to `1.0`.

Candidate arms use manual mode with explicit values derived from the
`current_auto` diagnostic snapshot. This isolates a policy decision without
adding benchmark-only production modes. When current auto selected the
split-only small-wide L2 rule, the harness reproduces it for the manual
candidate through the existing `ALLOYGBM_EXPERIMENT_SPLIT_L2` environment
setting. Before every fit, the harness snapshots the complete known set of
training-affecting `ALLOYGBM_EXPERIMENT_*` variables, clears all of them, sets
only split-L2 when the selected arm requires it, and restores every exact prior
value or absence in `finally`. This process-global API is explicitly
single-process and serial. It never substitutes estimator `lambda_l2`, which
would also change leaf regularization.

## Metrics and Selection Rules

Every record includes fixture, shape stratum, objective, seed, arm, effective
controls, completed rounds, fit seconds, and held-out metric.

Metrics:

- RMSE for regression;
- log loss and accuracy for binary and multiclass classification; and
- NDCG@10 for ranking.

Timing is descriptive and cannot make a lower-quality candidate pass.

A candidate is rejected if:

- any fit errors, produces non-finite output, or completes zero rounds;
- regression RMSE or classification log loss exceeds current auto by more
  than 3% in any protected shape/objective stratum;
- classification accuracy falls by more than 0.02 in any protected stratum;
- ranking NDCG@10 falls by more than 0.02 in any protected stratum; or
- its median normalized primary loss is worse than current auto within any
  shape stratum.

Among surviving candidates, a production change requires at least 1% lower
overall median normalized primary loss than current auto and no protected
stratum regression. If multiple candidates qualify within 0.5%, prefer the
one closest to current behavior, then the faster median fit.

If no candidate qualifies, the policy remains unchanged. The benchmark,
diagnostics, and resolution evidence still close the stale calibration
finding by demonstrating that a forced retune is not supported after the
earlier correctness fixes.

## CI Guardrail

The full Cartesian calibration is an offline evidence run. CI uses a compact
sentinel matrix that includes one fixture from each shape stratum and rotates
objectives across those shapes. It uses one fixed seed and fewer rounds.

CI gates:

- deterministic fixture and arm enumeration;
- complete finite records;
- diagnostic values matching the configured manual candidate;
- correct shape-stratum assignment;
- candidate-rejection logic on synthetic bad records; and
- current auto completing every sentinel fit with valid held-out metrics.

CI does not enforce wall-clock thresholds or require an experimental candidate
to beat production auto on one-seed reduced data.

## Engine and API Changes

1. Extract resolved-policy construction into a pure helper in
   `crates/engine/src/trainer/policy.rs`.
2. Store the snapshot in scalar and multiclass iteration summaries.
3. Add a Python-visible `NativeResolvedTrainingPolicy` pyclass nested in
   `NativeTrainingSummary`.
4. Assign `resolved_training_policy_` after scalar, binary, and multiclass
   fits, with compatibility fallback for mocked/older summaries.
5. Keep joint multi-output training out of scope because its trainer does not
   consume the same `IterationControls` auto-policy path.

No new constructor argument, artifact section, or prediction behavior is
introduced.

## Error Handling

- Resolved diagnostics are created only after parameter, dataset, and policy
  validation succeed.
- Invalid candidate records cause the benchmark gate to fail with fixture,
  seed, arm, and field context.
- Every record must satisfy the objective-specific metric schema, including
  finite ranges and the ranking identity `primary_metric == 1 - ndcg_at_10`.
- Candidate generation refuses missing or malformed resolved diagnostics
  rather than guessing effective controls.
- Manual user values remain lower bounds or explicit values exactly as in the
  production policy resolver.

## Testing

Rust tests cover:

- every row-count and feature-count boundary;
- ranking split-gain exemption;
- user overrides taking precedence;
- small-wide round-cap and L2 activation;
- manual-mode identity; and
- scalar/multiclass summaries carrying the exact resolved snapshot.

Python tests cover:

- fitted diagnostic keys and values for auto and manual estimators;
- binary and multiclass assignment;
- mocked-summary compatibility;
- no participation in `get_params`; and
- benchmark fixture determinism, shape classification, arm derivation,
  metric aggregation, and rejection rules.

Final verification includes:

- `cargo fmt --all -- --check`;
- `cargo clippy --workspace --all-targets -- -D warnings`;
- `cargo test --workspace`;
- release `maturin develop`;
- full Python and benchmark pytest suites;
- quick and full auto-policy benchmark gates;
- existing review guardrails;
- Sphinx with warnings as errors; and
- `git diff --check`.

## Documentation

The PR updates:

- `CHANGELOG.md`;
- `docs/benchmarks/README.md`;
- a generated `docs/benchmarks/auto_policy_calibration_v1.md` evidence report;
- the core review resolution ledger; and
- estimator documentation for `resolved_training_policy_`.

The evidence report publishes per-shape and per-objective results, candidate
selection outcome, environment metadata, and the exact command used. It states
explicitly whether production heuristics changed and why.

## Non-Goals

- changing `training_policy` defaults;
- adding a learned or validation-adaptive policy optimizer;
- tuning user-supplied manual parameters;
- changing early-stopping semantics;
- changing joint multi-output policy behavior;
- adding wall-clock CI thresholds; or
- using California Housing or any single dataset as the global calibration
  target.
