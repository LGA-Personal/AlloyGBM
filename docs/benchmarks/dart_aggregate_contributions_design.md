# DART Aggregate Contribution Design

Status: approved for implementation

Date: 2026-07-25

## Context

The July 2, 2026 special-modes review identified DART's repeated dropped-tree
prediction walks as a scaling problem. Each DART round currently:

1. walks every selected tree to subtract its weighted contribution before
   gradient computation;
2. builds the new tree;
3. walks every selected tree again to re-add its normalized contribution; and
4. repeats the same subtract/re-add work for validation data when present.

This costs `O(dropped_trees * rows * tree_depth)` twice per prediction domain.
The review explicitly recommends avoiding persistent per-tree prediction
columns because they require `O(rows)` memory for every trained tree.

The deterministic guardrail added by PR #123 confirms the scaling concern. On
the baseline branch, the `200 rounds / 0.20 drop rate / 50 max drops` stress
profile takes `7.34x` the matched standard fit time while completing all rounds.
Default-like DART profiles pass the existing quality contract.

## Objective

Reduce repeated DART tree traversal while preserving the existing public
dropout contract:

- identical dropout selection for the same seed and round;
- unchanged `dart_drop_rate`, `dart_max_drop`, `dart_normalize_type`, and
  `dart_sample_type` semantics and defaults;
- unchanged normalization factors and stored tree weights;
- unchanged warm-start, validation, artifact, and predictor contracts; and
- bounded reusable memory rather than persistent per-tree prediction caches.

This PR does not add or recalibrate an expected-drop cap. The current evidence
does not establish a defensible new threshold, especially for multiclass DART,
whose selection pool contains one class-tree per class and round.

## Selected Approach

Use a reusable aggregate contribution buffer for each active prediction
domain.

For selected trees `i`, define their pre-normalization weighted contribution:

```text
D(x) = sum_i w_old_i * f_i(x)
```

The existing dropout phase computes:

```text
P_drop(x) = P_full(x) - D(x)
```

Both DART normalization modes multiply every dropped tree's old weight by one
common factor:

```text
q = K / (K + 1)  for tree normalization
q = 1 / (K + 1)  for forest normalization
```

The post-build re-add therefore equals:

```text
sum_i w_new_i * f_i(x) = q * D(x)
```

During the required subtraction traversal, the trainer will also accumulate
`D(x)` into a scratch buffer. Finalization then applies `q * D(x)` with a
linear row loop instead of walking the dropped trees a second time.

This retains one tree traversal per dropped tree and prediction domain, plus
one `O(rows)` finalization pass.

## Rejected Alternatives

### Transient per-tree prediction columns

Caching each selected tree's row contributions would preserve the old re-add
order more closely, but requires `O(rows * dropped_trees)` temporary memory.
At the existing maximum of 50 drops, this is materially larger than the
selected `O(rows)` aggregate buffer and follows the storage strategy rejected
by the review.

### Expected-drop or default cap

Lowering the effective drop rate or default maximum would reduce both initial
and repeated traversal. It would also change selected trees, normalization,
trained models, warm-start continuations, and user-visible parameter behavior.
That requires a separate calibration matrix and API decision.

### Persistent per-tree prediction cache

Persisting every tree's train and validation predictions removes traversal but
grows as `O(total_trees * rows)` and complicates warm-start and validation
lifetimes. It is outside the acceptable memory contract.

## Components

### Scalar round helper

`crates/engine/src/round.rs` will gain a focused helper for applying one tree
to predictions while accumulating the tree's weighted contribution in a
same-length scratch slice.

The helper will reuse the current routing implementation for:

- learned missing-value direction;
- native categorical bitsets;
- scalar leaves; and
- piecewise-linear leaves with raw feature rows.

It will return `EngineError::ContractViolation` when prediction and scratch
lengths differ.

### Joint round helper

The joint trainer's output-major walker in
`crates/engine/src/joint/helpers.rs` will gain the equivalent optional
accumulation path. The scratch layout will match the existing
`outputs * rows` prediction layout exactly.

### Single-output trainer

The single-output trainer will allocate reusable train scratch storage and,
when validation is active, validation scratch storage outside the boosting
loop.

For each DART round it will:

1. clear the active scratch buffers;
2. select the same dropped tree IDs as before;
3. subtract each dropped tree and accumulate its old weighted contribution;
4. fit the new tree against the same dropped-out predictions;
5. scale the new tree contribution as before;
6. add the aggregate dropped contribution using the common normalization
   factor; and
7. commit the unchanged tree-weight bookkeeping.

The current full-prediction backups remain the rollback source for rejected
rounds and early exits.

### Multiclass trainer

The multiclass trainer will use output-major `classes * rows` scratch buffers.
Flat class-tree indexing, phantom zero-stump trees, level-wise and leaf-wise
tree slices, validation, and warm-start offsets remain unchanged.

Each selected flat tree contributes only to its class slice. The aggregate
buffer replaces both repeated train re-add walks and the corresponding
validation re-add walks.

### Joint trainer

The joint trainer will reuse one `outputs * rows` scratch buffer. Its existing
walker already updates all output leaves in one row traversal, so accumulation
will happen in the same traversal. Finalization adds the aggregate using the
normalization factor before committing the unchanged DART weights.

### DART state documentation

`DartState::tree_weights` documentation will be corrected to describe one
entry per logical tree, or one entry per flat class-tree in multiclass
training. No state layout changes.

## Numerical Contract

Dropout IDs, RNG inputs, normalization factors, and final tree weights must be
exactly unchanged.

The aggregate finalization groups floating-point additions differently from
the old repeated traversal. Helper-level results must match the reference
implementation within a tight `f32` tolerance, but trained artifacts are not
promised byte-identical. End-to-end deterministic quality and completion gates
must remain satisfied.

To minimize numerical drift:

- use `f32` scratch buffers, matching predictions and leaf values;
- preserve selected-tree traversal order;
- apply the normalization factor exactly once;
- do not parallelize accumulation in this PR; and
- keep all selection and weight calculations unchanged.

## Error and Rollback Contract

- Scratch length mismatches return a contract error rather than truncating or
  panicking.
- Scratch buffers are cleared before each dropout phase, including rounds with
  no selected trees.
- Empty and phantom trees contribute zero.
- Existing full-prediction backups remain authoritative for rejected rounds.
- Scratch storage is ephemeral and never serialized.
- Validation scratch exists only when validation predictions exist.

## Test Strategy

### Helper tests

Compare aggregate subtraction/finalization with the current repeated-walk
reference for:

- tree and forest normalization;
- empty and multi-stump trees;
- weighted contribution factors;
- missing-value routing;
- native categorical routing;
- piecewise-linear leaves; and
- joint multi-output leaves.

Add explicit mismatch-length error tests.

### State and trainer tests

- Assert dropout indices are unchanged for deterministic fixtures.
- Assert final DART tree weights are unchanged.
- Cover single-output validation, early stopping, warm start, ranking, and
  forest normalization.
- Cover multiclass multi-stump trees, leaf-wise growth, validation, warm
  start, and artifact round trips.
- Cover joint training-time versus predictor parity and warm start.

Existing suites remain the primary compatibility contract.

### Performance evidence

Run the full DART section of `benchmarks/review_guardrails.py` before and after
the implementation.

Required:

- all DART contract, completion, and quality gates pass;
- no timing threshold is added to CI; and
- the stress profile shows a material descriptive improvement over the
  recorded baseline, subject to normal local timing variance.

The resolution document will record both the behavioral scope and measured
before/after timing. If the remaining stress ratio is still excessive, an
expected-drop calibration PR remains open rather than being folded into this
change.

## Documentation

Update:

- the special-modes resolution document with the implementation and measured
  evidence;
- benchmark documentation with the new report or comparison;
- user-facing DART documentation only to clarify performance behavior, without
  adding parameters; and
- the unreleased changelog because training performance changes materially.

## Out of Scope

- changing dropout selection or RNG;
- changing any DART default;
- adding an expected-drop parameter or automatic policy;
- persistent per-tree prediction storage;
- artifact format changes;
- prediction-time changes; and
- unrelated scanner or histogram optimization.
