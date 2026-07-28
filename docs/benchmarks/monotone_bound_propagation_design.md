# Monotone Bound Propagation Design

Date: 2026-07-26

Status: approved for implementation

Review finding:
[`docs/reviews/2026-07-02-v0.12.10-core.md` section 1.4](../reviews/2026-07-02-v0.12.10-core.md#14-true-monotone-constraint-enforcement-bounds-propagation)

## Objective

Replace post-hoc sibling-order rejection with interval propagation so every
freshly trained scalar tree is globally monotone in each constrained numeric
feature. Descendant leaves in one side of a constrained split must remain
ordered relative to every descendant leaf on the other side, including cousin
subtrees.

This change must:

- support level-wise and leaf-wise growth;
- compose with standard and GOSS boosting, MorphBoost, DRO, missing-value routing,
  interaction constraints, row sampling, and column sampling;
- preserve the exact no-constraint path;
- leave model artifacts unchanged; align public Rust prediction arithmetic
  with compact artifact prediction; and
- reject combinations for which AlloyGBM cannot make the same global
  guarantee.

## Existing Failure

The current builders select a split, calculate absolute child outputs, and
reject the node when a constrained sibling pair is locally reversed. They do
not retain an ancestor interval. A later split can therefore produce a
locally ordered pair whose values cross a cousin subtree from the opposite
side of an earlier constrained split.

A deterministic fixture on current `main` demonstrates the defect:

- NumPy RNG seed `0`;
- 512 rows and three continuous features sampled uniformly from `[-2, 2]`;
- target
  `1.2*x0 + 2.5*sin(2.2*x1) + 1.5*x0*x2 + Normal(0, 0.35)`;
- twelve depth-four trees, learning rate `0.2`, manual policy, quantile
  binning, and `monotone_constraints=[1, 0, 0]`.

At fixed `x1=-2` and `x2=-2`, the fitted prediction falls from approximately
`2.2371` to `0.7982` as `x0` rises from `0.125` to `0.15625`. The regression
suite will pin this fixture rather than relying only on simple monotone
targets that the local sibling check already handles.

## Chosen Approach

Use builder-level scalar intervals, matching the basic LightGBM monotone
method. Split gain calculation remains unchanged in this PR. The selected
candidate is made feasible by bounding its emitted child outputs instead of
discarding the node.

Bound-aware candidate gain rescoring is deliberately separate. Implementing
it now would alter every standard, SIMD, categorical, MorphBoost, DRO, and
factor split scanner. It may improve constrained-model quality, but it is not
required for the global correctness guarantee and should be justified by
candidate evidence before expanding those hot paths.

## Interval Model

Add an engine-private copyable value:

```text
MonotoneBounds { lower: f32, upper: f32 }
```

The root interval is
`[-controls.max_abs_leaf_value, controls.max_abs_leaf_value]`. Every active
level-wise node and queued leaf-wise split carries its interval.

For a node interval `[a, b]`:

1. Calculate the existing raw absolute scalar child outputs.
2. Clamp both outputs to `[a, b]`.
3. Inspect the selected split feature's constraint.
4. For an unconstrained split, both children inherit `[a, b]`.
5. For a constrained split, calculate a finite midpoint in f64 from the two
   already bounded child outputs and clamp it to `[a, b]`.
6. For an increasing constraint:
   - left child interval: `[a, midpoint]`;
   - right child interval: `[midpoint, b]`.
7. For a decreasing constraint:
   - left child interval: `[midpoint, b]`;
   - right child interval: `[a, midpoint]`.
8. Clamp each child output to its child interval.
9. Calculate stored parent-relative deltas from those final absolute outputs.

When a sibling pair is already ordered, its outputs remain unchanged because
the midpoint lies between them. When it is reversed, both outputs meet at the
midpoint instead of rejecting the split. Descendants can move within their
own intervals but cannot cross the sibling boundary.

The leaf-magnitude check runs on the final bounded deltas. All bounds and
outputs must remain finite and ordered; violations are engine contract errors,
not silent fallback.

## Builder Integration

### Shared helper

Keep interval arithmetic in a small private module or focused section of the
tree-builder module. Both growth strategies must call the same helper for:

- root interval construction;
- inherited clamping;
- constrained midpoint partitioning; and
- final child output clamping.

Pure unit tests will cover the arithmetic independently of histogram and
queue behavior.

The same module will expose tree-level projection and validation helpers.
They reconstruct absolute scalar outputs from the existing parent-relative
stumps, traverse nodes in tree-local order with the same interval rules, and
either:

- rewrite the deltas to their projected bounded outputs; or
- report whether an existing tree already satisfies the contract.

This keeps builder, refinement, and warm-start semantics on one implementation
of the monotone mathematics.

### Level-wise growth

Extend the active-node state and `LevelNodeChildren` with a
`MonotoneBounds`. Node proposals receive the inherited interval before they
calculate scalar deltas. Accepted child entries carry the derived intervals
through both sequential and Rayon proposal paths. Ordered commit behavior
remains unchanged.

### Leaf-wise growth

Extend `PendingSplit` with a `MonotoneBounds`. Root queue insertion uses the
root interval; child queue entries use the derived intervals. Priority
ordering remains based on the existing split gain.

### Post-growth leaf refinement

Quantile refinement and the opt-in regression leaf-refinement experiment
replace scalar outputs after tree construction. After either refinement:

- run the shared monotone projection over each affected tree;
- rewrite parent-relative deltas before predictions are rebuilt; and
- return a contract error for malformed, non-scalar, or structurally
  incomplete trees instead of retaining an unverified result.

Quantile and refined squared-error models therefore retain the same global
guarantee. The projection may move an empirical leaf optimum to the nearest
feasible interval boundary; this is the required constrained optimization
tradeoff and will be included in quality evidence.

## Supported and Rejected Combinations

The strict contract applies to scalar tree outputs. The following
combinations are rejected with explicit configuration errors:

- `boosting_mode="dart"` with any nonzero monotone constraint. DART predictor
  weights can change exact f32 ordering after tree projection, so neither
  transient dropout predictions nor final weighted predictions carry the
  monotonicity guarantee. Empty or all-zero constraints remain valid and keep
  ordinary DART behavior unchanged.
- `leaf_model="linear"` with any nonzero monotone constraint. A bounded
  intercept does not constrain the leaf's linear terms or cross-leaf output
  ranges.
- Multiclass softmax with any nonzero monotone constraint. Applying one
  direction to every class logit does not guarantee that any transformed
  class probability is monotone.
- A nonzero monotone constraint on a feature actively using native
  categorical splitting. An arbitrary category subset has no lower/upper
  order compatible with numeric monotonicity.

Standard and GOSS scalar ensembles are covered. Binary classification remains
supported because sigmoid preserves logit ordering. Regression, ranking,
quantile, GLM, and scalar custom-objective outputs remain supported.

Target- or frequency-encoded categorical columns remain eligible when they
are trained as ordered numeric features rather than active native
categorical splits.

Joint multi-output training already rejects monotone constraints and is
unchanged.

## Warm Start and Persistence

No artifact section or metadata field is added. Final bounded scalar deltas
are stored in the existing tree representation. Both public Rust and compact
artifact prediction first evaluate a selected path into one zero-based local
f32 contribution per tree, then add that completed contribution once to the
model baseline. Public Rust prediction groups stumps by decoded tree ID first,
so manually interleaved public stump vectors retain the same arithmetic as the
compact predictor. Artifact bytes and compact encoding remain unchanged.

Warm-starting validates every retained scalar tree against the requested
constraint intervals before adding rounds. A model produced by this
implementation passes unchanged. A legacy initial model whose trees violate
the contract is rejected with an actionable error rather than being silently
presented as a monotone result. Existing legacy artifacts remain loadable and
predict identically when used for inference; only constrained warm-start
training adds this validation. A retained prefix with any non-unit DART tree
weight is also rejected under active constraints, regardless of the resumed
fit's current boosting mode.

## Testing

### Rust unit and integration tests

- Root and inherited interval clamping.
- Increasing and decreasing interval partitioning.
- Ordered siblings remain unchanged.
- Reversed siblings meet at the midpoint.
- Nested constrained splits cannot cross cousin intervals.
- Post-refinement projection restores valid deltas and cousin ordering.
- Structurally monotone warm-start trees pass without mutation.
- A legacy violating warm-start tree is rejected.
- Level-wise and leaf-wise builders carry child bounds.
- Empty/all-zero constraints retain existing artifact bytes.
- Core validation rejects linear leaves with active constraints.
- Multiclass training rejects active constraints.
- Tree builders reject native-categorical overlap.

### Python regression tests

- The deterministic seed-0 counterexample is nondecreasing on a dense `x0`
  grid for both growth modes.
- A mirrored decreasing fixture is nonincreasing.
- Multiple fixed values of the unconstrained features are checked; aggregate
  correlation is not an adequate assertion.
- Binary probabilities preserve the requested direction.
- Missing training values do not weaken finite-grid monotonicity.
- Representative GOSS, MorphBoost, and DRO fits remain monotone.
- Active DART constraints fail with an actionable configuration error, while
  all-zero constraints retain ordinary DART behavior.
- PL and multiclass combinations fail with stable actionable messages.
- Native-categorical overlap fails while an unconstrained categorical feature
  remains supported.
- Save/load and same-version warm start preserve the guarantee.
- Constrained warm start rejects a deliberately violating legacy fixture.

### Benchmark evidence

Add a deterministic offline benchmark and committed report spanning:

- small, medium, and large row counts;
- narrow and wide feature counts;
- increasing and decreasing constraints;
- regression and binary classification;
- level-wise and leaf-wise growth; and
- three fixed seeds.

The gate requires:

- complete unique records with finite metrics;
- zero finite-grid monotonicity violations;
- completion of requested rounds;
- constrained regression loss no worse than `1.25x` the matching
  unconstrained control;
- constrained classification error no more than `0.08` absolute worse than
  the matching unconstrained control; and
- every constrained model to beat its constant-predictor baseline.

Timing is descriptive only. The report records the exact source commit,
environment, shapes, objectives, growth modes, seeds, quality ratios, and
maximum observed adjacent violation.

## Documentation

Update:

- `CHANGELOG.md`;
- `docs/user/gbmregressor.md`;
- `docs/site/source/estimator.rst`;
- `docs/site/source/architecture.rst`;
- `docs/limitations.md` if its constraint wording needs correction;
- the benchmark index and committed evidence report; and
- `docs/reviews/2026-07-02-v0.12.10-core-resolutions.md` section 1.4.

The user docs must describe the scalar guarantee, rejected combinations,
finite-value interpretation for missing data, post-refinement projection, and
legacy warm-start validation.

## Non-Goals

- Bound-aware split-gain rescoring.
- Monotone piecewise-linear solvers.
- Multiclass probability-specific constraint vectors.
- Ordered native-categorical constraints.
- Joint multi-output monotone constraints.
- Artifact format changes.
