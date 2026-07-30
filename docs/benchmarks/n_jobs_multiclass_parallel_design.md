# Fit Thread Control and Parallel Multiclass Tree Builds

## Status

Approved design for PR #127, based on `main` at `99a61ea`.

This change closes two findings from the 2026-07-02 review:

- core review section 2.8: expose fit-time thread control through `n_jobs`;
- special-modes review section 6.4: build independent multiclass class trees in parallel.

## Goals

1. Add sklearn-style fit-time `n_jobs` control to every public estimator.
2. Isolate each native fit in one Rayon thread pool without mutating Rayon's global pool.
3. Parallelize independent per-class tree construction for multiclass softmax.
4. Keep nested Rayon work inside the same bounded pool so class-level and tree-level
   parallelism cannot multiply worker counts.
5. Preserve deterministic predictions, artifacts, class order, warm starts, DART
   bookkeeping, MorphBoost state, and early-stopping behavior across thread counts.
6. Add correctness, concurrency, and performance evidence across low/high class counts and
   tall/wide datasets.

## Non-Goals

- `n_jobs` does not control prediction, SHAP, metric evaluation after fitting, or user-owned
  Python/NumPy thread pools.
- This change does not coordinate multiple simultaneous estimator fits. Each fit owns its
  requested pool, so callers running concurrent fits remain responsible for assigning a
  suitable `n_jobs` to each estimator.
- This change does not alter objective mathematics, split scoring, tree growth, sampling,
  model artifacts, metadata, or schema versions.
- This change does not parallelize multiclass gradient calculation, DART mutation, validation
  prediction updates, loss evaluation, or round commit.
- This change does not introduce a second executor or dependency.

## Public API

`GBMRegressor.__init__` gains:

```python
n_jobs: int | None = None
```

`GBMClassifier` and `GBMRanker` inherit the parameter through their existing shared
constructor/signature machinery. `MultiLabelGBMRanker` accepts and forwards it through
`_per_label_kwargs` in independent mode and through the joint native bridge in joint mode.

The accepted values are:

| Value | Meaning |
| --- | --- |
| `None` | Preserve current behavior by using all logical CPUs visible to the process. |
| `-1` | Use all logical CPUs visible to the process. |
| positive integer | Create exactly that many Rayon workers for this fit. |

`0`, integers below `-1`, booleans, non-integral numbers, and non-numeric values raise a
Python `ValueError` naming `n_jobs`. Rust bridge entry points repeat the range validation so
direct native callers cannot bypass the contract.

`n_jobs` is returned by `get_params`, accepted by `set_params`, displayed by `repr`, preserved
by estimator pickle/state persistence, and visible through classifier/ranker signatures. It is
not written into the model artifact because it affects execution rather than inference.

## One-Pool Execution Model

Add `bindings/python/src/threading.rs` with two responsibilities:

```rust
pub(crate) fn resolve_fit_thread_count(n_jobs: Option<isize>) -> PyResult<usize>;

pub(crate) fn install_in_fit_pool<T, F>(
    n_jobs: Option<isize>,
    operation: F,
) -> PyResult<T>
where
    T: Send,
    F: FnOnce() -> PyResult<T> + Send;
```

Resolution uses `std::thread::available_parallelism`, falling back to one worker only if the
platform cannot report a value. `None` and `Some(-1)` resolve to that value. Positive values
are converted to `usize` without silently capping an explicit user request.

`install_in_fit_pool` builds a private `rayon::ThreadPool` with exactly the resolved count and
runs the operation through `ThreadPool::install`. Pool-construction errors become
`PyRuntimeError` values that identify the requested worker count.

Every PyO3 training entry point accepts `n_jobs=None` and wraps its detached native
quantization/training implementation in this helper. Parsing Python callables and dictionaries
continues while the GIL is held; CPU-heavy preparation and fitting run after `Python::detach`.
Existing custom objective/metric callbacks reattach to Python only at callback boundaries.

All Rayon iterators reached by the fit, including quantization, ranking gradients, histogram
tiles, node proposals, multiclass class builds, and joint gradients, execute in the installed
pool. Rayon nested iterators reuse the current pool and do not create another worker set.
Therefore a fit with `n_jobs=N` owns exactly `N` Rayon workers even when an outer class task
enters inner tree-building parallelism.

The global Rayon pool is never configured. Separate fits can use different `n_jobs` values in
sequence without a first-call-wins global setting.

## Multiclass Build Architecture

The current multiclass round computes all class gradients, then mutates each class tree and
candidate prediction serially. Replace only the class-tree build phase.

### Work Policy

Add:

```rust
const MIN_MULTICLASS_PARALLEL_WORK: usize = 16_384;

fn should_parallelize_multiclass_trees(
    class_count: usize,
    sampled_row_count: usize,
    sampled_feature_count: usize,
) -> bool;
```

The function returns true only when:

- at least two classes are present;
- `rayon::current_num_threads() >= 2`; and
- `class_count * sampled_row_count * max(sampled_feature_count, 1)` is at least
  `MIN_MULTICLASS_PARALLEL_WORK`, using saturating multiplication.

Small fits keep the current sequential class loop. Eligible fits use an indexed Rayon iterator,
so collection order is class order regardless of completion order.

### State Separation

Before dispatch:

1. Compute and, for GOSS, amplify all class gradient buffers exactly as today.
2. Copy current predictions into each class candidate buffer.
3. Update every MorphBoost EMA entry sequentially in ascending class order.
4. Capture immutable MorphBoost contexts after all EMA updates.
5. Record pre-round stump counts without mutating any class stump vector.

Each class worker receives:

- its class index;
- an immutable class gradient buffer;
- its own mutable candidate-prediction slice;
- immutable dataset, binned matrix, backend, feature tiles, parameters, controls, and factor
  exposures;
- an immutable class-specific MorphBoost context.

It returns:

```rust
struct MulticlassTreeBuildOutcome {
    class_index: usize,
    diagnostics: IterationDiagnostics,
    round_stumps: Vec<TrainedStump>,
}
```

The worker does not mutate `class_stumps`, DART state, round counters, losses, validation
predictions, or other classes' buffers.

After collection, the trainer consumes outcomes in ascending `class_index`, extends each
`class_stumps[class_index]`, and derives `any_tree_produced`. The existing full-row candidate
rebuild, DART normalization, loss gates, validation handling, early stopping, and commit logic
then run unchanged.

If multiple class workers fail, ordered collection returns the lowest class-index error. This
keeps failure selection deterministic.

### Nested Parallelism

Tree builders retain their adaptive histogram, split-feature, and node-level Rayon paths.
Those nested tasks execute in the same per-fit pool as the class iterator. No child
`ThreadPoolBuilder`, scoped OS thread, or per-class pool is allowed.

This shared-pool design is preferred to forcing every inner operation sequential:

- low-class-count, wide-feature fits can lend idle workers to feature scans;
- high-class-count fits naturally distribute workers across classes;
- Rayon work stealing balances uneven class trees;
- the worker ceiling remains `n_jobs`.

## Determinism and Compatibility

The implementation must preserve:

- exact class-major artifact layout;
- exact encoded tree IDs and per-class stump order;
- deterministic artifact bytes between `n_jobs=1`, `n_jobs=2`, and `n_jobs=-1`;
- prediction equality across thread counts;
- shared GOSS row samples and class-buffer amplification;
- MorphBoost EMA update order and warm-start snapshots;
- DART flat `round * K + class` indexing, phantom rounds, and weight stamping;
- validation and training early-stop truncation;
- level-wise and leaf-wise growth;
- native categorical, PL leaf, DRO, factor-neutralization, missing-value, interaction-constraint,
  and warm-start paths.

No field is added to `TrainParams` or model metadata. Thread control remains a Python binding
execution concern, while class-level parallel scheduling remains an engine concern.

## Tests

### Rust binding tests

- `resolve_fit_thread_count` accepts `None`, `-1`, `1`, and positive counts.
- It rejects `0` and values below `-1`.
- A two-worker private pool reports `rayon::current_num_threads() == 2`.
- Nested Rayon iterators observe only worker indices from that same two-worker pool.
- Two sequential private pools can use different sizes without global-pool interference.

### Rust engine tests

- A concurrency-recording backend observes overlapping multiclass root histogram builds under a
  four-worker pool on an eligible fixture.
- The same fixture does not overlap under a one-worker pool or below the work threshold.
- Parallel and sequential multiclass fits produce identical artifacts, predictions, round
  counts, diagnostics, and stop reasons.
- Equivalence covers level-wise, leaf-wise, GOSS, DART, MorphBoost, validation early stopping,
  warm start, and a class that produces no stumps.
- Error injection in two classes reports the lower class-index failure.

### Python tests

- Regressor, classifier, ranker, independent multi-label ranker, and joint multi-label ranker
  accept and retain `n_jobs`.
- Constructor, `set_params`, sklearn clone, repr, pickle, and save/load behavior are covered.
- Invalid values fail before native training.
- Binary/regression/ranking fits remain prediction-equivalent between one and multiple workers.
- Multiclass standard, GOSS, DART, MorphBoost, warm-start, and validation fits remain
  artifact/prediction-equivalent between one and multiple workers.

## Benchmark and Acceptance Evidence

Add `benchmarks/multiclass_parallelism_benchmark.py` and contract tests under
`benchmarks/tests/`.

The quick matrix uses one seed and reduced rounds. The full matrix uses three seeds and:

- tall/narrow: `32,768 x 8`, classes `3` and `12`;
- medium/wide: `4,096 x 128`, classes `3` and `12`;
- small control: `512 x 8`, classes `3` and `12`;
- level-wise and leaf-wise growth;
- `n_jobs=1` and `n_jobs=min(4, available CPUs)`.

Each record contains scenario identity, requested/resolved workers, fit seconds, completed
rounds, artifact SHA-256, prediction SHA-256, finite prediction status, multiclass log loss,
constant-class-prior log loss, and peak incremental RSS when available.

The gate requires:

- the exact canonical scenario matrix and unique record identities;
- all fits complete their requested rounds unless an identical stop reason occurs in both arms;
- finite predictions and losses;
- identical artifacts and predictions across thread-count arms;
- equal quality metrics across arms;
- every trained model beats the class-prior baseline;
- resolved workers never exceed the requested explicit positive value;
- at least one eligible high-class-count scenario is marked class-parallel.

Timing is descriptive in CI. The full local report must present per-shape medians and the
one-to-many-worker speedup. A production performance claim requires the high-class-count median
to improve without a median regression greater than 10% in low-class-count scenarios; otherwise
the implementation remains functionally correct but the class-parallel policy must be retuned
before this finding is marked resolved.

## Documentation and Review Closure

Update:

- `CHANGELOG.md`;
- `docs/user/gbmregressor.md`;
- `docs/site/source/estimator.rst`;
- benchmark indexes and generated evidence;
- `docs/reviews/2026-07-02-v0.12.10-core-resolutions.md` section 2.8;
- `docs/reviews/2026-07-02-v0.12.10-special-modes-resolutions.md` section 6.4.

The user documentation must state that `n_jobs` is fit-only, uses a private Rayon pool, does
not mutate global Rayon configuration, and does not coordinate concurrent estimators.

## Verification

Before opening the draft PR:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
maturin develop --release
.venv/bin/python -m pytest bindings/python/tests benchmarks/tests -q
.venv/bin/python benchmarks/review_guardrails.py --quick --gate
.venv/bin/python benchmarks/auto_policy_benchmark.py --quick --gate
.venv/bin/python benchmarks/monotone_constraints_benchmark.py --quick --gate
.venv/bin/python benchmarks/multiclass_parallelism_benchmark.py --quick --gate
.venv/bin/python -m sphinx -W -b html docs/site/source docs/site/build/html
```

The full multiclass benchmark must be regenerated from the final implementation commit, and
the resolution documents must cite that exact source commit.
