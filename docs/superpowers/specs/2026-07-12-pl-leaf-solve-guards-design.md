# PL Leaf Solve Guards Design

## Goal

Close the remaining numerical-stability portion of the v0.12.10 PL-leaf
review finding without changing the model artifact format, estimator API, or
the f32/SIMD histogram scan path.

The current PL split-gain and leaf-coefficient solves factor a regularized
8-by-8 normal-equation matrix in f32. A positive-definite but badly
conditioned matrix can therefore produce unstable finite coefficients instead
of taking the existing scalar-leaf fallback.

## Selected Approach

Keep `LinearHistogramBin` and the running f32 histogram accumulators unchanged.
Once a candidate child or selected leaf reaches a Cholesky solve, copy its
active statistics into fixed-size f64 work arrays. Run matrix assembly,
factorization, forward substitution, backward substitution, and gain
accumulation in f64. Convert only a verified finite final gain or coefficient
to f32.

Before factorization, inspect the regularized diagonal. A solve is rejected
when a diagonal entry is non-finite or non-positive, or when the ratio of the
largest to smallest diagonal exceeds `1e6`. During factorization, reject
non-finite or non-positive pivots and non-finite intermediate or final values.

The shared solve result makes the fallback explicit:

- Split-gain evaluation returns zero for a rejected side, so that candidate
  cannot outrank a valid positive-gain candidate.
- Leaf construction retains the existing scalar Newton intercept and uses an
  all-zero linear correction when its matrix is rejected.

## Components And Data Flow

`crates/backend_cpu/src/pl.rs` will gain a private f64 Cholesky helper that
accepts the existing f32 `xtg` and `xt_hx` arrays plus active dimension and
ridge value. It returns either the solved f64 vector or a rejection result.

`compute_pl_gain_one_side` will call the helper's factorization/forward-solve
path and compute `0.5 * y^T y` in f64. `cholesky_solve_alpha` will call the
same guarded factorization and solve `A alpha = -xtg` in f64. It preserves its
all-zero return contract on rejection. The existing public function signatures
and all callers remain unchanged.

The `1e6` diagonal-ratio limit applies after ridge regularization. It is
intentionally an internal fixed policy, not a new train parameter: selected
regressors are already standardized and the threshold exists only to reject
pathological matrices that cannot be trusted by the f32 artifact
representation.

## Error Handling And Compatibility

No user-visible errors are introduced. Invalid or numerically untrustworthy PL
systems use the existing scalar-leaf fallback, preserving the model's ability
to train. Valid well-conditioned inputs keep their current behavior, with
slightly more accurate solve arithmetic. Histograms, serialized artifacts,
prediction behavior, and old-model decoding remain unchanged.

## Tests

Backend unit tests will demonstrate:

1. A well-conditioned two-regressor matrix still produces the known gain and
   leaf coefficients.
2. A positive-definite matrix whose regularized diagonal ratio exceeds the
   limit produces zero gain and zero linear correction rather than unstable
   finite values.
3. An extreme but admissible finite matrix produces finite results through the
   f64 solve path.

The full Rust workspace and Python binding suite will run before opening the
draft PR. The special-modes resolutions document will mark the f64 selected
child solve and conditioning guard complete while retaining the unrelated PL
histogram-memory and benchmark work as open.

## Scope Boundaries

This PR does not change histogram accumulation precision, add top-k histogram
selection, add a user-facing condition threshold, or add benchmark CI jobs.
Those are separate review follow-ups with different performance and release
trade-offs.
