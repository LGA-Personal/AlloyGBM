# PR #131 sklearn Conformance Design

| Date | Reviewer | Version reviewed | Commit | Status |
|---|---|---|---|---|
| 2026-08-01 | OpenAI Codex | `main` after PR #130 | `1ed8fd6` | Approved for implementation |

## Objective

Make `GBMRegressor` and `GBMClassifier` pass every applicable scikit-learn estimator check on
the declared development floor, scikit-learn 1.8, and the current supported line,
scikit-learn 1.9. Give `GBMRanker` equivalent group-aware conformance coverage without weakening
its documented requirement that callers provide query groups.

This PR closes section 4.3 of the 2026-07-02 core review. It does not add sparse training,
multi-output regression/classification, or array API support. Unsupported inputs must be described
accurately by estimator tags and rejected with stable, explicit errors.

## Current Failure Inventory

At the reviewed commit, scikit-learn 1.9 runs only 47 generic checks against `GBMRegressor`
because its reversed mixin order leaves `estimator_type=None`; 19 checks fail and one
environment-driven array API check skips. Supplying correct regressor tags exposes 58 checks, of
which 25 fail. Supplying classifier tags exposes 61 checks for `GBMClassifier`, of which 29 fail.

The failures reduce to shared root causes rather than independent special cases:

- fitted attributes are created with `None` values in `__init__`, so `check_is_fitted` reports a
  fresh estimator as fitted;
- `predict` and related methods raise generic runtime errors instead of `NotFittedError`;
- constructor and `set_params` eagerly validate, coerce, and sometimes copy parameter values;
- the public fitted feature dimension is private (`_n_features_in`) rather than
  `n_features_in_`;
- custom array adapters produce inconsistent errors for sparse, complex, empty, one-dimensional,
  and non-array protocol inputs;
- sample weights reject zero rather than implementing zero weight as row omission;
- classifier inheritance combines regressor and classifier mixins in the wrong order, its tag
  method fails when `classifier_tags` is absent, and its label codec is integer-only.

## Estimator Architecture

Shared training, prediction, persistence, SHAP, quantization, and validation behavior will move
behind one private estimator core that terminates in scikit-learn's `BaseEstimator` when sklearn is
installed and in the existing no-dependency fallback otherwise. Public estimators become semantic
shells with specialized mixins before that core:

- `GBMRegressor(RegressorMixin, _GBMEstimatorCore)`;
- `GBMClassifier(ClassifierMixin, _GBMEstimatorCore)`;
- `GBMRanker(_GBMEstimatorCore)` with explicit target-required and dense-input tags.

`GBMClassifier` will not inherit `RegressorMixin`, and `GBMRanker` will not claim to be a generic
regressor. Existing public import paths, constructor signatures, native bridge calls, save/load
entry points, and pickle resolution remain stable. The package must continue to import and operate
without scikit-learn installed.

## Fitted-State Contract

Fresh estimators will contain constructor parameters and private runtime scaffolding only. Public
attributes ending in `_` will be created only after the corresponding state exists. In particular:

- successful fit and model load create `n_features_in_`;
- `feature_names_in_` exists only when all input feature names are strings;
- training diagnostics such as `best_iteration_`, `n_estimators_`, `evals_result_`, and
  `fit_timing_` do not exist before fit;
- resetting or replacing fitted state deletes stale public fitted attributes rather than assigning
  `None` placeholders;
- `predict`, probability methods, `score`, SHAP, feature importance, persistence, and artifact
  access use one fitted-state guard.

When scikit-learn is installed, the guard delegates to `check_is_fitted` and raises
`NotFittedError`. The no-sklearn fallback raises an equivalent AlloyGBM runtime error without
adding a mandatory sklearn dependency. Warm starts and loaded models continue to use private
artifact state, but they must also restore the public fitted schema consistently.

## Parameters And Validation Timing

`__init__` and `set_params` will be assignment-only for known parameters. They will preserve
supplied object identity, avoid numeric/string coercion, and avoid cross-field validation. Unknown
parameter names still fail immediately. A single `_validate_hyperparameters()` pass will run at
the start of `fit`, before training or fitted-state mutation, and will retain the existing value and
cross-field constraints.

This deliberately changes invalid-parameter timing: invalid known values are accepted by
construction and `set_params` but rejected by `fit`. The change is required for sklearn cloning,
parameter search, and `check_do_not_raise_errors_in_init_or_set_params`, and will be documented in
the changelog and estimator guides.

`get_params(deep=True)` will expose the exact constructor surface. `set_params` will use
BaseEstimator semantics when sklearn is available and an equivalent local implementation
otherwise. Classifier objective restrictions and all specialized-mode constraints move to fit-time
validation rather than bypassing this contract.

## Input And Feature Validation

One shared validation boundary will serve fit and inference while retaining AlloyGBM's categorical
and zero-copy fast paths.

For ordinary numeric inputs, sklearn validation will be used when available with these tags and
rules:

- two-dimensional dense arrays only;
- sparse inputs unsupported and rejected with an error explicitly containing `sparse`;
- NaN feature values allowed;
- infinities and complex values rejected;
- at least one sample and one feature required;
- feature count and string feature names recorded on fit and checked on inference.

The no-sklearn path will enforce equivalent shape, dtype, finiteness, sparse, and schema errors.
Categorical/DataFrame adapters remain supported, but share the same dimensionality, sparse,
feature-count, and feature-name boundary before entering their specialized conversion logic.
Validation must not force the established NumPy/native fast paths through Python lists.

Regression targets must be finite numeric one-dimensional arrays. A two-dimensional single-column
target is flattened with the standard sklearn warning; true multi-output targets remain unsupported.
Classifier targets use sklearn target typing when available and equivalent fallback logic otherwise.

## Sample-Weight Semantics

Sample weights must be one-dimensional, finite, nonnegative, and length-matched. An all-zero vector
raises a message stating that the weights sum to zero. Rows with zero weight are removed before
quantization, categorical encoding, grouping, factor neutralization, or native training so fitting
with a zero-weight row is equivalent to omitting that row.

The row filter must be applied consistently to `X`, `y`, groups, categorical value columns, time
indices, and factor exposures. Validation/evaluation weights follow the same nonnegative contract.
For ranking, groups that become empty after filtering disappear; surviving group IDs retain their
relative query ordering.

## Classifier Labels And Outputs

`GBMClassifier` will accept binary and multiclass targets with numeric, boolean, or string labels.
Continuous, multilabel, non-finite, mixed-incomparable, and malformed targets will be rejected with
sklearn-compatible target-type messages.

`classes_` will be a sorted one-dimensional NumPy array. `predict` will return a one-dimensional
NumPy array in the original label dtype, while `predict_proba` and `predict_log_proba` retain their
standard `(n_samples, n_classes)` numeric arrays. The binary and multiclass native objective
selection remains unchanged; only the Python label codec broadens. Pickle and save/load metadata
must preserve the class array and label mapping without converting string labels to integers.

## Tags And Mixin Order

The estimator shells will derive base tags from their sklearn mixins and adjust only supported
capabilities: deterministic state, required targets, dense two-dimensional input, NaN feature
support, single-output targets, and no sparse or array API support. Compatibility `_more_tags`
support may remain for older optional sklearn installations, but certification targets 1.8 and 1.9
dataclass tags.

Tags must describe behavior rather than suppress checks. The PR will not use `_skip_test`,
`poor_score`, false sparse/array API claims, or a false estimator type to reduce coverage.

## Group-Aware Ranker Harness

Generic estimator checks cannot call the public `GBMRanker.fit(X, y, *, group=...)` contract. The
test suite will therefore define a private adapter that delegates to the real ranker and derives
deterministic valid query groups for each generated check dataset when the check cannot supply
groups. The adapter exists only in tests and does not relax the public signature.

The generated generic checks will be supplemented by direct public-contract tests covering:

- cloning, `get_params`, and delayed `set_params` validation;
- rejection when `group` is omitted;
- successful grouped fit/predict and fitted-state behavior;
- `n_features_in_`, feature-name, and feature-mismatch behavior;
- sparse, complex, empty, and one-dimensional input errors;
- sample-weight row filtering while preserving query alignment;
- pickle and save/load restoration of fitted schema.

## Check Exclusion Policy

`GBMRegressor` and `GBMClassifier` begin and finish with empty expected-failure maps. Every check
selected by sklearn 1.8 and 1.9 for their truthful tags must pass. Environment-driven skips emitted
by sklearn itself are recorded but are not AlloyGBM exclusions.

The ranker harness also begins with no exclusions. If a generated check fundamentally depends on a
public `fit(X, y)` signature or scoring contract that cannot faithfully coexist with mandatory
query groups, it may be excluded only by exact check name with a stable reason citing the documented
`group=` contract. The harness will treat unexpected passes as failures so exclusions cannot become
stale. Version-based blanket exclusions and warning suppression are prohibited.

## Test And CI Matrix

A dedicated Python conformance module will run the complete legacy and current estimator check sets
for default `GBMRegressor` and `GBMClassifier` instances and the group-aware ranker adapter. Focused
contract tests will cover each repaired behavior independently so failures remain diagnosable.

CI will add a dedicated matrix with:

- `scikit-learn>=1.8,<1.9`;
- `scikit-learn>=1.9,<1.10`.

Each arm will build the native extension, run the conformance module, report per-estimator
pass/skip/expected-failure counts, and fail on any unlisted failure or unexpected pass. The normal
Python suite remains responsible for the complete estimator and special-mode regression surface.

The final local gate is:

- `cargo fmt --all -- --check`;
- `cargo clippy --workspace --all-targets -- -D warnings`;
- `cargo test --workspace`;
- the complete Python test suite;
- the complete benchmark contract suite;
- the sklearn 1.8 and 1.9 conformance matrix;
- Sphinx with warnings treated as errors.

## Documentation And Compatibility

The PR will update `CHANGELOG.md`, the Markdown and Sphinx estimator guides, CI documentation where
needed, and the 2026-07-02 core review resolution. Documentation will state the certified sklearn
versions, delayed parameter-validation timing, fitted-attribute lifecycle, unsupported sparse input,
classifier label support, and the ranker's mandatory group-aware contract.

Valid trained-model artifacts and native training/prediction math do not change. Existing pickles
and saved models remain loadable; restored objects gain the corrected public fitted attributes.
Code that reads fitted attributes before fit or expects invalid known parameters to fail in
`__init__`/`set_params` will observe the intentional sklearn-conformance behavior change.
