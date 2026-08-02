# PR #131 sklearn Conformance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

| Date | Reviewer | Version reviewed | Commit | Status |
|---|---|---|---|---|
| 2026-08-01 | OpenAI Codex | `main` after PR #130 | `1ed8fd6` | Approved for implementation |

**Goal:** Make `GBMRegressor` and `GBMClassifier` pass every applicable sklearn 1.8 and 1.9
estimator check, and give `GBMRanker` a truthful group-aware conformance harness without weakening
its mandatory query-group contract.

**Architecture:** A private estimator core owns AlloyGBM training, persistence, and prediction.
Thin public estimator shells put the applicable sklearn semantic mixin first in the MRO. A small
optional-sklearn compatibility module centralizes tags, fitted-state checks, and validation while
retaining equivalent behavior when sklearn is absent. Constructor state remains cloneable and
unvalidated until `fit`; fitted schema is established only after successful fit or load.

**Tech Stack:** Python 3.11-3.13, numpy, optional pandas, scikit-learn 1.8/1.9, PyO3-backed native
runtime, pytest, GitHub Actions, Sphinx.

## Global Constraints

- Preserve public import paths, constructor signatures, valid artifact bytes, native training
  behavior, predictions for existing numeric labels, and supported pickle/model-load compatibility.
- Keep sklearn optional at import and runtime; fallback paths must enforce the same public input and
  fitted-state contracts without importing sklearn.
- Put no learned trailing-underscore attributes on an estimator before a successful fit or load.
- Constructor and `set_params` must assign known values without coercion or semantic validation;
  fit-time validation must happen before fitted state is mutated.
- Advertise only capabilities that AlloyGBM actually supports. Do not use `_skip_test`,
  `poor_score`, false sparse/multi-output tags, or broad expected-failure lists.
- Keep regressor and classifier expected-failure maps empty on both supported sklearn minors.
- Preserve `GBMRanker.fit(..., *, group=...)` as a mandatory public contract; adaptation belongs
  only in the test harness.
- Use test-first changes and observe each focused test fail for the intended reason before editing
  production code.
- Do not change Rust code or the model artifact format in this PR.

---

### Task 1: Optional sklearn compatibility layer and semantic estimator shells

**Files:**
- Create: `bindings/python/alloygbm/_sklearn_compat.py`
- Modify: `bindings/python/alloygbm/_regressor/_base.py`
- Modify: `bindings/python/alloygbm/_regressor/_core.py`
- Modify: `bindings/python/alloygbm/_regressor/__init__.py`
- Modify: `bindings/python/alloygbm/classifier.py`
- Modify: `bindings/python/alloygbm/ranker.py`
- Test: `bindings/python/tests/test_sklearn_conformance.py`

**Interfaces:**
- Produces `_BaseEstimator`, `_RegressorMixin`, `_ClassifierMixin`,
  `_SKLEARN_AVAILABLE`, and compatibility helpers without making sklearn mandatory.
- Produces private `_GBMEstimatorCore` and public shells with MROs ending in
  `RegressorMixin -> ... -> BaseEstimator`, `ClassifierMixin -> ... -> BaseEstimator`, and
  `_GBMEstimatorCore -> BaseEstimator`, respectively.
- Preserves `alloygbm.GBMRegressor`, `alloygbm.GBMClassifier`, and `alloygbm.GBMRanker` imports and
  their introspected constructor signatures.

- [ ] **Step 1: Add failing MRO, estimator-type, and import-fallback tests**

Assert `RegressorMixin`/`ClassifierMixin` precede `BaseEstimator`, classifier MRO contains no
`RegressorMixin`, `is_regressor`/`is_classifier` are truthful, and public modules/signatures are
unchanged. Add an import-isolation test that blocks `sklearn` imports and verifies estimators can
still be constructed.

```python
def test_classifier_has_only_classifier_semantics():
    mro = GBMClassifier.__mro__
    assert ClassifierMixin in mro
    assert RegressorMixin not in mro
    assert mro.index(ClassifierMixin) < mro.index(BaseEstimator)
    assert is_classifier(GBMClassifier())
```

- [ ] **Step 2: Verify the MRO test fails**

Run: `.venv/bin/python -m pytest bindings/python/tests/test_sklearn_conformance.py \
  -k 'mro or classifier_semantics or sklearn_optional' -q`

Expected: classifier includes the regressor mixin and its sklearn tags fail to identify it as a
classifier.

- [ ] **Step 3: Extract the private core and add thin semantic shells**

Move optional sklearn imports into `_sklearn_compat.py`, rename the existing implementation class
to `_GBMEstimatorCore`, and define public shells over it. Keep the existing late injection used by
the validation/quantization modules and preserve `GBMRegressor.__module__`.

- [ ] **Step 4: Run architecture and existing contract tests**

Run: `.venv/bin/python -m pytest bindings/python/tests/test_sklearn_conformance.py \
  bindings/python/tests/test_regressor_contract.py bindings/python/tests/test_classifier_and_metrics.py \
  bindings/python/tests/test_ranker.py -q`

- [ ] **Step 5: Commit the estimator architecture**

```bash
git add bindings/python/alloygbm/_sklearn_compat.py \
  bindings/python/alloygbm/_regressor/_base.py bindings/python/alloygbm/_regressor/_core.py \
  bindings/python/alloygbm/_regressor/__init__.py bindings/python/alloygbm/classifier.py \
  bindings/python/alloygbm/ranker.py bindings/python/tests/test_sklearn_conformance.py
git commit -m "refactor: separate sklearn estimator semantics"
```

### Task 2: Fitted-state and learned-schema lifecycle

**Files:**
- Modify: `bindings/python/alloygbm/_sklearn_compat.py`
- Modify: `bindings/python/alloygbm/_regressor/_core.py`
- Modify: `bindings/python/alloygbm/_regressor/_persistence.py`
- Modify: `bindings/python/alloygbm/_regressor/_shap.py`
- Modify: `bindings/python/alloygbm/classifier.py`
- Test: `bindings/python/tests/test_sklearn_conformance.py`
- Test: `bindings/python/tests/test_regressor_contract.py`
- Test: `bindings/python/tests/test_classifier_and_metrics.py`

**Interfaces:**
- Produces `_require_fitted()` and `__sklearn_is_fitted__()`.
- Creates `n_features_in_` after fit/load and `feature_names_in_` only for all-string named input.
- Deletes learned attributes during reset instead of assigning sentinel values.

- [ ] **Step 1: Add failing lifecycle tests**

Cover no public trailing-underscore attributes after construction, `NotFittedError` from every
prediction/explanation/importance method, `n_features_in_` after fit and load, ndarray
`feature_names_in_` for string DataFrame columns, absence for array/non-string columns, failed-fit
cleanup, and warm-start schema preservation.

- [ ] **Step 2: Verify representative lifecycle tests fail**

Run: `.venv/bin/python -m pytest bindings/python/tests/test_sklearn_conformance.py \
  -k 'unfitted or n_features_in or feature_names or failed_fit' -q`

Expected: pre-fit learned attributes exist and prediction raises AlloyGBM-specific runtime errors
instead of sklearn's fitted-state error.

- [ ] **Step 3: Centralize fitted-state checks and schema publication**

Keep private runtime fields available from construction, but delete all learned public attributes
in `_reset_fitted_state`. Publish schema only after native fit succeeds. Restore and validate the
same schema when loading current and legacy persistence payloads. Replace direct optional
`feature_names_in_` reads with `getattr`.

- [ ] **Step 4: Run focused lifecycle and persistence coverage**

Run: `.venv/bin/python -m pytest bindings/python/tests/test_sklearn_conformance.py \
  bindings/python/tests/test_regressor_contract.py bindings/python/tests/test_classifier_and_metrics.py \
  -k 'fitted or feature or persist or pickle or save or load' -q`

- [ ] **Step 5: Commit fitted-state lifecycle changes**

```bash
git add bindings/python/alloygbm/_sklearn_compat.py bindings/python/alloygbm/_regressor/_core.py \
  bindings/python/alloygbm/_regressor/_persistence.py bindings/python/alloygbm/_regressor/_shap.py \
  bindings/python/alloygbm/classifier.py bindings/python/tests/test_sklearn_conformance.py \
  bindings/python/tests/test_regressor_contract.py bindings/python/tests/test_classifier_and_metrics.py
git commit -m "fix: align estimator fitted-state lifecycle"
```

### Task 3: Clone-safe parameter handling and fit-time validation

**Files:**
- Modify: `bindings/python/alloygbm/_regressor/_core.py`
- Modify: `bindings/python/alloygbm/classifier.py`
- Modify: `bindings/python/alloygbm/ranker.py`
- Test: `bindings/python/tests/test_sklearn_conformance.py`
- Test: `bindings/python/tests/test_regressor_contract.py`
- Test: `bindings/python/tests/test_classifier_and_metrics.py`
- Test: `bindings/python/tests/test_ranker.py`

**Interfaces:**
- Produces assignment-only constructors and `set_params` for known parameters.
- Produces `_validate_hyperparameters()` called at the start of each public `fit`.
- Preserves object identity through `get_params(deep=False)`, `clone`, and `set_params`.

- [ ] **Step 1: Add failing parameter protocol tests**

Use opaque identity-bearing values to verify construction and `set_params` do not coerce them.
Verify invalid known values are accepted by construction/`set_params` and rejected by `fit`, while
unknown names fail immediately. Cover classifier objective/neutralization and ranker objective,
sigma, truncation, and normalization constraints.

- [ ] **Step 2: Verify clone and deferred-validation tests fail**

Run: `.venv/bin/python -m pytest bindings/python/tests/test_sklearn_conformance.py \
  -k 'parameter or clone or set_params' -q`

Expected: constructors and `set_params` eagerly coerce or reject known values.

- [ ] **Step 3: Move semantic validation to fit without changing validated values**

Extract the existing validation rules into `_validate_hyperparameters`; keep constructor
assignments exact. Replace custom eager `set_params` logic with BaseEstimator-compatible assignment
and a dependency-free fallback. Ensure all validation completes before `_reset_fitted_state` or
native training begins.

- [ ] **Step 4: Update legacy eager-validation tests and CI smoke expectations**

Tests must call `fit` to assert invalid known parameter failures. Unknown parameter tests remain
immediate. Do not weaken any existing error condition.

- [ ] **Step 5: Run parameter suites and commit**

Run: `.venv/bin/python -m pytest bindings/python/tests/test_sklearn_conformance.py \
  bindings/python/tests/test_regressor_contract.py bindings/python/tests/test_classifier_and_metrics.py \
  bindings/python/tests/test_ranker.py -k 'param or init or objective or neutralization' -q`

```bash
git add bindings/python/alloygbm/_regressor/_core.py bindings/python/alloygbm/classifier.py \
  bindings/python/alloygbm/ranker.py bindings/python/tests/test_sklearn_conformance.py \
  bindings/python/tests/test_regressor_contract.py bindings/python/tests/test_classifier_and_metrics.py \
  bindings/python/tests/test_ranker.py
git commit -m "fix: defer estimator parameter validation to fit"
```

### Task 4: Shared dense input, target, and feature-schema validation

**Files:**
- Modify: `bindings/python/alloygbm/_sklearn_compat.py`
- Modify: `bindings/python/alloygbm/_regressor/_validation.py`
- Modify: `bindings/python/alloygbm/_regressor/_core.py`
- Modify: `bindings/python/alloygbm/classifier.py`
- Modify: `bindings/python/alloygbm/ranker.py`
- Test: `bindings/python/tests/test_sklearn_conformance.py`
- Test: `bindings/python/tests/test_regressor_contract.py`

**Interfaces:**
- Produces shared fit/predict validation for dense numeric inputs while retaining categorical and
  DataFrame adapters.
- Allows NaN feature values, rejects infinities and complex data, and emits an explicit error
  containing `sparse` for sparse matrices.
- Enforces 2D nonempty X, feature schema agreement, finite one-dimensional regression targets, and
  sklearn-compatible column-vector handling.

- [ ] **Step 1: Add failing input-protocol tests**

Cover sparse CSR/CSC, complex values, empty rows/features, one-dimensional X, infinity, NaN
features, non-array wrappers, feature-count mismatch, DataFrame feature-name mismatch, one-
dimensional predict input, nonfinite y, and `(n, 1)` versus `(n, 2)` targets.

- [ ] **Step 2: Verify sparse and shape tests fail with current messages**

Run: `.venv/bin/python -m pytest bindings/python/tests/test_sklearn_conformance.py \
  -k 'sparse or complex or empty or dimensional or feature_mismatch' -q`

Expected: late conversion/type errors do not meet sklearn's explicit input contract.

- [ ] **Step 3: Route numeric arrays through shared validation**

Use sklearn validation helpers when installed and equivalent numpy checks otherwise. Avoid forced
Python-list conversion. Keep the existing categorical encoding and quantization paths after a
common preflight and schema step. Warn and flatten only `(n, 1)` targets; reject true multi-output.

- [ ] **Step 4: Run focused validation and categorical regressions**

Run: `.venv/bin/python -m pytest bindings/python/tests/test_sklearn_conformance.py \
  bindings/python/tests/test_regressor_contract.py bindings/python/tests/test_native_categorical_splits.py \
  -k 'sparse or shape or feature or target or dataframe or categorical or nan' -q`

- [ ] **Step 5: Commit input validation changes**

```bash
git add bindings/python/alloygbm/_sklearn_compat.py \
  bindings/python/alloygbm/_regressor/_validation.py bindings/python/alloygbm/_regressor/_core.py \
  bindings/python/alloygbm/classifier.py bindings/python/alloygbm/ranker.py \
  bindings/python/tests/test_sklearn_conformance.py bindings/python/tests/test_regressor_contract.py
git commit -m "fix: standardize estimator input validation"
```

### Task 5: Sample-weight protocol and zero-weight row equivalence

**Files:**
- Modify: `bindings/python/alloygbm/_regressor/_validation.py`
- Modify: `bindings/python/alloygbm/_regressor/_core.py`
- Modify: `bindings/python/alloygbm/classifier.py`
- Modify: `bindings/python/alloygbm/ranker.py`
- Test: `bindings/python/tests/test_sklearn_conformance.py`
- Test: `bindings/python/tests/test_regressor_contract.py`
- Test: `bindings/python/tests/test_classifier_and_metrics.py`
- Test: `bindings/python/tests/test_ranker.py`

**Interfaces:**
- Accepts finite nonnegative one-dimensional sample weights with exact row alignment.
- Rejects all-zero weights with an error stating that the weight sum is zero.
- Removes zero-weight rows before quantization, categorical encoding, query sorting, time ordering,
  and factor preparation, including aligned evaluation inputs.

- [ ] **Step 1: Add failing weight validation and equivalence tests**

Cover scalar/2D/wrong-length/negative/nonfinite/all-zero weights and compare predictions from a
zero-weight fit with predictions from physically omitted rows. Repeat equivalence for regressor,
classifier, ranker groups, categorical side channels, time indices, factor exposures, and eval
weights using small deterministic fixtures.

- [ ] **Step 2: Verify the omission-equivalence test fails**

Run: `.venv/bin/python -m pytest bindings/python/tests/test_sklearn_conformance.py \
  -k 'sample_weight or zero_weight' -q`

Expected: zero-weight rows still influence quantile cuts or aligned preprocessing.

- [ ] **Step 3: Implement one aligned row-filtering primitive**

Validate weights before any fit preprocessing, derive a keep mask, and apply it consistently to X,
y, group, category columns, time indices, and factor exposures. Apply the same contract to eval
data. Preserve all-positive-weight behavior exactly.

- [ ] **Step 4: Run weighted and ranking regressions**

Run: `.venv/bin/python -m pytest bindings/python/tests/test_sklearn_conformance.py \
  bindings/python/tests/test_regressor_contract.py bindings/python/tests/test_classifier_and_metrics.py \
  bindings/python/tests/test_ranker.py -k 'weight or group or categorical or factor or time' -q`

- [ ] **Step 5: Commit sample-weight semantics**

```bash
git add bindings/python/alloygbm/_regressor/_validation.py \
  bindings/python/alloygbm/_regressor/_core.py bindings/python/alloygbm/classifier.py \
  bindings/python/alloygbm/ranker.py bindings/python/tests/test_sklearn_conformance.py \
  bindings/python/tests/test_regressor_contract.py bindings/python/tests/test_classifier_and_metrics.py \
  bindings/python/tests/test_ranker.py
git commit -m "fix: make zero sample weights equivalent to row omission"
```

### Task 6: Classifier target typing, labels, outputs, and persistence

**Files:**
- Modify: `bindings/python/alloygbm/classifier.py`
- Modify: `bindings/python/alloygbm/_regressor/_persistence.py`
- Test: `bindings/python/tests/test_sklearn_conformance.py`
- Test: `bindings/python/tests/test_classifier_and_metrics.py`

**Interfaces:**
- Accepts finite numeric, boolean, and string class labels.
- Rejects continuous, multilabel, malformed, and one-class targets with sklearn-compatible errors.
- Publishes ndarray `classes_`, returns ndarray labels in the original dtype, and preserves arbitrary
  labels through pickle and `save_model`/`load_model`.

- [ ] **Step 1: Add failing classifier label tests**

Cover binary/multiclass integer, boolean, and string labels; one-class and continuous targets;
multilabel/2D/nonfinite y; ndarray output shape/dtype; probability shape/order; and string-label
pickle/model-file round trips.

- [ ] **Step 2: Verify string-label and ndarray-output tests fail**

Run: `.venv/bin/python -m pytest bindings/python/tests/test_sklearn_conformance.py \
  bindings/python/tests/test_classifier_and_metrics.py -k 'string or label or classes or output' -q`

Expected: labels are cast to integers, `classes_` is a list, and `predict` returns a list.

- [ ] **Step 3: Replace the integer-only codec with an ndarray class codec**

Use sklearn target classification when available and a dependency-free equivalent otherwise. Keep
native training labels encoded as contiguous numeric indices. Persist classes as an ordered value
sequence rather than JSON object keys so string and boolean labels round-trip losslessly.

- [ ] **Step 4: Run classifier, SHAP, and persistence regressions**

Run: `.venv/bin/python -m pytest bindings/python/tests/test_sklearn_conformance.py \
  bindings/python/tests/test_classifier_and_metrics.py bindings/python/tests/test_shap_multiclass_multioutput.py \
  -k 'class or label or predict or persist or pickle or shap' -q`

- [ ] **Step 5: Commit classifier protocol changes**

```bash
git add bindings/python/alloygbm/classifier.py \
  bindings/python/alloygbm/_regressor/_persistence.py \
  bindings/python/tests/test_sklearn_conformance.py \
  bindings/python/tests/test_classifier_and_metrics.py
git commit -m "fix: support sklearn classifier label contracts"
```

### Task 7: Truthful tags and full regressor/classifier estimator checks

**Files:**
- Modify: `bindings/python/alloygbm/_regressor/_core.py`
- Modify: `bindings/python/alloygbm/classifier.py`
- Modify: `bindings/python/tests/test_sklearn_conformance.py`

**Interfaces:**
- Produces truthful sklearn 1.8/1.9 tags: estimator type, required target, dense-only input, NaN
  feature support, single-output targets, and non-determinism state.
- Produces empty regressor/classifier expected-failure maps and fails on skips/xfails not explicitly
  caused by the test environment.

- [ ] **Step 1: Add the aggregate estimator-check runner**

Run `check_estimator(..., legacy=True, on_fail=None, on_skip=None)` for small deterministic
regressor and classifier instances. Normalize 1.8/1.9 result records, report every failing check by
name/message, assert no expected failures, and keep environment-only skips visible.

```python
@pytest.mark.parametrize("estimator", [
    GBMRegressor(n_estimators=6, seed=0, deterministic=True),
    GBMClassifier(n_estimators=6, seed=0, deterministic=True),
])
def test_all_applicable_sklearn_checks_pass(estimator):
    failures, xfails = run_estimator_checks(estimator)
    assert failures == []
    assert xfails == []
```

- [ ] **Step 2: Verify the aggregate harness reports remaining failures**

Run: `.venv/bin/python -m pytest bindings/python/tests/test_sklearn_conformance.py \
  -k 'all_applicable_sklearn_checks_pass' -vv`

Expected: the harness names any residual check-specific defects after Tasks 1-6.

- [ ] **Step 3: Fix residual applicable checks without capability exceptions**

Address each residual at the narrowest shared contract boundary. Add a direct regression test for
every discovered failure before the fix. Do not mark checks expected-failing to avoid implementable
protocol behavior.

- [ ] **Step 4: Prove the complete regressor/classifier conformance gate**

Run: `.venv/bin/python -m pytest bindings/python/tests/test_sklearn_conformance.py \
  -k 'regressor or classifier or applicable' -vv`

Expected: all applicable checks pass; expected-failure maps are empty.

- [ ] **Step 5: Commit conformance closure**

```bash
git add bindings/python/alloygbm/_regressor/_core.py bindings/python/alloygbm/classifier.py \
  bindings/python/tests/test_sklearn_conformance.py
git commit -m "test: enforce sklearn estimator conformance"
```

### Task 8: Group-aware ranker conformance harness

**Files:**
- Modify: `bindings/python/alloygbm/ranker.py`
- Modify: `bindings/python/tests/test_sklearn_conformance.py`
- Modify: `bindings/python/tests/test_ranker.py`

**Interfaces:**
- Produces a private test-only ranker adapter that synthesizes deterministic query groups only when
  a generic sklearn check omits them.
- Preserves and directly tests the public mandatory `group` keyword contract.
- Produces an exact ranker expected-failure map containing only checks that fundamentally cannot
  represent grouped ranking, with a reason per check; unexpected passes are failures.

- [ ] **Step 1: Add failing public-contract and adapter tests**

Assert `GBMRanker.fit(X, y)` without `group` fails, explicit groups train normally, the adapter
derives stable groups, clone preserves ranker parameters, and generic checks cannot mutate the
public contract.

- [ ] **Step 2: Run the initial ranker check inventory**

Run: `.venv/bin/python -m pytest bindings/python/tests/test_sklearn_conformance.py \
  -k 'ranker' -vv`

Expected: the unadapted ranker fails checks that call `fit` without query groups.

- [ ] **Step 3: Implement the test-only group adapter and exact result accounting**

Derive contiguous two-or-more-row query IDs from the current check input, preserving row count and
ordering. Run applicable generic checks against the adapter. If a check still fundamentally
contradicts mandatory grouped fit, record only that exact name and an explicit reason; fail on
unknown failures and unexpected passes.

- [ ] **Step 4: Run ranker and multi-label regressions**

Run: `.venv/bin/python -m pytest bindings/python/tests/test_sklearn_conformance.py \
  bindings/python/tests/test_ranker.py bindings/python/tests/test_multi_label_ranker.py -q`

- [ ] **Step 5: Commit the ranker harness**

```bash
git add bindings/python/alloygbm/ranker.py bindings/python/tests/test_sklearn_conformance.py \
  bindings/python/tests/test_ranker.py
git commit -m "test: add group-aware ranker conformance harness"
```

### Task 9: sklearn 1.8/1.9 CI certification

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `requirements-dev.txt`
- Modify: `bindings/python/tests/test_sklearn_conformance.py`

**Interfaces:**
- Produces explicit `scikit-learn>=1.8,<1.9` and `scikit-learn>=1.9,<1.10` CI jobs.
- Keeps the broad OS/Python smoke matrix while running the expensive estimator checks once per
  supported sklearn minor.

- [ ] **Step 1: Add a local version-window assertion**

Parameterize the conformance runner's report with the installed sklearn version and assert the
tested release is in the supported `[1.8, 1.10)` window. Keep version parsing dependency-free.

- [ ] **Step 2: Add a dedicated Linux/Python 3.13 sklearn matrix job**

Build/install one wheel, then test both sklearn constraints. Run
`test_sklearn_conformance.py` and the focused estimator contract files in each matrix leg. Change
the broad smoke job to install the canonical supported latest minor explicitly rather than an
unbounded release.

- [ ] **Step 3: Update smoke assertions for the new documented contracts**

Change invalid-known-parameter smoke coverage to assert failure at `fit`, and compare classifier
`classes_` with numpy array semantics.

- [ ] **Step 4: Verify both dependency environments locally**

Create temporary virtual environments outside the repository, install the built wheel plus each
sklearn constraint, and run:

```bash
python -m pytest bindings/python/tests/test_sklearn_conformance.py \
  bindings/python/tests/test_regressor_contract.py \
  bindings/python/tests/test_classifier_and_metrics.py bindings/python/tests/test_ranker.py -q
```

Expected: both sklearn 1.8.x and 1.9.x environments pass with no conformance xfails.

- [ ] **Step 5: Commit CI certification**

```bash
git add .github/workflows/ci.yml requirements-dev.txt \
  bindings/python/tests/test_sklearn_conformance.py
git commit -m "ci: certify sklearn 1.8 and 1.9"
```

### Task 10: User documentation and review resolution

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `docs/user/gbmregressor.md`
- Modify: `docs/user/gbmclassifier.md`
- Modify: `docs/user/gbmranker.md`
- Modify: `docs/site/source/estimator.rst`
- Modify: `docs/site/source/classifier.rst`
- Modify: `docs/site/source/ranker.rst`
- Modify: `docs/site/source/release.rst`
- Modify: `docs/reviews/2026-07-02-v0.12.10-core-resolutions.md`

**Interfaces:**
- Documents fitted attributes, dense/sparse behavior, deferred parameter validation, supported
  classifier labels, and the mandatory ranker group contract.
- Records exact sklearn versions, per-estimator pass/skip/xfail counts, and the reason for every
  ranker exclusion, if any.

- [ ] **Step 1: Correct inheritance and compatibility claims**

Replace statements that classifier/ranker extend `GBMRegressor` with the semantic-shell model.
Document that sklearn is optional but 1.8 and 1.9 are certified when installed.

- [ ] **Step 2: Document intentional behavior changes**

Record fit-time validation for known parameters, explicit sparse rejection, `n_features_in_` and
conditional `feature_names_in_`, arbitrary classifier labels/ndarray outputs, and zero-weight row
omission semantics.

- [ ] **Step 3: Add the review resolution entry**

Tie PR #131 to the remaining Python estimator/API validation finding in the 2026-07-02 core review.
Include exact check accounting produced by the two CI environments; do not claim generic
`check_estimator` compatibility for the public ranker because checks cannot supply query groups.

- [ ] **Step 4: Verify documentation builds and claims match tests**

Run: `.venv/bin/python -m sphinx -W -b html docs/site/source /tmp/alloygbm-docs-pr131`

Run: `rg -n "extends .*GBMRegressor|inherits .*GBMRegressor|full sklearn" \
  docs/user docs/site/source CHANGELOG.md`

- [ ] **Step 5: Commit documentation**

```bash
git add CHANGELOG.md docs/user/gbmregressor.md docs/user/gbmclassifier.md \
  docs/user/gbmranker.md docs/site/source/estimator.rst docs/site/source/classifier.rst \
  docs/site/source/ranker.rst docs/site/source/release.rst \
  docs/reviews/2026-07-02-v0.12.10-core-resolutions.md
git commit -m "docs: record sklearn conformance closure"
```

### Task 11: Final verification and draft PR

**Files:**
- Verify all modified files
- Create: no additional production files

- [ ] **Step 1: Run formatting and static repository checks**

Run: `cargo fmt -- --check`

Run: `cargo clippy --workspace --exclude alloygbm-python --all-targets -- -D warnings`

- [ ] **Step 2: Run the complete Rust suite**

Run: `cargo test --workspace`

Expected: all workspace tests and doctests pass; no Rust behavior changed.

- [ ] **Step 3: Rebuild and run the complete Python suite**

Run: `.venv/bin/maturin develop --release`

Run: `.venv/bin/python -m pytest bindings/python/tests/ -q`

Expected: the full suite passes, including every conformance check.

- [ ] **Step 4: Re-run both isolated sklearn version gates**

Use the temporary environments from Task 9 and record installed sklearn version plus exact
pass/skip/xfail counts for regressor, classifier, and ranker. Any xfail requires a documented
fundamental contract conflict; regressor/classifier xfails are forbidden.

- [ ] **Step 5: Audit diff and compatibility**

Run: `git diff main...HEAD --check`

Run: `git status --short`

Run focused current/legacy pickle and model-file tests again, and inspect the diff for accidental
Rust, artifact-format, generated-file, or unrelated documentation changes.

- [ ] **Step 6: Push and open draft PR #131**

Push `codex/pr-131-sklearn-conformance` and open a draft PR with the behavioral contract changes,
exact 1.8/1.9 conformance accounting, full verification results, and any ranker-only exclusions.
Stop before merge so independent reviewers can inspect it.
