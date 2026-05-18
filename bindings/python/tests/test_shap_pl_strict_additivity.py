"""Regression tests for v0.7.4 PL-leaf SHAP strict additivity.

Pre-v0.7.4, ``shap_values()`` on ``leaf_model="linear"`` artifacts only
credited the **terminal** leaf's linear deviation ``Σⱼ wⱼ·(xⱼ − μⱼ)``,
while the predictor accumulates ``leaf.eval_row(row)`` at **every visited
node along the path**.  Internal nodes' linear deviations were uncredited,
producing additivity gaps on the order of the predictions themselves
(scaling with ``n_estimators × max_depth``).

The internal Rust ``verify_additivity`` exempted PL-leaf models from the
strict tolerance check entirely.  v0.7.4 fixes the path walk in
``distribute_linear_terms_for_row`` and tightens the additivity check
(strict for any caller that passes a ``BinningContext`` — i.e. the
default Python path for continuous features).

These tests fix the contract in place:

* Strict additivity for ``leaf_model="linear"`` regressors under every
  combination of binning strategy, max-bin width, regularization,
  tree depth, ensemble size, feature-scale heterogeneity, and
  training_mode (``manual`` and ``morph``).
* ``GBMRanker`` strict additivity.
* ``GBMClassifier`` strict additivity via the internal check (the raw
  margin is not exposed in Python, but ``shap_values()`` raises if the
  internal ``verify_additivity`` ever fails).
* ``feature_importances()`` exercises the TreeSHAP polynomial path; the
  same fix lives in the shared ``distribute_linear_terms_for_row``
  helper so this is covered without a separate path.
* Mixed scalar/linear-leaf trees stay correct (constant leaves are
  no-ops in the linear-deviation walk).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor


# Matches the constants in crates/shap/src/lib.rs.
ADDITIVITY_ATOL = 1e-5
ADDITIVITY_RTOL = 1e-4


def _assert_strict_additivity(model, X, *, label: str) -> None:
    """Assert ``Σ shap_values + expected_value ≈ predict(x)`` per row."""
    preds = np.asarray(model.predict(X), dtype=np.float64)
    ev, shap_vals = model.shap_values(X, include_expected_value=True)
    shap_vals = np.asarray(shap_vals, dtype=np.float64)
    recon = float(ev) + shap_vals.sum(axis=1)
    tol = ADDITIVITY_ATOL + ADDITIVITY_RTOL * np.abs(preds)
    diff = np.abs(preds - recon)
    assert np.all(diff <= tol), (
        f"[{label}] max additivity drift {diff.max():.2e} exceeded tolerance "
        f"{tol[diff.argmax()]:.2e}; row {int(diff.argmax())} pred="
        f"{preds[diff.argmax()]:.4f} recon={recon[diff.argmax()]:.4f}"
    )


def _make_linear_data(
    n_rows: int = 256,
    n_features: int = 6,
    feature_scales=None,
    seed: int = 7,
):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_features)).astype(np.float32)
    if feature_scales is not None:
        X = X * np.asarray(feature_scales, dtype=np.float32)
    coeffs = rng.standard_normal(n_features).astype(np.float32)
    y = X @ coeffs + 0.1 * rng.standard_normal(n_rows).astype(np.float32)
    return X, y.astype(np.float32)


# ── Regressor: combinatoric matrix of configurations ─────────────────────────


@pytest.mark.parametrize(
    "binning_strategy,max_bins",
    [
        ("quantile", 256),
        ("quantile", 512),
        ("linear", 256),
    ],
)
@pytest.mark.parametrize("lambda_l2", [0.01, 1.0])
@pytest.mark.parametrize("max_depth", [4, 6, 8])
@pytest.mark.parametrize("n_estimators", [50, 200])
def test_regressor_strict_additivity(
    binning_strategy, max_bins, lambda_l2, max_depth, n_estimators
):
    X, y = _make_linear_data(seed=7)
    model = GBMRegressor(
        leaf_model="linear",
        learning_rate=0.05,
        max_depth=max_depth,
        n_estimators=n_estimators,
        lambda_l2=lambda_l2,
        continuous_binning_strategy=binning_strategy,
        continuous_binning_max_bins=max_bins,
        seed=7,
        deterministic=True,
    )
    model.fit(X, y)
    _assert_strict_additivity(
        model, X,
        label=f"binning={binning_strategy} bins={max_bins} λ={lambda_l2} "
              f"depth={max_depth} n={n_estimators}",
    )


def test_regressor_skewed_feature_scales():
    """Features ranging over 5 orders of magnitude: the deviation sum is
    dominated by the large-scale features; if any internal-node linear
    term is dropped the gap explodes."""
    X, y = _make_linear_data(
        feature_scales=[0.01, 0.1, 1.0, 10.0, 100.0, 1.0], seed=7
    )
    model = GBMRegressor(
        leaf_model="linear",
        learning_rate=0.05,
        max_depth=6,
        n_estimators=100,
        lambda_l2=0.01,
        seed=7,
        deterministic=True,
    )
    model.fit(X, y)
    _assert_strict_additivity(model, X, label="skewed-feature-scales")


@pytest.mark.parametrize("training_mode", ["manual", "morph"])
def test_regressor_training_mode_compose(training_mode):
    X, y = _make_linear_data(seed=7)
    model = GBMRegressor(
        leaf_model="linear",
        learning_rate=0.05,
        max_depth=6,
        n_estimators=100,
        lambda_l2=0.01,
        training_mode=training_mode,
        seed=7,
        deterministic=True,
    )
    model.fit(X, y)
    _assert_strict_additivity(model, X, label=f"training_mode={training_mode}")


def test_regressor_with_interaction_constraints():
    X, y = _make_linear_data(seed=7)
    model = GBMRegressor(
        leaf_model="linear",
        learning_rate=0.05,
        max_depth=6,
        n_estimators=100,
        lambda_l2=0.01,
        interaction_constraints=[[0, 1, 2], [3, 4, 5]],
        seed=7,
        deterministic=True,
    )
    model.fit(X, y)
    _assert_strict_additivity(model, X, label="interaction-constraints")


def test_feature_importances_smoke_brute_force_path():
    """``feature_importances`` invokes the SHAP machinery; for models with
    distinct split features <= ``MAX_EXACT_SPLIT_FEATURES`` (=25, see
    ``crates/shap/src/lib.rs``) this exercises the brute-force exact
    path.  This case keeps n_features small (6) so we stay on
    brute-force; the TreeSHAP polynomial path is exercised by
    ``test_strict_additivity_via_tree_shap_polynomial_path`` below."""
    X, y = _make_linear_data(seed=7)
    model = GBMRegressor(
        leaf_model="linear",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        lambda_l2=0.01,
        seed=7,
        deterministic=True,
    )
    model.fit(X, y)
    importance = model.feature_importances(X)
    assert len(importance) == X.shape[1]
    # Importances are non-negative by construction.
    assert all(score >= 0.0 for _, score in importance)


def test_strict_additivity_via_tree_shap_polynomial_path():
    """Drive the polynomial TreeSHAP path by training on more than
    ``MAX_EXACT_SPLIT_FEATURES`` (=25) distinct split features.
    ``explain_rows_from_model`` switches to ``explain_rows_tree_shap``
    when ``distinct_split_feature_count > MAX_EXACT_SPLIT_FEATURES``;
    with 32 features and a deep enough ensemble every feature shows up
    as a split at least once.

    Pinned by an ``@xfail(strict=True)`` regression test in v0.7.4 to
    flag a pre-existing TreeSHAP polynomial-path additivity drift
    (Limitation 5).  Closed in v0.7.5 — the bug was in
    ``ts_unextend_path`` which incorrectly shifted ``pweight`` along
    with the feature_index / zero_fraction / one_fraction tuple when
    a duplicate feature was removed from the path.  The reference
    implementation in slundberg/shap stores those four fields as four
    parallel arrays and only shifts the first three, preserving the
    post-unwind pweights computed in place by the unwind loop."""
    n_features = 32  # > MAX_EXACT_SPLIT_FEATURES = 25
    X, y = _make_linear_data(n_features=n_features, seed=7)
    model = GBMRegressor(
        leaf_model="linear",
        learning_rate=0.05,
        max_depth=6,
        n_estimators=100,
        lambda_l2=0.01,
        seed=7,
        deterministic=True,
    )
    model.fit(X, y)
    _assert_strict_additivity(model, X, label="tree-shap-polynomial-path")


# ── Classifier: internal check is the ground truth ───────────────────────────


def test_classifier_strict_additivity_via_internal_check():
    """``GBMClassifier.shap_values`` explains the raw margin (logit).  The
    margin is not exposed in Python; instead, ``shap_values()`` invokes the
    internal Rust ``verify_additivity`` for every row and raises if the
    strict tolerance is violated.  Reaching this assertion proves
    additivity holds in raw-margin space — even in cases where reversing
    ``predict_proba`` via ``log(p/(1-p))`` loses precision near
    saturated probabilities."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((256, 6)).astype(np.float32)
    coeffs = rng.standard_normal(6).astype(np.float32)
    p = 1.0 / (1.0 + np.exp(-X @ coeffs))
    y = (rng.random(256) < p).astype(np.int32)

    model = GBMClassifier(
        leaf_model="linear",
        learning_rate=0.05,
        max_depth=6,
        n_estimators=100,
        lambda_l2=0.01,
        seed=7,
        deterministic=True,
    )
    model.fit(X, y)
    # No exception → internal verify_additivity passed for all 256 rows.
    ev, shap_vals = model.shap_values(X, include_expected_value=True)
    assert np.asarray(shap_vals).shape == (256, 6)
    assert math.isfinite(float(ev))


# ── Ranker ───────────────────────────────────────────────────────────────────


def test_ranker_strict_additivity():
    rng = np.random.default_rng(7)
    n_queries, rows_per_query = 16, 16
    n_rows = n_queries * rows_per_query
    X = rng.standard_normal((n_rows, 6)).astype(np.float32)
    coeffs = rng.standard_normal(6).astype(np.float32)
    relevance = X @ coeffs + 0.2 * rng.standard_normal(n_rows)
    y = np.clip(np.round(relevance), 0, 4).astype(np.int32)
    group = np.repeat(np.arange(n_queries, dtype=np.int32), rows_per_query)

    model = GBMRanker(
        ranking_objective="rank:ndcg",
        leaf_model="linear",
        learning_rate=0.05,
        max_depth=6,
        n_estimators=100,
        lambda_l2=0.01,
        seed=7,
        deterministic=True,
    )
    model.fit(X, y, group=group)
    _assert_strict_additivity(model, X, label="ranker")


# ── Mixed scalar + linear leaves (regression of no-op path) ──────────────────


def test_scalar_leaves_unchanged_by_pl_fix():
    """The PL-leaf fix adds a per-visited-node walk that is a no-op for
    scalar leaves.  Scalar-only models must keep passing additivity
    exactly — i.e., the fix introduced no scalar-side regression."""
    X, y = _make_linear_data(seed=7)
    model = GBMRegressor(
        leaf_model="constant",  # scalar leaves only
        learning_rate=0.05,
        max_depth=6,
        n_estimators=100,
        seed=7,
        deterministic=True,
    )
    model.fit(X, y)
    _assert_strict_additivity(model, X, label="scalar-leaves-baseline")
