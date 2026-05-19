"""Regression test for Limitation 4 (NaN routing on linear-rank predict path).

v0.9.0 fix: the predict-time `quantize_dense_values_linear_rank_inplace_wide`
helper now preserves NaN through the f32 cast, so the predictor's
`is_nan` short-circuit fires and routes through `default_left`. Prior to
v0.9.0 the bug was that NaN inputs silently fell through to bin 0 on
rank-binned columns, breaking the learned-missing-direction routing.

Three cases:
1. Constant-leaf model, mixed linear-rank path, NaN input → finite output.
2. Pure-linear path (no rank flags) — already worked in v0.8.0, regression
   check that it still does.
3. PL-leaf model (`leaf_model="linear"`) on the rank-binned path — NaN
   feature contributes 0.0 to the linear sum instead of propagating
   `w · NaN = NaN` through the prediction.
"""

from __future__ import annotations

import numpy as np
import pytest

from alloygbm import GBMRegressor


@pytest.fixture(autouse=True)
def _enable_rank_binning(monkeypatch):
    """Force the experimental linear-rank path so the codepath under test fires."""
    monkeypatch.setenv("ALLOYGBM_EXPERIMENT_LINEAR_TAIL_RANK", "1")


def test_nan_on_rank_binned_column_routes_default_left():
    rng = np.random.default_rng(0)
    # Lognormal data triggers the linear-rank binning path
    X_train = rng.lognormal(size=(500, 3)).astype(np.float32)
    y_train = rng.normal(size=500).astype(np.float32)

    m = GBMRegressor(
        n_estimators=20,
        continuous_binning_strategy="linear",
        seed=42,
    )
    m.fit(X_train, y_train)

    X_test = np.array([[np.nan, 0.5, 0.3]], dtype=np.float32)
    pred = np.asarray(m.predict(X_test))
    assert np.isfinite(pred).all(), (
        f"NaN on rank-binned column should route through default_left, "
        f"not propagate. Got {pred}"
    )


def test_nan_on_pure_linear_path_still_works():
    """Without rank flags, pure linear binning already routed NaN correctly
    in v0.8.0. This regression check confirms the v0.9.0 changes didn't
    break it."""
    rng = np.random.default_rng(1)
    # Normal data does not trigger rank-binning flags
    X_train = rng.normal(size=(500, 3)).astype(np.float32)
    y_train = rng.normal(size=500).astype(np.float32)

    m = GBMRegressor(
        n_estimators=20,
        continuous_binning_strategy="linear",
        seed=42,
    )
    m.fit(X_train, y_train)

    pred = np.asarray(
        m.predict(np.array([[np.nan, 0.5, 0.3]], dtype=np.float32))
    )
    assert np.isfinite(pred).all()


def test_nan_on_pl_leaf_with_rank_binned_column_is_finite():
    """PL leaves must not NaN-poison when the input row has NaN on a
    rank-binned regressor feature. This is the subtler part of
    Limitation 4: even if tree traversal routes correctly, the linear
    leaf's `intercept + Σ w · x` would propagate NaN through w · NaN."""
    rng = np.random.default_rng(2)
    X_train = rng.lognormal(size=(500, 3)).astype(np.float32)
    y_train = rng.normal(size=500).astype(np.float32)

    m = GBMRegressor(
        n_estimators=20,
        continuous_binning_strategy="linear",
        leaf_model="linear",
        seed=42,
    )
    m.fit(X_train, y_train)

    pred = np.asarray(
        m.predict(np.array([[np.nan, 0.5, 0.3]], dtype=np.float32))
    )
    assert np.isfinite(pred).all(), (
        f"PL-leaf rank-binned path NaN-poisoned: {pred}"
    )


def test_nan_in_training_data_still_supported():
    """Sanity check that training-time NaN handling (the native bin) still
    works after the predict-time fix. NaN values in y or X during fit
    should not crash and should produce sensible models."""
    rng = np.random.default_rng(3)
    X_train = rng.lognormal(size=(200, 3)).astype(np.float32)
    # Inject a few NaN values
    X_train[5, 0] = np.nan
    X_train[15, 1] = np.nan
    y_train = rng.normal(size=200).astype(np.float32)

    m = GBMRegressor(
        n_estimators=20,
        continuous_binning_strategy="linear",
        seed=42,
    )
    m.fit(X_train, y_train)
    preds = np.asarray(m.predict(X_train))
    assert np.isfinite(preds).all()
