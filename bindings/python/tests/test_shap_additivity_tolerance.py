"""Regression tests for the SHAP additivity tolerance fix (v0.7.3 bug 2).

Prior to v0.7.3 the additivity check used a fixed absolute tolerance
of ``1e-5``.  Accumulated f32 round-off across a large explanation
batch (e.g. ``feature_importances()`` over ~1000 rows of California
Housing with ``n_estimators=200``) could exceed it by a few ulps even
on healthy ``leaf_model="constant"`` artifacts, raising::

    RuntimeError: row N additivity check failed:
        predicted=X, reconstructed=Y, tolerance=0.00001

The fix switches to numpy-style ``atol + rtol * |predicted|`` so the
tolerance scales with the prediction magnitude.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

from alloygbm import GBMRegressor


@pytest.mark.skipif(not HAVE_SKLEARN, reason="sklearn not installed")
def test_feature_importances_large_sample_california_housing() -> None:
    """The pre-v0.7.3 reproduction: California Housing, 200 estimators,
    full test set.  This exceeded the 1e-5 absolute tolerance on at
    least one row.  After the fix this must succeed for all 1000+
    rows."""
    data = fetch_california_housing(as_frame=False)
    X_train, X_test, y_train, _ = train_test_split(
        data.data, data.target, test_size=0.2, random_state=7
    )

    model = GBMRegressor(
        learning_rate=0.05,
        max_depth=6,
        n_estimators=200,
        training_policy="manual",
        deterministic=True,
        seed=7,
    )
    model.fit(X_train, y_train)

    # Pre-v0.7.3 this raised at row 231 with a predicted value of
    # ~4.643 and a ~1.2e-5 reconstruction drift.  The new tolerance
    # `atol=1e-5 + rtol=1e-4 * |4.643|` = ~4.7e-4 absorbs it.
    importance = model.feature_importances(X_test)
    assert len(importance) == X_test.shape[1]
    assert all(score >= 0.0 for _, score in importance), \
        "SHAP importance must be non-negative"


@pytest.mark.skipif(not HAVE_SKLEARN, reason="sklearn not installed")
def test_shap_additivity_holds_under_new_tolerance() -> None:
    """Every per-row reconstruction is within `atol + rtol * |predict|`."""
    data = fetch_california_housing(as_frame=False)
    X_train, X_test, y_train, _ = train_test_split(
        data.data, data.target, test_size=0.2, random_state=7
    )

    model = GBMRegressor(
        n_estimators=150,
        training_policy="manual",
        deterministic=True,
        seed=7,
    )
    model.fit(X_train, y_train)

    # Take a 300-row sample so this is fast but still exercises the
    # round-off accumulation path.
    rows = X_test[:300]
    expected_value, raw_values = model.shap_values(
        rows, include_expected_value=True
    )
    values = np.asarray(raw_values, dtype=np.float64)
    preds = np.asarray(model.predict(rows), dtype=np.float64)

    reconstructed = values.sum(axis=1) + float(expected_value)
    # Same idiom as the Rust additivity check.
    atol, rtol = 1e-5, 1e-4
    tol = atol + rtol * np.abs(preds)
    diff = np.abs(preds - reconstructed)
    assert np.all(diff <= tol), (
        f"max additivity drift {diff.max():.2e} exceeded tolerance "
        f"{tol[diff.argmax()]:.2e}"
    )
