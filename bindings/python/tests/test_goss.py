"""Smoke + sanity tests for v0.8.0 GOSS gradient-based one-side sampling.

GOSS (LightGBM): keep the top-`goss_top_rate` rows by `|gradient|`, sample
`goss_other_rate` from the rest, amplify sampled-low gradients by
`(1 - goss_top_rate) / goss_other_rate`.  Default boosting_mode is
"standard" (byte-identical to v0.7.5); `boosting_mode="goss"` opts in.

The Rust engine rejects GOSS on multiclass objectives with a clear error
message — verified below.
"""

from __future__ import annotations

import numpy as np
import pytest

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor
from alloygbm.evaluation import rmse


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(20260518)
    X = rng.normal(size=(2_000, 10)).astype("float32")
    y = (X[:, 0] - 0.5 * X[:, 1] + 0.1 * rng.normal(size=2_000)).astype("float32")
    return X, y


def test_goss_regressor_trains_and_predicts(regression_data):
    X, y = regression_data
    model = GBMRegressor(
        n_estimators=50,
        boosting_mode="goss",
        goss_top_rate=0.2,
        goss_other_rate=0.1,
    ).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (2_000,) if hasattr(preds, "shape") else len(preds) == 2_000
    arr = np.asarray(preds, dtype=np.float64)
    assert np.all(np.isfinite(arr))
    assert arr.std() > 0.1


def test_goss_does_not_regress_significantly_vs_uniform_subsample(regression_data):
    """GOSS keeping top 20% + sampling 10% (30% total) should be no worse
    than uniform 30% row subsampling under typical conditions.  Loose
    tolerance (1.30x) is deliberate — this is a smoke check, not a
    benchmark.
    """
    X, y = regression_data
    goss = GBMRegressor(
        n_estimators=80,
        boosting_mode="goss",
        goss_top_rate=0.2,
        goss_other_rate=0.1,
    ).fit(X, y)
    uniform = GBMRegressor(n_estimators=80, row_subsample=0.3).fit(X, y)
    goss_rmse = float(rmse(y, np.asarray(goss.predict(X), dtype=np.float64)))
    uniform_rmse = float(rmse(y, np.asarray(uniform.predict(X), dtype=np.float64)))
    assert goss_rmse <= uniform_rmse * 1.30, (
        f"goss_rmse={goss_rmse:.5f} should be within 30% of "
        f"uniform_rmse={uniform_rmse:.5f}"
    )


def test_uniform_row_subsample_updates_all_training_rows_after_commit():
    rng = np.random.default_rng(20260702)
    X = rng.normal(size=(1_200, 6)).astype("float32")
    y = (
        1.5 * X[:, 0]
        - 0.8 * X[:, 1]
        + 0.25 * np.sin(2.0 * X[:, 2])
        + 0.05 * rng.normal(size=X.shape[0])
    ).astype("float32")
    baseline_rmse = float(rmse(y, np.full_like(y, float(np.mean(y)))))

    model = GBMRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        row_subsample=0.5,
        continuous_binning_strategy="quantile",
        training_policy="manual",
        seed=17,
    ).fit(X, y)

    train_rmse = float(rmse(y, np.asarray(model.predict(X), dtype=np.float64)))
    assert model.n_estimators_ == 120
    assert train_rmse < baseline_rmse * 0.55, (
        f"row_subsample fit should beat the mean predictor by a wide margin; "
        f"train_rmse={train_rmse:.5f}, baseline_rmse={baseline_rmse:.5f}"
    )


def test_goss_regressor_predictions_do_not_overshoot_simple_target_range():
    rng = np.random.default_rng(20260703)
    X = rng.normal(size=(1_200, 6)).astype("float32")
    y = (
        1.5 * X[:, 0]
        - 0.8 * X[:, 1]
        + 0.25 * np.sin(2.0 * X[:, 2])
        + 0.05 * rng.normal(size=X.shape[0])
    ).astype("float32")
    baseline_rmse = float(rmse(y, np.full_like(y, float(np.mean(y)))))

    model = GBMRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        boosting_mode="goss",
        goss_top_rate=0.2,
        goss_other_rate=0.2,
        continuous_binning_strategy="quantile",
        training_policy="manual",
        seed=17,
    ).fit(X, y)

    preds = np.asarray(model.predict(X), dtype=np.float64)
    train_rmse = float(rmse(y, preds))
    target_span = float(np.max(y) - np.min(y))
    allowed_low = float(np.min(y)) - 0.25 * target_span
    allowed_high = float(np.max(y)) + 0.25 * target_span
    assert model.n_estimators_ == 120
    assert train_rmse < baseline_rmse * 0.65, (
        f"GOSS should not diverge beyond a mean predictor; "
        f"train_rmse={train_rmse:.5f}, baseline_rmse={baseline_rmse:.5f}"
    )
    assert float(np.min(preds)) >= allowed_low
    assert float(np.max(preds)) <= allowed_high


def test_goss_classifier_binary_trains_and_predicts():
    rng = np.random.default_rng(20260518)
    X = rng.normal(size=(1_000, 6)).astype("float32")
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    model = GBMClassifier(
        n_estimators=30,
        boosting_mode="goss",
        goss_top_rate=0.2,
        goss_other_rate=0.1,
    ).fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 1_000
    assert set(np.asarray(preds).tolist()).issubset({0, 1})


def test_goss_ranker_trains_and_predicts():
    rng = np.random.default_rng(20260518)
    X = rng.normal(size=(800, 6)).astype("float32")
    y = rng.integers(0, 4, size=800).astype("float32")
    group = np.array([0] * 200 + [1] * 200 + [2] * 200 + [3] * 200, dtype="int32")
    model = GBMRanker(
        n_estimators=30,
        boosting_mode="goss",
        goss_top_rate=0.2,
        goss_other_rate=0.1,
    ).fit(X, y, group=group)
    preds = model.predict(X)
    assert len(preds) == 800
    arr = np.asarray(preds, dtype=np.float64)
    assert np.all(np.isfinite(arr))


def test_goss_now_supported_on_multiclass():
    """v0.10.1 enabled multiclass softmax + GOSS using per-row scoring
    `s_i = sum_k |g_{i,k}|` (LightGBM convention). The v0.8.0 rejection
    test was inverted in v0.10.1; the new positive coverage lives in
    `test_multiclass_goss.py`."""
    rng = np.random.default_rng(20260518)
    X = rng.normal(size=(300, 4)).astype("float32")
    y = rng.integers(0, 3, size=300).astype(int)
    model = GBMClassifier(
        n_estimators=4,
        boosting_mode="goss",
        goss_top_rate=0.2,
        goss_other_rate=0.1,
    )
    model.fit(X, y)  # Should NOT raise.
    assert model.predict_proba(X).shape == (300, 3)


def test_goss_rejects_invalid_rate_ranges():
    with pytest.raises(ValueError, match=r"goss_top_rate"):
        GBMRegressor(boosting_mode="goss", goss_top_rate=0.0, goss_other_rate=0.1)
    with pytest.raises(ValueError, match=r"goss_other_rate"):
        GBMRegressor(boosting_mode="goss", goss_top_rate=0.2, goss_other_rate=1.5)
    with pytest.raises(ValueError, match=r"<= 1.0"):
        GBMRegressor(boosting_mode="goss", goss_top_rate=0.7, goss_other_rate=0.5)


def test_dart_basic_construction_works():
    """v0.9.0: DART is fully wired through for the single-output trainer."""
    m = GBMRegressor(boosting_mode="dart", dart_drop_rate=0.1, dart_max_drop=5)
    assert m.boosting_mode == "dart"
    assert m.dart_drop_rate == 0.1
    assert m.dart_max_drop == 5


def test_unknown_boosting_mode_rejected():
    with pytest.raises(ValueError, match=r"boosting_mode"):
        GBMRegressor(boosting_mode="not_a_mode")


def test_default_boosting_mode_is_standard_and_bytewise_compatible():
    """Standard mode (default) must produce the same predictions as
    omitting boosting_mode entirely — proves byte-compat with v0.7.5.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 5)).astype("float32")
    y = (X[:, 0] - X[:, 1] + 0.1 * rng.standard_normal(300)).astype("float32")
    legacy = GBMRegressor(n_estimators=20, seed=42).fit(X, y)
    explicit = GBMRegressor(n_estimators=20, seed=42, boosting_mode="standard").fit(X, y)
    np.testing.assert_array_equal(
        np.asarray(legacy.predict(X)),
        np.asarray(explicit.predict(X)),
        err_msg="boosting_mode='standard' should be byte-identical to default",
    )
