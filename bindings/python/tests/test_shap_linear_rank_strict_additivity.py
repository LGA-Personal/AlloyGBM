"""Strict-additivity regression for the mixed linear-rank SHAP path.

Closes Limitation 4: when `continuous_binning_strategy="linear"` triggers
per-feature rank-based binning on at least one column, SHAP should still
use the predictor-aligned binning context (not the legacy quantize-then-walk
fallback) so strict additivity holds for `leaf_model="linear"` as well as
`leaf_model="constant"`.
"""

from __future__ import annotations

import numpy as np
import pytest

from alloygbm import GBMRegressor


@pytest.fixture(autouse=True)
def _enable_linear_tail_rank(monkeypatch):
    """The mixed linear-rank binning policy is gated by an experiment flag.

    These tests are about the SHAP code path that runs when the flag is on
    and at least one feature is selected for rank-based binning; enable it
    for the test session and let the per-feature policy decide which
    columns actually use rank binning.
    """
    monkeypatch.setenv("ALLOYGBM_EXPERIMENT_LINEAR_TAIL_RANK", "1")


@pytest.fixture
def skewed_data():
    rng = np.random.default_rng(20260517)
    # Lognormal columns are skewed enough that the linear-binning policy
    # auto-selects per-feature rank binning on most features.
    X = rng.lognormal(size=(500, 6)).astype("float32")
    y = (X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=500)).astype("float32")
    return X, y


@pytest.mark.parametrize("leaf_model", ["constant", "linear"])
def test_mixed_linear_rank_uses_predictor_aligned_binning(skewed_data, leaf_model):
    """Architectural contract: when rank flags fire, SHAP must use the
    predictor-aligned binning path (not the legacy quantize-then-walk
    fallback that exempts linear leaves from strict additivity).
    """
    X, y = skewed_data
    model = GBMRegressor(
        n_estimators=30,
        continuous_binning_strategy="linear",
        continuous_binning_max_bins=64,
        leaf_model=leaf_model,
    ).fit(X, y)

    flags = model._continuous_feature_linear_rank_flags
    assert flags is not None and any(flags), (
        "fixture should exercise mixed linear-rank binning"
    )

    # Contract: SHAP binning kwargs must not be None for this path.
    kwargs = model._shap_binning_kwargs()
    assert kwargs is not None, (
        "mixed linear-rank path must use predictor-aligned SHAP binning"
    )
    assert kwargs["binning_kind"] == "linear_rank", (
        f"expected binning_kind='linear_rank', got {kwargs.get('binning_kind')!r}"
    )


@pytest.mark.parametrize("leaf_model", ["constant", "linear"])
def test_mixed_linear_rank_strict_additivity(skewed_data, leaf_model):
    X, y = skewed_data
    model = GBMRegressor(
        n_estimators=30,
        continuous_binning_strategy="linear",
        continuous_binning_max_bins=64,
        leaf_model=leaf_model,
    ).fit(X, y)

    # The fixture is chosen so that at least one column triggers the
    # rank-based linear binning policy.  Guard the test invariant.
    flags = model._continuous_feature_linear_rank_flags
    assert flags is not None, (
        "expected linear-binning policy to populate rank flags; got None"
    )
    assert any(flags), (
        "expected at least one feature to use rank-based linear binning"
    )

    expected, phi = model.shap_values(X, include_expected_value=True)
    preds = model.predict(X)
    reconstructed = np.asarray(phi, dtype=np.float64).sum(axis=1) + float(expected)
    np.testing.assert_allclose(
        reconstructed,
        preds,
        atol=1e-5,
        rtol=1e-4,
        err_msg=(
            "mixed linear-rank SHAP failed strict additivity for "
            f"leaf_model={leaf_model!r}"
        ),
    )


def test_mixed_linear_rank_flags_are_populated(skewed_data):
    """Sanity guard: skewed lognormal data exercises the mixed code path."""
    X, y = skewed_data
    model = GBMRegressor(
        n_estimators=5,
        continuous_binning_strategy="linear",
        continuous_binning_max_bins=64,
    ).fit(X, y)
    flags = model._continuous_feature_linear_rank_flags
    assert flags is not None
    assert any(flags)
