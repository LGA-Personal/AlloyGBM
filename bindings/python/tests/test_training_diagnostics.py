"""Tests for `diagnostics_per_round_` on the estimator surface (v0.7.1).

Each completed training round writes one `IterationDiagnostics` snapshot
containing gradient/hessian magnitudes and — when factor neutralization is
active — a "neutralization effectiveness" score.  The Python wrapper
materializes those as a list of dicts on every estimator.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor


EXPECTED_KEYS = {
    "gradient_l2_norm",
    "gradient_variance",
    "hessian_l2_norm",
    "original_gradient_l2_norm",
    "projected_gradient_l2_norm",
    "neutralization_effectiveness",
    "n_active_rows",
    "n_active_features",
}


def _shape_assertions(diagnostics, expected_rounds, *, expect_projection: bool) -> None:
    assert diagnostics is not None
    assert len(diagnostics) == expected_rounds
    for entry in diagnostics:
        assert set(entry.keys()) == EXPECTED_KEYS
        assert math.isfinite(entry["gradient_l2_norm"])
        assert entry["gradient_l2_norm"] >= 0.0
        assert entry["gradient_variance"] >= 0.0
        assert entry["hessian_l2_norm"] >= 0.0
        assert entry["n_active_rows"] > 0
        assert entry["n_active_features"] > 0
        if expect_projection:
            assert entry["original_gradient_l2_norm"] is not None
            assert entry["projected_gradient_l2_norm"] is not None
            eff = entry["neutralization_effectiveness"]
            assert eff is None or 0.0 <= eff <= 1.0
        else:
            assert entry["original_gradient_l2_norm"] is None
            assert entry["projected_gradient_l2_norm"] is None
            assert entry["neutralization_effectiveness"] is None


class TestRegressorDiagnostics:
    def test_unfitted_regressor_has_no_diagnostics(self) -> None:
        m = GBMRegressor(n_estimators=3)
        assert m.diagnostics_per_round_ is None

    def test_fitted_regressor_emits_one_entry_per_round(self) -> None:
        rng = np.random.default_rng(7)
        X = rng.standard_normal((120, 3)).astype("float32")
        y = (X[:, 0] * 0.7 + 0.1 * rng.standard_normal(120)).astype("float32")
        m = GBMRegressor(n_estimators=4, max_depth=2).fit(X, y)
        _shape_assertions(
            m.diagnostics_per_round_, m.rounds_completed_, expect_projection=False
        )

    def test_gradient_norm_decreases_for_learnable_target(self) -> None:
        rng = np.random.default_rng(11)
        X = rng.standard_normal((200, 2)).astype("float32")
        y = (X[:, 0] * 2.0 - X[:, 1] * 1.0).astype("float32")
        m = GBMRegressor(n_estimators=6, max_depth=3).fit(X, y)
        norms = [d["gradient_l2_norm"] for d in m.diagnostics_per_round_]
        # Final round's gradient norm should be smaller than the first; we
        # don't require strict monotonicity round-to-round.
        assert norms[-1] < norms[0]


class TestClassifierDiagnostics:
    def test_binary_classifier_diagnostics(self) -> None:
        rng = np.random.default_rng(13)
        X = rng.standard_normal((150, 4)).astype("float32")
        y = (X[:, 0] > 0).astype("int32")
        m = GBMClassifier(n_estimators=3).fit(X, y)
        _shape_assertions(
            m.diagnostics_per_round_, m.rounds_completed_, expect_projection=False
        )


class TestRankerDiagnostics:
    def test_ranker_diagnostics(self) -> None:
        rng = np.random.default_rng(17)
        n_groups = 10
        per_group = 5
        n = n_groups * per_group
        X = rng.standard_normal((n, 3)).astype("float32")
        relevance = (X[:, 0] * 2).astype("int32").clip(0, 4)
        group = np.repeat(np.arange(n_groups), per_group)
        m = GBMRanker(n_estimators=3).fit(X, relevance, group=group)
        _shape_assertions(
            m.diagnostics_per_round_, m.rounds_completed_, expect_projection=False
        )


class TestNeutralizationDiagnostics:
    def test_per_round_gradient_neutralization_populates_projection_fields(self) -> None:
        rng = np.random.default_rng(23)
        X = rng.standard_normal((200, 5)).astype("float32")
        # Target depends on features 0 and 1; we neutralize against features
        # 2 and 3, which are independent of y.  The projection therefore
        # removes only sampling noise from the gradient, leaving the signal
        # intact so the model still completes the requested rounds.
        y = (X[:, 0] * 1.5 - X[:, 1] * 0.5 + 0.1 * rng.standard_normal(200)).astype(
            "float32"
        )
        exposures = X[:, 2:4].copy()
        m = GBMRegressor(
            n_estimators=4,
            max_depth=2,
            neutralization="per_round_gradient",
            factor_neutralization_lambda=1e-4,
        ).fit(X, y, factor_exposures=exposures)

        assert m.rounds_completed_ > 0
        _shape_assertions(
            m.diagnostics_per_round_, m.rounds_completed_, expect_projection=True
        )
        # The projected-gradient norm should never exceed the original; we
        # check this on every round we recorded.
        for d in m.diagnostics_per_round_:
            assert d["projected_gradient_l2_norm"] <= d["original_gradient_l2_norm"] + 1e-4

    def test_pre_target_neutralization_omits_projection_fields(self) -> None:
        # `pre_target` residualizes targets once before fit; gradients are
        # never projected per round, so the diagnostics' projection fields
        # must stay `None`.
        rng = np.random.default_rng(29)
        X = rng.standard_normal((150, 3)).astype("float32")
        y = X[:, 2].astype("float32")  # target uncorrelated with exposures below
        exposures = X[:, :1].copy()
        m = GBMRegressor(
            n_estimators=3,
            neutralization="pre_target",
            factor_neutralization_lambda=1e-4,
        ).fit(X, y, factor_exposures=exposures)
        assert m.rounds_completed_ > 0
        _shape_assertions(
            m.diagnostics_per_round_, m.rounds_completed_, expect_projection=False
        )
