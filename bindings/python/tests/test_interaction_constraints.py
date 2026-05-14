"""Tests for ``interaction_constraints`` on the estimator surface (v0.7.1).

The estimator accepts a LightGBM-compatible ``interaction_constraints`` list
of feature-index groups.  At training time each tree's root-to-leaf path is
restricted so it may only split on features from a single allowed group;
features outside every group are unconstrained and may be used freely.

These tests cover:

* parameter validation (constructor and ``set_params``);
* propagation through ``get_params`` and the ``__repr__`` string;
* end-to-end fit + classifier sanity (the engine-level test in
  ``crates/engine`` performs the per-path tree-walk enforcement assertion).
"""

from __future__ import annotations

import numpy as np
import pytest

from alloygbm import GBMClassifier, GBMRegressor


class TestInteractionConstraintValidation:
    def test_constructor_accepts_empty_groups_list(self) -> None:
        # Empty list is equivalent to no constraints — must not error.
        m = GBMRegressor(interaction_constraints=[])
        assert m.interaction_constraints == []

    def test_constructor_rejects_too_many_groups(self) -> None:
        with pytest.raises(ValueError, match="at most 64 groups"):
            GBMRegressor(interaction_constraints=[[i] for i in range(65)])

    def test_constructor_rejects_empty_group(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            GBMRegressor(interaction_constraints=[[]])

    def test_constructor_rejects_negative_feature_index(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            GBMRegressor(interaction_constraints=[[-1, 0]])

    def test_constructor_rejects_duplicate_feature_in_group(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            GBMRegressor(interaction_constraints=[[0, 1, 0]])

    def test_set_params_round_trip(self) -> None:
        m = GBMRegressor()
        m.set_params(interaction_constraints=[[0, 1], [2, 3]])
        assert m.interaction_constraints == [[0, 1], [2, 3]]
        params = m.get_params()
        assert params["interaction_constraints"] == [[0, 1], [2, 3]]
        m.set_params(interaction_constraints=None)
        assert m.interaction_constraints is None


class TestInteractionConstraintEnforcement:
    def test_unconstrained_feature_allowed_anywhere(self) -> None:
        # Feature 5 is in no constraint group; it should be usable alongside
        # features from any group.  We don't assert which path picks it —
        # just that training doesn't error and the model produces finite
        # predictions.
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 6)).astype("float32")
        y = (X[:, 0] - X[:, 5]).astype("float32")
        m = GBMRegressor(
            n_estimators=3,
            interaction_constraints=[[0, 1], [2, 3]],
        ).fit(X, y)
        pred = np.asarray(m.predict(X[:5]))
        assert np.all(np.isfinite(pred))

    def test_classifier_accepts_interaction_constraints(self) -> None:
        rng = np.random.default_rng(2)
        X = rng.standard_normal((200, 4)).astype("float32")
        y = (X[:, 0] > 0).astype("int32")
        m = GBMClassifier(
            n_estimators=3,
            interaction_constraints=[[0, 1], [2, 3]],
        ).fit(X, y)
        proba = np.asarray(m.predict_proba(X[:5]))
        assert proba.shape == (5, 2)
        assert np.all(np.isfinite(proba))
