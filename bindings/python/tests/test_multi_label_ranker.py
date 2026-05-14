"""Tests for ``MultiLabelGBMRanker`` (v0.7.1).

The wrapper trains one :class:`GBMRanker` per label using a shared ``group``
and (optional) ``factor_exposures``, then stacks predictions to match the
``(n_rows, n_labels)`` shape.  Joint shared-tree multi-label training is a
v0.7.2 follow-up; this test suite asserts the wrapper API contract and
documents the per-label independence semantics.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from alloygbm import GBMRanker, MultiLabelGBMRanker


def _ranking_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_groups, per_group = 15, 5
    n = n_groups * per_group
    X = rng.standard_normal((n, 4)).astype("float32")
    y_a = (X[:, 0] * 2 + 0.3 * rng.standard_normal(n)).astype("float32")
    y_b = (X[:, 1] - X[:, 0] + 0.2 * rng.standard_normal(n)).astype("float32")
    y = np.stack([y_a, y_b], axis=1)
    group = np.repeat(np.arange(n_groups), per_group)
    return X, y, group


class TestMultiLabelRankerAPI:
    def test_rejects_1d_y(self) -> None:
        X = np.zeros((10, 2), dtype="float32")
        y = np.zeros(10, dtype="float32")
        with pytest.raises(ValueError, match="2-D"):
            MultiLabelGBMRanker().fit(X, y, group=np.zeros(10, dtype="int32"))

    def test_rejects_zero_labels(self) -> None:
        X = np.zeros((10, 2), dtype="float32")
        y = np.zeros((10, 0), dtype="float32")
        with pytest.raises(ValueError, match="at least one label"):
            MultiLabelGBMRanker().fit(X, y, group=np.zeros(10, dtype="int32"))

    def test_positional_labels_inferred_from_y_shape(self) -> None:
        X, y, group = _ranking_data(seed=1)
        m = MultiLabelGBMRanker(n_estimators=3).fit(X, y, group=group)
        assert m.n_labels_ == 2
        assert m.ranking_labels_ == ["label_0", "label_1"]

    def test_named_labels(self) -> None:
        X, y, group = _ranking_data(seed=2)
        m = MultiLabelGBMRanker(
            ranking_labels=["ctr", "conv"], n_estimators=2
        ).fit(X, y, group=group)
        assert m.ranking_labels_ == ["ctr", "conv"]

    def test_label_count_mismatch_raises(self) -> None:
        X, y, group = _ranking_data(seed=3)
        m = MultiLabelGBMRanker(ranking_labels=["only_one"])  # but y has 2 cols
        with pytest.raises(ValueError, match="ranking_labels length"):
            m.fit(X, y, group=group)

    def test_predict_shape(self) -> None:
        X, y, group = _ranking_data(seed=4)
        m = MultiLabelGBMRanker(n_estimators=3).fit(X, y, group=group)
        pred = m.predict(X[:7])
        assert pred.shape == (7, 2)
        assert np.all(np.isfinite(pred))

    def test_predict_before_fit_raises(self) -> None:
        m = MultiLabelGBMRanker()
        with pytest.raises(RuntimeError, match="fit before predict"):
            m.predict(np.zeros((3, 2)))


class TestMultiLabelRankerObjectives:
    def test_single_objective_applies_to_every_label(self) -> None:
        X, y, group = _ranking_data(seed=5)
        m = MultiLabelGBMRanker(
            ranking_objective="rank:pairwise", n_estimators=2
        ).fit(X, y, group=group)
        for sub in m.sub_rankers_:
            assert sub.ranking_objective == "rank:pairwise"

    def test_heterogeneous_objectives(self) -> None:
        X, y, group = _ranking_data(seed=6)
        m = MultiLabelGBMRanker(
            ranking_objective=["rank:ndcg", "rank:pairwise"], n_estimators=2
        ).fit(X, y, group=group)
        assert m.sub_rankers_[0].ranking_objective == "rank:ndcg"
        assert m.sub_rankers_[1].ranking_objective == "rank:pairwise"

    def test_objective_list_length_mismatch_raises(self) -> None:
        X, y, group = _ranking_data(seed=7)
        m = MultiLabelGBMRanker(
            ranking_objective=["rank:ndcg", "rank:pairwise", "rank:xendcg"]
        )
        with pytest.raises(ValueError, match="ranking_objective list length"):
            m.fit(X, y, group=group)


class TestMultiLabelRankerIndependenceContract:
    def test_predictions_match_independent_per_label_fits(self) -> None:
        # The v0.7.1 wrapper is documented as numerically equivalent to
        # training each label independently.  Verify by comparing wrapper
        # predictions against two `GBMRanker` instances trained per label
        # with identical hyper-parameters.
        X, y, group = _ranking_data(seed=11)
        kwargs = dict(
            n_estimators=4,
            max_depth=3,
            seed=42,
            deterministic=True,
        )
        wrapper = MultiLabelGBMRanker(**kwargs).fit(X, y, group=group)
        wrap_pred = wrapper.predict(X[:10])

        for label_idx in range(y.shape[1]):
            single = GBMRanker(**kwargs).fit(X, y[:, label_idx], group=group)
            single_pred = np.asarray(single.predict(X[:10]))
            np.testing.assert_allclose(
                wrap_pred[:, label_idx], single_pred, atol=1e-5
            )


class TestMultiLabelRankerPersistence:
    def test_save_load_round_trip(self) -> None:
        X, y, group = _ranking_data(seed=13)
        m = MultiLabelGBMRanker(
            ranking_labels=["alpha", "beta"], n_estimators=3
        ).fit(X, y, group=group)
        before = m.predict(X[:5])

        with tempfile.NamedTemporaryFile(suffix=".mlrk", delete=False) as f:
            path = f.name
        try:
            m.save_model(path)
            restored = MultiLabelGBMRanker.load_model(path)
            after = restored.predict(X[:5])
        finally:
            os.unlink(path)

        np.testing.assert_allclose(after, before, atol=1e-6)
        assert restored.ranking_labels_ == ["alpha", "beta"]
        assert restored.n_labels_ == 2

    def test_save_before_fit_raises(self) -> None:
        m = MultiLabelGBMRanker()
        with tempfile.NamedTemporaryFile(suffix=".mlrk", delete=False) as f:
            path = f.name
        try:
            with pytest.raises(RuntimeError, match="fit before save_model"):
                m.save_model(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestMultiLabelRankerParamsRoundTrip:
    def test_get_params_includes_ranking_labels_and_objective(self) -> None:
        m = MultiLabelGBMRanker(
            ranking_labels=["a", "b"],
            ranking_objective="rank:ndcg",
            n_estimators=5,
        )
        p = m.get_params()
        assert p["ranking_labels"] == ["a", "b"]
        assert p["ranking_objective"] == "rank:ndcg"
        assert p["n_estimators"] == 5

    def test_set_params_updates_kwargs(self) -> None:
        m = MultiLabelGBMRanker(n_estimators=2)
        m.set_params(n_estimators=10, ranking_objective="rank:pairwise")
        assert m._per_label_kwargs["n_estimators"] == 10
        assert m.ranking_objective == "rank:pairwise"


class TestMultiLabelRankerEvalSet:
    """`eval_set=(X_val, y_val)` is per-label-sliced inside the wrapper so
    every per-label fit sees a 1-D validation target.  Without slicing,
    ``GBMRegressor._validate_targets`` would try to cast each row vector to
    a float and reject the 2-D ``y_val``."""

    def test_eval_set_with_2d_y_val_enables_early_stopping(self) -> None:
        X, y, group = _ranking_data(seed=51)
        X_val, y_val, val_group = _ranking_data(seed=52)
        m = MultiLabelGBMRanker(
            n_estimators=5,
            early_stopping_rounds=2,
            min_validation_improvement=0.0,
        ).fit(
            X,
            y,
            group=group,
            eval_set=(X_val, y_val),
            eval_group=val_group,
        )
        # Each sub-ranker honored its own early-stopping signal, so per-label
        # round counts can differ but must be > 0 and ≤ n_estimators.
        for rounds in m.rounds_completed_ or []:
            assert 0 < rounds <= 5

    def test_eval_set_column_count_mismatch_raises(self) -> None:
        X, y, group = _ranking_data(seed=53)
        X_val, _, val_group = _ranking_data(seed=54)
        bad_y_val = np.zeros((X_val.shape[0], 3), dtype="float32")  # 3 ≠ 2
        m = MultiLabelGBMRanker(
            n_estimators=3,
            early_stopping_rounds=1,
            min_validation_improvement=0.0,
        )
        with pytest.raises(ValueError, match="label columns"):
            m.fit(X, y, group=group, eval_set=(X_val, bad_y_val), eval_group=val_group)


class TestMultiLabelRankerLoadRestoresWrapperConfig:
    """``load_model`` must restore the wrapper-level configuration
    (``ranking_objective``, the per-label kwargs) so sklearn-style
    introspection / clone behavior matches the saved model.  Before the
    v0.7.1 fix this dropped everything except ``ranking_labels``."""

    def test_load_preserves_heterogeneous_ranking_objectives(self) -> None:
        X, y, group = _ranking_data(seed=61)
        m = MultiLabelGBMRanker(
            ranking_objective=["rank:ndcg", "rank:pairwise"],
            n_estimators=2,
            max_depth=2,
            seed=7,
        ).fit(X, y, group=group)
        with tempfile.NamedTemporaryFile(suffix=".mlrk", delete=False) as f:
            path = f.name
        try:
            m.save_model(path)
            restored = MultiLabelGBMRanker.load_model(path)
        finally:
            os.unlink(path)
        params = restored.get_params()
        assert params["ranking_objective"] == ["rank:ndcg", "rank:pairwise"]
        assert params["n_estimators"] == 2
        assert params["max_depth"] == 2
        assert params["seed"] == 7

    def test_load_collapses_homogeneous_objectives_to_string(self) -> None:
        X, y, group = _ranking_data(seed=63)
        m = MultiLabelGBMRanker(
            ranking_objective="rank:pairwise",
            n_estimators=2,
            max_depth=3,
        ).fit(X, y, group=group)
        with tempfile.NamedTemporaryFile(suffix=".mlrk", delete=False) as f:
            path = f.name
        try:
            m.save_model(path)
            restored = MultiLabelGBMRanker.load_model(path)
        finally:
            os.unlink(path)
        params = restored.get_params()
        # All sub-rankers share the same objective so the load collapses to
        # a single string (matches the original constructor input).
        assert params["ranking_objective"] == "rank:pairwise"
        assert params["max_depth"] == 3
