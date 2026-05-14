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
