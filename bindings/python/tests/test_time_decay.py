"""Opt-in time-decay row weighting (``time_decay_half_life``)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor, MultiLabelGBMRanker
from alloygbm._regressor import _auto_rounds


def _panel(rows_per_period: int = 40, periods: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    period = np.repeat(np.arange(periods), rows_per_period)
    X = rng.normal(size=(len(period), 4))
    slope = np.linspace(-1.0, 1.0, periods)[period]
    y = slope * X[:, 0] + 0.5 * X[:, 1] + 0.1 * rng.normal(size=len(period))
    return X, y, period


def _decay(period: np.ndarray, half_life: float) -> np.ndarray:
    return 0.5 ** ((period.max() - period) / half_life)


def test_default_is_off_and_matches_explicit_weights_free_fit():
    X, y, period = _panel()
    plain = GBMRegressor(n_estimators=20, seed=3).fit(X, y)
    with_index = GBMRegressor(n_estimators=20, seed=3).fit(X, y, time_index=period)
    assert plain.time_decay_half_life is None
    assert plain.time_decay_ is None
    assert plain.artifact_bytes == with_index.artifact_bytes


def test_decay_equals_the_same_sample_weight():
    X, y, period = _panel()
    user_weight = np.random.default_rng(1).uniform(0.5, 2.0, size=len(y))
    decayed = GBMRegressor(n_estimators=20, seed=3, time_decay_half_life=6.0).fit(
        X, y, time_index=period, sample_weight=user_weight
    )
    weighted = GBMRegressor(n_estimators=20, seed=3).fit(
        X, y, sample_weight=user_weight * _decay(period, 6.0)
    )
    np.testing.assert_allclose(decayed.predict(X), weighted.predict(X), rtol=1e-6, atol=1e-6)


def test_time_decay_summary_reports_effective_sample_size():
    X, y, period = _panel()
    model = GBMRegressor(n_estimators=5, time_decay_half_life=6.0).fit(X, y, time_index=period)
    weights = _decay(period, 6.0)
    summary = model.time_decay_
    assert summary["half_life"] == 6.0
    assert summary["time_max"] == int(period.max())
    assert summary["rows"] == len(y)
    assert summary["min_decay_weight"] == pytest.approx(weights.min())
    assert summary["effective_sample_size"] == pytest.approx(
        weights.sum() ** 2 / np.square(weights).sum()
    )
    assert summary["effective_sample_size"] < len(y)


def test_auto_rounds_holds_out_the_latest_periods():
    X, y, period = _panel()
    model = GBMRegressor(time_decay_half_life=6.0, seed=2).fit(X, y, time_index=period)
    info = model.auto_rounds_
    assert info["mode"] == "holdout"
    assert info["holdout_split"] == "latest_periods"
    # 30 periods of 40 rows: the newest 3 periods are the 10% holdout.
    assert info["validation_rows"] == 120
    # Under decay the refit gains less than the row share suggests.
    assert 0.0 <= info["validation_fraction"] < 0.1
    assert model.time_decay_["rows"] == len(y)


def test_infinite_half_life_is_the_time_holdout_without_decay():
    X, y, period = _panel()
    fixed = dict(n_estimators=20, seed=3)
    plain = GBMRegressor(**fixed).fit(X, y)
    undecayed = GBMRegressor(**fixed, time_decay_half_life=float("inf")).fit(
        X, y, time_index=period
    )
    assert undecayed.artifact_bytes == plain.artifact_bytes
    assert undecayed.time_decay_["effective_sample_size"] == pytest.approx(len(y))
    auto = GBMRegressor(time_decay_half_life=float("inf")).fit(X, y, time_index=period)
    assert auto.auto_rounds_["holdout_split"] == "latest_periods"
    assert auto.auto_rounds_["validation_fraction"] == pytest.approx(0.1)


def test_time_index_without_decay_keeps_the_legacy_auto_rounds():
    X, y, period = _panel()
    model = GBMRegressor(seed=2).fit(X, y, time_index=period)
    assert model.auto_rounds_["mode"] == "legacy"


def test_time_holdout_split_takes_whole_newest_periods():
    period = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 3])
    train, valid = _auto_rounds.time_holdout_split(period, 0.25)
    np.testing.assert_array_equal(valid, [7, 8, 9])
    np.testing.assert_array_equal(train, np.arange(7))
    assert _auto_rounds.time_holdout_split(np.zeros(10), 0.1) is None
    # Never holds out every period.
    train, valid = _auto_rounds.time_holdout_split(np.array([0, 1, 1, 1]), 0.9)
    np.testing.assert_array_equal(train, [0])
    # A query group on both sides of the cut makes the split unusable.
    groups = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3])
    assert _auto_rounds.time_holdout_split(period, 0.25, group_ids=groups) is None


def test_requires_time_index():
    X, y, _ = _panel()
    with pytest.raises(ValueError, match="requires time_index"):
        GBMRegressor(time_decay_half_life=4.0).fit(X, y)
    with pytest.raises(ValueError, match="requires time_index"):
        GBMRegressor(n_estimators=5, time_decay_half_life=4.0).fit(X, y)


@pytest.mark.parametrize("value", [0, -1.0, float("nan"), float("-inf"), True, "x"])
def test_rejects_bad_half_life(value):
    X, y, period = _panel()
    with pytest.raises(ValueError, match="time_decay_half_life"):
        GBMRegressor(time_decay_half_life=value).fit(X, y, time_index=period)


def test_params_roundtrip_and_pickle():
    model = GBMRegressor(time_decay_half_life=12.0)
    assert model.get_params()["time_decay_half_life"] == 12.0
    assert "time_decay_half_life=12.0" in repr(model)
    model.set_params(time_decay_half_life=None)
    assert model.time_decay_half_life is None
    X, y, period = _panel()
    model.set_params(time_decay_half_life=8.0, n_estimators=5)
    model.fit(X, y, time_index=period)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.time_decay_ == model.time_decay_
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    state = model.__getstate__()
    state.pop("time_decay_half_life")
    old = GBMRegressor.__new__(GBMRegressor)
    old.__setstate__(state)
    assert old.time_decay_half_life is None


def test_classifier_binary_and_multiclass():
    X, y, period = _panel()
    binary = GBMClassifier(time_decay_half_life=6.0).fit(X, (y > 0).astype(int), time_index=period)
    assert binary.auto_rounds_["holdout_split"] == "latest_periods"
    assert "time_decay_half_life=6.0" in repr(binary)
    labels = np.digitize(y, [-0.4, 0.4])
    multiclass = GBMClassifier(n_estimators=10, time_decay_half_life=6.0).fit(
        X, labels, time_index=period
    )
    assert multiclass.predict_proba(X).shape == (len(y), 3)
    assert multiclass.time_decay_["effective_sample_size"] < len(y)


def test_ranker_with_weighted_objective():
    X, y, period = _panel()
    ranker = GBMRanker(ranking_objective="queryrmse", time_decay_half_life=6.0).fit(
        X, y, group=period, time_index=period
    )
    assert ranker.auto_rounds_["holdout_split"] == "latest_periods"
    assert "time_decay_half_life=6.0" in repr(ranker)


def test_ranker_rejects_objectives_that_ignore_weights():
    X, y, period = _panel()
    with pytest.raises(ValueError, match="ignores"):
        GBMRanker(n_estimators=5, time_decay_half_life=6.0).fit(
            X, (y > 0).astype(int), group=period, time_index=period
        )


def test_multilabel_independent_forwards_and_joint_rejects():
    X, y, period = _panel()
    targets = np.column_stack([y, -y])
    independent = MultiLabelGBMRanker(
        ranking_objective="queryrmse", n_estimators=5, time_decay_half_life=6.0
    ).fit(X, targets, group=period, time_index=period)
    assert independent.predict(X).shape == (len(y), 2)
    with pytest.raises(NotImplementedError):
        MultiLabelGBMRanker(
            multi_label_mode="joint",
            ranking_objective="queryrmse",
            n_estimators=5,
            time_decay_half_life=6.0,
        ).fit(X, targets, group=period)
