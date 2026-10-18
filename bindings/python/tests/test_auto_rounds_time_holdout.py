"""``n_estimators="auto"`` holds out the latest periods when given ``time_index``."""

from __future__ import annotations

import numpy as np

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor
from alloygbm._regressor import _auto_rounds


def _panel(rows_per_period: int = 40, periods: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    period = np.repeat(np.arange(periods), rows_per_period)
    X = rng.normal(size=(len(period), 4))
    slope = np.linspace(-1.0, 1.0, periods)[period]
    y = slope * X[:, 0] + 0.5 * X[:, 1] + 0.1 * rng.normal(size=len(period))
    return X, y, period


def test_time_index_holds_out_the_latest_periods():
    X, y, period = _panel()
    model = GBMRegressor(seed=2).fit(X, y, time_index=period)
    info = model.auto_rounds_
    assert info["mode"] == "holdout"
    assert info["holdout_split"] == "latest_periods"
    # 30 periods of 40 rows: the newest 3 periods are the 10% holdout.
    assert info["validation_rows"] == 120
    assert info["validation_fraction"] == 0.1


def test_without_time_index_the_holdout_stays_random():
    X, y, _ = _panel()
    model = GBMRegressor(seed=2).fit(X, y)
    assert model.auto_rounds_["holdout_split"] == "random"


def test_explicit_rounds_ignore_time_index():
    X, y, period = _panel()
    plain = GBMRegressor(n_estimators=20, seed=3).fit(X, y)
    timed = GBMRegressor(n_estimators=20, seed=3).fit(X, y, time_index=period)
    assert plain.artifact_bytes == timed.artifact_bytes


def test_single_period_falls_back_to_legacy_rounds():
    X, y, _ = _panel()
    model = GBMRegressor(seed=2).fit(X, y, time_index=np.zeros(len(y), dtype=int))
    assert model.auto_rounds_["mode"] == "legacy"


def test_categorical_time_aware_keeps_legacy_rounds():
    X, y, period = _panel()
    model = GBMRegressor(categorical_time_aware=True, seed=2).fit(X, y, time_index=period)
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


def test_classifier_and_ranker_use_the_time_holdout():
    X, y, period = _panel()
    binary = GBMClassifier(seed=2).fit(X, (y > 0).astype(int), time_index=period)
    assert binary.auto_rounds_["holdout_split"] == "latest_periods"
    multiclass = GBMClassifier(seed=2).fit(X, np.digitize(y, [-0.4, 0.4]), time_index=period)
    assert multiclass.auto_rounds_["holdout_split"] == "latest_periods"
    ranker = GBMRanker(seed=2).fit(X, (y > 0).astype(int), group=period, time_index=period)
    assert ranker.auto_rounds_["holdout_split"] == "latest_periods"
