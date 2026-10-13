"""All-integer feature matrices with values beyond the bin range (prices,
counts, IDs) train with continuous binning instead of failing as
"pre-binned" data (found on OpenML-CTR23 brazilian_houses)."""

from __future__ import annotations

import numpy as np

from alloygbm import GBMClassifier, GBMRegressor


def _data(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 200_000, (n, 3)).astype(float)
    y = X[:, 0] / 1e5 + 0.1 * rng.normal(size=n)
    return X, y


def test_regressor_trains_and_predicts_on_large_integers():
    X, y = _data()
    model = GBMRegressor(n_estimators=100, seed=0).fit(X[:1500], y[:1500])
    assert model._uses_continuous_binning
    rmse = float(np.sqrt(np.mean((model.predict(X[1500:]) - y[1500:]) ** 2)))
    assert rmse < 0.15


def test_classifier_trains_on_large_integers():
    X, y = _data(seed=1)
    labels = (y > 1.0).astype(int)
    model = GBMClassifier(n_estimators=50, seed=0).fit(X, labels)
    assert np.mean(model.predict(X) == labels) > 0.9


def test_small_integers_still_use_the_pre_binned_path():
    rng = np.random.default_rng(2)
    X = rng.integers(0, 50, (500, 2)).astype(float)
    y = X[:, 0] + rng.normal(size=500)
    model = GBMRegressor(n_estimators=20, seed=0).fit(X, y)
    assert not model._uses_continuous_binning


def test_integers_between_the_bin_range_and_u16_use_continuous_binning():
    # 255..=65535 used to stay "pre-binned" at the default 256 bins: 255
    # collided with the missing bin and larger values were never split on.
    rng = np.random.default_rng(3)
    X = rng.integers(0, 10_000, (2000, 2)).astype(float)
    y = X[:, 0] / 1e3 + 0.1 * rng.normal(size=2000)
    model = GBMRegressor(n_estimators=100, seed=0).fit(X[:1500], y[:1500])
    assert model._uses_continuous_binning
    rmse = float(np.sqrt(np.mean((model.predict(X[1500:]) - y[1500:]) ** 2)))
    assert rmse < 0.5


def test_uint8_pixels_keep_255_out_of_the_missing_bin():
    rng = np.random.default_rng(4)
    X = rng.integers(0, 256, (3000, 2)).astype(float)
    X[:300, 0] = 255.0
    y = (X[:, 0] >= 250).astype(float) + 0.05 * rng.normal(size=3000)
    model = GBMRegressor(n_estimators=50, seed=0).fit(X, y)
    assert model._uses_continuous_binning
    top = model.predict(np.array([[255.0, 0.0], [252.0, 0.0]]))
    low = model.predict(np.array([[10.0, 0.0]]))
    assert np.all(top > 0.8) and low[0] < 0.2


def test_pre_binned_limit_follows_max_bins():
    rng = np.random.default_rng(5)
    X = rng.integers(0, 100, (500, 2)).astype(float)
    y = X[:, 0] + rng.normal(size=500)
    small = GBMRegressor(n_estimators=5, continuous_binning_max_bins=64, seed=0)
    assert small.fit(X, y)._uses_continuous_binning
    wide = GBMRegressor(n_estimators=5, continuous_binning_max_bins=512, seed=0)
    assert not wide.fit(X, y)._uses_continuous_binning
