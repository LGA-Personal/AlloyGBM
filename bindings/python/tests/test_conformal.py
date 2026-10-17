"""Tests for the leaf-index API (``apply``) and conformal prediction intervals."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor
from alloygbm.conformal import LoBoost, SplitConformal, _conformal_quantile


def _heteroscedastic(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    noise = 0.1 + 2.0 * (X[:, 0] > 0.5)
    y = 3.0 * X[:, 0] + X[:, 1] + rng.normal(size=n) * noise
    return X, y


def _model(**params) -> GBMRegressor:
    base = dict(n_estimators=40, max_depth=4, learning_rate=0.1, seed=0)
    base.update(params)
    return GBMRegressor(**base)


# -- apply ---------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["quantile", "linear", "rank"])
def test_apply_leaf_outputs_reconstruct_predictions(strategy: str) -> None:
    X, y = _heteroscedastic(600)
    X[::17, 2] = np.nan
    model = _model(continuous_binning_strategy=strategy).fit(X, y)

    leaves = model.apply(X)
    assert leaves.shape == (600, model._leaf_assignments(X)[0].shape[1])
    assert leaves.dtype == np.int32
    assert leaves.min() >= 0
    # Each tree's leaves are numbered densely from zero.
    assert all(len(np.unique(col)) <= col.max() + 1 for col in leaves.T)

    _, outputs = model._leaf_assignments(X)
    offset = model.predict(X) - outputs.sum(axis=1)
    np.testing.assert_allclose(offset, offset[0], atol=1e-4)


def test_apply_matches_after_pickle_roundtrip() -> None:
    X, y = _heteroscedastic(300)
    model = _model().fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.apply(X), model.apply(X))


def test_apply_with_linear_leaves() -> None:
    X, y = _heteroscedastic(400)
    model = _model(leaf_model="linear", n_estimators=10).fit(X, y)
    _, outputs = model._leaf_assignments(X)
    offset = model.predict(X) - outputs.sum(axis=1)
    np.testing.assert_allclose(offset, offset[0], atol=1e-4)


def test_apply_rows_with_identical_leaves_share_outputs() -> None:
    X, y = _heteroscedastic(400)
    model = _model().fit(X, y)
    leaves, outputs = model._leaf_assignments(X)
    tree = 0
    for leaf in np.unique(leaves[:, tree]):
        np.testing.assert_allclose(outputs[leaves[:, tree] == leaf, tree], outputs[leaves[:, tree] == leaf, tree][0])


def test_apply_binary_classifier_and_ranker() -> None:
    X, y = _heteroscedastic(200)
    clf = GBMClassifier(n_estimators=5).fit(X, (y > 0).astype(int))
    assert clf.apply(X).shape == (200, 5)
    ranker = GBMRanker(n_estimators=5).fit(X, (y > 0).astype(int), group=np.repeat(np.arange(20), 10))
    assert ranker.apply(X).shape[0] == 200


def test_apply_rejects_multiclass_and_bad_width() -> None:
    X, y = _heteroscedastic(150)
    clf = GBMClassifier(n_estimators=3).fit(X, np.digitize(y, [-1.0, 1.0]))
    with pytest.raises(NotImplementedError):
        clf.apply(X)
    model = _model(n_estimators=3).fit(X, y)
    with pytest.raises(ValueError):
        model.apply(X[:, :3])


# -- conformal -----------------------------------------------------------------


def test_conformal_quantile_finite_sample_rank() -> None:
    scores = np.arange(1.0, 11.0)
    # ceil(11 * 0.9) = 10 -> the largest of ten scores.
    assert _conformal_quantile(scores, 0.1) == 10.0
    # ceil(11 * 0.95) = 11 > 10 -> unbounded interval.
    assert _conformal_quantile(scores, 0.05) == np.inf


def test_split_conformal_covers_at_target_level() -> None:
    X, y = _heteroscedastic(6000, seed=1)
    model = _model().fit(X[:2000], y[:2000])
    lo, hi = SplitConformal(model).calibrate(X[2000:4000], y[2000:4000], alpha=0.1).predict_interval(X[4000:])
    coverage = np.mean((y[4000:] >= lo) & (y[4000:] <= hi))
    assert 0.87 <= coverage <= 0.93
    np.testing.assert_allclose(hi - lo, (hi - lo)[0])


def test_loboost_covers_and_adapts_width() -> None:
    X, y = _heteroscedastic(8000, seed=2)
    model = _model().fit(X[:3000], y[:3000])
    calibrator = LoBoost(model).calibrate(X[3000:5500], y[3000:5500], alpha=0.1)
    lo, hi = calibrator.predict_interval(X[5500:])
    y_test = y[5500:]
    covered = (y_test >= lo) & (y_test <= hi)
    assert 0.87 <= covered.mean() <= 0.93
    assert calibrator.n_regions_ > 1
    assert all(len(s) >= calibrator.min_group_size_ for s in calibrator._region_scores)
    assert np.isfinite(hi - lo).all()
    # Noise is 20x larger where x0 > 0.5, and the model splits on x0, so the
    # local intervals there must be wider than elsewhere.
    noisy = X[5500:, 0] > 0.5
    assert (hi - lo)[noisy].mean() > 2.0 * (hi - lo)[~noisy].mean()


def test_loboost_beats_split_conformal_interval_score_on_heteroscedastic_data() -> None:
    X, y = _heteroscedastic(8000, seed=3)
    model = _model().fit(X[:3000], y[:3000])
    y_test = y[5500:]

    def interval_score(lo: np.ndarray, hi: np.ndarray) -> float:
        return float(np.mean((hi - lo) + 20.0 * (np.maximum(lo - y_test, 0) + np.maximum(y_test - hi, 0))))

    split = interval_score(*SplitConformal(model).calibrate(X[3000:5500], y[3000:5500]).predict_interval(X[5500:]))
    local = interval_score(*LoBoost(model).calibrate(X[3000:5500], y[3000:5500]).predict_interval(X[5500:]))
    assert local < 0.9 * split


def test_loboost_small_calibration_set_reduces_to_split_conformal() -> None:
    X, y = _heteroscedastic(400, seed=4)
    model = _model().fit(X[:300], y[:300])
    lob = LoBoost(model).calibrate(X[300:360], y[300:360])
    assert lob.n_regions_ == 1
    split = SplitConformal(model).calibrate(X[300:360], y[300:360])
    np.testing.assert_allclose(lob.predict_interval(X[360:])[0], split.predict_interval(X[360:])[0])


def test_loboost_alpha_override_and_pickle() -> None:
    X, y = _heteroscedastic(3000, seed=5)
    model = _model().fit(X[:1000], y[:1000])
    calibrator = LoBoost(model, n_part=40).calibrate(X[1000:2000], y[1000:2000], alpha=0.1)
    lo90, hi90 = calibrator.predict_interval(X[2000:])
    lo80, hi80 = calibrator.predict_interval(X[2000:], alpha=0.2)
    assert np.all(hi80 - lo80 <= hi90 - lo90 + 1e-9)
    restored = pickle.loads(pickle.dumps(calibrator))
    np.testing.assert_allclose(restored.predict_interval(X[2000:])[1], hi90)


def test_loboost_unseen_leaf_falls_back_to_matched_ancestor() -> None:
    X, y = _heteroscedastic(3000, seed=6)
    model = _model().fit(X[:1000], y[:1000])
    # Calibrate only on rows with x0 < 0 so rows with large x0 reach leaves
    # no calibration row reached.
    cal = slice(1000, 2500)
    keep = X[cal, 0] < 0.0
    calibrator = LoBoost(model).calibrate(X[cal][keep], y[cal][keep])
    lo, hi = calibrator.predict_interval(np.abs(X[2500:]) + 1.0)
    assert np.isfinite(hi - lo).all()


def test_loboost_merge_prefers_closest_weighted_path() -> None:
    # Three regions over two trees; tree 0 has a much larger output range.
    leaves = np.array([[0, 0], [0, 1], [1, 0], [0, 0], [1, 1]])
    outputs = np.array([[0.0, 0.0], [0.0, 0.1], [5.0, 0.0], [0.0, 0.0], [5.0, 0.1]])
    region_rows = [np.array([0, 3]), np.array([1]), np.array([2, 4])]
    merged = LoBoost._merge_undersized(
        leaves, outputs, region_rows, np.array([2, 2, 1]), np.array([2, 1, 2]), min_size=2
    )
    # Region 1 (path 0,1) differs from region 0 only on the low-weight tree.
    assert merged == [0, 0, 2]


def test_conformal_rejects_classifiers_and_bad_alpha() -> None:
    X, y = _heteroscedastic(200)
    clf = GBMClassifier(n_estimators=3).fit(X, (y > 0).astype(int))
    with pytest.raises(TypeError):
        LoBoost(clf).calibrate(X, y)
    model = _model(n_estimators=3).fit(X, y)
    with pytest.raises(ValueError):
        SplitConformal(model).calibrate(X, y, alpha=1.5)
    with pytest.raises(RuntimeError):
        LoBoost(model).predict_interval(X)
