"""Target-encoded categoricals (``max_cat_threshold=0``, or above the
threshold) must predict on the bins they were trained on."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from alloygbm import GBMClassifier, GBMRegressor  # noqa: E402

EFFECTS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, -2.0, 1.0, 0.5])


def _data(n=2000, levels=8, seed=0):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, levels, n)
    x = rng.normal(size=n)
    effects = EFFECTS if levels == len(EFFECTS) else rng.normal(size=levels) * 2.0
    y = effects[codes] + 0.3 * x + 0.05 * rng.normal(size=n)
    X = pd.DataFrame({"x": x, "c": pd.Categorical(codes, categories=np.arange(levels))})
    return X, y, codes


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def test_regressor_target_encoding_matches_native_split_quality():
    X, y, _ = _data()
    te = GBMRegressor(n_estimators=100, seed=0).fit(X, y)
    native = GBMRegressor(n_estimators=100, seed=0, max_cat_threshold=255).fit(X, y)
    assert te._target_encoding_bins_ and not native._target_encoding_bins_
    assert _rmse(te.predict(X), y) < 0.15
    assert _rmse(te.predict(X), y) < 1.5 * _rmse(native.predict(X), y)


def test_classifier_target_encoding_predicts_on_training_bins():
    X, y, _ = _data()
    labels = (y > 0).astype(int)
    model = GBMClassifier(n_estimators=100, seed=0).fit(X, labels)
    assert np.mean(model.predict(X) == labels) > 0.97


def test_multiclass_target_encoding():
    X, y, _ = _data()
    labels = np.digitize(y, [-1.0, 1.0])
    model = GBMClassifier(n_estimators=100, seed=0).fit(X, labels)
    assert np.mean(model.predict(X) == labels) > 0.9


def test_held_out_rows_and_unseen_category():
    X, y, codes = _data(n=4000)
    train, test = slice(0, 3000), slice(3000, None)
    model = GBMRegressor(n_estimators=100, seed=0).fit(X.iloc[train], y[train])
    assert _rmse(model.predict(X.iloc[test]), y[test]) < 0.2
    unseen = X.iloc[test].copy()
    unseen["c"] = pd.Categorical(np.full(len(unseen), 99), categories=[99])
    assert np.all(np.isfinite(model.predict(unseen)))


def test_missing_category_values():
    X, y, _ = _data()
    X.loc[::7, "c"] = np.nan
    model = GBMRegressor(n_estimators=100, seed=0).fit(X, y)
    known = X["c"].notna().to_numpy()
    assert _rmse(model.predict(X)[known], y[known]) < 0.15


def test_early_stopping_validation_uses_training_bins():
    X, y, _ = _data(n=4000)
    model = GBMRegressor(n_estimators=300, seed=0, early_stopping_rounds=20)
    model.fit(X.iloc[:3000], y[:3000], eval_set=(X.iloc[3000:], y[3000:]))
    assert model.best_score_ is not None and model.best_score_ < 0.2
    assert _rmse(model.predict(X.iloc[3000:]), y[3000:]) < 0.2


def test_more_levels_than_bins():
    X, y, _ = _data(n=6000, levels=600)
    model = GBMRegressor(n_estimators=150, seed=0).fit(X, y)
    bins = model._target_encoding_bins_[1]
    assert len(bins["bins"]) == 600
    assert max(bins["bins"].values()) <= 254
    assert _rmse(model.predict(X), y) < 0.6 * float(np.std(y))


def test_numpy_codes_follow_sorted_category_order():
    X, y, codes = _data()
    names = [f"level_{c}" for c in codes]  # sorts the same as the codes
    matrix = np.column_stack([X["x"].to_numpy(), codes.astype(float)])
    model = GBMRegressor(n_estimators=100, seed=0, categorical_feature_indices=[1])
    model.fit(matrix, y, categorical_feature_values_list=[names])
    assert _rmse(model.predict(matrix), y) < 0.15


@pytest.mark.parametrize("labels", [np.arange(12), np.arange(12) * 10 + 10])
def test_numpy_integer_categories_match_by_value(labels):
    # 12 levels: "10" sorts before "2" as a string, and labels 10..120 are
    # not positions, so numeric input must match categories by value.
    rng = np.random.default_rng(6)
    codes = rng.integers(0, 12, 3000)
    effects = rng.normal(size=12) * 2.0
    y = effects[codes] + 0.05 * rng.normal(size=3000)
    matrix = np.column_stack([rng.normal(size=3000), labels[codes].astype(float)])
    names = [str(v) for v in labels[codes]]
    model = GBMRegressor(n_estimators=100, seed=0, categorical_feature_indices=[1])
    model.fit(matrix, y, categorical_feature_values_list=[names])
    assert _rmse(model.predict(matrix), y) < 0.15


def test_numpy_nan_uses_the_missing_category_bin():
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 4, 3000).astype(float)
    codes[::5] = np.nan
    y = np.where(np.isnan(codes), 5.0, np.nan_to_num(codes)) + 0.05 * rng.normal(size=3000)
    matrix = np.column_stack([rng.normal(size=3000), codes])
    names = [str(c) for c in codes]  # NaN becomes the category "nan"
    model = GBMRegressor(n_estimators=100, seed=0, categorical_feature_indices=[1])
    model.fit(matrix, y, categorical_feature_values_list=[names])
    assert np.all(np.abs(model.predict(matrix[np.isnan(codes)]) - 5.0) < 0.3)


def test_non_finite_category_values_are_placeholders():
    X, y, _ = _data()
    names = ["inf" if i % 9 == 0 else str(c) for i, c in enumerate(X["c"])]
    rows = [[float(x), name] for x, name in zip(X["x"], names)]
    model = GBMRegressor(n_estimators=20, seed=0, categorical_feature_indices=[1])
    model.fit(rows, y, categorical_feature_values_list=[names])
    assert np.all(np.isfinite(model.predict(rows)))


def test_save_load_and_pickle_keep_target_encoding(tmp_path):
    X, y, _ = _data()
    model = GBMRegressor(n_estimators=50, seed=0).fit(X, y)
    expected = model.predict(X)
    path = tmp_path / "model.alloy"
    model.save_model(str(path))
    loaded = GBMRegressor.load_model(str(path))
    np.testing.assert_allclose(loaded.predict(X), expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(pickle.loads(pickle.dumps(model)).predict(X), expected,
                               rtol=0, atol=1e-6)


def test_shap_values_add_up_to_predictions():
    X, y, _ = _data(n=500)
    model = GBMRegressor(n_estimators=30, seed=0).fit(X, y)
    expected, values = model.shap_values(X.iloc[:20], include_expected_value=True)
    reconstructed = expected + np.asarray(values).sum(axis=1)
    np.testing.assert_allclose(reconstructed, model.predict(X.iloc[:20]), atol=1e-3)
