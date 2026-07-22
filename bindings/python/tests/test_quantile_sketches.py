from __future__ import annotations

import inspect

import numpy as np
import pytest

from alloygbm import GBMClassifier, GBMRegressor, MultiLabelGBMRanker


def _continuous_fixture() -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(
        [[row + 0.25, (row % 5) + 0.125] for row in range(24)],
        dtype=np.float32,
    )
    y = np.asarray([0.2 * row + (row % 3) for row in range(24)], dtype=np.float32)
    return x, y


def test_quantile_sketch_parameter_round_trips_and_validates() -> None:
    model = GBMRegressor(quantile_sketch_max_rows=17)

    assert model.get_params()["quantile_sketch_max_rows"] == 17
    assert "quantile_sketch_max_rows=17" in repr(model)
    assert "quantile_sketch_max_rows" in inspect.signature(
        GBMClassifier.__init__
    ).parameters
    assert model.set_params(quantile_sketch_max_rows=None) is model
    assert model.quantile_sketch_max_rows is None

    with pytest.raises(ValueError, match="quantile_sketch_max_rows"):
        GBMRegressor(quantile_sketch_max_rows=0)
    with pytest.raises(ValueError, match="quantile_sketch_max_rows"):
        model.set_params(quantile_sketch_max_rows=-1)
    with pytest.raises(ValueError, match="quantile_sketch_max_rows"):
        MultiLabelGBMRanker(quantile_sketch_max_rows=0)


def test_exact_quantile_fit_reports_methods_and_persists(tmp_path) -> None:
    x, y = _continuous_fixture()
    model = GBMRegressor(n_estimators=2, max_depth=2).fit(x, y)

    assert model.feature_quantile_cut_methods_ == ["exact", "exact"]

    path = tmp_path / "exact-quantile.agbm"
    model.save_model(path)
    restored = GBMRegressor.load_model(path)
    assert restored.quantile_sketch_max_rows is None
    assert restored.feature_quantile_cut_methods_ == ["exact", "exact"]
    np.testing.assert_array_equal(restored.predict(x), model.predict(x))


def test_classifier_and_independent_multilabel_expose_exact_methods() -> None:
    x, y = _continuous_fixture()
    classifier = GBMClassifier(n_estimators=2, max_depth=2).fit(
        x, (y > np.median(y)).astype(np.int32)
    )
    assert classifier.feature_quantile_cut_methods_ == ["exact", "exact"]

    multilabel = MultiLabelGBMRanker(
        ranking_objective="queryrmse",
        n_estimators=2,
        max_depth=2,
    ).fit(
        x,
        np.column_stack((y, y[::-1])),
        group=np.repeat(np.arange(6), 4),
    )
    assert multilabel.feature_quantile_cut_methods_ == ["exact", "exact"]


def test_sketch_activation_is_deterministic_and_persists(tmp_path) -> None:
    x, y = _continuous_fixture()
    params = {
        "n_estimators": 2,
        "max_depth": 2,
        "quantile_sketch_max_rows": 8,
    }
    first = GBMRegressor(**params).fit(x, y)
    second = GBMRegressor(**params).fit(x, y)

    assert first.feature_quantile_cut_methods_ == ["sketch", "sketch"]
    assert first._continuous_feature_quantile_cuts == second._continuous_feature_quantile_cuts
    assert all(len(cuts) <= 7 for cuts in first._continuous_feature_quantile_cuts)

    path = tmp_path / "sketched-quantile.agbm"
    first.save_model(path)
    restored = GBMRegressor.load_model(path)
    assert restored.quantile_sketch_max_rows == 8
    assert restored.feature_quantile_cut_methods_ == ["sketch", "sketch"]
    assert (
        restored._continuous_feature_quantile_cuts
        == first._continuous_feature_quantile_cuts
    )
    np.testing.assert_array_equal(restored.predict(x), first.predict(x))


def test_sketch_limit_at_row_count_keeps_exact_cuts() -> None:
    x, y = _continuous_fixture()
    exact = GBMRegressor(n_estimators=2).fit(x, y)
    bounded = GBMRegressor(
        n_estimators=2, quantile_sketch_max_rows=len(x)
    ).fit(x, y)

    assert bounded.feature_quantile_cut_methods_ == ["exact", "exact"]
    assert bounded._continuous_feature_quantile_cuts == exact._continuous_feature_quantile_cuts


def test_changing_sketch_limit_after_fit_requires_refit() -> None:
    x, y = _continuous_fixture()
    model = GBMRegressor(n_estimators=2).fit(x, y)

    model.set_params(quantile_sketch_max_rows=8)

    with pytest.raises(RuntimeError, match="must be fit"):
        model.predict(x)
