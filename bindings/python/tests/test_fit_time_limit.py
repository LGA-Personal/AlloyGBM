"""Fit deadlines are resource controls, never model-selection parameters."""

import numpy as np
import pytest

from alloygbm import GBMClassifier, GBMRegressor


def data(classes=0):
    rng = np.random.default_rng(91)
    x = rng.normal(size=(240, 6)).astype(np.float32)
    y = x[:, 0] + 0.2 * x[:, 1] if not classes else np.arange(len(x)) % classes
    return x, y


@pytest.mark.parametrize("cls", [GBMRegressor, GBMClassifier])
@pytest.mark.parametrize("limit", [0, -1, float("nan"), float("inf"), True])
def test_invalid_fit_time_limit(cls, limit):
    x, y = data(2 if cls is GBMClassifier else 0)
    with pytest.raises((ValueError, TypeError), match="time_limit"):
        cls(n_estimators=3).fit(x, y, time_limit=limit)


@pytest.mark.parametrize(
    "classes,strategy",
    [(0, "auto"), (2, "auto"), (3, "per_class_trees"), (3, "vector_leaves")],
)
def test_unrestricted_and_generous_deadline_are_identical(classes, strategy):
    x, y = data(classes)
    cls = GBMClassifier if classes else GBMRegressor
    params = dict(n_estimators=4, max_depth=3, n_jobs=1, multiclass_strategy=strategy)
    ordinary = cls(**params).fit(x, y)
    for limit in (None, 60):
        bounded = cls(**params).fit(x, y, time_limit=limit)
        assert ordinary.artifact_bytes == bounded.artifact_bytes
        np.testing.assert_array_equal(ordinary.predict(x), bounded.predict(x))


@pytest.mark.parametrize("classes", [0, 2, 3])
def test_expired_fit_budget_raises_without_publishing_model(classes):
    x, y = data(classes)
    model = (GBMClassifier if classes else GBMRegressor)(n_estimators=10000, n_jobs=1)
    with pytest.raises(TimeoutError, match="time"):
        model.fit(x, y, time_limit=1e-12)
    assert not getattr(model, "_is_fitted", False)


def test_automatic_rounds_uses_provided_validation_with_budget():
    x, y = data()
    model = GBMRegressor(max_depth=2, n_jobs=1).fit(
        x[:180], y[:180], eval_set=(x[180:], y[180:]), time_limit=30
    )
    assert model.auto_rounds_["mode"] == "eval_set"
    assert "time_limit" not in model.get_params()


@pytest.mark.parametrize(
    "classes,strategy",
    [(0, "auto"), (2, "auto"), (3, "per_class_trees"), (3, "vector_leaves")],
)
def test_budget_stop_keeps_a_complete_serializable_model(classes, strategy, tmp_path):
    x, y = data(classes)
    cls = GBMClassifier if classes else GBMRegressor
    model = cls(
        n_estimators=100000,
        n_jobs=1,
        training_policy="manual",
        max_depth=3,
        learning_rate=0.001,
        multiclass_strategy=strategy,
    )
    model.fit(x, y, time_limit=0.1)
    assert model.stop_reason_ == "TimeBudgetReached"
    assert 0 < model.n_estimators_ < 100000
    path = tmp_path / "bounded.alloy"
    model.save_model(str(path))
    restored = cls.load_model(str(path))
    np.testing.assert_array_equal(model.predict(x), restored.predict(x))
