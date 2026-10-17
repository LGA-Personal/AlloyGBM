"""Tests for ``multiclass_strategy`` and ``multiclass_hessian``."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from alloygbm import GBMClassifier


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def make_fixture(rows: int = 300, classes: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(rows, 5)).astype(np.float32)
    X[rng.random(rows) < 0.05, 2] = np.nan
    logits = np.stack(
        [X[:, 0] * (k - 1.5) + X[:, 1] * (1 if k % 2 else -1) for k in range(classes)],
        axis=1,
    )
    p = softmax(logits)
    y = np.array([rng.choice(classes, p=row) for row in p], dtype=np.int32)
    return X, y


def fit(**kwargs: object) -> GBMClassifier:
    X, y = make_fixture()
    params = dict(n_estimators=8, max_depth=3, learning_rate=0.2, seed=7)
    params.update(kwargs)
    return GBMClassifier(**params).fit(X, y)


def test_defaults_are_auto_strategy_with_diagonal_hessian() -> None:
    model = GBMClassifier()

    assert model.multiclass_strategy == "auto"
    assert model.multiclass_hessian == "diagonal"
    params = model.get_params()
    assert params["multiclass_strategy"] == "auto"
    assert params["multiclass_hessian"] == "diagonal"


def test_params_round_trip_through_get_set_params_and_repr() -> None:
    model = GBMClassifier(multiclass_strategy="vector_leaves", multiclass_hessian="full")

    assert "multiclass_strategy='vector_leaves'" in repr(model)
    assert "multiclass_hessian='full'" in repr(model)
    clone = GBMClassifier(**model.get_params())
    assert clone.multiclass_strategy == "vector_leaves"
    assert clone.multiclass_hessian == "full"
    model.set_params(multiclass_strategy="per_class_trees", multiclass_hessian="abs_residual")
    assert model.multiclass_strategy == "per_class_trees"
    assert model.multiclass_hessian == "abs_residual"


@pytest.mark.parametrize(
    "param, value",
    [
        ("multiclass_strategy", "one_tree"),
        ("multiclass_strategy", None),
        ("multiclass_hessian", "exact"),
        ("multiclass_hessian", 1),
    ],
)
def test_invalid_values_are_rejected_at_fit(param: str, value: object) -> None:
    X, y = make_fixture(rows=60)
    for model in (
        GBMClassifier(n_estimators=2, **{param: value}),
        GBMClassifier(n_estimators=2).set_params(**{param: value}),
    ):
        with pytest.raises(ValueError, match=param):
            model.fit(X, y)


def test_default_strategy_is_unchanged() -> None:
    X, _ = make_fixture()
    implicit = fit()
    explicit = fit(multiclass_strategy="per_class_trees", multiclass_hessian="diagonal")

    np.testing.assert_array_equal(implicit.predict_proba(X), explicit.predict_proba(X))


@pytest.mark.parametrize("hessian", ["diagonal", "full", "abs_residual"])
@pytest.mark.parametrize("tree_growth", ["level", "leaf"])
def test_vector_leaves_fit_predict_and_learn(hessian: str, tree_growth: str) -> None:
    X, y = make_fixture()
    model = fit(
        multiclass_strategy="vector_leaves",
        multiclass_hessian=hessian,
        tree_growth=tree_growth,
        max_leaves=6 if tree_growth == "leaf" else None,
    )
    proba = model.predict_proba(X)

    assert proba.shape == (X.shape[0], 4)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    uniform = -np.log(0.25)
    log_loss = -np.mean(np.log(proba[np.arange(len(y)), y]))
    assert log_loss < uniform
    assert not np.array_equal(proba, fit().predict_proba(X))


def test_vector_leaves_are_deterministic() -> None:
    X, _ = make_fixture()
    first = fit(multiclass_strategy="vector_leaves").predict_proba(X)
    second = fit(multiclass_strategy="vector_leaves").predict_proba(X)

    np.testing.assert_array_equal(first, second)


def test_vector_leaves_survive_pickle_and_save_load(tmp_path) -> None:
    X, _ = make_fixture()
    model = fit(multiclass_strategy="vector_leaves", multiclass_hessian="full")
    expected = model.predict_proba(X)

    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict_proba(X), expected, rtol=1e-6)
    assert restored.multiclass_strategy == "vector_leaves"
    assert restored.multiclass_hessian == "full"

    path = tmp_path / "vector.alloy"
    model.save_model(str(path))
    loaded = GBMClassifier.load_model(str(path))
    np.testing.assert_allclose(loaded.predict_proba(X), expected, rtol=1e-6)


def test_vector_leaves_shap_is_additive() -> None:
    X, _ = make_fixture(rows=120)
    X = X[~np.isnan(X).any(axis=1)]
    model = fit(multiclass_strategy="vector_leaves")

    expected, shap_vals = model.shap_values(X, include_expected_value=True)
    logits = np.stack(
        [np.asarray(shap_vals[c]).sum(axis=1) + expected[c] for c in range(4)], axis=1
    )
    np.testing.assert_allclose(
        softmax(logits), model.predict_proba(X), atol=1e-5, rtol=1e-4
    )


def test_vector_leaves_support_early_stopping() -> None:
    X, y = make_fixture(rows=400)
    model = GBMClassifier(
        n_estimators=500,
        learning_rate=0.3,
        multiclass_strategy="vector_leaves",
        early_stopping_rounds=5,
        seed=7,
    ).fit(X[:300], y[:300], eval_set=(X[300:], y[300:]))

    assert model.predict_proba(X).shape == (400, 4)


@pytest.mark.parametrize("hessian", ["diagonal", "full"])
def test_vector_leaves_auto_n_estimators(hessian: str) -> None:
    X, y = make_fixture(rows=400)
    model = GBMClassifier(
        multiclass_strategy="vector_leaves", multiclass_hessian=hessian, seed=7
    ).fit(X, y)

    assert model.n_estimators == "auto"
    assert model.auto_rounds_["mode"] == "holdout"
    assert model.n_estimators_ == model.auto_rounds_["refit_rounds"] > 0
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    assert -np.mean(np.log(proba[np.arange(len(y)), y])) < -np.log(0.25)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {"multiclass_strategy": "per_class_trees", "multiclass_hessian": "full"},
            "requires multiclass_strategy='vector_leaves'",
        ),
        (
            {"multiclass_strategy": "vector_leaves", "training_mode": "morph"},
            "does not support",
        ),
    ],
)
def test_unsupported_combinations_raise(kwargs: dict, message: str) -> None:
    with pytest.raises(Exception, match=message):
        fit(**kwargs)
