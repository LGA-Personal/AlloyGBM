"""Tests for ``multiclass_sketch`` and ``multiclass_sketch_dim`` (SketchBoost)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from alloygbm import GBMClassifier

SKETCHES = ("top_outputs", "random_sampling", "random_projection")


def make_fixture(rows: int = 400, classes: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(rows, 5)).astype(np.float32)
    X[rng.random(rows) < 0.05, 2] = np.nan
    logits = np.stack(
        [X[:, 0] * (k - classes / 2) + X[:, 1] * (1 if k % 2 else -1) for k in range(classes)],
        axis=1,
    )
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(classes, p=row) for row in p], dtype=np.int32)
    return X, y


def fit(**kwargs: object) -> GBMClassifier:
    X, y = make_fixture()
    params = dict(n_estimators=10, max_depth=3, learning_rate=0.2, seed=7)
    params.update(kwargs)
    return GBMClassifier(**params).fit(X, y)


def test_defaults_are_off() -> None:
    model = GBMClassifier()

    assert model.multiclass_sketch == "none"
    assert model.multiclass_sketch_dim == 5
    params = model.get_params()
    assert params["multiclass_sketch"] == "none"
    assert params["multiclass_sketch_dim"] == 5


def test_params_round_trip_through_get_set_params_and_repr() -> None:
    model = GBMClassifier(multiclass_sketch="random_projection", multiclass_sketch_dim=2)

    assert "multiclass_sketch='random_projection'" in repr(model)
    assert "multiclass_sketch_dim=2" in repr(model)
    clone = GBMClassifier(**model.get_params())
    assert clone.multiclass_sketch == "random_projection"
    assert clone.multiclass_sketch_dim == 2
    model.set_params(multiclass_sketch="top_outputs", multiclass_sketch_dim=3)
    assert model.multiclass_sketch == "top_outputs"
    assert model.multiclass_sketch_dim == 3


@pytest.mark.parametrize(
    "param, value",
    [
        ("multiclass_sketch", "gaussian"),
        ("multiclass_sketch", None),
        ("multiclass_sketch_dim", 0),
        ("multiclass_sketch_dim", 1.5),
        ("multiclass_sketch_dim", True),
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


@pytest.mark.parametrize("sketch", SKETCHES)
@pytest.mark.parametrize("tree_growth", ["level", "leaf"])
def test_sketched_fit_predicts_and_learns(sketch: str, tree_growth: str) -> None:
    X, y = make_fixture()
    max_leaves = 6 if tree_growth == "leaf" else None
    model = fit(
        multiclass_sketch=sketch,
        multiclass_sketch_dim=2,
        tree_growth=tree_growth,
        max_leaves=max_leaves,
    )
    proba = model.predict_proba(X)

    assert model.multiclass_strategy_ == "vector_leaves"
    assert proba.shape == (len(y), 6)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    prior = np.bincount(y, minlength=6) / len(y)
    prior_loss = -np.mean(np.log(prior[y]))
    loss = -np.mean(np.log(proba[np.arange(len(y)), y]))
    assert loss < prior_loss


@pytest.mark.parametrize("sketch", SKETCHES)
def test_sketched_fits_are_deterministic_and_differ_from_unsketched(sketch: str) -> None:
    X, _ = make_fixture()
    a = fit(multiclass_sketch=sketch, multiclass_sketch_dim=1)
    b = fit(multiclass_sketch=sketch, multiclass_sketch_dim=1)
    plain = fit(multiclass_strategy="vector_leaves")

    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))
    assert not np.array_equal(a.predict_proba(X), plain.predict_proba(X))


def test_sketch_as_wide_as_the_classes_matches_unsketched() -> None:
    X, _ = make_fixture()
    wide = fit(multiclass_sketch="random_projection", multiclass_sketch_dim=6)
    plain = fit(multiclass_strategy="vector_leaves")

    np.testing.assert_array_equal(wide.predict_proba(X), plain.predict_proba(X))


def test_sketch_selects_vector_leaves_under_auto() -> None:
    model = fit(multiclass_sketch="top_outputs", multiclass_sketch_dim=2)

    assert model.multiclass_strategy == "auto"
    assert model.multiclass_strategy_ == "vector_leaves"


def test_sketch_with_per_class_trees_raises() -> None:
    with pytest.raises(ValueError, match="multiclass_sketch"):
        fit(multiclass_strategy="per_class_trees", multiclass_sketch="random_projection")


def test_binary_ignores_sketch() -> None:
    X, y = make_fixture(rows=200, classes=2)
    sketched = GBMClassifier(n_estimators=3, multiclass_sketch="random_projection").fit(X, y)
    plain = GBMClassifier(n_estimators=3).fit(X, y)

    assert sketched.multiclass_strategy_ is None
    np.testing.assert_array_equal(sketched.predict_proba(X), plain.predict_proba(X))


def test_sketched_model_survives_pickle_and_save_load(tmp_path) -> None:
    X, _ = make_fixture()
    model = fit(multiclass_sketch="random_sampling", multiclass_sketch_dim=2)
    expected = model.predict_proba(X)

    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict_proba(X), expected)
    assert restored.multiclass_sketch == "random_sampling"
    path = tmp_path / "model.alloy"
    model.save_model(str(path))
    loaded = GBMClassifier.load_model(str(path))
    np.testing.assert_array_equal(loaded.predict_proba(X), expected)
