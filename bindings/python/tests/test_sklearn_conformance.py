"""Conformance coverage for AlloyGBM's sklearn estimator contracts."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    clone,
    is_classifier,
    is_regressor,
)
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor


def _small_regression_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    y = np.asarray([0.0, 1.0, 1.0, 2.0], dtype=np.float32)
    return X, y


def test_regressor_mixin_precedes_base_estimator() -> None:
    mro = GBMRegressor.__mro__

    assert RegressorMixin in mro
    assert mro.index(RegressorMixin) < mro.index(BaseEstimator)
    assert is_regressor(GBMRegressor())
    assert not is_classifier(GBMRegressor())


def test_classifier_has_only_classifier_semantics() -> None:
    mro = GBMClassifier.__mro__

    assert ClassifierMixin in mro
    assert RegressorMixin not in mro
    assert mro.index(ClassifierMixin) < mro.index(BaseEstimator)
    assert is_classifier(GBMClassifier())
    assert not is_regressor(GBMClassifier())


def test_ranker_has_no_regression_or_classification_mixin() -> None:
    mro = GBMRanker.__mro__

    assert RegressorMixin not in mro
    assert ClassifierMixin not in mro
    assert BaseEstimator in mro
    assert not is_regressor(GBMRanker())
    assert not is_classifier(GBMRanker())


def test_public_estimator_paths_and_parameter_signatures_are_stable() -> None:
    assert GBMRegressor.__module__ == "alloygbm.regressor"
    assert GBMClassifier.__module__ == "alloygbm.classifier"
    assert GBMRanker.__module__ == "alloygbm.ranker"

    regressor_parameters = inspect.signature(GBMRegressor.__init__).parameters
    classifier_parameters = inspect.signature(GBMClassifier.__init__).parameters
    ranker_parameters = inspect.signature(GBMRanker.__init__).parameters

    assert classifier_parameters == regressor_parameters
    assert set(regressor_parameters) <= set(ranker_parameters)
    assert {
        "ranking_objective",
        "ranking_sigma",
        "lambdarank_truncation_level",
        "lambdarank_normalize",
    } <= set(ranker_parameters)


def test_estimators_remain_importable_without_sklearn() -> None:
    package_root = Path(__file__).resolve().parents[1]
    script = """
import importlib.abc
import sys

class BlockSklearn(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sklearn" or fullname.startswith("sklearn."):
            raise ImportError("sklearn blocked for optional-dependency test")
        return None

sys.meta_path.insert(0, BlockSklearn())

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor

assert GBMRegressor().get_params()["n_estimators"] == 6
assert GBMClassifier().get_params()["n_estimators"] == 6
assert GBMRanker().get_params()["ranking_objective"] == "rank:ndcg"
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package_root)

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("estimator", [GBMRegressor(), GBMClassifier(), GBMRanker()])
def test_constructor_does_not_publish_learned_attributes(estimator: object) -> None:
    learned = sorted(
        name
        for name in vars(estimator)
        if name.endswith("_") and not name.startswith("_")
    )

    assert learned == []


@pytest.mark.parametrize(
    ("estimator", "operation"),
    [
        (GBMRegressor(), lambda model: model.predict([[0.0, 1.0]])),
        (GBMRegressor(), lambda model: model.shap_values([[0.0, 1.0]])),
        (GBMRegressor(), lambda model: model.feature_importances([[0.0, 1.0]])),
        (GBMRegressor(), lambda model: model.artifact_bytes),
        (GBMClassifier(), lambda model: model.predict([[0.0, 1.0]])),
        (GBMClassifier(), lambda model: model.predict_proba([[0.0, 1.0]])),
        (GBMRanker(), lambda model: model.predict([[0.0, 1.0]])),
    ],
)
def test_unfitted_operations_raise_not_fitted_error(estimator, operation) -> None:
    with pytest.raises(NotFittedError):
        operation(estimator)

    with pytest.raises(NotFittedError):
        check_is_fitted(estimator)


def test_successful_fit_and_load_publish_feature_schema(tmp_path: Path) -> None:
    X, y = _small_regression_data()
    model = GBMRegressor(n_estimators=1, max_depth=1).fit(X, y)

    check_is_fitted(model)
    assert model.n_features_in_ == 2
    assert not hasattr(model, "feature_names_in_")

    path = tmp_path / "schema.alloygbm"
    model.save_model(str(path))
    loaded = GBMRegressor.load_model(str(path))

    check_is_fitted(loaded)
    assert loaded.n_features_in_ == 2
    assert not hasattr(loaded, "feature_names_in_")


def test_feature_names_are_published_only_for_all_string_columns() -> None:
    pd = pytest.importorskip("pandas")
    _, y = _small_regression_data()

    named = GBMRegressor(n_estimators=1, max_depth=1).fit(
        pd.DataFrame([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], columns=["a", "b"]),
        y,
    )
    mixed = GBMRegressor(n_estimators=1, max_depth=1).fit(
        pd.DataFrame([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], columns=["a", 1]),
        y,
    )

    np.testing.assert_array_equal(named.feature_names_in_, np.asarray(["a", "b"], dtype=object))
    assert not hasattr(mixed, "feature_names_in_")


def test_failed_fit_does_not_publish_partial_feature_schema() -> None:
    pd = pytest.importorskip("pandas")
    _, y = _small_regression_data()
    X = pd.DataFrame(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        columns=["a", "b"],
    )
    model = GBMRegressor(
        categorical_feature_indices=[5],
        n_estimators=1,
        max_depth=1,
    )

    with pytest.raises(ValueError, match="feature bounds"):
        model.fit(X, y, categorical_feature_values_list=[["x", "y", "x", "y"]])

    assert not hasattr(model, "n_features_in_")
    assert not hasattr(model, "feature_names_in_")
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


def test_constructor_preserves_parameter_identity_without_validation() -> None:
    categorical_indices = [0]
    monotone_constraints = {0: 1}
    feature_weights = {0: 2.0}
    interaction_constraints = [[0, 1]]

    model = GBMRegressor(
        learning_rate=0.0,
        categorical_feature_indices=categorical_indices,
        monotone_constraints=monotone_constraints,
        feature_weights=feature_weights,
        interaction_constraints=interaction_constraints,
    )

    assert model.learning_rate == 0.0
    assert model.categorical_feature_indices is categorical_indices
    assert model.monotone_constraints is monotone_constraints
    assert model.feature_weights is feature_weights
    assert model.interaction_constraints is interaction_constraints


def test_set_params_assigns_known_values_without_validation_or_coercion() -> None:
    model = GBMRegressor()
    value = [0]

    returned = model.set_params(
        learning_rate=0.0,
        categorical_feature_indices=value,
    )

    assert returned is model
    assert model.learning_rate == 0.0
    assert model.categorical_feature_indices is value
    with pytest.raises(ValueError, match="Invalid parameter"):
        model.set_params(does_not_exist=1)


@pytest.mark.parametrize(
    ("estimator_factory", "fit_kwargs", "message"),
    [
        (lambda: GBMRegressor(learning_rate=0.0), {}, "learning_rate"),
        (lambda: GBMClassifier(objective="squared_error"), {}, "auto-detected"),
        (
            lambda: GBMRanker(ranking_objective="not-a-ranking-objective"),
            {"group": [0, 0, 1, 1]},
            "ranking_objective",
        ),
    ],
)
def test_invalid_known_parameters_are_rejected_at_fit(
    estimator_factory, fit_kwargs: dict[str, object], message: str
) -> None:
    X, y = _small_regression_data()
    estimator = estimator_factory()

    with pytest.raises((TypeError, ValueError), match=message):
        estimator.fit(X, y, **fit_kwargs)

    assert not hasattr(estimator, "n_features_in_")


def test_sklearn_clone_preserves_the_complete_parameter_surface() -> None:
    model = GBMRegressor(
        n_estimators=2,
        categorical_feature_indices=[0],
        monotone_constraints={1: -1},
        interaction_constraints=[[0, 1]],
    )

    cloned = clone(model)

    assert cloned is not model
    assert cloned.get_params(deep=False) == model.get_params(deep=False)
