"""Conformance coverage for AlloyGBM's sklearn estimator contracts."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest
import sklearn
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    clone,
    is_classifier,
    is_regressor,
)
from sklearn.exceptions import NotFittedError
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.estimator_checks import estimator_checks_generator
from sklearn.utils.validation import check_is_fitted

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor


def _check_name(check: object) -> str:
    function = getattr(check, "func", check)
    return getattr(function, "__name__", repr(function))


def _run_estimator_checks(estimator: object) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    skips: list[str] = []
    for checked_estimator, check in estimator_checks_generator(estimator, legacy=True):
        try:
            check(checked_estimator)
        except Exception as exc:  # sklearn checks raise heterogeneous assertion types
            result = f"{_check_name(check)}: {type(exc).__name__}: {exc}"
            if type(exc).__name__ == "SkipTest":
                skips.append(result)
            else:
                failures.append(result)
    return failures, skips


class _GroupAwareRanker(GBMRanker):
    """Test-only adapter supplying groups to generic sklearn checks."""

    @staticmethod
    def _default_groups(X: object) -> np.ndarray:
        shape = getattr(X, "shape", None)
        row_count = int(shape[0]) if shape is not None and len(shape) > 0 else len(np.asarray(X))
        return np.arange(row_count, dtype=np.uint32) // 2

    def fit(
        self,
        X: object,
        y: object,
        *,
        group: object | None = None,
        sample_weight: object | None = None,
        eval_set: tuple[object, object] | None = None,
        eval_sample_weight: object | None = None,
        eval_group: object | None = None,
        eval_time_index: object | None = None,
        categorical_feature_values: object | None = None,
        categorical_feature_values_list: object | None = None,
        time_index: object | None = None,
        init_model: GBMRegressor | None = None,
        eval_metric: object | None = None,
        factor_exposures: object | None = None,
    ) -> "_GroupAwareRanker":
        return super().fit(
            X,
            y,
            group=self._default_groups(X) if group is None else group,
            sample_weight=sample_weight,
            eval_set=eval_set,
            eval_sample_weight=eval_sample_weight,
            eval_group=eval_group,
            eval_time_index=eval_time_index,
            categorical_feature_values=categorical_feature_values,
            categorical_feature_values_list=categorical_feature_values_list,
            time_index=time_index,
            init_model=init_model,
            eval_metric=eval_metric,
            factor_exposures=factor_exposures,
        )


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
    # Import alloygbm the same way the parent process does (installed package),
    # not from the source tree — a wheel-built install keeps the compiled
    # `_alloygbm` extension only in the installed package, not in `bindings/python`.
    env = os.environ.copy()

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
        pl_split_candidates=3,
        categorical_feature_indices=[0],
        monotone_constraints={1: -1},
        interaction_constraints=[[0, 1]],
    )

    cloned = clone(model)

    assert cloned is not model
    assert cloned.get_params(deep=False) == model.get_params(deep=False)
    assert cloned.pl_split_candidates == 3


class _ArrayConvertible:
    def __init__(self, values: object) -> None:
        self.values = np.asarray(values)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.values, dtype=dtype)


@pytest.mark.parametrize("sparse_format", ["csr", "csc"])
def test_sparse_input_is_rejected_explicitly(sparse_format: str) -> None:
    sparse = pytest.importorskip("scipy.sparse")
    X, y = _small_regression_data()
    sparse_X = getattr(sparse, f"{sparse_format}_matrix")(X)

    with pytest.raises((TypeError, ValueError), match="[Ss]parse"):
        GBMRegressor(n_estimators=1).fit(sparse_X, y)


def test_numeric_input_rejects_complex_infinite_empty_and_one_dimensional() -> None:
    X, y = _small_regression_data()

    with pytest.raises(ValueError, match="[Cc]omplex"):
        GBMRegressor().fit(X.astype(np.complex64) + 1j, y)
    with pytest.raises(ValueError, match="infinity|infinite|finite"):
        GBMRegressor().fit(np.asarray([[0.0], [np.inf]], dtype=np.float32), [0.0, 1.0])
    with pytest.raises(ValueError, match="0 sample|minimum of 1|0 feature"):
        GBMRegressor().fit(np.empty((0, 2), dtype=np.float32), np.empty(0))
    with pytest.raises(ValueError, match="0 feature|minimum of 1"):
        GBMRegressor().fit(np.empty((2, 0), dtype=np.float32), [0.0, 1.0])
    with pytest.raises(ValueError, match="2D array|dimensional"):
        GBMRegressor().fit(np.asarray([0.0, 1.0]), [0.0, 1.0])


def test_numeric_input_accepts_nan_and_array_convertible_wrappers() -> None:
    X, y = _small_regression_data()
    X[0, 0] = np.nan

    model = GBMRegressor(n_estimators=1).fit(
        _ArrayConvertible(X),
        _ArrayConvertible(y),
    )

    predictions = model.predict(_ArrayConvertible(X))
    assert predictions.shape == (4,)
    assert np.isfinite(predictions).all()


def test_regression_target_shape_and_finiteness_contract() -> None:
    X, y = _small_regression_data()

    with pytest.warns(DataConversionWarning):
        model = GBMRegressor(n_estimators=1).fit(X, y.reshape(-1, 1))
    assert model.predict(X).shape == (4,)

    with pytest.raises(ValueError, match="1d|1-dimensional|shape"):
        GBMRegressor().fit(X, np.column_stack([y, y]))
    with pytest.raises(ValueError, match="infinity|infinite|finite"):
        GBMRegressor().fit(X, np.asarray([0.0, 1.0, np.inf, 2.0]))
    with pytest.raises(ValueError, match="y|target"):
        GBMRegressor().fit(X, None)


def test_prediction_rejects_one_dimensional_and_mismatched_features() -> None:
    X, y = _small_regression_data()
    model = GBMRegressor(n_estimators=1).fit(X, y)

    with pytest.raises(ValueError, match="2D array|dimensional"):
        model.predict(np.asarray([0.0, 1.0], dtype=np.float32))
    with pytest.raises(ValueError, match="features|feature count|n_features"):
        model.predict(np.ones((2, 3), dtype=np.float32))


@pytest.mark.parametrize("estimator", [GBMRegressor(n_estimators=1), GBMClassifier(n_estimators=1)])
def test_sample_weight_accepts_array_wrappers_and_rejects_invalid_shapes(estimator) -> None:
    X, y = _small_regression_data()
    if isinstance(estimator, GBMClassifier):
        y = np.asarray([0, 0, 1, 1])

    fitted = estimator.fit(X, _ArrayConvertible(y), sample_weight=_ArrayConvertible([1, 2, 1, 2]))
    assert fitted.predict(X).shape == (4,)

    with pytest.raises(ValueError, match="sample_weight"):
        estimator.fit(X, y, sample_weight=np.ones((4, 1)))
    with pytest.raises(ValueError, match="weight.*zero|zero.*weight"):
        estimator.fit(X, y, sample_weight=np.zeros(4))


@pytest.mark.parametrize("estimator", [GBMRegressor(n_estimators=1), GBMClassifier(n_estimators=1)])
def test_zero_sample_weight_matches_removing_the_sample(estimator) -> None:
    X, y = _small_regression_data()
    if isinstance(estimator, GBMClassifier):
        y = np.asarray([0, 0, 1, 1])
    weights = np.asarray([1.0, 0.0, 1.0, 1.0])

    weighted = estimator.fit(X, y, sample_weight=weights)
    retained = clone(estimator).fit(X[weights > 0], y[weights > 0])

    np.testing.assert_allclose(weighted.predict(X), retained.predict(X), rtol=0.0, atol=1e-6)


def test_classifier_supports_string_and_noncanonical_numeric_labels(tmp_path: Path) -> None:
    X, _ = _small_regression_data()

    strings = GBMClassifier(n_estimators=1).fit(X, ["low", "low", "high", "high"])
    numeric = GBMClassifier(n_estimators=1).fit(X, [-1, -1, 7, 7])

    np.testing.assert_array_equal(strings.classes_, ["high", "low"])
    np.testing.assert_array_equal(numeric.classes_, [-1, 7])
    assert strings.predict(X).shape == (4,)
    assert numeric.predict(X).shape == (4,)
    assert set(strings.predict(X)) <= {"high", "low"}
    assert set(numeric.predict(X)) <= {-1, 7}

    pickled = pickle.loads(pickle.dumps(strings))
    path = tmp_path / "string-labels.alloygbm"
    strings.save_model(str(path))
    loaded = GBMClassifier.load_model(str(path))
    np.testing.assert_array_equal(pickled.predict(X), strings.predict(X))
    np.testing.assert_array_equal(loaded.predict(X), strings.predict(X))
    np.testing.assert_array_equal(loaded.classes_, strings.classes_)


def test_classifier_target_validation_matches_sklearn_contract() -> None:
    X, _ = _small_regression_data()

    with pytest.warns(DataConversionWarning):
        model = GBMClassifier(n_estimators=1).fit(X, np.asarray([[0], [0], [1], [1]]))
    assert model.predict(X).shape == (4,)

    with pytest.raises(ValueError, match="Unknown label type: continuous"):
        GBMClassifier().fit(X, [0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="infinity|infinite|finite"):
        GBMClassifier().fit(X, [0, 1, np.inf, 1])
    with pytest.raises(ValueError, match="y|array-like"):
        GBMClassifier().fit(X, None)


def test_supported_sklearn_version_window() -> None:
    major, minor = (int(part) for part in sklearn.__version__.split(".")[:2])
    assert (major, minor) in {(1, 8), (1, 9)}


@pytest.mark.parametrize("estimator", [GBMRegressor(), GBMClassifier()])
def test_all_applicable_sklearn_checks_pass(estimator: object) -> None:
    failures, skips = _run_estimator_checks(estimator)

    assert failures == []
    assert all("check_array_api_input" in skip and "SCIPY_ARRAY_API" in skip for skip in skips)


def test_public_ranker_keeps_mandatory_group_contract() -> None:
    X, y = _small_regression_data()

    with pytest.raises(TypeError, match="group"):
        GBMRanker(n_estimators=1).fit(X, y)

    fitted = GBMRanker(n_estimators=1).fit(X, y, group=[0, 0, 1, 1])
    assert fitted.predict(X).shape == (4,)
    sized = GBMRanker(n_estimators=1).fit(X, y, group=[2, 2])
    assert sized.predict(X).shape == (4,)
    assert GBMRanker._normalize_group_input([2, 2], 4) == [0, 0, 1, 1]

    with pytest.raises(ValueError, match="positive integer|summing to the number of rows"):
        GBMRanker(n_estimators=1).fit(X, y, group=[0, 0, 1])
    with pytest.raises(ValueError, match="non-negative"):
        GBMRanker(n_estimators=1).fit(X, y, group=[0, 0, -1, -1])


def test_group_aware_ranker_adapter_is_cloneable_and_deterministic() -> None:
    X, _ = _small_regression_data()
    adapter = _GroupAwareRanker(ranking_objective="queryrmse", n_estimators=1)
    cloned = clone(adapter)

    np.testing.assert_array_equal(adapter._default_groups(X), [0, 0, 1, 1])
    assert cloned.get_params(deep=False) == adapter.get_params(deep=False)


def test_group_aware_ranker_passes_applicable_generic_checks() -> None:
    failures, skips = _run_estimator_checks(
        _GroupAwareRanker(ranking_objective="queryrmse")
    )

    assert failures == []
    assert all("check_array_api_input" in skip and "SCIPY_ARRAY_API" in skip for skip in skips)
