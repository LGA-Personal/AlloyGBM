"""Focused contract tests for the AlloyGBM Python regressor baseline."""

from __future__ import annotations

import importlib.util
import os
from types import SimpleNamespace
import unittest
from array import array
from pathlib import Path
from unittest.mock import patch


def load_regressor_module():
    import alloygbm._regressor._core as module
    return module


regressor_module = load_regressor_module()
GBMRegressor = regressor_module.GBMRegressor


from contextlib import contextmanager


@contextmanager
def _force_legacy_train_path():
    """Block the _with_summary training paths so tests that mock
    ``_load_native_train_regression_artifact`` hit the legacy bridge."""
    orig_ws = regressor_module._load_native_train_regression_artifact_with_summary
    orig_dws = regressor_module._load_native_train_regression_artifact_dense_with_summary
    orig_ds = getattr(regressor_module, "_load_native_train_regression_artifact_dense", None)

    def _raise():
        raise RuntimeError("blocked by test helper")

    regressor_module._load_native_train_regression_artifact_with_summary = _raise
    regressor_module._load_native_train_regression_artifact_dense_with_summary = _raise
    if orig_ds is not None:
        regressor_module._load_native_train_regression_artifact_dense = _raise
    try:
        yield
    finally:
        regressor_module._load_native_train_regression_artifact_with_summary = orig_ws
        regressor_module._load_native_train_regression_artifact_dense_with_summary = orig_dws
        if orig_ds is not None:
            regressor_module._load_native_train_regression_artifact_dense = orig_ds


class _FakeNumpyLike:
    def __init__(self, values: object) -> None:
        self._values = values

    def tolist(self) -> object:
        return self._values


class _FakePandasLikeFrame:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def to_numpy(self) -> _FakeNumpyLike:
        return _FakeNumpyLike(self._rows)


class _FakePolarsLikeFrame:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def to_numpy(self) -> list[list[float]]:
        return self._rows


class _FakePolarsLikeSeries:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def to_list(self) -> list[float]:
        return self._values


class _NonConvertible:
    pass


class _FakeCategoricalSeries:
    def __init__(self, values: list[object]) -> None:
        self._values = values

    def to_list(self) -> list[object]:
        return self._values


class _FakeCategoricalFrame:
    def __init__(
        self,
        rows: list[list[object]],
        columns: list[str],
        dtypes: list[object],
    ) -> None:
        self._rows = rows
        self.columns = columns
        self.dtypes = dtypes
        self._column_map = {
            column: [row[index] for row in rows] for index, column in enumerate(columns)
        }

    def to_numpy(self) -> list[list[object]]:
        return self._rows

    def __getitem__(self, key: object) -> _FakeCategoricalSeries:
        return _FakeCategoricalSeries(self._column_map[str(key)])


def _dense_memoryview(
    values: list[float] | list[int], rows: int, cols: int, *, typecode: str = "f"
) -> memoryview:
    base = memoryview(array(typecode, values))
    return base.cast("B").cast(typecode, shape=[rows, cols])


class GBMRegressorContractTests(unittest.TestCase):
    def test_constructor_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            GBMRegressor(learning_rate=0.0)
        with self.assertRaisesRegex(ValueError, "max_depth"):
            GBMRegressor(max_depth=0)
        with self.assertRaisesRegex(ValueError, "n_estimators"):
            GBMRegressor(n_estimators=0)
        with self.assertRaisesRegex(ValueError, "row_subsample"):
            GBMRegressor(row_subsample=0.0)
        with self.assertRaisesRegex(ValueError, "col_subsample"):
            GBMRegressor(col_subsample=1.5)
        with self.assertRaisesRegex(ValueError, "early_stopping_rounds"):
            GBMRegressor(early_stopping_rounds=0)
        with self.assertRaisesRegex(ValueError, "min_validation_improvement"):
            GBMRegressor(min_validation_improvement=-0.1)
        with self.assertRaisesRegex(ValueError, "min_data_in_leaf"):
            GBMRegressor(min_data_in_leaf=0)
        with self.assertRaisesRegex(ValueError, "lambda_l1"):
            GBMRegressor(lambda_l1=-0.1)
        with self.assertRaisesRegex(ValueError, "lambda_l2"):
            GBMRegressor(lambda_l2=-0.1)
        with self.assertRaisesRegex(ValueError, "min_child_hessian"):
            GBMRegressor(min_child_hessian=-0.1)
        with self.assertRaisesRegex(ValueError, "categorical_feature_index"):
            GBMRegressor(categorical_feature_index=-1)
        with self.assertRaisesRegex(ValueError, "categorical_smoothing"):
            GBMRegressor(categorical_smoothing=-0.1)
        with self.assertRaisesRegex(ValueError, "categorical_min_samples_leaf"):
            GBMRegressor(categorical_min_samples_leaf=0)
        with self.assertRaisesRegex(ValueError, "continuous_binning_strategy"):
            GBMRegressor(continuous_binning_strategy="invalid")
        with self.assertRaisesRegex(ValueError, "continuous_binning_max_bins"):
            GBMRegressor(continuous_binning_max_bins=1)
        with self.assertRaisesRegex(ValueError, "continuous_binning_max_bins"):
            GBMRegressor(continuous_binning_max_bins=65536)
        with self.assertRaisesRegex(ValueError, "leaf_solver"):
            GBMRegressor(leaf_solver="invalid")
        with self.assertRaisesRegex(ValueError, "dro_radius"):
            GBMRegressor(leaf_solver="dro", dro_radius=-0.1)
        with self.assertRaisesRegex(ValueError, "dro_metric"):
            GBMRegressor(leaf_solver="dro", dro_metric="kl")
        with self.assertRaisesRegex(ValueError, "leaf_solver='dro'.*leaf_model='constant'"):
            GBMRegressor(leaf_solver="dro", leaf_model="linear")

    def test_get_params_and_set_params_roundtrip(self) -> None:
        model = GBMRegressor()
        params = model.get_params()
        self.assertEqual(params["learning_rate"], 0.1)
        self.assertEqual(params["max_depth"], 6)
        self.assertEqual(params["n_estimators"], 6)
        self.assertEqual(params["row_subsample"], 1.0)
        self.assertEqual(params["col_subsample"], 1.0)
        self.assertIsNone(params["early_stopping_rounds"])
        self.assertEqual(params["min_validation_improvement"], 0.0)
        self.assertEqual(params["min_data_in_leaf"], 1)
        self.assertEqual(params["lambda_l1"], 0.0)
        self.assertEqual(params["lambda_l2"], 0.0)
        self.assertEqual(params["min_child_hessian"], 0.0)
        self.assertEqual(params["seed"], 0)
        self.assertTrue(params["deterministic"])
        self.assertEqual(params["continuous_binning_strategy"], "linear")
        self.assertEqual(params["continuous_binning_max_bins"], 256)
        self.assertIsNone(params["categorical_feature_index"])
        self.assertEqual(params["training_policy"], "auto")
        self.assertFalse(params["store_node_stats"])
        self.assertEqual(params["categorical_smoothing"], 20.0)
        self.assertEqual(params["categorical_min_samples_leaf"], 1)
        self.assertFalse(params["categorical_time_aware"])
        self.assertEqual(params["leaf_solver"], "standard")
        self.assertEqual(params["dro_radius"], 0.05)
        self.assertEqual(params["dro_metric"], "wasserstein")

        updated = model.set_params(
            learning_rate=0.2,
            max_depth=4,
            n_estimators=9,
            row_subsample=0.75,
            col_subsample=0.5,
            early_stopping_rounds=4,
            min_validation_improvement=0.01,
            min_data_in_leaf=3,
            lambda_l1=0.2,
            lambda_l2=0.4,
            min_child_hessian=0.6,
            seed=7,
            deterministic=False,
            continuous_binning_strategy="rank",
            continuous_binning_max_bins=128,
            categorical_feature_index=1,
            training_policy="manual",
            store_node_stats=True,
            categorical_smoothing=5.0,
            categorical_min_samples_leaf=2,
            categorical_time_aware=True,
            leaf_solver="dro",
            dro_radius=0.2,
            dro_metric="wasserstein",
        )
        self.assertIs(updated, model)
        self.assertEqual(model.get_params()["learning_rate"], 0.2)
        self.assertEqual(model.get_params()["max_depth"], 4)
        self.assertEqual(model.get_params()["n_estimators"], 9)
        self.assertEqual(model.get_params()["leaf_solver"], "dro")
        self.assertEqual(model.get_params()["dro_radius"], 0.2)
        self.assertEqual(model.get_params()["dro_metric"], "wasserstein")
        self.assertEqual(model.get_params()["row_subsample"], 0.75)
        self.assertEqual(model.get_params()["col_subsample"], 0.5)
        self.assertEqual(model.get_params()["early_stopping_rounds"], 4)
        self.assertEqual(model.get_params()["min_validation_improvement"], 0.01)
        self.assertEqual(model.get_params()["min_data_in_leaf"], 3)
        self.assertEqual(model.get_params()["lambda_l1"], 0.2)
        self.assertEqual(model.get_params()["lambda_l2"], 0.4)
        self.assertEqual(model.get_params()["min_child_hessian"], 0.6)
        self.assertEqual(model.get_params()["seed"], 7)
        self.assertFalse(model.get_params()["deterministic"])
        self.assertEqual(model.get_params()["continuous_binning_strategy"], "rank")
        self.assertEqual(model.get_params()["continuous_binning_max_bins"], 128)
        self.assertEqual(model.get_params()["categorical_feature_index"], 1)
        self.assertEqual(model.get_params()["training_policy"], "manual")
        self.assertTrue(model.get_params()["store_node_stats"])
        self.assertEqual(model.get_params()["categorical_smoothing"], 5.0)
        self.assertEqual(model.get_params()["categorical_min_samples_leaf"], 2)
        self.assertTrue(model.get_params()["categorical_time_aware"])

    def test_set_params_rejects_unknown_parameter(self) -> None:
        model = GBMRegressor()
        with self.assertRaisesRegex(ValueError, "Unknown parameter"):
            model.set_params(unknown=1)  # type: ignore[arg-type]

    def test_set_params_rejects_invalid_continuous_binning_strategy(self) -> None:
        model = GBMRegressor()
        with self.assertRaisesRegex(ValueError, "continuous_binning_strategy"):
            model.set_params(continuous_binning_strategy="bad")

    def test_set_params_rejects_invalid_continuous_binning_max_bins(self) -> None:
        model = GBMRegressor()
        with self.assertRaisesRegex(ValueError, "continuous_binning_max_bins"):
            model.set_params(continuous_binning_max_bins=0)

    def test_set_params_rejects_invalid_n_estimators(self) -> None:
        model = GBMRegressor()
        with self.assertRaisesRegex(ValueError, "n_estimators"):
            model.set_params(n_estimators=0)

    def test_predict_requires_fit(self) -> None:
        model = GBMRegressor()
        with self.assertRaisesRegex(RuntimeError, "must be fit"):
            model.predict([])

    def test_shap_values_requires_fit(self) -> None:
        model = GBMRegressor()
        with self.assertRaisesRegex(RuntimeError, "must be fit"):
            model.shap_values([[1.0, 0.0]])

    def test_feature_importances_requires_fit(self) -> None:
        model = GBMRegressor()
        with self.assertRaisesRegex(RuntimeError, "must be fit"):
            model.feature_importances([[1.0, 0.0]])

    def test_fit_and_predict_use_native_bridges(self) -> None:
        with _force_legacy_train_path():
            train_calls: list[dict[str, object]] = []
            predict_calls: list[tuple[bytes, list[list[float]]]] = []

            def fake_train(**kwargs: object) -> bytes:
                train_calls.append(kwargs)
                return b"trained-artifact"

            def fake_predictor(
                artifact_bytes: bytes, rows: list[list[float]]
            ) -> list[float]:
                predict_calls.append((artifact_bytes, rows))
                return [0.5 + row[0] for row in rows]

            original_train_loader = regressor_module._load_native_train_regression_artifact
            original_predict_loader = (
                regressor_module._load_native_predictor_predict_batch_canonical
            )
            regressor_module._load_native_train_regression_artifact = lambda: fake_train
            regressor_module._load_native_predictor_predict_batch_canonical = (
                lambda: fake_predictor
            )
            try:
                model = GBMRegressor(
                    learning_rate=0.2,
                    max_depth=4,
                    n_estimators=9,
                    row_subsample=0.75,
                    col_subsample=0.5,
                    min_validation_improvement=0.01,
                    seed=7,
                    deterministic=False,
                )
                fitted = model.fit([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], [2.0, 4.0, 6.0])
                predictions = model.predict([[1.0, 0.0], [2.0, 0.0]])
            finally:
                regressor_module._load_native_train_regression_artifact = (
                    original_train_loader
                )
                regressor_module._load_native_predictor_predict_batch_canonical = (
                    original_predict_loader
                )

            self.assertIs(fitted, model)
            self.assertEqual(predictions, [1.5, 2.5])
            self.assertEqual(
                train_calls,
                [
                    {
                        "rows": [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
                        "targets": [2.0, 4.0, 6.0],
                        "learning_rate": 0.2,
                        "max_depth": 4,
                        "row_subsample": 0.75,
                        "col_subsample": 0.5,
                        "min_validation_improvement": 0.01,
                        "seed": 7,
                        "deterministic": False,
                        "rounds": 9,
                        "early_stopping_rounds": None,
                        "categorical_feature_index": None,
                        "categorical_feature_values": None,
                        "training_policy": "auto",
                        "store_node_stats": False,
                        "categorical_smoothing": 20.0,
                        "categorical_min_samples_leaf": 1,
                        "categorical_time_aware": False,
                        "time_index": None,
                        "continuous_binning_strategy": "linear",
                        "continuous_binning_max_bins": 256,
                        "objective": "squared_error",
                        "leaf_model": "constant",
                        "leaf_solver": "standard",
                        "dro_radius": 0.05,
                        "dro_metric": "wasserstein",
                        "neutralization": "none",
                        "factor_neutralization_lambda": 1e-6,
                        "factor_penalty": 0.0,
                        "factor_exposure_values": None,
                        "factor_exposure_row_count": None,
                        "factor_exposure_factor_count": None,
                        "boosting_mode": "standard",
                        "goss_top_rate": None,
                        "goss_other_rate": None,
                        "dart_drop_rate": None,
                        "dart_max_drop": None,
                        "dart_normalize_type": None,
                        "dart_sample_type": None,
                        "tweedie_variance_power": None,
                        "quantile_alpha": None,
                    }
                ],
            )
            self.assertEqual(
                predict_calls,
                [(b"trained-artifact", [[1.0, 0.0], [2.0, 0.0]])],
            )

    def test_fit_requires_eval_set_when_early_stopping_is_enabled(self) -> None:
        model = GBMRegressor(early_stopping_rounds=3)
        with self.assertRaisesRegex(ValueError, "early_stopping_rounds"):
            model.fit([[1.0], [2.0]], [1.0, 2.0])

    def test_fit_with_eval_set_populates_training_summary_attributes(self) -> None:
        result = SimpleNamespace(
            artifact_bytes=b"artifact",
            summary=SimpleNamespace(
                rounds_requested=12,
                rounds_completed=5,
                best_validation_round=3,
                best_validation_loss=0.16,
                train_rmse=[1.0, 0.8, 0.6, 0.5, 0.4],
                validation_rmse=[1.1, 0.9, 0.7, 0.6, 0.5],
                train_loss=[1.0, 0.64, 0.36, 0.25, 0.16],
                validation_loss=[1.21, 0.81, 0.49, 0.36, 0.25],
                objective="squared_error",
                stop_reason="ValidationLossPlateau",
                bridge_prepare_seconds=0.01,
                native_train_seconds=0.02,
            ),
            continuous_binning_metadata=SimpleNamespace(
                uses_continuous_binning=True,
                feature_mins=[0.0],
                feature_maxs=[2.0],
                feature_sorted_values=None,
                feature_quantile_cuts=None,
                feature_linear_rank_flags=None,
            ),
        )
        original_loader = (
            regressor_module._load_native_train_regression_artifact_dense_with_summary
        )
        regressor_module._load_native_train_regression_artifact_dense_with_summary = (
            lambda: (lambda **_kwargs: result)
        )
        # The bytes-path function (train_regression_artifact_dense_with_summary_bytes) is
        # tried first and now succeeds on this all-integer training data.  Force it to
        # raise AttributeError so the code falls through to the mocked list path.
        bytes_patch = "alloygbm._alloygbm.train_regression_artifact_dense_with_summary_bytes"
        try:
            with patch(bytes_patch, side_effect=AttributeError("mocked for test")):
                model = GBMRegressor(
                    early_stopping_rounds=2,
                    min_data_in_leaf=4,
                    lambda_l1=0.1,
                    lambda_l2=0.2,
                    min_child_hessian=0.3,
                )
                fitted = model.fit(
                    _dense_memoryview([0.0, 1.0, 2.0], 3, 1),
                    [0.0, 1.0, 2.0],
                    eval_set=(_dense_memoryview([0.5, 1.5], 2, 1), [0.4, 1.4]),
                )
        finally:
            regressor_module._load_native_train_regression_artifact_dense_with_summary = (
                original_loader
            )

        self.assertIs(fitted, model)
        self.assertEqual(model.best_iteration_, 3)
        self.assertEqual(model.best_score_, 0.16)
        self.assertEqual(model.n_estimators_, 5)
        self.assertEqual(
            model.evals_result_,
            {
                "train": {
                    "rmse": [1.0, 0.8, 0.6, 0.5, 0.4],
                    "mse": [1.0, 0.64, 0.36, 0.25, 0.16],
                },
                "validation": {
                    "rmse": [1.1, 0.9, 0.7, 0.6, 0.5],
                    "mse": [1.21, 0.81, 0.49, 0.36, 0.25],
                },
            },
        )
        self.assertEqual(model.fit_timing_["native_bridge_prepare_seconds"], 0.01)
        self.assertEqual(model.fit_timing_["native_train_seconds"], 0.02)

    def test_fit_quantizes_continuous_rows_before_native_training(self) -> None:
        with _force_legacy_train_path():
            train_calls: list[dict[str, object]] = []

            def fake_train(**kwargs: object) -> bytes:
                train_calls.append(kwargs)
                return b"artifact"

            original_train_loader = regressor_module._load_native_train_regression_artifact
            regressor_module._load_native_train_regression_artifact = lambda: fake_train
            try:
                GBMRegressor().fit(
                    [[-1.2, 0.49], [0.51, 300.1], [2.6, 12.9]],
                    [0.0, 1.0, 2.0],
                )
            finally:
                regressor_module._load_native_train_regression_artifact = (
                    original_train_loader
                )

            self.assertEqual(len(train_calls), 1)
            self.assertEqual(
                train_calls[0]["rows"],
                [[0.0, 0.0], [114.0, 254.0], [254.0, 11.0]],
            )

    def test_fit_linear_tail_rank_fallback_quantizes_heavy_tail_features(self) -> None:
        with _force_legacy_train_path():
            train_calls: list[dict[str, object]] = []

            def fake_train(**kwargs: object) -> bytes:
                train_calls.append(kwargs)
                return b"artifact"

            original_train_loader = regressor_module._load_native_train_regression_artifact
            original_tail_rank = os.environ.get("ALLOYGBM_EXPERIMENT_LINEAR_TAIL_RANK")
            original_tail_ratio = os.environ.get(
                "ALLOYGBM_EXPERIMENT_LINEAR_TAIL_CORE_SPAN_RATIO"
            )
            regressor_module._load_native_train_regression_artifact = lambda: fake_train
            os.environ["ALLOYGBM_EXPERIMENT_LINEAR_TAIL_RANK"] = "1"
            os.environ["ALLOYGBM_EXPERIMENT_LINEAR_TAIL_CORE_SPAN_RATIO"] = "0.05"
            try:
                GBMRegressor().fit(
                    [
                        [0.1, 0.10],
                        [0.2, 0.20],
                        [0.3, 0.25],
                        [0.4, 0.26],
                        [1000.0, 0.50],
                    ],
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                )
            finally:
                regressor_module._load_native_train_regression_artifact = (
                    original_train_loader
                )
                if original_tail_rank is None:
                    os.environ.pop("ALLOYGBM_EXPERIMENT_LINEAR_TAIL_RANK", None)
                else:
                    os.environ["ALLOYGBM_EXPERIMENT_LINEAR_TAIL_RANK"] = (
                        original_tail_rank
                    )
                if original_tail_ratio is None:
                    os.environ.pop("ALLOYGBM_EXPERIMENT_LINEAR_TAIL_CORE_SPAN_RATIO", None)
                else:
                    os.environ["ALLOYGBM_EXPERIMENT_LINEAR_TAIL_CORE_SPAN_RATIO"] = (
                        original_tail_ratio
                    )

            self.assertEqual(len(train_calls), 1)
            self.assertEqual(
                train_calls[0]["rows"],
                [
                    [0.0, 0.0],
                    [64.0, 64.0],
                    [127.0, 95.0],
                    [191.0, 102.0],
                    [254.0, 254.0],
                ],
            )

    def test_predict_quantizes_rows_when_model_fitted_on_continuous_inputs(self) -> None:
        with _force_legacy_train_path():
            predict_calls: list[list[list[float]]] = []

            def fake_predictor(_artifact_bytes: bytes, rows: list[list[float]]) -> list[float]:
                predict_calls.append(rows)
                return [0.0] * len(rows)

            original_train_loader = regressor_module._load_native_train_regression_artifact
            original_predict_loader = (
                regressor_module._load_native_predictor_predict_batch_canonical
            )
            regressor_module._load_native_train_regression_artifact = lambda: (
                lambda *_args, **_kwargs: b"artifact"
            )
            regressor_module._load_native_predictor_predict_batch_canonical = (
                lambda: fake_predictor
            )
            try:
                model = GBMRegressor().fit(
                    [[-1.2, 0.49], [0.51, 300.1], [2.6, 12.9]],
                    [0.0, 1.0, 2.0],
                )
                model.predict([[-0.7, 0.2], [4.4, 1200.4]])
            finally:
                regressor_module._load_native_train_regression_artifact = (
                    original_train_loader
                )
                regressor_module._load_native_predictor_predict_batch_canonical = (
                    original_predict_loader
                )

            self.assertEqual(predict_calls, [[[33.0, 0.0], [254.0, 254.0]]])

    def test_predict_preserves_linear_tail_rank_fallback_after_fit(self) -> None:
        with _force_legacy_train_path():
            predict_calls: list[list[list[float]]] = []

            def fake_predictor(
                _artifact_bytes: bytes, rows: list[list[float]]
            ) -> list[float]:
                predict_calls.append(rows)
                return [0.0] * len(rows)

            original_train_loader = regressor_module._load_native_train_regression_artifact
            original_predict_loader = (
                regressor_module._load_native_predictor_predict_batch_canonical
            )
            original_tail_rank = os.environ.get("ALLOYGBM_EXPERIMENT_LINEAR_TAIL_RANK")
            original_tail_ratio = os.environ.get(
                "ALLOYGBM_EXPERIMENT_LINEAR_TAIL_CORE_SPAN_RATIO"
            )
            regressor_module._load_native_train_regression_artifact = lambda: (
                lambda *_args, **_kwargs: b"artifact"
            )
            regressor_module._load_native_predictor_predict_batch_canonical = (
                lambda: fake_predictor
            )
            os.environ["ALLOYGBM_EXPERIMENT_LINEAR_TAIL_RANK"] = "1"
            os.environ["ALLOYGBM_EXPERIMENT_LINEAR_TAIL_CORE_SPAN_RATIO"] = "0.05"
            try:
                model = GBMRegressor().fit(
                    [
                        [0.1, 0.10],
                        [0.2, 0.20],
                        [0.3, 0.25],
                        [0.4, 0.26],
                        [1000.0, 0.50],
                    ],
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                )
                os.environ["ALLOYGBM_EXPERIMENT_LINEAR_TAIL_RANK"] = "0"
                model.predict([[0.35, 0.24], [2.0, 0.30]])
            finally:
                regressor_module._load_native_train_regression_artifact = (
                    original_train_loader
                )
                regressor_module._load_native_predictor_predict_batch_canonical = (
                    original_predict_loader
                )
                if original_tail_rank is None:
                    os.environ.pop("ALLOYGBM_EXPERIMENT_LINEAR_TAIL_RANK", None)
                else:
                    os.environ["ALLOYGBM_EXPERIMENT_LINEAR_TAIL_RANK"] = (
                        original_tail_rank
                    )
                if original_tail_ratio is None:
                    os.environ.pop("ALLOYGBM_EXPERIMENT_LINEAR_TAIL_CORE_SPAN_RATIO", None)
                else:
                    os.environ["ALLOYGBM_EXPERIMENT_LINEAR_TAIL_CORE_SPAN_RATIO"] = (
                        original_tail_ratio
                    )

            self.assertEqual(predict_calls, [[[127.0, 89.0], [191.0, 127.0]]])

    def test_fit_quantizes_continuous_rows_with_rank_strategy(self) -> None:
        with _force_legacy_train_path():
            train_calls: list[dict[str, object]] = []

            def fake_train(**kwargs: object) -> bytes:
                train_calls.append(kwargs)
                return b"artifact"

            original_train_loader = regressor_module._load_native_train_regression_artifact
            regressor_module._load_native_train_regression_artifact = lambda: fake_train
            try:
                GBMRegressor(continuous_binning_strategy="rank").fit(
                    [[-1.2, 0.49], [0.51, 300.1], [2.6, 12.9]],
                    [0.0, 1.0, 2.0],
                )
            finally:
                regressor_module._load_native_train_regression_artifact = (
                    original_train_loader
                )

            self.assertEqual(len(train_calls), 1)
            self.assertEqual(
                train_calls[0]["rows"],
                [[0.0, 0.0], [127.0, 254.0], [254.0, 127.0]],
            )

    def test_fit_quantizes_continuous_rows_with_quantile_strategy(self) -> None:
        with _force_legacy_train_path():
            train_calls: list[dict[str, object]] = []

            def fake_train(**kwargs: object) -> bytes:
                train_calls.append(kwargs)
                return b"artifact"

            original_train_loader = regressor_module._load_native_train_regression_artifact
            regressor_module._load_native_train_regression_artifact = lambda: fake_train
            try:
                GBMRegressor(continuous_binning_strategy="quantile").fit(
                    [[-1.2, 0.49], [0.51, 300.1], [2.6, 12.9]],
                    [0.0, 1.0, 2.0],
                )
            finally:
                regressor_module._load_native_train_regression_artifact = (
                    original_train_loader
                )

            self.assertEqual(len(train_calls), 1)
            self.assertEqual(
                train_calls[0]["rows"],
                [[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]],
            )

    def test_set_params_binning_strategy_after_fit_requires_refit(self) -> None:
        model = GBMRegressor().fit([[1.0, 0.0], [2.0, 0.0]], [1.0, 2.0])
        model.set_params(continuous_binning_strategy="rank")
        with self.assertRaisesRegex(RuntimeError, "must be fit"):
            model.predict([[1.0, 0.0]])

    def test_shap_values_use_native_bridge_with_optional_expected_value(self) -> None:
        shap_calls: list[tuple[bytes, list[list[float]]]] = []

        def fake_shap(
            artifact_bytes: bytes, rows: list[list[float]]
        ) -> tuple[float, list[list[float]]]:
            shap_calls.append((artifact_bytes, rows))
            return (1.25, [[0.1, -0.1], [0.2, -0.2]])

        original_shap_loader = regressor_module._load_native_shap_explain_rows
        regressor_module._load_native_shap_explain_rows = lambda: fake_shap
        try:
            model = GBMRegressor().fit([[1.0, 0.0], [2.0, 0.0]], [1.0, 2.0])
            values = model.shap_values([[1.0, 0.0], [2.0, 0.0]])
            with_expected = model.shap_values(
                [[1.0, 0.0], [2.0, 0.0]], include_expected_value=True
            )
        finally:
            regressor_module._load_native_shap_explain_rows = original_shap_loader

        self.assertEqual(values, [[0.1, -0.1], [0.2, -0.2]])
        self.assertEqual(with_expected, (1.25, [[0.1, -0.1], [0.2, -0.2]]))
        self.assertEqual(len(shap_calls), 2)
        self.assertEqual(shap_calls[0][0], model._artifact_bytes)
        self.assertEqual(shap_calls[0][1], [[1.0, 0.0], [2.0, 0.0]])
        self.assertEqual(shap_calls[1][0], model._artifact_bytes)
        self.assertEqual(shap_calls[1][1], [[1.0, 0.0], [2.0, 0.0]])

    def test_shap_values_reject_feature_count_mismatch(self) -> None:
        model = GBMRegressor().fit([[1.0, 0.0], [2.0, 0.0]], [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "feature count"):
            model.shap_values([[1.0]])

    def test_feature_importances_use_native_shap_global_bridge(self) -> None:
        calls: list[tuple[bytes, list[list[float]]]] = []

        def fake_importance(
            artifact_bytes: bytes, rows: list[list[float]]
        ) -> list[tuple[str, float]]:
            calls.append((artifact_bytes, rows))
            return [("f0", 0.6), ("f1", 0.05)]

        original_importance_loader = regressor_module._load_native_shap_global_importance
        regressor_module._load_native_shap_global_importance = lambda: fake_importance
        try:
            model = GBMRegressor().fit([[1.0, 0.0], [2.0, 0.0]], [1.0, 2.0])
            importance = model.feature_importances([[1.0, 0.0], [2.0, 0.0]])
        finally:
            regressor_module._load_native_shap_global_importance = (
                original_importance_loader
            )

        self.assertEqual(importance, [("f0", 0.6), ("f1", 0.05)])
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0][0], bytes)
        self.assertEqual(calls[0][0], model._artifact_bytes)
        self.assertEqual(calls[0][1], [[1.0, 0.0], [2.0, 0.0]])

    def test_feature_importances_reject_unsupported_method(self) -> None:
        model = GBMRegressor().fit([[1.0, 0.0], [2.0, 0.0]], [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "unsupported feature importance method"):
            model.feature_importances([[1.0, 0.0]], method="gain")

    def test_feature_importances_reject_feature_count_mismatch(self) -> None:
        model = GBMRegressor().fit([[1.0, 0.0], [2.0, 0.0]], [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "feature count"):
            model.feature_importances([[1.0]])

    def test_fit_rejects_mismatched_lengths(self) -> None:
        model = GBMRegressor()
        with self.assertRaisesRegex(ValueError, "same number of rows"):
            model.fit([[1.0], [2.0]], [1.0])

    def test_fit_rejects_out_of_bounds_categorical_feature_index(self) -> None:
        model = GBMRegressor(categorical_feature_index=2)
        with self.assertRaisesRegex(ValueError, "within fitted feature bounds"):
            model.fit(
                [[1.0, 0.0], [2.0, 0.0]],
                [1.0, 2.0],
                categorical_feature_values=["A", "B"],
            )

    def test_fit_rejects_missing_categorical_values(self) -> None:
        model = GBMRegressor(categorical_feature_index=1)
        with self.assertRaisesRegex(ValueError, "categorical_feature_values must be provided"):
            model.fit([[1.0, 0.0], [2.0, 0.0]], [1.0, 2.0])

    def test_fit_rejects_time_aware_categorical_without_time_index(self) -> None:
        model = GBMRegressor(
            categorical_feature_index=1,
            categorical_time_aware=True,
        )
        with self.assertRaisesRegex(ValueError, "time_index must be provided"):
            model.fit(
                [[1.0, 0.0], [2.0, 0.0]],
                [1.0, 2.0],
                categorical_feature_values=["A", "B"],
            )

    def test_fit_passes_categorical_bridge_arguments(self) -> None:
        with _force_legacy_train_path():
            train_calls: list[dict[str, object]] = []

            def fake_train(**kwargs: object) -> bytes:
                train_calls.append(kwargs)
                return b"artifact"

            original_train_loader = regressor_module._load_native_train_regression_artifact
            regressor_module._load_native_train_regression_artifact = lambda: fake_train
            try:
                model = GBMRegressor(
                    categorical_feature_index=1,
                    categorical_smoothing=3.0,
                    categorical_min_samples_leaf=2,
                    categorical_time_aware=True,
                )
                model.fit(
                    [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
                    [1.0, 2.0, 3.0],
                    categorical_feature_values=["A", "B", "A"],
                    time_index=[1, 2, 3],
                )
            finally:
                regressor_module._load_native_train_regression_artifact = (
                    original_train_loader
                )

            self.assertEqual(len(train_calls), 1)
            self.assertEqual(train_calls[0]["categorical_feature_index"], 1)
            self.assertEqual(train_calls[0]["categorical_feature_values"], ["A", "B", "A"])
            self.assertEqual(train_calls[0]["categorical_smoothing"], 3.0)
            self.assertEqual(train_calls[0]["categorical_min_samples_leaf"], 2)
            self.assertTrue(train_calls[0]["categorical_time_aware"])
            self.assertEqual(train_calls[0]["time_index"], [1, 2, 3])

    def test_fit_rejects_non_convertible_adapter_inputs(self) -> None:
        model = GBMRegressor()
        with self.assertRaisesRegex(TypeError, "to_numpy/to_list/tolist"):
            model.fit(_NonConvertible(), [1.0])  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "to_numpy/to_list/tolist"):
            model.fit([[1.0, 0.0]], _NonConvertible())  # type: ignore[arg-type]

    def test_fit_accepts_numpy_pandas_polars_like_inputs(self) -> None:
        train_calls: list[tuple[list[list[float]], list[float]]] = []

        def fake_train(
            rows: list[list[float]],
            targets: list[float],
            *_args: object,
            **_kwargs: object,
        ) -> bytes:
            train_calls.append((rows, targets))
            return b"artifact"

        original_train_loader = regressor_module._load_native_train_regression_artifact
        original_summary_loader = regressor_module._load_native_train_regression_artifact_with_summary
        original_dense_summary_loader = regressor_module._load_native_train_regression_artifact_dense_with_summary
        regressor_module._load_native_train_regression_artifact = lambda: fake_train
        regressor_module._load_native_train_regression_artifact_with_summary = (
            lambda: (_ for _ in ()).throw(RuntimeError("force legacy path"))
        )
        regressor_module._load_native_train_regression_artifact_dense_with_summary = (
            lambda: (_ for _ in ()).throw(RuntimeError("force legacy path"))
        )
        try:
            model = GBMRegressor()
            model.fit(
                _FakeNumpyLike([[1.0, 0.0], [2.0, 0.0]]),
                _FakeNumpyLike([1.0, 2.0]),
            )
            model.fit(
                _FakePandasLikeFrame([[3.0, 0.0], [4.0, 0.0]]),
                _FakePolarsLikeSeries([3.0, 4.0]),
            )
            model.fit(
                _FakePolarsLikeFrame([[5.0, 0.0], [6.0, 0.0]]),
                _FakeNumpyLike([5.0, 6.0]),
            )
        finally:
            regressor_module._load_native_train_regression_artifact = (
                original_train_loader
            )
            regressor_module._load_native_train_regression_artifact_with_summary = (
                original_summary_loader
            )
            regressor_module._load_native_train_regression_artifact_dense_with_summary = (
                original_dense_summary_loader
            )

        self.assertEqual(
            train_calls,
            [
                ([[1.0, 0.0], [2.0, 0.0]], [1.0, 2.0]),
                ([[3.0, 0.0], [4.0, 0.0]], [3.0, 4.0]),
                ([[5.0, 0.0], [6.0, 0.0]], [5.0, 6.0]),
            ],
        )

    def test_fit_uses_dense_native_training_bridge_for_integer_buffer_inputs(self) -> None:
        model = GBMRegressor()
        model.fit(
            _dense_memoryview([1, 0, 2, 0], 2, 2, typecode="I"),
            [1.0, 2.0],
        )
        self.assertTrue(model._is_fitted)
        self.assertEqual(model._n_features_in, 2)

    def test_fit_infers_single_explicit_categorical_column(self) -> None:
        with _force_legacy_train_path():
            train_calls: list[dict[str, object]] = []

            def fake_train(**kwargs: object) -> bytes:
                train_calls.append(kwargs)
                return b"artifact"

            original_train_loader = regressor_module._load_native_train_regression_artifact
            regressor_module._load_native_train_regression_artifact = lambda: fake_train
            try:
                model = GBMRegressor()
                model.fit(
                    _FakeCategoricalFrame(
                        rows=[["A", 1.0], ["B", 2.0], ["A", 3.0]],
                        columns=["kind", "value"],
                        dtypes=["category", "float64"],
                    ),
                    [1.0, 2.0, 3.0],
                )
            finally:
                regressor_module._load_native_train_regression_artifact = (
                    original_train_loader
                )

            self.assertEqual(train_calls[0]["categorical_feature_index"], 0)
            self.assertEqual(train_calls[0]["categorical_feature_values"], ["A", "B", "A"])
            self.assertEqual(train_calls[0]["rows"], [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])

    def test_fit_auto_infers_multiple_explicit_categorical_columns(
        self,
    ) -> None:
        """Multiple categorical columns in a DataFrame should be auto-inferred."""
        model = GBMRegressor(n_estimators=3, training_policy="manual")
        model.fit(
            _FakeCategoricalFrame(
                rows=[["A", "X", 1.0], ["B", "Y", 2.0], ["A", "X", 3.0]],
                columns=["kind_a", "kind_b", "value"],
                dtypes=["category", "categorical", "float64"],
            ),
            [1.0, 2.0, 3.0],
        )
        preds = model.predict([[0.0, 0.0, 1.5]])
        self.assertEqual(len(preds), 1)

    def test_predict_rejects_feature_count_mismatch(self) -> None:
        with _force_legacy_train_path():
            original_train_loader = regressor_module._load_native_train_regression_artifact
            regressor_module._load_native_train_regression_artifact = lambda: (
                lambda *_args, **_kwargs: b"artifact"
            )
            try:
                model = GBMRegressor().fit([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0])
                with self.assertRaisesRegex(ValueError, "feature count"):
                    model.predict([[1.0]])
            finally:
                regressor_module._load_native_train_regression_artifact = (
                    original_train_loader
                )

    def test_predict_accepts_pandas_like_rows(self) -> None:
        with _force_legacy_train_path():
            predict_calls: list[list[list[float]]] = []

            def fake_predictor(artifact_bytes: bytes, rows: list[list[float]]) -> list[float]:
                del artifact_bytes
                predict_calls.append(rows)
                return [0.0] * len(rows)

            original_train_loader = regressor_module._load_native_train_regression_artifact
            original_predict_loader = (
                regressor_module._load_native_predictor_predict_batch_canonical
            )
            regressor_module._load_native_train_regression_artifact = lambda: (
                lambda *_args, **_kwargs: b"artifact"
            )
            regressor_module._load_native_predictor_predict_batch_canonical = (
                lambda: fake_predictor
            )
            try:
                model = GBMRegressor().fit([[1.0, 0.0], [2.0, 0.0]], [1.0, 2.0])
                predictions = model.predict(_FakePandasLikeFrame([[3.0, 0.0], [4.0, 0.0]]))
            finally:
                regressor_module._load_native_train_regression_artifact = (
                    original_train_loader
                )
                regressor_module._load_native_predictor_predict_batch_canonical = (
                    original_predict_loader
                )

            self.assertEqual(predictions, [0.0, 0.0])
            self.assertEqual(predict_calls, [[[3.0, 0.0], [4.0, 0.0]]])

    def test_predict_uses_cached_native_predictor_handle_when_available(self) -> None:
        with _force_legacy_train_path():
            handle_inits: list[tuple[bytes, bool]] = []
            handle_predict_calls: list[list[list[float]]] = []

            class FakeHandle:
                def __init__(self, artifact_bytes: bytes, strict: bool = True) -> None:
                    handle_inits.append((artifact_bytes, strict))

                def predict_batch(self, rows: list[list[float]]) -> list[float]:
                    handle_predict_calls.append(rows)
                    return [8.0 + row[0] for row in rows]

            original_train_loader = regressor_module._load_native_train_regression_artifact
            original_handle_loader = regressor_module._load_native_predictor_handle_class
            original_canonical_loader = (
                regressor_module._load_native_predictor_predict_batch_canonical
            )
            regressor_module._load_native_train_regression_artifact = lambda: (
                lambda *_args, **_kwargs: b"artifact"
            )
            regressor_module._load_native_predictor_handle_class = lambda: FakeHandle
            regressor_module._load_native_predictor_predict_batch_canonical = lambda: (
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("canonical loader should not be used when handle is cached")
                )
            )
            try:
                model = GBMRegressor().fit([[1.0, 0.0], [2.0, 0.0]], [1.0, 2.0])
                predictions = model.predict([[3.0, 0.0], [4.0, 0.0]])
            finally:
                regressor_module._load_native_train_regression_artifact = (
                    original_train_loader
                )
                regressor_module._load_native_predictor_handle_class = original_handle_loader
                regressor_module._load_native_predictor_predict_batch_canonical = (
                    original_canonical_loader
                )

            self.assertEqual(handle_inits, [(b"artifact", True)])
            self.assertEqual(handle_predict_calls, [[[3.0, 0.0], [4.0, 0.0]]])
            self.assertEqual(predictions, [11.0, 12.0])

    def test_predict_falls_back_to_canonical_when_cached_handle_runtime_errors(self) -> None:
        with _force_legacy_train_path():
            handle_predict_calls: list[list[list[float]]] = []
            canonical_calls: list[tuple[bytes, list[list[float]]]] = []

            class FailingHandle:
                def __init__(self, _artifact_bytes: bytes, strict: bool = True) -> None:
                    self.strict = strict

                def predict_batch(self, rows: list[list[float]]) -> list[float]:
                    handle_predict_calls.append(rows)
                    raise RuntimeError("stale handle")

            def fake_canonical(
                artifact_bytes: bytes, rows: list[list[float]]
            ) -> list[float]:
                canonical_calls.append((artifact_bytes, rows))
                return [5.0] * len(rows)

            original_train_loader = regressor_module._load_native_train_regression_artifact
            original_handle_loader = regressor_module._load_native_predictor_handle_class
            original_canonical_loader = (
                regressor_module._load_native_predictor_predict_batch_canonical
            )
            regressor_module._load_native_train_regression_artifact = lambda: (
                lambda *_args, **_kwargs: b"artifact"
            )
            regressor_module._load_native_predictor_handle_class = lambda: FailingHandle
            regressor_module._load_native_predictor_predict_batch_canonical = (
                lambda: fake_canonical
            )
            try:
                model = GBMRegressor().fit([[1.0, 0.0], [2.0, 0.0]], [1.0, 2.0])
                first = model.predict([[3.0, 0.0]])
                second = model.predict([[4.0, 0.0]])
            finally:
                regressor_module._load_native_train_regression_artifact = (
                    original_train_loader
                )
                regressor_module._load_native_predictor_handle_class = original_handle_loader
                regressor_module._load_native_predictor_predict_batch_canonical = (
                    original_canonical_loader
                )

            self.assertEqual(first, [5.0])
            self.assertEqual(second, [5.0])
            self.assertEqual(handle_predict_calls, [[[3.0, 0.0]]])
            self.assertEqual(
                canonical_calls,
                [(b"artifact", [[3.0, 0.0]]), (b"artifact", [[4.0, 0.0]])],
            )

    def test_predict_uses_canonical_loader_not_compatibility_loader(self) -> None:
        with _force_legacy_train_path():
            calls: list[tuple[bytes, list[list[float]]]] = []

            def fake_canonical(artifact_bytes: bytes, rows: list[list[float]]) -> list[float]:
                calls.append((artifact_bytes, rows))
                return [3.0] * len(rows)

            original_train_loader = regressor_module._load_native_train_regression_artifact
            original_compat_loader = regressor_module._load_native_predictor_predict_batch
            original_canonical_loader = (
                regressor_module._load_native_predictor_predict_batch_canonical
            )
            regressor_module._load_native_train_regression_artifact = lambda: (
                lambda *_args, **_kwargs: b"artifact"
            )
            regressor_module._load_native_predictor_predict_batch = lambda: (
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("compatibility loader should not be used for predict")
                )
            )
            regressor_module._load_native_predictor_predict_batch_canonical = (
                lambda: fake_canonical
            )
            try:
                model = GBMRegressor().fit([[1.0, 0.0], [2.0, 0.0]], [1.0, 2.0])
                predictions = model.predict([[1.0, 0.0], [2.0, 0.0]])
            finally:
                regressor_module._load_native_train_regression_artifact = (
                    original_train_loader
                )
                regressor_module._load_native_predictor_predict_batch = original_compat_loader
                regressor_module._load_native_predictor_predict_batch_canonical = (
                    original_canonical_loader
                )

            self.assertEqual(predictions, [3.0, 3.0])
            self.assertEqual(calls, [(b"artifact", [[1.0, 0.0], [2.0, 0.0]])])

    def test_predict_from_artifact_uses_compatibility_loader_not_canonical(self) -> None:
        calls: list[tuple[bytes, list[list[float]]]] = []

        def fake_compatibility(
            artifact_bytes: bytes, rows: list[list[float]]
        ) -> list[float]:
            calls.append((artifact_bytes, rows))
            return [4.0] * len(rows)

        original_compat_loader = regressor_module._load_native_predictor_predict_batch
        original_canonical_loader = (
            regressor_module._load_native_predictor_predict_batch_canonical
        )
        regressor_module._load_native_predictor_predict_batch = lambda: fake_compatibility
        regressor_module._load_native_predictor_predict_batch_canonical = lambda: (
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("canonical loader should not be used for predict_from_artifact")
            )
        )
        try:
            predictions = GBMRegressor.predict_from_artifact(
                b"artifact", [[1.0, 0.0], [2.0, 0.0]]
            )
        finally:
            regressor_module._load_native_predictor_predict_batch = original_compat_loader
            regressor_module._load_native_predictor_predict_batch_canonical = (
                original_canonical_loader
            )

        self.assertEqual(predictions, [4.0, 4.0])
        self.assertEqual(calls, [(b"artifact", [[1.0, 0.0], [2.0, 0.0]])])

    def test_predict_from_artifact_rejects_non_bytes_payload(self) -> None:
        with self.assertRaisesRegex(TypeError, "artifact_bytes"):
            GBMRegressor.predict_from_artifact("artifact", [[1.0]])  # type: ignore[arg-type]

    def test_predict_from_artifact_accepts_bytearray_payload(self) -> None:
        calls: list[tuple[bytes, list[list[float]]]] = []

        def fake_predictor(
            artifact_bytes: bytes, rows: list[list[float]]
        ) -> list[float]:
            calls.append((artifact_bytes, rows))
            return [1.0] * len(rows)

        original_loader = regressor_module._load_native_predictor_predict_batch
        regressor_module._load_native_predictor_predict_batch = lambda: fake_predictor
        try:
            predictions = GBMRegressor.predict_from_artifact(
                bytearray(b"artifact"), [[1.0, 0.0], [2.0, 0.0]]
            )
        finally:
            regressor_module._load_native_predictor_predict_batch = original_loader

        self.assertEqual(predictions, [1.0, 1.0])
        self.assertEqual(calls, [(b"artifact", [[1.0, 0.0], [2.0, 0.0]])])

    def test_predict_from_artifact_accepts_memoryview_payload(self) -> None:
        calls: list[tuple[bytes, list[list[float]]]] = []

        def fake_predictor(
            artifact_bytes: bytes, rows: list[list[float]]
        ) -> list[float]:
            calls.append((artifact_bytes, rows))
            return [2.0] * len(rows)

        original_loader = regressor_module._load_native_predictor_predict_batch
        regressor_module._load_native_predictor_predict_batch = lambda: fake_predictor
        try:
            predictions = GBMRegressor.predict_from_artifact(
                memoryview(b"artifact"), [[1.0, 0.0], [2.0, 0.0]]
            )
        finally:
            regressor_module._load_native_predictor_predict_batch = original_loader

        self.assertEqual(predictions, [2.0, 2.0])
        self.assertEqual(calls, [(b"artifact", [[1.0, 0.0], [2.0, 0.0]])])

    def test_predict_from_artifact_rejects_non_convertible_rows(self) -> None:
        with self.assertRaisesRegex(TypeError, "to_numpy/to_list/tolist"):
            GBMRegressor.predict_from_artifact(b"artifact", _NonConvertible())  # type: ignore[arg-type]

    def test_predict_from_artifact_accepts_polars_like_rows(self) -> None:
        calls: list[tuple[bytes, list[list[float]]]] = []

        def fake_predictor(
            artifact_bytes: bytes, rows: list[list[float]]
        ) -> list[float]:
            calls.append((artifact_bytes, rows))
            return [2.5] * len(rows)

        original_loader = regressor_module._load_native_predictor_predict_batch
        regressor_module._load_native_predictor_predict_batch = lambda: fake_predictor
        try:
            predictions = GBMRegressor.predict_from_artifact(
                b"artifact", _FakePolarsLikeFrame([[1.0, 0.0], [2.0, 0.0]])
            )
        finally:
            regressor_module._load_native_predictor_predict_batch = original_loader

        self.assertEqual(predictions, [2.5, 2.5])
        self.assertEqual(calls, [(b"artifact", [[1.0, 0.0], [2.0, 0.0]])])

    def test_predict_from_artifact_uses_native_bridge(self) -> None:
        calls: list[tuple[bytes, list[list[float]]]] = []

        def fake_predictor(
            artifact_bytes: bytes, rows: list[list[float]]
        ) -> list[float]:
            calls.append((artifact_bytes, rows))
            return [1.5] * len(rows)

        original_loader = regressor_module._load_native_predictor_predict_batch
        regressor_module._load_native_predictor_predict_batch = lambda: fake_predictor
        try:
            predictions = GBMRegressor.predict_from_artifact(
                b"artifact", [[1.0, 0.0], [2.0, 0.0]]
            )
        finally:
            regressor_module._load_native_predictor_predict_batch = original_loader

        self.assertEqual(predictions, [1.5, 1.5])
        self.assertEqual(
            calls,
            [(b"artifact", [[1.0, 0.0], [2.0, 0.0]])],
        )

    def test_predict_from_artifact_propagates_native_bridge_error(self) -> None:
        original_loader = regressor_module._load_native_predictor_predict_batch
        regressor_module._load_native_predictor_predict_batch = lambda: (_ for _ in ()).throw(
            RuntimeError("native bridge unavailable")
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "native bridge unavailable"):
                GBMRegressor.predict_from_artifact(b"artifact", [[1.0, 0.0]])
        finally:
            regressor_module._load_native_predictor_predict_batch = original_loader


if __name__ == "__main__":
    unittest.main()
