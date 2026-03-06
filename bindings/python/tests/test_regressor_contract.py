"""Focused contract tests for the AlloyGBM Python regressor baseline."""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path


def load_regressor_module():
    regressor_path = (
        Path(__file__).resolve().parents[1] / "alloygbm" / "regressor.py"
    )
    spec = importlib.util.spec_from_file_location(
        "alloygbm_regressor", regressor_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load alloygbm regressor module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


regressor_module = load_regressor_module()
GBMRegressor = regressor_module.GBMRegressor


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
            GBMRegressor(continuous_binning_max_bins=257)

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
        self.assertEqual(params["seed"], 0)
        self.assertTrue(params["deterministic"])
        self.assertEqual(params["continuous_binning_strategy"], "linear")
        self.assertEqual(params["continuous_binning_max_bins"], 256)
        self.assertIsNone(params["categorical_feature_index"])
        self.assertEqual(params["categorical_smoothing"], 20.0)
        self.assertEqual(params["categorical_min_samples_leaf"], 1)
        self.assertFalse(params["categorical_time_aware"])

        updated = model.set_params(
            learning_rate=0.2,
            max_depth=4,
            n_estimators=9,
            row_subsample=0.75,
            col_subsample=0.5,
            early_stopping_rounds=4,
            min_validation_improvement=0.01,
            seed=7,
            deterministic=False,
            continuous_binning_strategy="rank",
            continuous_binning_max_bins=128,
            categorical_feature_index=1,
            categorical_smoothing=5.0,
            categorical_min_samples_leaf=2,
            categorical_time_aware=True,
        )
        self.assertIs(updated, model)
        self.assertEqual(model.get_params()["learning_rate"], 0.2)
        self.assertEqual(model.get_params()["max_depth"], 4)
        self.assertEqual(model.get_params()["n_estimators"], 9)
        self.assertEqual(model.get_params()["row_subsample"], 0.75)
        self.assertEqual(model.get_params()["col_subsample"], 0.5)
        self.assertEqual(model.get_params()["early_stopping_rounds"], 4)
        self.assertEqual(model.get_params()["min_validation_improvement"], 0.01)
        self.assertEqual(model.get_params()["seed"], 7)
        self.assertFalse(model.get_params()["deterministic"])
        self.assertEqual(model.get_params()["continuous_binning_strategy"], "rank")
        self.assertEqual(model.get_params()["continuous_binning_max_bins"], 128)
        self.assertEqual(model.get_params()["categorical_feature_index"], 1)
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
                early_stopping_rounds=4,
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
                    "early_stopping_rounds": 4,
                    "categorical_feature_index": None,
                    "categorical_feature_values": None,
                    "categorical_smoothing": 20.0,
                    "categorical_min_samples_leaf": 1,
                    "categorical_time_aware": False,
                    "time_index": None,
                }
            ],
        )
        self.assertEqual(
            predict_calls,
            [(b"trained-artifact", [[1.0, 0.0], [2.0, 0.0]])],
        )

    def test_fit_quantizes_continuous_rows_before_native_training(self) -> None:
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
            [[0.0, 0.0], [115.0, 255.0], [255.0, 11.0]],
        )

    def test_fit_linear_tail_rank_fallback_quantizes_heavy_tail_features(self) -> None:
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
                [128.0, 96.0],
                [191.0, 102.0],
                [255.0, 255.0],
            ],
        )

    def test_predict_quantizes_rows_when_model_fitted_on_continuous_inputs(self) -> None:
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

        self.assertEqual(predict_calls, [[[34.0, 0.0], [255.0, 255.0]]])

    def test_predict_preserves_linear_tail_rank_fallback_after_fit(self) -> None:
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

        self.assertEqual(predict_calls, [[[128.0, 89.0], [191.0, 127.0]]])

    def test_fit_quantizes_continuous_rows_with_rank_strategy(self) -> None:
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
            [[0.0, 0.0], [128.0, 255.0], [255.0, 128.0]],
        )

    def test_fit_quantizes_continuous_rows_with_quantile_strategy(self) -> None:
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
        original_train_loader = regressor_module._load_native_train_regression_artifact
        original_predict_loader = (
            regressor_module._load_native_predictor_predict_batch_canonical
        )
        regressor_module._load_native_train_regression_artifact = lambda: (
            lambda *_args, **_kwargs: b"artifact"
        )
        regressor_module._load_native_predictor_predict_batch_canonical = (
            lambda: (lambda *_args, **_kwargs: [0.0])
        )
        try:
            model = GBMRegressor().fit([[1.0, 0.0], [2.0, 0.0]], [1.0, 2.0])
            model.set_params(continuous_binning_strategy="rank")
            with self.assertRaisesRegex(RuntimeError, "must be fit"):
                model.predict([[1.0, 0.0]])
        finally:
            regressor_module._load_native_train_regression_artifact = (
                original_train_loader
            )
            regressor_module._load_native_predictor_predict_batch_canonical = (
                original_predict_loader
            )

    def test_shap_values_use_native_bridge_with_optional_expected_value(self) -> None:
        shap_calls: list[tuple[bytes, list[list[float]]]] = []

        def fake_shap(
            artifact_bytes: bytes, rows: list[list[float]]
        ) -> tuple[float, list[list[float]]]:
            shap_calls.append((artifact_bytes, rows))
            return (1.25, [[0.1, -0.1], [0.2, -0.2]])

        original_train_loader = regressor_module._load_native_train_regression_artifact
        original_shap_loader = regressor_module._load_native_shap_explain_rows
        regressor_module._load_native_train_regression_artifact = lambda: (
            lambda *_args, **_kwargs: b"artifact"
        )
        regressor_module._load_native_shap_explain_rows = lambda: fake_shap
        try:
            model = GBMRegressor().fit([[1.0, 0.0], [2.0, 0.0]], [1.0, 2.0])
            values = model.shap_values([[1.0, 0.0], [2.0, 0.0]])
            with_expected = model.shap_values(
                [[1.0, 0.0], [2.0, 0.0]], include_expected_value=True
            )
        finally:
            regressor_module._load_native_train_regression_artifact = (
                original_train_loader
            )
            regressor_module._load_native_shap_explain_rows = original_shap_loader

        self.assertEqual(values, [[0.1, -0.1], [0.2, -0.2]])
        self.assertEqual(with_expected, (1.25, [[0.1, -0.1], [0.2, -0.2]]))
        self.assertEqual(
            shap_calls,
            [
                (b"artifact", [[1.0, 0.0], [2.0, 0.0]]),
                (b"artifact", [[1.0, 0.0], [2.0, 0.0]]),
            ],
        )

    def test_shap_values_reject_feature_count_mismatch(self) -> None:
        original_train_loader = regressor_module._load_native_train_regression_artifact
        original_shap_loader = regressor_module._load_native_shap_explain_rows
        regressor_module._load_native_train_regression_artifact = lambda: (
            lambda *_args, **_kwargs: b"artifact"
        )
        regressor_module._load_native_shap_explain_rows = lambda: (
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("native SHAP loader should not be called on shape mismatch")
            )
        )
        try:
            model = GBMRegressor().fit([[1.0, 0.0], [2.0, 0.0]], [1.0, 2.0])
            with self.assertRaisesRegex(ValueError, "feature count"):
                model.shap_values([[1.0]])
        finally:
            regressor_module._load_native_train_regression_artifact = (
                original_train_loader
            )
            regressor_module._load_native_shap_explain_rows = original_shap_loader

    def test_feature_importances_use_native_shap_global_bridge(self) -> None:
        calls: list[tuple[bytes, list[list[float]]]] = []

        def fake_importance(
            artifact_bytes: bytes, rows: list[list[float]]
        ) -> list[tuple[str, float]]:
            calls.append((artifact_bytes, rows))
            return [("f0", 0.6), ("f1", 0.05)]

        original_train_loader = regressor_module._load_native_train_regression_artifact
        original_importance_loader = regressor_module._load_native_shap_global_importance
        regressor_module._load_native_train_regression_artifact = lambda: (
            lambda *_args, **_kwargs: b"artifact"
        )
        regressor_module._load_native_shap_global_importance = lambda: fake_importance
        try:
            model = GBMRegressor().fit([[1.0, 0.0], [2.0, 0.0]], [1.0, 2.0])
            importance = model.feature_importances([[1.0, 0.0], [2.0, 0.0]])
        finally:
            regressor_module._load_native_train_regression_artifact = (
                original_train_loader
            )
            regressor_module._load_native_shap_global_importance = (
                original_importance_loader
            )

        self.assertEqual(importance, [("f0", 0.6), ("f1", 0.05)])
        self.assertEqual(calls, [(b"artifact", [[1.0, 0.0], [2.0, 0.0]])])

    def test_feature_importances_reject_unsupported_method(self) -> None:
        original_train_loader = regressor_module._load_native_train_regression_artifact
        regressor_module._load_native_train_regression_artifact = lambda: (
            lambda *_args, **_kwargs: b"artifact"
        )
        try:
            model = GBMRegressor().fit([[1.0, 0.0], [2.0, 0.0]], [1.0, 2.0])
            with self.assertRaisesRegex(ValueError, "unsupported feature importance method"):
                model.feature_importances([[1.0, 0.0]], method="gain")
        finally:
            regressor_module._load_native_train_regression_artifact = (
                original_train_loader
            )

    def test_feature_importances_reject_feature_count_mismatch(self) -> None:
        original_train_loader = regressor_module._load_native_train_regression_artifact
        original_importance_loader = regressor_module._load_native_shap_global_importance
        regressor_module._load_native_train_regression_artifact = lambda: (
            lambda *_args, **_kwargs: b"artifact"
        )
        regressor_module._load_native_shap_global_importance = lambda: (
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError(
                    "native SHAP global-importance loader should not be called on shape mismatch"
                )
            )
        )
        try:
            model = GBMRegressor().fit([[1.0, 0.0], [2.0, 0.0]], [1.0, 2.0])
            with self.assertRaisesRegex(ValueError, "feature count"):
                model.feature_importances([[1.0]])
        finally:
            regressor_module._load_native_train_regression_artifact = (
                original_train_loader
            )
            regressor_module._load_native_shap_global_importance = (
                original_importance_loader
            )

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
        regressor_module._load_native_train_regression_artifact = lambda: fake_train
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

        self.assertEqual(
            train_calls,
            [
                ([[1.0, 0.0], [2.0, 0.0]], [1.0, 2.0]),
                ([[3.0, 0.0], [4.0, 0.0]], [3.0, 4.0]),
                ([[5.0, 0.0], [6.0, 0.0]], [5.0, 6.0]),
            ],
        )

    def test_predict_rejects_feature_count_mismatch(self) -> None:
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
