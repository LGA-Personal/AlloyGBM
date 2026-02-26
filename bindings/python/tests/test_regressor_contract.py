"""Focused contract tests for the AlloyGBM Python regressor baseline."""

from __future__ import annotations

import importlib.util
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
        with self.assertRaisesRegex(ValueError, "row_subsample"):
            GBMRegressor(row_subsample=0.0)
        with self.assertRaisesRegex(ValueError, "col_subsample"):
            GBMRegressor(col_subsample=1.5)
        with self.assertRaisesRegex(ValueError, "early_stopping_rounds"):
            GBMRegressor(early_stopping_rounds=0)
        with self.assertRaisesRegex(ValueError, "min_validation_improvement"):
            GBMRegressor(min_validation_improvement=-0.1)

    def test_get_params_and_set_params_roundtrip(self) -> None:
        model = GBMRegressor()
        params = model.get_params()
        self.assertEqual(params["learning_rate"], 0.1)
        self.assertEqual(params["max_depth"], 6)
        self.assertEqual(params["row_subsample"], 1.0)
        self.assertEqual(params["col_subsample"], 1.0)
        self.assertIsNone(params["early_stopping_rounds"])
        self.assertEqual(params["min_validation_improvement"], 0.0)
        self.assertEqual(params["seed"], 0)
        self.assertTrue(params["deterministic"])

        updated = model.set_params(
            learning_rate=0.2,
            max_depth=4,
            row_subsample=0.75,
            col_subsample=0.5,
            early_stopping_rounds=4,
            min_validation_improvement=0.01,
            seed=7,
            deterministic=False,
        )
        self.assertIs(updated, model)
        self.assertEqual(model.get_params()["learning_rate"], 0.2)
        self.assertEqual(model.get_params()["max_depth"], 4)
        self.assertEqual(model.get_params()["row_subsample"], 0.75)
        self.assertEqual(model.get_params()["col_subsample"], 0.5)
        self.assertEqual(model.get_params()["early_stopping_rounds"], 4)
        self.assertEqual(model.get_params()["min_validation_improvement"], 0.01)
        self.assertEqual(model.get_params()["seed"], 7)
        self.assertFalse(model.get_params()["deterministic"])

    def test_set_params_rejects_unknown_parameter(self) -> None:
        model = GBMRegressor()
        with self.assertRaisesRegex(ValueError, "Unknown parameter"):
            model.set_params(unknown=1)  # type: ignore[arg-type]

    def test_predict_requires_fit(self) -> None:
        model = GBMRegressor()
        with self.assertRaisesRegex(RuntimeError, "must be fit"):
            model.predict([])

    def test_fit_and_predict_use_native_bridges(self) -> None:
        train_calls: list[tuple[object, ...]] = []
        predict_calls: list[tuple[bytes, list[list[float]]]] = []

        def fake_train(
            rows: list[list[float]],
            targets: list[float],
            learning_rate: float,
            max_depth: int,
            row_subsample: float,
            col_subsample: float,
            early_stopping_rounds: int | None,
            min_validation_improvement: float,
            seed: int,
            deterministic: bool,
        ) -> bytes:
            train_calls.append(
                (
                    rows,
                    targets,
                    learning_rate,
                    max_depth,
                    row_subsample,
                    col_subsample,
                    early_stopping_rounds,
                    min_validation_improvement,
                    seed,
                    deterministic,
                )
            )
            return b"trained-artifact"

        def fake_predictor(
            artifact_bytes: bytes, rows: list[list[float]]
        ) -> list[float]:
            predict_calls.append((artifact_bytes, rows))
            return [0.5 + row[0] for row in rows]

        original_train_loader = regressor_module._load_native_train_regression_artifact
        original_predict_loader = regressor_module._load_native_predictor_predict_batch
        regressor_module._load_native_train_regression_artifact = lambda: fake_train
        regressor_module._load_native_predictor_predict_batch = lambda: fake_predictor
        try:
            model = GBMRegressor(
                learning_rate=0.2,
                max_depth=4,
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
            regressor_module._load_native_predictor_predict_batch = (
                original_predict_loader
            )

        self.assertIs(fitted, model)
        self.assertEqual(predictions, [1.5, 2.5])
        self.assertEqual(
            train_calls,
            [
                (
                    [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
                    [2.0, 4.0, 6.0],
                    0.2,
                    4,
                    0.75,
                    0.5,
                    4,
                    0.01,
                    7,
                    False,
                )
            ],
        )
        self.assertEqual(
            predict_calls,
            [(b"trained-artifact", [[1.0, 0.0], [2.0, 0.0]])],
        )

    def test_fit_rejects_mismatched_lengths(self) -> None:
        model = GBMRegressor()
        with self.assertRaisesRegex(ValueError, "same number of rows"):
            model.fit([[1.0], [2.0]], [1.0])

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
        original_predict_loader = regressor_module._load_native_predictor_predict_batch
        regressor_module._load_native_train_regression_artifact = lambda: (
            lambda *_args, **_kwargs: b"artifact"
        )
        regressor_module._load_native_predictor_predict_batch = lambda: fake_predictor
        try:
            model = GBMRegressor().fit([[1.0, 0.0], [2.0, 0.0]], [1.0, 2.0])
            predictions = model.predict(_FakePandasLikeFrame([[3.0, 0.0], [4.0, 0.0]]))
        finally:
            regressor_module._load_native_train_regression_artifact = (
                original_train_loader
            )
            regressor_module._load_native_predictor_predict_batch = (
                original_predict_loader
            )

        self.assertEqual(predictions, [0.0, 0.0])
        self.assertEqual(predict_calls, [[[3.0, 0.0], [4.0, 0.0]]])

    def test_predict_from_artifact_rejects_non_bytes_payload(self) -> None:
        with self.assertRaisesRegex(TypeError, "artifact_bytes"):
            GBMRegressor.predict_from_artifact("artifact", [[1.0]])  # type: ignore[arg-type]

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
