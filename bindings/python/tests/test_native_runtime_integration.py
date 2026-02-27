"""Python runtime integration checks for the native alloygbm extension."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

FIXTURE_ARTIFACT_HEX = (
    "4147424d010000000200000047000000010000007f00000000000000d000000000000000"
    "020000004f010000000000000c000000000000007b22666f726d61745f76657273696f6e"
    "223a312c22666561747572655f6e616d6573223a5b226630222c226631225d2c22747261"
    "696e65645f646576696365223a22637075227d0100000002000000060000000000000000"
    "000000000000000200000097999941979919bfea51b83e03000000050000000100000000"
    "00000001000000e8ffbf3ffaff3fbf909999be0200000001000000020000000000000005"
    "0000003a44b440c9cccc3dfaff3f3f03000000020000000000100000000000040000009e"
    "1dbc401afb4bbee6fba93e0500000003000000010010000000000000000000188d9b3f9a"
    "70fdbe8c4100be010000000400000002001000000000000600000040a06b3fe1a55b3ee3"
    "26113f0200000001000000010000000200000001000000"
)
FIXTURE_ARTIFACT_BYTES = bytes.fromhex(FIXTURE_ARTIFACT_HEX)
FIXTURE_ROWS = [[1.0, 0.0], [6.0, 0.0], [3.5, 0.0]]
FIXTURE_PREDICTIONS = [-1.674449, 1.656500, 0.135550]
FIT_ROWS = [
    [0.0, 0.0],
    [1.0, 0.0],
    [2.0, 0.0],
    [3.0, 0.0],
    [4.0, 0.0],
    [5.0, 0.0],
    [6.0, 0.0],
    [7.0, 0.0],
]
FIT_TARGETS = [-3.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0]


class _RuntimeNumpyLike:
    def __init__(self, values: object) -> None:
        self._values = values

    def tolist(self) -> object:
        return self._values


class _RuntimePandasLikeFrame:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def to_numpy(self) -> _RuntimeNumpyLike:
        return _RuntimeNumpyLike(self._rows)


class _RuntimePolarsLikeSeries:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def to_list(self) -> list[float]:
        return self._values


class NativeRuntimeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._repo_root = Path(__file__).resolve().parents[3]
        cls._tempdir = tempfile.TemporaryDirectory(prefix="alloygbm-runtime-")
        temp_root = Path(cls._tempdir.name)
        wheel_out = temp_root / "wheelhouse"
        install_target = temp_root / "site-packages"
        wheel_out.mkdir(parents=True, exist_ok=True)
        install_target.mkdir(parents=True, exist_ok=True)

        common_env = dict(os.environ)
        common_env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

        subprocess.run(
            [
                sys.executable,
                "-m",
                "maturin",
                "build",
                "--manifest-path",
                str(cls._repo_root / "bindings/python/Cargo.toml"),
                "--interpreter",
                sys.executable,
                "--out",
                str(wheel_out),
                "-q",
            ],
            cwd=cls._repo_root,
            env=common_env,
            check=True,
        )

        wheels = sorted(wheel_out.glob("alloygbm-*.whl"))
        if not wheels:
            raise RuntimeError("maturin build did not produce an alloygbm wheel")
        wheel = wheels[-1]

        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-cache-dir",
                "--target",
                str(install_target),
                str(wheel),
            ],
            cwd=cls._repo_root,
            env=common_env,
            check=True,
            stdout=subprocess.DEVNULL,
        )

        cls._install_path = str(install_target)
        sys.path.insert(0, cls._install_path)

        for module_name in ("alloygbm", "alloygbm._alloygbm", "alloygbm.regressor"):
            sys.modules.pop(module_name, None)

        import alloygbm

        cls.alloygbm = alloygbm

    @classmethod
    def tearDownClass(cls) -> None:
        for module_name in ("alloygbm", "alloygbm._alloygbm", "alloygbm.regressor"):
            sys.modules.pop(module_name, None)

        if hasattr(cls, "_install_path") and cls._install_path in sys.path:
            sys.path.remove(cls._install_path)

        if hasattr(cls, "_tempdir"):
            cls._tempdir.cleanup()

    def test_runtime_import_exposes_native_runtime_info(self) -> None:
        info = self.alloygbm.native_runtime_info()
        self.assertEqual(info.name, "alloygbm")
        self.assertRegex(info.version, re.compile(r"^\d+\.\d+\.\d+"))

    def test_runtime_import_exposes_metric_helpers(self) -> None:
        self.assertTrue(callable(self.alloygbm.rmse))
        self.assertTrue(callable(self.alloygbm.mae))
        self.assertTrue(callable(self.alloygbm.r2_score))
        self.assertTrue(callable(self.alloygbm.pearson_correlation))
        self.assertTrue(callable(self.alloygbm.rank_ic))
        self.assertTrue(callable(self.alloygbm.hit_rate))
        self.assertTrue(callable(self.alloygbm.icir))

        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        self.assertEqual(self.alloygbm.rmse(y_true, y_pred), 0.0)
        self.assertEqual(self.alloygbm.mae(y_true, y_pred), 0.0)
        self.assertEqual(self.alloygbm.r2_score(y_true, y_pred), 1.0)
        self.assertEqual(self.alloygbm.pearson_correlation(y_true, y_pred), 1.0)
        self.assertEqual(self.alloygbm.rank_ic(y_true, y_pred), 1.0)
        self.assertEqual(self.alloygbm.hit_rate(y_true, y_pred), 1.0)
        self.assertAlmostEqual(
            self.alloygbm.icir([0.1, 0.2, 0.0, -0.1]), 0.44721359549995787, places=12
        )

    def test_runtime_native_predictor_entrypoint_executes(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "serialization|artifact|header"):
            self.alloygbm._alloygbm.predictor_predict_batch(
                b"invalid-artifact", [[1.0, 0.0]]
            )

    def test_runtime_train_bridge_rejects_zero_rounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "rounds"):
            self.alloygbm._alloygbm.train_regression_artifact(
                rows=FIT_ROWS,
                targets=FIT_TARGETS,
                learning_rate=0.3,
                max_depth=2,
                row_subsample=1.0,
                col_subsample=1.0,
                min_validation_improvement=0.0,
                seed=7,
                deterministic=True,
                rounds=0,
                early_stopping_rounds=None,
            )

    def test_public_regressor_bridge_uses_native_extension_runtime(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "serialization|artifact|header"):
            self.alloygbm.GBMRegressor.predict_from_artifact(
                b"invalid-artifact", [[1.0, 0.0]]
            )

    def test_runtime_native_predictor_entrypoint_returns_expected_values(self) -> None:
        predictions = list(
            self.alloygbm._alloygbm.predictor_predict_batch(
                FIXTURE_ARTIFACT_BYTES, FIXTURE_ROWS
            )
        )
        self.assertEqual(len(predictions), len(FIXTURE_PREDICTIONS))
        for actual, expected in zip(predictions, FIXTURE_PREDICTIONS):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_public_regressor_bridge_matches_native_success_path(self) -> None:
        native_predictions = list(
            self.alloygbm._alloygbm.predictor_predict_batch(
                FIXTURE_ARTIFACT_BYTES, FIXTURE_ROWS
            )
        )
        bridge_predictions = self.alloygbm.GBMRegressor.predict_from_artifact(
            FIXTURE_ARTIFACT_BYTES, FIXTURE_ROWS
        )
        self.assertEqual(len(bridge_predictions), len(native_predictions))
        for bridge_value, native_value in zip(bridge_predictions, native_predictions):
            self.assertAlmostEqual(bridge_value, native_value, places=6)

    def test_public_regressor_fit_predict_is_native_backed_and_deterministic(self) -> None:
        model_a = self.alloygbm.GBMRegressor(
            learning_rate=0.3,
            max_depth=2,
            n_estimators=6,
            row_subsample=1.0,
            col_subsample=1.0,
            early_stopping_rounds=None,
            min_validation_improvement=0.0,
            seed=7,
            deterministic=True,
        )
        fitted = model_a.fit(FIT_ROWS, FIT_TARGETS)
        self.assertIs(fitted, model_a)
        predictions_a = model_a.predict([[1.0, 0.0], [6.0, 0.0], [3.0, 0.0]])

        model_b = self.alloygbm.GBMRegressor(
            learning_rate=0.3,
            max_depth=2,
            n_estimators=6,
            row_subsample=1.0,
            col_subsample=1.0,
            early_stopping_rounds=None,
            min_validation_improvement=0.0,
            seed=7,
            deterministic=True,
        )
        predictions_b = model_b.fit(FIT_ROWS, FIT_TARGETS).predict(
            [[1.0, 0.0], [6.0, 0.0], [3.0, 0.0]]
        )

        self.assertEqual(len(predictions_a), 3)
        self.assertGreater(len({round(value, 6) for value in predictions_a}), 1)
        for value_a, value_b in zip(predictions_a, predictions_b):
            self.assertAlmostEqual(value_a, value_b, places=6)

    def test_public_regressor_n_estimators_controls_training_rounds(self) -> None:
        model_short = self.alloygbm.GBMRegressor(
            learning_rate=0.3,
            max_depth=2,
            n_estimators=1,
            row_subsample=1.0,
            col_subsample=1.0,
            early_stopping_rounds=None,
            min_validation_improvement=0.0,
            seed=7,
            deterministic=True,
        )
        model_long = self.alloygbm.GBMRegressor(
            learning_rate=0.3,
            max_depth=2,
            n_estimators=8,
            row_subsample=1.0,
            col_subsample=1.0,
            early_stopping_rounds=None,
            min_validation_improvement=0.0,
            seed=7,
            deterministic=True,
        )

        predictions_short = model_short.fit(FIT_ROWS, FIT_TARGETS).predict(
            [[1.0, 0.0], [6.0, 0.0], [3.0, 0.0]]
        )
        predictions_long = model_long.fit(FIT_ROWS, FIT_TARGETS).predict(
            [[1.0, 0.0], [6.0, 0.0], [3.0, 0.0]]
        )

        self.assertNotEqual(model_short._artifact_bytes, model_long._artifact_bytes)
        self.assertTrue(
            any(
                abs(short_value - long_value) > 1e-6
                for short_value, long_value in zip(predictions_short, predictions_long)
            )
        )

    def test_public_regressor_accepts_dataframe_like_adapters(self) -> None:
        model = self.alloygbm.GBMRegressor(
            learning_rate=0.3,
            max_depth=2,
            n_estimators=6,
            row_subsample=1.0,
            col_subsample=1.0,
            early_stopping_rounds=None,
            min_validation_improvement=0.0,
            seed=7,
            deterministic=True,
        )
        model.fit(_RuntimePandasLikeFrame(FIT_ROWS), _RuntimePolarsLikeSeries(FIT_TARGETS))
        predictions = model.predict(_RuntimePandasLikeFrame([[1.0, 0.0], [6.0, 0.0]]))
        self.assertEqual(len(predictions), 2)
        self.assertNotAlmostEqual(predictions[0], predictions[1], places=6)

        native_predictions = list(
            self.alloygbm._alloygbm.predictor_predict_batch(
                FIXTURE_ARTIFACT_BYTES, FIXTURE_ROWS
            )
        )
        adapter_predictions = self.alloygbm.GBMRegressor.predict_from_artifact(
            FIXTURE_ARTIFACT_BYTES, _RuntimePandasLikeFrame(FIXTURE_ROWS)
        )
        self.assertEqual(len(adapter_predictions), len(native_predictions))
        for adapter_value, native_value in zip(adapter_predictions, native_predictions):
            self.assertAlmostEqual(adapter_value, native_value, places=6)


if __name__ == "__main__":
    unittest.main()
