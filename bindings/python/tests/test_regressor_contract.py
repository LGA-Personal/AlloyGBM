"""Focused contract tests for the v0.0.3 Python regressor baseline."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def load_regressor_class() -> type:
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
    return module.GBMRegressor


GBMRegressor = load_regressor_class()


class GBMRegressorContractTests(unittest.TestCase):
    def test_constructor_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            GBMRegressor(learning_rate=0.0)
        with self.assertRaisesRegex(ValueError, "max_depth"):
            GBMRegressor(max_depth=0)

    def test_get_params_and_set_params_roundtrip(self) -> None:
        model = GBMRegressor()
        params = model.get_params()
        self.assertEqual(params["learning_rate"], 0.1)
        self.assertEqual(params["max_depth"], 6)
        self.assertEqual(params["seed"], 0)
        self.assertTrue(params["deterministic"])

        updated = model.set_params(
            learning_rate=0.2, max_depth=4, seed=7, deterministic=False
        )
        self.assertIs(updated, model)
        self.assertEqual(model.get_params()["learning_rate"], 0.2)
        self.assertEqual(model.get_params()["max_depth"], 4)
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

    def test_fit_and_predict_constant_baseline(self) -> None:
        model = GBMRegressor()
        fitted = model.fit([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], [2.0, 4.0, 6.0])
        self.assertIs(fitted, model)
        predictions = model.predict([[10.0, 0.0], [20.0, 0.0]])
        self.assertEqual(len(predictions), 2)
        self.assertAlmostEqual(predictions[0], 4.0)
        self.assertAlmostEqual(predictions[1], 4.0)

    def test_fit_rejects_mismatched_lengths(self) -> None:
        model = GBMRegressor()
        with self.assertRaisesRegex(ValueError, "same number of rows"):
            model.fit([[1.0], [2.0]], [1.0])

    def test_predict_rejects_feature_count_mismatch(self) -> None:
        model = GBMRegressor().fit([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "feature count"):
            model.predict([[1.0]])


if __name__ == "__main__":
    unittest.main()
