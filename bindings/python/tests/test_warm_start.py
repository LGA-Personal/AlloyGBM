"""Tests for warm-starting / incremental training."""

import struct
import tempfile
import unittest

import numpy as np
import pytest

from alloygbm import GBMRegressor


def _tree_records(artifact_bytes: bytes) -> list[bytes]:
    section_count = struct.unpack_from("<I", artifact_bytes, 8)[0]
    for section_index in range(section_count):
        descriptor_offset = 16 + section_index * 20
        kind, payload_offset, payload_length = struct.unpack_from(
            "<IQQ", artifact_bytes, descriptor_offset
        )
        if kind != 1:
            continue
        payload = artifact_bytes[payload_offset : payload_offset + payload_length]
        stump_count = struct.unpack_from("<I", payload, 8)[0]
        return [
            payload[16 + stump_index * 32 : 16 + (stump_index + 1) * 32]
            for stump_index in range(stump_count)
        ]
    raise AssertionError("artifact is missing its Trees section")


def _logical_tree_count(records: list[bytes]) -> int:
    tree_ids = {struct.unpack_from("<I", record)[0] // (1 << 20) for record in records}
    return len(tree_ids)


def _make_dataset(n=200, seed=42):
    """Produce a simple dataset for warm-start testing."""
    import random

    rng = random.Random(seed)
    X = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(n)]
    y = [row[0] * 2.0 + row[1] * 0.5 + rng.gauss(0, 0.1) for row in X]
    return X, y


def _make_monotone_dataset(n=384, seed=7):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2.0, 2.0, size=(n, 3)).astype(np.float32)
    y = (
        1.5 * X[:, 0]
        + 2.0 * np.sin(X[:, 1] * 2.3)
        + X[:, 0] * X[:, 2]
        + 0.2 * rng.standard_normal(n)
    ).astype(np.float32)
    grid = np.column_stack(
        [
            np.linspace(-2.0, 2.0, 101, dtype=np.float32),
            np.full(101, -1.5, dtype=np.float32),
            np.full(101, -1.5, dtype=np.float32),
        ]
    )
    return X, y, grid


def _assert_nondecreasing_predictions(model, grid):
    predictions = np.asarray(model.predict(grid))
    assert np.all(np.isfinite(predictions))
    assert np.all(np.diff(predictions) >= -1e-6)


class TestWarmStartParams(unittest.TestCase):
    """Test parameter handling for warm_start."""

    def test_default_warm_start_is_false(self):
        """Default warm_start should be False."""
        m = GBMRegressor(n_estimators=3)
        self.assertFalse(m.warm_start)

    def test_warm_start_accepted(self):
        """Constructor should accept warm_start=True."""
        m = GBMRegressor(n_estimators=3, warm_start=True)
        self.assertTrue(m.warm_start)

    def test_get_params_includes_warm_start(self):
        """get_params() should include warm_start."""
        m = GBMRegressor(n_estimators=3, warm_start=True)
        params = m.get_params()
        self.assertTrue(params["warm_start"])

    def test_set_params_warm_start(self):
        """set_params() should update warm_start."""
        m = GBMRegressor(n_estimators=3)
        m.set_params(warm_start=True)
        self.assertTrue(m.warm_start)

    def test_repr_includes_warm_start(self):
        """__repr__ should include warm_start."""
        m = GBMRegressor(n_estimators=3, warm_start=True)
        r = repr(m)
        self.assertIn("warm_start=True", r)

    def test_clone_preserves_warm_start(self):
        """get_params/set_params roundtrip should preserve warm_start."""
        m1 = GBMRegressor(n_estimators=3, warm_start=True)
        m2 = GBMRegressor(**m1.get_params())
        self.assertTrue(m2.warm_start)


class TestWarmStartTraining(unittest.TestCase):
    """Test warm-start training end-to-end."""

    def test_warm_start_basic(self):
        """warm_start=True should continue training from previous state."""
        X, y = _make_dataset()
        m = GBMRegressor(
            n_estimators=10,
            max_depth=4,
            training_policy="manual",
            seed=42,
            warm_start=True,
        )
        # First fit
        m.fit(X, y)
        preds_after_10 = m.predict(X[:5])
        self.assertEqual(len(preds_after_10), 5)
        self.assertEqual(m.n_estimators_, 10)

        # Warm-start with more rounds
        m.n_estimators = 20
        m.fit(X, y)
        preds_after_20 = m.predict(X[:5])
        self.assertEqual(len(preds_after_20), 5)
        # Should have more trees now
        self.assertIsNotNone(m.n_estimators_)

    def test_warm_start_improves_quality(self):
        """Warm-started model should fit better than the original."""
        X, y = _make_dataset(n=200)
        m = GBMRegressor(
            n_estimators=5,
            max_depth=4,
            training_policy="manual",
            seed=42,
            warm_start=True,
        )
        m.fit(X, y)
        preds_5 = m.predict(X)
        mse_5 = sum((p - t) ** 2 for p, t in zip(preds_5, y)) / len(y)

        # Continue with more rounds
        m.n_estimators = 30
        m.fit(X, y)
        preds_30 = m.predict(X)
        mse_30 = sum((p - t) ** 2 for p, t in zip(preds_30, y)) / len(y)

        # More rounds should reduce training error
        self.assertLess(mse_30, mse_5, "Warm-start should improve training MSE")

    def test_warm_start_false_resets(self):
        """When warm_start=False, fit() should start fresh each time."""
        X, y = _make_dataset()
        m = GBMRegressor(
            n_estimators=10,
            max_depth=4,
            training_policy="manual",
            seed=42,
            warm_start=False,
        )
        m.fit(X, y)
        preds1 = m.predict(X[:5])

        # Second fit should start fresh and produce same results
        m.fit(X, y)
        preds2 = m.predict(X[:5])

        for p1, p2 in zip(preds1, preds2):
            self.assertAlmostEqual(p1, p2, places=6)

    def test_init_model_basic(self):
        """init_model should continue training from another model."""
        X, y = _make_dataset()
        m1 = GBMRegressor(
            n_estimators=10,
            max_depth=4,
            training_policy="manual",
            seed=42,
        )
        m1.fit(X, y)

        m2 = GBMRegressor(
            n_estimators=10,
            max_depth=4,
            training_policy="manual",
            seed=42,
        )
        m2.fit(X, y, init_model=m1)
        preds = m2.predict(X[:5])
        self.assertEqual(len(preds), 5)

    def test_init_model_improves_quality(self):
        """init_model should allow continuing training to reduce error."""
        X, y = _make_dataset(n=200)
        m1 = GBMRegressor(
            n_estimators=5,
            max_depth=4,
            training_policy="manual",
            seed=42,
        )
        m1.fit(X, y)
        preds1 = m1.predict(X)
        mse1 = sum((p - t) ** 2 for p, t in zip(preds1, y)) / len(y)

        m2 = GBMRegressor(
            n_estimators=20,
            max_depth=4,
            training_policy="manual",
            seed=42,
        )
        m2.fit(X, y, init_model=m1)
        preds2 = m2.predict(X)
        mse2 = sum((p - t) ** 2 for p, t in zip(preds2, y)) / len(y)

        self.assertLess(mse2, mse1, "init_model continuation should reduce MSE")

    def test_init_model_takes_priority_over_warm_start(self):
        """init_model should take priority when warm_start=True."""
        X, y = _make_dataset()
        m_base = GBMRegressor(
            n_estimators=5,
            max_depth=4,
            training_policy="manual",
            seed=42,
        )
        m_base.fit(X, y)

        m = GBMRegressor(
            n_estimators=10,
            max_depth=4,
            training_policy="manual",
            seed=42,
            warm_start=True,
        )
        # First fit
        m.fit(X, y)
        # Warm-start with init_model (should use init_model, not self)
        m.fit(X, y, init_model=m_base)
        preds = m.predict(X[:5])
        self.assertEqual(len(preds), 5)

    def test_init_model_unfitted_raises(self):
        """init_model with unfitted model should raise ValueError."""
        X, y = _make_dataset()
        m = GBMRegressor(n_estimators=3, training_policy="manual")
        unfitted = GBMRegressor(n_estimators=3)
        with self.assertRaises(ValueError):
            m.fit(X, y, init_model=unfitted)

    def test_warm_start_with_validation(self):
        """Warm-start should work with eval_set."""
        X, y = _make_dataset(n=200)
        split = 150
        m = GBMRegressor(
            n_estimators=10,
            max_depth=4,
            training_policy="manual",
            seed=42,
            warm_start=True,
            early_stopping_rounds=5,
        )
        m.fit(X[:split], y[:split], eval_set=(X[split:], y[split:]))
        preds = m.predict(X[:5])
        self.assertEqual(len(preds), 5)

        # Continue training with more rounds
        m.n_estimators = 30
        m.fit(X[:split], y[:split], eval_set=(X[split:], y[split:]))
        preds2 = m.predict(X[:5])
        self.assertEqual(len(preds2), 5)

    def test_monotone_init_model_continuation(self):
        X, y, grid = _make_monotone_dataset()
        prior = GBMRegressor(
            n_estimators=5,
            max_depth=4,
            learning_rate=0.2,
            monotone_constraints=[1, 0, 0],
            training_policy="manual",
            seed=0,
        ).fit(X, y)
        continued = GBMRegressor(
            n_estimators=5,
            max_depth=4,
            learning_rate=0.2,
            monotone_constraints=[1, 0, 0],
            training_policy="manual",
            seed=0,
        ).fit(X, y, init_model=prior)

        _assert_nondecreasing_predictions(continued, grid)

    def test_monotone_rejects_dart_weighted_init_model_under_standard_mode(self):
        X, y, _ = _make_monotone_dataset()
        prior = GBMRegressor(
            n_estimators=20,
            max_depth=3,
            boosting_mode="dart",
            dart_drop_rate=0.3,
            dart_max_drop=5,
            training_policy="manual",
            seed=11,
        ).fit(X, y)
        continued = GBMRegressor(
            n_estimators=2,
            max_depth=3,
            boosting_mode="standard",
            monotone_constraints=[1, 0, 0],
            training_policy="manual",
            seed=11,
        )

        with pytest.raises(
            ValueError,
            match="warm_start.*DART tree weights.*monotone_constraints",
        ):
            continued.fit(X, y, init_model=prior)

    def test_monotone_rejects_dart_weighted_estimator_warm_start(self):
        X, y, _ = _make_monotone_dataset()
        model = GBMRegressor(
            n_estimators=20,
            max_depth=3,
            boosting_mode="dart",
            dart_drop_rate=0.3,
            dart_max_drop=5,
            training_policy="manual",
            seed=11,
            warm_start=True,
        ).fit(X, y)
        model.set_params(
            n_estimators=2,
            boosting_mode="standard",
            monotone_constraints=[1, 0, 0],
        )

        with pytest.raises(
            ValueError,
            match="warm_start.*DART tree weights.*monotone_constraints",
        ):
            model.fit(X, y)

    def test_monotone_estimator_warm_start_continuation(self):
        X, y, grid = _make_monotone_dataset()
        model = GBMRegressor(
            n_estimators=5,
            max_depth=4,
            learning_rate=0.2,
            monotone_constraints=[1, 0, 0],
            training_policy="manual",
            seed=0,
            warm_start=True,
        ).fit(X, y)
        prior_rounds = model.rounds_completed_
        prior_records = _tree_records(model.artifact_bytes)
        model.n_estimators = 5
        model.fit(X, y)
        continued_records = _tree_records(model.artifact_bytes)

        self.assertEqual(prior_rounds, 5)
        self.assertEqual(model.rounds_completed_, 5)
        self.assertEqual(_logical_tree_count(prior_records), prior_rounds)
        self.assertEqual(
            _logical_tree_count(continued_records),
            prior_rounds + model.rounds_completed_,
        )
        self.assertGreater(len(continued_records), len(prior_records))
        self.assertEqual(continued_records[: len(prior_records)], prior_records)
        _assert_nondecreasing_predictions(model, grid)

    def test_monotone_save_load_is_exact_before_continuation(self):
        X, y, grid = _make_monotone_dataset()
        prior = GBMRegressor(
            n_estimators=5,
            max_depth=4,
            learning_rate=0.2,
            monotone_constraints=[1, 0, 0],
            training_policy="manual",
            seed=0,
        ).fit(X, y)
        predictions_before = prior.predict(grid)

        with tempfile.NamedTemporaryFile(suffix=".agbm") as model_file:
            prior.save_model(model_file.name)
            loaded = GBMRegressor.load_model(model_file.name)
        np.testing.assert_array_equal(predictions_before, loaded.predict(grid))

        continued = GBMRegressor(
            n_estimators=3,
            max_depth=4,
            learning_rate=0.2,
            monotone_constraints=[1, 0, 0],
            training_policy="manual",
            seed=0,
        ).fit(X, y, init_model=loaded)
        _assert_nondecreasing_predictions(continued, grid)

    def test_monotone_rejects_violating_unconstrained_init_model(self):
        X, _, _ = _make_monotone_dataset()
        violating_targets = (-3.0 * X[:, 0]).astype(np.float32)
        prior = GBMRegressor(
            n_estimators=5,
            max_depth=3,
            learning_rate=0.5,
            training_policy="manual",
            seed=0,
        ).fit(X, violating_targets)
        constrained = GBMRegressor(
            n_estimators=2,
            max_depth=3,
            monotone_constraints=[1, 0, 0],
            training_policy="manual",
            seed=0,
        )

        with pytest.raises(
            ValueError, match="warm_start.*monotone_constraints"
        ):
            constrained.fit(X, violating_targets, init_model=prior)


class TestWarmStartEdgeCases(unittest.TestCase):
    """Edge cases for warm-start."""

    def test_warm_start_single_round_increment(self):
        """Warm-start should work with just 1 additional round."""
        X, y = _make_dataset(n=100)
        m = GBMRegressor(
            n_estimators=5,
            max_depth=4,
            training_policy="manual",
            seed=42,
            warm_start=True,
        )
        m.fit(X, y)
        m.n_estimators = 1
        m.fit(X, y)
        preds = m.predict(X[:3])
        self.assertEqual(len(preds), 3)

    def test_warm_start_preserves_predictions_format(self):
        """Predictions from warm-started model should be valid floats."""
        X, y = _make_dataset(n=100)
        m = GBMRegressor(
            n_estimators=5,
            max_depth=4,
            training_policy="manual",
            seed=42,
            warm_start=True,
        )
        m.fit(X, y)
        m.n_estimators = 10
        m.fit(X, y)
        preds = m.predict(X)
        self.assertIsInstance(preds, np.ndarray)
        self.assertTrue(np.issubdtype(preds.dtype, np.floating))
        self.assertTrue(np.all(np.abs(preds) < 100), "Predictions seem unreasonably large")


if __name__ == "__main__":
    unittest.main()
