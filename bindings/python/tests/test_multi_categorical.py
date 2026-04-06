"""Tests for multiple categorical column support."""

import unittest

from alloygbm import GBMRegressor


class _FakeCategoricalSeries:
    """Minimal Series-like that supports to_list()."""

    def __init__(self, values: list) -> None:
        self._values = values

    def to_list(self) -> list:
        return self._values


class _FakeCategoricalFrame:
    """Minimal DataFrame-like object that exposes .dtypes, .columns, to_numpy()."""

    def __init__(
        self,
        rows: list[list],
        columns: list[str],
        dtypes: list[str],
    ) -> None:
        self._rows = rows
        self.columns = columns
        self.dtypes = dtypes
        self._column_map = {
            column: [row[index] for row in rows]
            for index, column in enumerate(columns)
        }

    def to_numpy(self) -> list[list]:
        return self._rows

    def __getitem__(self, key) -> _FakeCategoricalSeries:
        return _FakeCategoricalSeries(self._column_map[str(key)])


def _make_multi_cat_dataset(n=100, seed=42):
    """Dataset with two categorical features and one numeric feature."""
    import random

    rng = random.Random(seed)
    categories_a = ["cat", "dog", "bird"]
    categories_b = ["red", "blue", "green", "yellow"]
    X = []
    y = []
    for _ in range(n):
        cat_a = rng.choice(categories_a)
        cat_b = rng.choice(categories_b)
        numeric = rng.gauss(0, 1)
        X.append([cat_a, cat_b, numeric])
        # Target depends on categories and numeric feature
        base = {"cat": 1.0, "dog": 2.0, "bird": 3.0}[cat_a]
        color_effect = {"red": 0.5, "blue": -0.5, "green": 1.0, "yellow": -1.0}[cat_b]
        y.append(base + color_effect + 0.3 * numeric + rng.gauss(0, 0.1))
    return X, y


class TestMultiCategoricalParams(unittest.TestCase):
    """Test parameter handling for multi-categorical."""

    def test_categorical_feature_indices_accepted(self):
        """Constructor should accept categorical_feature_indices."""
        m = GBMRegressor(n_estimators=3, categorical_feature_indices=[0, 1])
        self.assertEqual(m.categorical_feature_indices, [0, 1])

    def test_default_is_none(self):
        """Default categorical_feature_indices should be None."""
        m = GBMRegressor(n_estimators=3)
        self.assertIsNone(m.categorical_feature_indices)

    def test_get_params_includes_categorical_feature_indices(self):
        """get_params() should include categorical_feature_indices."""
        m = GBMRegressor(n_estimators=3, categorical_feature_indices=[0, 2])
        params = m.get_params()
        self.assertEqual(params["categorical_feature_indices"], [0, 2])

    def test_set_params_categorical_feature_indices(self):
        """set_params() should update categorical_feature_indices."""
        m = GBMRegressor(n_estimators=3)
        m.set_params(categorical_feature_indices=[1, 3])
        self.assertEqual(m.categorical_feature_indices, [1, 3])

    def test_set_params_none_clears(self):
        """set_params(categorical_feature_indices=None) should clear."""
        m = GBMRegressor(n_estimators=3, categorical_feature_indices=[0])
        m.set_params(categorical_feature_indices=None)
        self.assertIsNone(m.categorical_feature_indices)

    def test_repr_includes_categorical_feature_indices(self):
        """__repr__ should include categorical_feature_indices."""
        m = GBMRegressor(n_estimators=3, categorical_feature_indices=[0, 1])
        r = repr(m)
        self.assertIn("categorical_feature_indices=[0, 1]", r)

    def test_clone_preserves_categorical_feature_indices(self):
        """get_params/set_params roundtrip should preserve."""
        m1 = GBMRegressor(n_estimators=3, categorical_feature_indices=[2, 5])
        m2 = GBMRegressor(**m1.get_params())
        self.assertEqual(m2.categorical_feature_indices, [2, 5])


class TestMultiCategoricalValidation(unittest.TestCase):
    """Test validation rules for multi-categorical parameters."""

    def test_mutual_exclusion_with_singular(self):
        """Setting both singular and plural should raise."""
        with self.assertRaises(ValueError):
            GBMRegressor(
                n_estimators=3,
                categorical_feature_index=0,
                categorical_feature_indices=[0, 1],
            )

    def test_set_params_mutual_exclusion(self):
        """set_params with both singular and plural should raise."""
        m = GBMRegressor(n_estimators=3, categorical_feature_index=0)
        with self.assertRaises(ValueError):
            m.set_params(categorical_feature_indices=[0, 1])

    def test_negative_index_raises(self):
        """Negative indices should raise ValueError."""
        with self.assertRaises(ValueError):
            GBMRegressor(n_estimators=3, categorical_feature_indices=[-1, 0])

    def test_duplicate_indices_raises(self):
        """Duplicate indices should raise ValueError."""
        with self.assertRaises(ValueError):
            GBMRegressor(n_estimators=3, categorical_feature_indices=[0, 0])

    def test_non_list_raises(self):
        """Non-list type should raise TypeError."""
        with self.assertRaises(TypeError):
            GBMRegressor(n_estimators=3, categorical_feature_indices="0,1")

    def test_fit_values_list_and_values_mutual_exclusion(self):
        """categorical_feature_values and categorical_feature_values_list are mutually exclusive."""
        m = GBMRegressor(
            n_estimators=3,
            categorical_feature_indices=[0, 1],
        )
        with self.assertRaises(ValueError):
            m.fit(
                [[0.0, 0.0, 1.0]],
                [1.0],
                categorical_feature_values=["A"],
                categorical_feature_values_list=[["A"], ["B"]],
            )


class TestMultiCategoricalTraining(unittest.TestCase):
    """Test multi-categorical training end-to-end."""

    def test_explicit_indices_with_list_data(self):
        """Train with explicit indices and list-of-lists data."""
        X, y = _make_multi_cat_dataset(n=60)
        m = GBMRegressor(
            n_estimators=10,
            max_depth=4,
            categorical_feature_indices=[0, 1],
            training_policy="manual",
        )
        # Extract categorical values for each column
        cat_values_0 = [str(row[0]) for row in X]
        cat_values_1 = [str(row[1]) for row in X]
        # Convert categorical columns to 0.0 for float matrix
        X_numeric = [[0.0, 0.0, row[2]] for row in X]
        m.fit(
            X_numeric,
            y,
            categorical_feature_values_list=[cat_values_0, cat_values_1],
        )
        preds = m.predict(X_numeric[:5])
        self.assertEqual(len(preds), 5)
        for p in preds:
            self.assertIsInstance(p, float)

    def test_auto_inferred_from_dataframe(self):
        """Train with DataFrame-like that has multiple categorical columns."""
        X_raw, y = _make_multi_cat_dataset(n=60)
        frame = _FakeCategoricalFrame(
            rows=X_raw,
            columns=["species", "color", "size"],
            dtypes=["category", "category", "float64"],
        )
        m = GBMRegressor(
            n_estimators=10,
            max_depth=4,
            training_policy="manual",
        )
        m.fit(frame, y)
        preds = m.predict([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        self.assertEqual(len(preds), 2)

    def test_quality_multi_categorical(self):
        """Model with multiple categorical columns should learn meaningful patterns."""
        X_raw, y = _make_multi_cat_dataset(n=200, seed=99)
        cat_values_0 = [str(row[0]) for row in X_raw]
        cat_values_1 = [str(row[1]) for row in X_raw]
        X_numeric = [[0.0, 0.0, row[2]] for row in X_raw]
        m = GBMRegressor(
            n_estimators=50,
            max_depth=4,
            categorical_feature_indices=[0, 1],
            training_policy="manual",
            seed=42,
        )
        m.fit(
            X_numeric,
            y,
            categorical_feature_values_list=[cat_values_0, cat_values_1],
        )
        preds = m.predict(X_numeric)
        mse = sum((p - t) ** 2 for p, t in zip(preds, y)) / len(y)
        rmse = mse**0.5
        self.assertLess(rmse, 3.0, f"Multi-categorical RMSE {rmse:.4f} too high")

    def test_with_validation_set(self):
        """Multi-categorical should work with eval_set."""
        X_raw, y = _make_multi_cat_dataset(n=100, seed=42)
        split = 70
        cat_values_0 = [str(row[0]) for row in X_raw]
        cat_values_1 = [str(row[1]) for row in X_raw]
        X_numeric = [[0.0, 0.0, row[2]] for row in X_raw]

        m = GBMRegressor(
            n_estimators=30,
            max_depth=4,
            categorical_feature_indices=[0, 1],
            training_policy="manual",
            early_stopping_rounds=5,
        )
        # Pass raw X (with string columns) for eval_set so categorical
        # values can be extracted automatically.
        m.fit(
            X_numeric[:split],
            y[:split],
            categorical_feature_values_list=[
                cat_values_0[:split],
                cat_values_1[:split],
            ],
            eval_set=(X_raw[split:], y[split:]),
        )
        preds = m.predict(X_numeric[:5])
        self.assertEqual(len(preds), 5)

    def test_singular_backward_compat(self):
        """Singular categorical_feature_index should still work."""
        import random

        rng = random.Random(42)
        categories = ["cat", "dog", "bird"]
        n = 60
        X_numeric = []
        cat_values = []
        y = []
        for _ in range(n):
            cat = rng.choice(categories)
            numeric_1 = rng.gauss(0, 1)
            numeric_2 = rng.gauss(0, 1)
            X_numeric.append([0.0, numeric_1, numeric_2])
            cat_values.append(cat)
            y.append({"cat": 1.0, "dog": 2.0, "bird": 3.0}[cat] + numeric_1 * 0.5)
        m = GBMRegressor(
            n_estimators=5,
            max_depth=4,
            categorical_feature_index=0,
            training_policy="manual",
        )
        m.fit(
            X_numeric,
            y,
            categorical_feature_values=cat_values,
        )
        preds = m.predict(X_numeric[:3])
        self.assertEqual(len(preds), 3)


class TestMultiCategoricalEdgeCases(unittest.TestCase):
    """Edge cases for multi-categorical."""

    def test_single_index_in_indices_list(self):
        """categorical_feature_indices=[0] should work like categorical_feature_index=0."""
        X_raw, y = _make_multi_cat_dataset(n=60)
        cat_values = [str(row[0]) for row in X_raw]
        X_numeric = [[0.0, float(hash(row[1]) % 10), row[2]] for row in X_raw]
        m = GBMRegressor(
            n_estimators=5,
            max_depth=4,
            categorical_feature_indices=[0],
            training_policy="manual",
        )
        m.fit(
            X_numeric,
            y,
            categorical_feature_values_list=[cat_values],
        )
        preds = m.predict(X_numeric[:3])
        self.assertEqual(len(preds), 3)

    def test_non_contiguous_indices(self):
        """categorical_feature_indices with non-contiguous indices should work."""
        import random

        rng = random.Random(42)
        n = 60
        X = []
        y = []
        categories = ["A", "B", "C"]
        for _ in range(n):
            cat_0 = rng.choice(categories)
            numeric_1 = rng.gauss(0, 1)
            numeric_2 = rng.gauss(0, 1)
            cat_3 = rng.choice(categories)
            X.append([cat_0, numeric_1, numeric_2, cat_3])
            y.append({"A": 1.0, "B": 2.0, "C": 3.0}[cat_0] + numeric_1 * 0.5)

        cat_values_0 = [str(row[0]) for row in X]
        cat_values_3 = [str(row[3]) for row in X]
        X_numeric = [[0.0, row[1], row[2], 0.0] for row in X]

        m = GBMRegressor(
            n_estimators=5,
            max_depth=4,
            categorical_feature_indices=[0, 3],
            training_policy="manual",
        )
        m.fit(
            X_numeric,
            y,
            categorical_feature_values_list=[cat_values_0, cat_values_3],
        )
        preds = m.predict(X_numeric[:3])
        self.assertEqual(len(preds), 3)

    def test_empty_indices_list_treated_as_none(self):
        """Empty categorical_feature_indices should behave like None."""
        # Empty list should work the same as no categorical features
        m = GBMRegressor(
            n_estimators=3,
            categorical_feature_indices=[],
            training_policy="manual",
        )
        # categorical_feature_indices=[] is stored but treated as "no categoricals"
        # during fit (just like None)
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        y = [1.0, 2.0, 3.0]
        m.fit(X, y)
        preds = m.predict(X)
        self.assertEqual(len(preds), 3)


if __name__ == "__main__":
    unittest.main()
