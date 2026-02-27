"""Deterministic tests for leakage-aware validation split helpers."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def load_validation_module():
    validation_path = (
        Path(__file__).resolve().parents[1] / "alloygbm" / "validation.py"
    )
    spec = importlib.util.spec_from_file_location("alloygbm_validation", validation_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load alloygbm validation module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validation_module = load_validation_module()
purged_panel_splits = validation_module.purged_panel_splits
purged_time_series_splits = validation_module.purged_time_series_splits


class _FakeNumpyLike:
    def __init__(self, values: object) -> None:
        self._values = values

    def tolist(self) -> object:
        return self._values


class _FakePandasLikeSeries:
    def __init__(self, values: list[object]) -> None:
        self._values = values

    def to_numpy(self) -> _FakeNumpyLike:
        return _FakeNumpyLike(self._values)


class ValidationSplitTests(unittest.TestCase):
    def test_time_series_splits_are_deterministic(self) -> None:
        time_index = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        first = purged_time_series_splits(
            time_index, n_splits=3, purge_gap=1, embargo=1
        )
        second = purged_time_series_splits(
            time_index, n_splits=3, purge_gap=1, embargo=1
        )
        self.assertEqual(first, second)

    def test_time_series_splits_enforce_no_overlap_and_windows(self) -> None:
        time_index = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        splits = purged_time_series_splits(time_index, n_splits=3, purge_gap=1, embargo=1)
        unique_periods = sorted(set(time_index))
        period_to_position = {period: position for position, period in enumerate(unique_periods)}

        for train_indices, test_indices in splits:
            self.assertTrue(train_indices)
            self.assertTrue(test_indices)
            self.assertTrue(set(train_indices).isdisjoint(test_indices))

            test_positions = [period_to_position[time_index[index]] for index in test_indices]
            min_test_position = min(test_positions)
            max_test_position = max(test_positions)

            for train_index in train_indices:
                train_position = period_to_position[time_index[train_index]]
                self.assertTrue(
                    train_position < (min_test_position - 1)
                    or train_position > (max_test_position + 1)
                )

    def test_panel_splits_use_time_buckets_across_groups(self) -> None:
        time_index = [1, 1, 2, 2, 3, 3, 4, 4]
        group_index = ["A", "B", "A", "B", "A", "B", "A", "B"]

        panel_splits = purged_panel_splits(
            time_index,
            group_index,
            n_splits=2,
            purge_gap=0,
            embargo=0,
        )
        time_only_splits = purged_time_series_splits(
            time_index,
            n_splits=2,
            purge_gap=0,
            embargo=0,
        )
        self.assertEqual(panel_splits, time_only_splits)

        for _train_indices, test_indices in panel_splits:
            test_times = sorted({time_index[index] for index in test_indices})
            for period in test_times:
                period_groups = {
                    group_index[index] for index in test_indices if time_index[index] == period
                }
                self.assertEqual(period_groups, {"A", "B"})

    def test_split_helpers_accept_sequence_like_adapters(self) -> None:
        time_series = _FakePandasLikeSeries([0, 1, 2, 3, 4, 5])
        group_series = _FakeNumpyLike(["A", "B", "A", "B", "A", "B"])

        splits = purged_time_series_splits(time_series, n_splits=3)
        self.assertEqual(len(splits), 3)

        panel_splits = purged_panel_splits(time_series, group_series, n_splits=3)
        self.assertEqual(len(panel_splits), 3)

    def test_invalid_split_parameters_raise_value_error(self) -> None:
        time_index = [0, 1, 2, 3]

        with self.assertRaisesRegex(ValueError, "n_splits"):
            purged_time_series_splits(time_index, n_splits=1)
        with self.assertRaisesRegex(ValueError, "n_splits"):
            purged_time_series_splits(time_index, n_splits=10)
        with self.assertRaisesRegex(ValueError, "purge_gap"):
            purged_time_series_splits(time_index, purge_gap=-1)
        with self.assertRaisesRegex(ValueError, "embargo"):
            purged_time_series_splits(time_index, embargo=-1)

    def test_invalid_data_shapes_raise_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one value"):
            purged_time_series_splits([], n_splits=2)

        with self.assertRaisesRegex(ValueError, "same number of rows"):
            purged_panel_splits([1, 2, 3], ["A", "B"], n_splits=2)

        with self.assertRaisesRegex(ValueError, "orderable"):
            purged_time_series_splits([1, "2"], n_splits=2)

    def test_extreme_windows_that_remove_training_rows_raise_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "no training rows"):
            purged_time_series_splits([0, 1, 2, 3], n_splits=2, purge_gap=5, embargo=0)


if __name__ == "__main__":
    unittest.main()
