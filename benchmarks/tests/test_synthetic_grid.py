"""Tests for the controlled accuracy-at-depth scenario grid generator."""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _shared.synthetic_grid import write_grid_dataset  # noqa: E402


def _read(path: Path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_grid_cells_are_class_balanced(tmp_path):
    """Every cell must be 50/50 so the axis under test is the only difference.

    An earlier version thresholded at a fixed zero and produced 1-3% positive
    rates, which would have made each cell differ in class balance as well.
    """
    for signal, feats in ((4, 16), (16, 16), (16, 64)):
        path = write_grid_dataset(
            "unit", rows=2000, feature_count=feats, signal_count=signal,
            seed=7, repo_root=tmp_path,
        )
        rows = _read(path)
        positives = sum(1 for row in rows if row["target"] == "1")
        assert positives == len(rows) // 2, (signal, feats, positives)


def test_grid_shape_matches_request(tmp_path):
    path = write_grid_dataset(
        "unit", rows=500, feature_count=32, signal_count=8, seed=3, repo_root=tmp_path
    )
    rows = _read(path)
    assert len(rows) == 500
    assert len(rows[0]) == 33  # 32 features + target


def test_grid_is_deterministic_for_a_seed(tmp_path):
    first = _read(write_grid_dataset(
        "a", rows=300, feature_count=16, signal_count=4, seed=11, repo_root=tmp_path))
    second = _read(write_grid_dataset(
        "b", rows=300, feature_count=16, signal_count=4, seed=11, repo_root=tmp_path))
    assert first == second


def test_grid_noise_features_are_independent_of_target(tmp_path):
    """Features beyond signal_count must carry no signal, or the axis is a lie."""
    path = write_grid_dataset(
        "unit", rows=8000, feature_count=12, signal_count=4, seed=5, repo_root=tmp_path
    )
    rows = _read(path)
    for index in range(4, 12):
        positives = [float(r[f"f{index}"]) for r in rows if r["target"] == "1"]
        negatives = [float(r[f"f{index}"]) for r in rows if r["target"] == "0"]
        mean_gap = abs(sum(positives) / len(positives) - sum(negatives) / len(negatives))
        spread = max(max(positives), max(negatives)) - min(min(positives), min(negatives))
        assert mean_gap < 0.05 * spread, f"f{index} separates the classes"


@pytest.mark.parametrize("signal, feats", [(20, 16), (1, 16)])
def test_grid_rejects_impossible_configurations(tmp_path, signal, feats):
    with pytest.raises(ValueError):
        write_grid_dataset(
            "unit", rows=10, feature_count=feats, signal_count=signal,
            seed=1, repo_root=tmp_path,
        )
