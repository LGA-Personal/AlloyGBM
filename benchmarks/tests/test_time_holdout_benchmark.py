"""Offline checks for benchmarks/time_holdout_benchmark.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time_holdout_benchmark as bench  # noqa: E402


def test_rolling_origin_run_and_summary(tmp_path):
    args = argparse.Namespace(blocks=5, first_test=3, seeds=2, cache_dir=tmp_path)
    rows = bench.run_dataset("synth_drift=0.1", args, log=lambda line: None)
    assert [r["arm"] for r in rows[:2]] == ["random", "time"]
    assert len(rows) == 2 * 2 * 2
    # The test block is the future of the training block: 80 periods in 5
    # blocks of 16, so fold 0 trains on 48 periods of 250 rows.
    assert rows[0]["train_rows"] == 48 * 250
    comparison = bench.compare(rows)
    assert comparison["periods"] == 32
    assert comparison["ci_low"] <= comparison["mean_delta"] <= comparison["ci_high"]
    report = bench.summarize({"datasets": {"synth_drift=0.1": rows}})
    assert "| synth_drift=0.1 | MSE |" in report


def test_block_bootstrap_interval_covers_a_shifted_mean():
    deltas = np.random.default_rng(0).normal(loc=-1.0, size=200)
    low, high = bench.block_bootstrap_ci(deltas)
    assert low < deltas.mean() < high
    assert high < 0.0
