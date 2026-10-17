"""Offline checks for benchmarks/time_decay_benchmark.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time_decay_benchmark as bench  # noqa: E402


def test_rolling_origin_run_and_summary(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    period = np.repeat(np.arange(40), 25)
    X = rng.normal(size=(len(period), 3))
    y = np.linspace(-1, 1, 40)[period] * X[:, 0] + 0.1 * rng.normal(size=len(period))
    monkeypatch.setattr(
        bench, "load_dataset", lambda name, cache_dir: (X, y, period, "regression")
    )
    args = argparse.Namespace(
        datasets=["toy"],
        blocks=4,
        first_test=3,
        seeds=1,
        half_life_fractions=[0.5],
        cache_dir=tmp_path,
        out=None,
    )
    results = bench.run(args)
    rows = results["datasets"]["toy"]["rows"]
    assert [r["arm"] for r in rows] == ["baseline", "time_holdout", "hl=0.5", "hl=0.5/random"]
    # The test block is the future of the training block.
    assert all(r["train_rows"] == 750 and r["test_rows"] == 250 for r in rows)
    held, decayed = rows[1], rows[2]
    assert held["effective_sample_size"] == 750
    assert decayed["effective_sample_size"] < 750
    report = bench.summarize(results)
    assert "| time_holdout |" in report and "| hl=0.5/random |" in report
