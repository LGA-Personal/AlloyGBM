"""Contract tests for the per-mode quality gate.

These test the *gate itself* — that its invariants actually fire on the
failure shapes it exists to catch — rather than re-testing the modes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = REPO_ROOT / "benchmarks" / "mode_quality_gate.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "mode_quality_gate_module", BENCHMARK_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load benchmark from {BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = _load_benchmark()


def _result(**overrides):
    defaults = {
        "name": "case",
        "metric": "rmse",
        "value": 0.5,
        "constant_reference": 2.0,
        "baseline_value": 0.5,
        "baseline_ratio_ceiling": GATE.BASELINE_RATIO_CEILING,
    }
    defaults.update(overrides)
    return GATE.ModeResult(**defaults)


def test_healthy_regression_mode_passes():
    assert GATE._check(_result()) == []


def test_divergent_mode_is_rejected():
    """The subsampling/GOSS regression shape: worse than predicting a constant."""
    failures = GATE._check(_result(value=2.19, constant_reference=1.15))
    assert failures, "a mode worse than the constant predictor must fail"
    assert any("constant-predictor" in f for f in failures)


def test_mode_far_worse_than_baseline_is_rejected():
    """A mode that still beats a constant but is much worse than plain boosting."""
    failures = GATE._check(
        _result(value=1.60, constant_reference=10.0, baseline_value=0.50)
    )
    assert any("plain-boosting" in f for f in failures)


def test_non_finite_metric_is_rejected():
    assert GATE._check(_result(value=float("nan")))
    assert GATE._check(_result(value=float("inf")))


def test_classification_floor_and_baseline_slack():
    healthy = _result(
        metric="accuracy", value=0.90, constant_reference=0.5, baseline_value=0.92
    )
    assert GATE._check(healthy) == []

    below_floor = _result(
        metric="accuracy", value=0.55, constant_reference=0.5, baseline_value=0.90
    )
    assert any("floor" in f for f in GATE._check(below_floor))

    below_majority = _result(
        metric="accuracy", value=0.45, constant_reference=0.5, baseline_value=0.90
    )
    assert any("majority-class" in f for f in GATE._check(below_majority))

    below_baseline = _result(
        metric="accuracy", value=0.75, constant_reference=0.5, baseline_value=0.92
    )
    assert any("below the baseline" in f for f in GATE._check(below_baseline))


def test_gate_covers_every_advertised_mode():
    """Each opt-in training mode must have a row; this is the point of the gate."""
    names = {case.name for case in GATE.regression_modes()}
    names |= {case.name for case in GATE.classification_modes()}
    for expected in (
        "row_subsample_0.5",
        "goss",
        "dart",
        "leaf_wise",
        "morph",
        "dro_leaves",
        "pl_leaves_unregularized",
        "pl_leaves_regularized",
        "quantile_objective",
        "multiclass_goss",
    ):
        assert expected in names, f"{expected} is not covered by the mode gate"


def test_quick_run_produces_a_result_per_mode_and_passes():
    results = GATE.run(quick=True)
    expected = (
        1  # regression baseline
        + len(GATE.regression_modes())
        + 2  # binary + multiclass baselines
        + len(GATE.classification_modes())
    )
    assert len(results) == expected
    assert all(np.isfinite(r.value) for r in results)
    failures = {r.name: r.failures for r in results if not r.ok}
    assert failures == {}, f"modes regressed: {failures}"
