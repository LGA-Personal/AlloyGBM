from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import default_quality as dq  # noqa: E402


def _result(arm_losses: dict[str, dict[tuple[str, int], float]]) -> dict:
    datasets = sorted({ds for losses in arm_losses.values() for ds, _ in losses})
    return {
        "arms": [{"name": arm} for arm in arm_losses],
        "datasets": [{"name": ds} for ds in datasets],
        "records": [
            {"dataset": ds, "seed": seed, "arm": arm, "loss": loss}
            for arm, losses in arm_losses.items()
            for (ds, seed), loss in losses.items()
        ],
    }


def _uniform(ratio: float, datasets: int = 8, seeds: int = 3, jitter: float = 0.0):
    base, cand = {}, {}
    for d in range(datasets):
        for s in range(seeds):
            value = 1.0 + 0.1 * d
            base[(f"d{d}", s)] = value
            cand[(f"d{d}", s)] = value * ratio * (1.0 + jitter * ((d + s) % 3 - 1))
    return base, cand


def test_parse_arm_builtin_and_custom():
    assert dq.parse_arm("alloy").library == "alloygbm"
    arm = dq.parse_arm("reg=alloygbm:lambda_l2=1,training_policy=manual,max_depth=4")
    assert arm.name == "reg"
    assert arm.params == {"lambda_l2": 1, "training_policy": "manual", "max_depth": 4}
    with pytest.raises(ValueError):
        dq.parse_arm("nope")
    with pytest.raises(ValueError):
        dq.parse_arm("x=sklearn:")
    with pytest.raises(ValueError):
        dq.parse_arm("x=alloygbm:lambda_l2")


def test_summarize_normalizes_against_best_arm():
    result = _result({
        "a": {("d0", 0): 1.0, ("d0", 1): 1.0, ("d1", 0): 4.0},
        "b": {("d0", 0): 2.0, ("d0", 1): 2.0, ("d1", 0): 2.0},
    })
    summary = dq.summarize(result)
    assert summary["normalized_loss"]["d0"] == {"a": 1.0, "b": 2.0}
    assert summary["normalized_loss"]["d1"] == {"a": 2.0, "b": 1.0}
    assert summary["aggregate"]["a"]["geomean_normalized_loss"] == pytest.approx(math.sqrt(2.0))
    assert summary["aggregate"]["a"]["wins"] == 1


def test_compare_passes_clear_uniform_improvement():
    base, cand = _uniform(0.95, jitter=0.01)
    cmp = dq.compare(_result({"base": base}), _result({"cand": cand}), "base", "cand")
    assert cmp["geomean_ratio"] == pytest.approx(0.95, rel=0.01)
    assert cmp["gate_passed"], cmp["gate_reasons"]


def test_compare_fails_when_no_improvement():
    base, cand = _uniform(1.0)
    cmp = dq.compare(_result({"base": base}), _result({"cand": cand}), "base", "cand")
    assert not cmp["gate_passed"]


def test_compare_flags_significant_dataset_regression():
    base, cand = _uniform(0.9, jitter=0.01)
    for s in range(3):
        cand[("d0", s)] = base[("d0", s)] * 1.2
    cmp = dq.compare(_result({"base": base}), _result({"cand": cand}), "base", "cand")
    assert not cmp["gate_passed"]
    assert any("d0" in reason for reason in cmp["gate_reasons"])


def test_compare_fails_closed_on_missing_pairs():
    base, cand = _uniform(0.9, jitter=0.01)
    del cand[("d1", 0)]
    cmp = dq.compare(_result({"base": base}), _result({"cand": cand}), "base", "cand")
    assert not cmp["gate_passed"]
    assert cmp["missing"] == [("d1", 0)]


def test_quick_suite_shapes_and_scale_variants():
    suite = dq.build_suite(quick=True)
    names = {ds.name for ds in suite}
    assert "diabetes@scale0.001" in names
    scaled = next(ds for ds in suite if ds.name == "diabetes@scale0.001")
    assert scaled.target_scale == pytest.approx(1e-3)
    for ds in suite:
        assert ds.X.shape[0] == ds.y.shape[0]
        assert ds.task in dq.TASKS


def test_scale_variant_scores_in_original_units():
    ds = dq.Dataset("lin", dq.REGRESSION, "scale", np.arange(40.0).reshape(-1, 1),
                    np.arange(40.0), target_scale=1e-3)
    arm = dq.Arm("alloy", "alloygbm", {"n_estimators": 30})
    pytest.importorskip("alloygbm")
    loss = dq.fit_and_score(arm, ds, seed=0, threads=1)["loss"]
    # Scored in the original units (targets 0..39), not the rescaled ones.
    assert 0.0 < loss < 20.0
