import numpy as np
import pytest
from benchmarks.morph_ablation import Result, evaluate_calibration_gates
from alloygbm._morph import (
    apply_interaction_importance_bonus,
    build_morph_config_dict,
    compute_morph_fingerprint,
    resolve_lr_schedule,
)


def test_fingerprint_fast_mode_returns_paper_constants():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    y = rng.standard_normal(100)
    fp = compute_morph_fingerprint(X, y, fast_mode=True)
    assert fp["complexity"] == 0.2
    assert fp["interaction_strength"] == 0.15
    assert fp["suggested_max_depth"] == 8


def test_fingerprint_full_mode_returns_finite_floats():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    y = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.1 * rng.standard_normal(100)
    fp = compute_morph_fingerprint(X, y, fast_mode=False)
    assert np.isfinite(fp["complexity"])
    assert np.isfinite(fp["non_linearity"])
    assert np.isfinite(fp["interaction_strength"])
    assert fp["suggested_max_depth"] in (8, 10)


def test_resolve_constant_schedule():
    out = resolve_lr_schedule("constant", 10, 0.1)
    assert len(out) == 10
    assert all(abs(x - 0.1) < 1e-6 for x in out)


def test_resolve_warmup_cosine_schedule_shape():
    out = resolve_lr_schedule("warmup_cosine", 100, 0.1, warmup_frac=0.1)
    assert len(out) == 100
    assert out[0] < 0.1
    assert out[-1] < 0.05


def test_resolve_unknown_schedule_raises():
    with pytest.raises(ValueError):
        resolve_lr_schedule("nonsense", 10, 0.1)


def test_build_morph_config_dict_defaults():
    d = build_morph_config_dict()
    assert d["morph_rate"] == 0.1
    assert d["info_score_weight"] == 0.1
    assert d["lr_schedule"] == "constant"
    assert d["balance_penalty"] is True


def test_morph_ablation_gate_accepts_bounded_regression():
    gates = evaluate_calibration_gates([
        Result("baseline_auto", "regression", 1.0, "RMSE", 0.1),
        Result("morph_full", "regression", 1.34, "RMSE", 0.1),
    ])
    assert len(gates) == 1
    assert gates[0].passed


def test_morph_ablation_gate_rejects_material_regression():
    gates = evaluate_calibration_gates([
        Result("baseline_auto", "binary_classification", 0.90, "Accuracy", 0.1),
        Result("morph_full", "binary_classification", 0.81, "Accuracy", 0.1),
    ])
    assert len(gates) == 1
    assert not gates[0].passed


def test_apply_interaction_importance_bonus_passthrough():
    importances = np.array([0.3, 0.5, 0.2])
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))
    result = apply_interaction_importance_bonus(importances, X)
    np.testing.assert_array_equal(result, importances)


def test_apply_interaction_importance_bonus_list_passthrough():
    importances = [0.4, 0.6]
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 2))
    result = apply_interaction_importance_bonus(importances, X)
    assert result == importances
