"""Contract tests for the MorphBoost performance and calibration harness."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ACCEPTANCE = _load_module(
    "morph_acceptance_module", REPO_ROOT / "benchmarks" / "morph_acceptance.py"
)
ABLATION = _load_module(
    "morph_ablation_module", REPO_ROOT / "benchmarks" / "morph_ablation.py"
)


def _record(
    *,
    arm: str,
    value: float,
    dataset: str = "regression-small",
    family: str = "regression",
    shape: str = "small-narrow",
    seed: int = 0,
    metric: str = "rmse",
    higher_is_better: bool = False,
):
    return ACCEPTANCE.MorphBenchmarkRecord(
        arm=arm,
        dataset=dataset,
        task_family=family,
        shape=shape,
        seed=seed,
        primary_metric=metric,
        primary_value=value,
        higher_is_better=higher_is_better,
        secondary_metrics={},
        fit_seconds=0.1,
    )


def _paired_records(changes: list[float]) -> list[object]:
    rows: list[object] = []
    for seed, change in enumerate(changes):
        control = 1.0
        candidate = control * (1.0 - change)
        rows.extend(
            [
                _record(arm="morph_current", value=control, seed=seed),
                _record(arm="candidate", value=candidate, seed=seed),
            ]
        )
    return rows


def test_ranking_fixture_returns_query_sizes() -> None:
    X_tr, y_tr, group_tr, X_te, y_te, group_te = ABLATION._ranking_dataset(
        n=240, n_features=8, n_groups=12, seed=7
    )
    assert all(size > 0 for size in [*group_tr, *group_te])
    assert sum(group_tr) == len(X_tr) == len(y_tr)
    assert sum(group_te) == len(X_te) == len(y_te)


def test_regularized_profile_covers_l1_levels_and_dro_families() -> None:
    specs = ACCEPTANCE.regularized_specs()
    assert {spec.lambda_l1 for spec in specs} == {0.1, 0.5}
    assert {spec.task_family for spec in specs if spec.leaf_solver == "dro"} == {
        "regression",
        "binary",
        "multiclass",
        "ranking",
    }


def test_normalized_improvement_respects_metric_direction() -> None:
    assert ACCEPTANCE.normalized_improvement(2.0, 1.5, False) == pytest.approx(0.25)
    assert ACCEPTANCE.normalized_improvement(0.8, 0.84, True) == pytest.approx(0.05)


def test_candidate_gate_accepts_broad_small_improvement() -> None:
    gate = ACCEPTANCE.evaluate_candidate(
        _paired_records([0.01, 0.012, 0.009, 0.011, 0.008]),
        control_arm="morph_current",
        candidate_arm="candidate",
    )
    assert gate.passed, gate.reasons
    assert gate.win_or_tie_fraction == 1.0
    assert gate.bootstrap_low > -0.0025


def test_candidate_gate_rejects_bad_worst_case() -> None:
    gate = ACCEPTANCE.evaluate_candidate(
        _paired_records([0.01, 0.02, -0.031, 0.01, 0.02]),
        control_arm="morph_current",
        candidate_arm="candidate",
    )
    assert not gate.passed
    assert any("worst paired change" in reason for reason in gate.reasons)


def test_candidate_gate_rejects_task_family_regression() -> None:
    rows: list[object] = []
    for seed in range(5):
        rows.extend(
            [
                _record(arm="morph_current", value=1.0, seed=seed),
                _record(arm="candidate", value=0.97, seed=seed),
                _record(
                    arm="morph_current",
                    value=0.8,
                    dataset="ranking-large-query",
                    family="ranking",
                    shape="tall-narrow",
                    metric="ndcg_at_10",
                    higher_is_better=True,
                    seed=seed,
                ),
                _record(
                    arm="candidate",
                    value=0.79,
                    dataset="ranking-large-query",
                    family="ranking",
                    shape="tall-narrow",
                    metric="ndcg_at_10",
                    higher_is_better=True,
                    seed=seed,
                ),
            ]
        )
    gate = ACCEPTANCE.evaluate_candidate(
        rows, control_arm="morph_current", candidate_arm="candidate"
    )
    assert not gate.passed
    assert gate.family_means["ranking"] < -0.005


def test_practical_tie_boundary_counts_as_tie() -> None:
    gate = ACCEPTANCE.evaluate_candidate(
        _paired_records([0.001, -0.001, 0.001, -0.001, 0.001]),
        control_arm="morph_current",
        candidate_arm="candidate",
    )
    assert gate.win_or_tie_fraction == 1.0


def test_bootstrap_is_deterministic() -> None:
    records = _paired_records([0.01, 0.02, 0.005, 0.015, 0.007])
    first = ACCEPTANCE.evaluate_candidate(
        records,
        control_arm="morph_current",
        candidate_arm="candidate",
        bootstrap_seed=132,
    )
    second = ACCEPTANCE.evaluate_candidate(
        records,
        control_arm="morph_current",
        candidate_arm="candidate",
        bootstrap_seed=132,
    )
    assert first.bootstrap_low == second.bootstrap_low


def test_relabel_and_merge_reject_duplicate_pair_keys() -> None:
    control = [_record(arm="morph_current", value=1.0)]
    candidate = ACCEPTANCE.relabel_arm(control, "morph_current", "morph_simd")
    merged = ACCEPTANCE.merge_record_sets(control, candidate)
    assert [row.arm for row in merged] == ["morph_current", "morph_simd"]
    with pytest.raises(ValueError, match="duplicate benchmark record"):
        ACCEPTANCE.merge_record_sets(control, control)
