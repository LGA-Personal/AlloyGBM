"""Contract tests for PR #137 DART expected-drop calibration."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "benchmarks" / "dart_policy_calibration.py"
SPEC = importlib.util.spec_from_file_location("dart_policy_calibration", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"failed to load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _record(
    spec,
    *,
    seed: int,
    cap: int,
    primary: float | None = None,
    secondary: float | None = None,
        fit_seconds: float | None = None,
    peak_rss_bytes: int | None = 128 * 1024 * 1024,
    completed_rounds: int | None = None,
):
    if spec.task == "ranking":
        primary_metric = "ndcg@10"
        secondary_metric = "ndcg@5"
        default_primary = 0.80
        default_secondary = 0.75
    elif spec.task in {"binary", "multiclass"}:
        primary_metric = "log_loss"
        secondary_metric = "accuracy"
        default_primary = 0.70
        default_secondary = 0.80
    else:
        primary_metric = "rmse"
        secondary_metric = "mae"
        default_primary = 1.00
        default_secondary = 0.70
    return MODULE.DartPolicyRecord(
        fixture=spec.name,
        seed=seed,
        cap=cap,
        arm=f"cap-{cap}",
        task=spec.task,
        n_rows=spec.n_rows,
        n_features=spec.n_features,
        n_estimators=spec.n_estimators,
        drop_rate=spec.drop_rate,
        sample_type=spec.sample_type,
        normalize_type=spec.normalize_type,
        tree_growth=spec.tree_growth,
        stress=spec.stress,
        trees_per_round=spec.trees_per_round,
        requested_rounds=spec.n_estimators,
        completed_rounds=(
            spec.n_estimators if completed_rounds is None else completed_rounds
        ),
        primary_metric=primary_metric,
        primary_value=default_primary if primary is None else primary,
        secondary_metric=secondary_metric,
        secondary_value=default_secondary if secondary is None else secondary,
        fit_seconds=(1.0 if cap == MODULE.INCUMBENT_CAP else 0.80)
        if fit_seconds is None
        else fit_seconds,
        peak_rss_bytes=peak_rss_bytes,
        configured_pressure=MODULE.configured_dropout_pressure(
            spec.n_estimators,
            spec.drop_rate,
            cap,
            spec.trees_per_round,
        ),
        prediction_sha256="a" * 64,
        artifact_sha256="b" * 64,
        source_commit="c" * 40,
    )


def _matrix(*, mutate=None):
    specs = MODULE.full_specs()
    records = [
        _record(spec, seed=seed, cap=cap)
        for spec in specs
        for seed in MODULE.FULL_SEEDS
        for cap in (*MODULE.CANDIDATE_CAPS, MODULE.INCUMBENT_CAP)
    ]
    if mutate is not None:
        records = mutate(records, specs)
    return specs, records


def _assessment(decision, cap):
    return next(item for item in decision.assessments if item.cap == cap)


def test_fixed_caps_seeds_and_fixture_catalog_are_pinned():
    assert MODULE.CANDIDATE_CAPS == (2, 5, 10, 20)
    assert MODULE.INCUMBENT_CAP == 50
    assert MODULE.FULL_SEEDS == (0, 1, 2, 3, 4)

    specs = MODULE.full_specs()
    assert [spec.name for spec in specs] == [
        "reg-small-narrow",
        "reg-small-wide",
        "reg-tall-narrow",
        "reg-tall-wide-leaf",
        "reg-long-stress",
        "binary-medium",
        "multiclass-four",
        "ranking-groups",
        "reg-weighted",
        "reg-forest",
    ]
    assert [
        (
            spec.task,
            spec.n_rows,
            spec.n_features,
            spec.n_estimators,
            spec.drop_rate,
            spec.sample_type,
            spec.normalize_type,
            spec.tree_growth,
            spec.stress,
        )
        for spec in specs
    ] == [
        ("regression", 640, 8, 100, 0.10, "uniform", "tree", "level", False),
        ("regression", 640, 64, 100, 0.10, "uniform", "tree", "level", False),
        ("regression", 4096, 12, 200, 0.10, "uniform", "tree", "level", True),
        ("regression", 3072, 64, 200, 0.10, "uniform", "tree", "leaf", True),
        ("regression", 2048, 24, 300, 0.20, "uniform", "tree", "level", True),
        ("binary", 2048, 24, 150, 0.10, "uniform", "tree", "level", False),
        ("multiclass", 1600, 20, 100, 0.10, "uniform", "tree", "level", True),
        ("ranking", 2400, 16, 120, 0.10, "uniform", "tree", "level", False),
        ("regression", 1536, 16, 200, 0.10, "weighted", "tree", "level", True),
        ("regression", 1536, 16, 200, 0.10, "uniform", "forest", "level", True),
    ]
    assert MODULE.quick_specs()
    assert {spec.name for spec in MODULE.quick_specs()} <= {
        spec.name for spec in specs
    }


def test_multiclass_pressure_uses_four_tree_pool():
    scalar = MODULE.configured_dropout_pressure(100, 0.10, 50, 1)
    multiclass = MODULE.configured_dropout_pressure(100, 0.10, 50, 4)
    assert multiclass > scalar
    assert all(
        spec.trees_per_round == 4 if spec.task == "multiclass" else spec.trees_per_round == 1
        for spec in MODULE.full_specs()
    )


def test_ranking_fixture_splits_by_whole_contiguous_groups():
    spec = next(item for item in MODULE.full_specs() if item.task == "ranking")
    fixture = MODULE.make_fixture(spec, seed=0)
    assert fixture.train_group_sizes == (30,) * 64
    assert fixture.test_group_sizes == (30,) * 16
    assert tuple(sorted(set(fixture.train_group_ids))) == tuple(range(64))
    assert tuple(sorted(set(fixture.test_group_ids))) == tuple(range(64, 80))
    assert fixture.train_group_ids[:30] == (0,) * 30
    assert fixture.test_group_ids[-30:] == (79,) * 30
    assert fixture.X_train.shape == (1920, 16)
    assert fixture.X_test.shape == (480, 16)


def test_selection_chooses_largest_passing_cap_and_keeps_reasons():
    specs, records = _matrix(
        mutate=lambda rows, specs: [
            replace(
                row,
                configured_pressure=(
                    MODULE.configured_dropout_pressure(
                        next(spec for spec in specs if spec.name == row.fixture).n_estimators,
                        next(spec for spec in specs if spec.name == row.fixture).drop_rate,
                        MODULE.INCUMBENT_CAP,
                        next(spec for spec in specs if spec.name == row.fixture).trees_per_round,
                    )
                    * (0.40 if row.cap == 10 else 0.60)
                ),
            )
            if row.stress and row.cap in (10, 20)
            else row
            for row in rows
        ]
    )
    decision = MODULE.evaluate_candidate_caps(records, specs=specs)
    assert decision.selected_cap == 10
    assert _assessment(decision, 2).passed
    assert _assessment(decision, 5).passed
    assert _assessment(decision, 10).passed
    assert not _assessment(decision, 20).passed
    assert any("pressure" in reason for reason in _assessment(decision, 20).reasons)


def test_selection_falls_back_to_incumbent_with_rejection_reasons():
    specs, records = _matrix(
        mutate=lambda rows, specs: [
            replace(row, primary_value=row.primary_value * 1.03)
            if row.cap in MODULE.CANDIDATE_CAPS
            else row
            for row in rows
        ]
    )
    decision = MODULE.evaluate_candidate_caps(records, specs=specs)
    assert decision.selected_cap == MODULE.INCUMBENT_CAP
    assert decision.fallback
    assert all(not assessment.passed for assessment in decision.assessments)
    assert any("median quality" in reason for reason in _assessment(decision, 2).reasons)


@pytest.mark.parametrize(
    ("label", "mutate", "reason_fragment"),
    [
        (
            "median-quality",
            lambda rows, specs: [
                replace(row, primary_value=1.0201)
                if row.cap == 2 and row.fixture == "reg-small-narrow"
                else row
                for row in rows
            ],
            "median quality",
        ),
        (
            "seed-quality",
            lambda rows, specs: [
                replace(row, primary_value=1.1001)
                if row.cap == 2
                and row.fixture == "reg-small-narrow"
                and row.seed == 0
                else row
                for row in rows
            ],
            "individual-seed quality",
        ),
        (
            "accuracy",
            lambda rows, specs: [
                replace(row, secondary_value=0.7799)
                if row.cap == 2 and row.task == "binary"
                else row
                for row in rows
            ],
            "accuracy",
        ),
        (
            "ndcg",
            lambda rows, specs: [
                replace(row, primary_value=0.7899)
                if row.cap == 2 and row.task == "ranking"
                else row
                for row in rows
            ],
            "NDCG",
        ),
        (
            "pressure",
            lambda rows, specs: [
                replace(
                    row,
                    configured_pressure=MODULE.configured_dropout_pressure(
                        specs[2].n_estimators,
                        specs[2].drop_rate,
                        MODULE.INCUMBENT_CAP,
                        specs[2].trees_per_round,
                    )
                    * 0.5001,
                )
                if row.cap == 2 and row.stress
                else row
                for row in rows
            ],
            "pressure",
        ),
        (
            "time",
            lambda rows, specs: [
                replace(row, fit_seconds=0.8501)
                if row.cap == 2 and row.stress
                else row
                for row in rows
            ],
            "fit time",
        ),
        (
            "rss",
            lambda rows, specs: [
                    replace(row, peak_rss_bytes=161 * 1024 * 1024)
                if row.cap == 2 and row.stress
                else row
                for row in rows
            ],
            "RSS",
        ),
        (
            "completion",
            lambda rows, specs: [
                replace(row, completed_rounds=row.requested_rounds - 1)
                if row.cap == 2 and row.fixture == "reg-small-narrow"
                else row
                for row in rows
            ],
            "completed",
        ),
    ],
)
def test_each_selection_gate_rejects_independently(label, mutate, reason_fragment):
    specs, records = _matrix(mutate=mutate)
    decision = MODULE.evaluate_candidate_caps(records, specs=specs)
    assessment = _assessment(decision, 2)
    assert not assessment.passed, label
    assert any(reason_fragment.lower() in reason.lower() for reason in assessment.reasons)


def test_comparator_rejects_missing_duplicate_and_nonfinite_matrix_values():
    specs, records = _matrix()
    missing = MODULE.evaluate_candidate_caps(records[:-1], specs=specs)
    assert not _assessment(missing, 50 if 50 in MODULE.CANDIDATE_CAPS else 2).passed
    assert any("coverage" in reason for reason in _assessment(missing, 2).reasons)

    duplicate = MODULE.evaluate_candidate_caps(records + records[:1], specs=specs)
    assert any("duplicate" in reason for reason in _assessment(duplicate, 2).reasons)

    nonfinite = list(records)
    index = next(i for i, row in enumerate(nonfinite) if row.cap == 2)
    nonfinite[index] = replace(nonfinite[index], primary_value=float("nan"))
    invalid = MODULE.evaluate_candidate_caps(nonfinite, specs=specs)
    assert any("non-finite" in reason for reason in _assessment(invalid, 2).reasons)


def test_metric_orientation_makes_lower_ndcg_worse():
    specs, records = _matrix(
        mutate=lambda rows, specs: [
            replace(row, primary_value=0.75)
            if row.cap == 2 and row.task == "ranking"
            else row
            for row in rows
        ]
    )
    decision = MODULE.evaluate_candidate_caps(records, specs=specs)
    assessment = _assessment(decision, 2)
    assert not assessment.passed
    ratio = dict(assessment.fixture_quality_ratios)["ranking-groups"]
    assert ratio > 1.02


def test_eight_gates_are_explicit_and_compatibility_can_fail():
    specs, records = _matrix()
    decision = MODULE.evaluate_candidate_caps(
        records,
        specs=specs,
        compatibility={"passed": False, "reasons": ["warm-start parity failed"]},
    )
    assessment = _assessment(decision, 2)
    assert set(dict(assessment.gates)) == set(MODULE.GATE_NAMES)
    assert not dict(assessment.gates)["compatibility"]
    assert any("warm-start" in reason for reason in assessment.reasons)


def test_candidate_compatibility_allows_calibrated_default_to_differ_from_cap50():
    specs = tuple(
        spec for spec in MODULE.full_specs() if spec.name in MODULE.COMPAT_FIXTURE_NAMES
    )
    records = []
    for spec in specs:
        for seed in MODULE.COMPAT_SEEDS:
            selected = replace(
                _record(spec, seed=seed, cap=5),
                prediction_sha256="d" * 64,
                artifact_sha256="e" * 64,
            )
            records.append(replace(selected, arm="default"))
            records.append(replace(selected, arm="cap-selected"))
            records.append(replace(_record(spec, seed=seed, cap=50), arm="cap50"))
    checks = MODULE._compat_checks(
        records,
        arms=("default", "cap-selected", "cap50"),
        seeds=MODULE.COMPAT_SEEDS,
        specs=specs,
        require_default_cap50_parity=False,
    )
    assert checks["passed"]
    assert checks["default_cap50_equal"] is False


def test_json_round_trip_is_sorted_and_rejects_schema_or_nonfinite(tmp_path):
    specs, records = _matrix()
    path = tmp_path / "matrix.json"
    MODULE.write_matrix(path, specs, records, caps=(*MODULE.CANDIDATE_CAPS, 50), seeds=MODULE.FULL_SEEDS)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == MODULE.MATRIX_SCHEMA_VERSION
    assert payload["records"] == sorted(
        payload["records"], key=MODULE.record_sort_key
    )
    assert MODULE.read_matrix(path)[1] == tuple(
        sorted(records, key=MODULE.record_sort_key)
    )

    malformed = dict(payload)
    malformed["schema_version"] = -1
    path.write_text(json.dumps(malformed))
    with pytest.raises(ValueError, match="schema"):
        MODULE.read_matrix(path)

    bad = dict(payload)
    bad["records"] = [dict(payload["records"][0], primary_value=float("nan"))]
    path.write_text(json.dumps(bad, allow_nan=True))
    with pytest.raises(ValueError, match="finite"):
        MODULE.read_matrix(path)


def test_committed_compatibility_captures_pin_cap50_and_selected_default_parity():
    production = MODULE.read_compat(
        REPO_ROOT / "benchmarks" / "results" / "pr137_dart_policy_production_compat.json"
    )
    candidate = MODULE.read_compat(
        REPO_ROOT / "benchmarks" / "results" / "pr137_dart_policy_candidate_compat.json"
    )
    production_by_key = {
        (record.fixture, record.seed, record.arm): record
        for record in MODULE._compat_payload_records(production)
    }
    candidate_by_key = {
        (record.fixture, record.seed, record.arm): record
        for record in MODULE._compat_payload_records(candidate)
    }
    for fixture in MODULE.COMPAT_FIXTURE_NAMES:
        for seed in MODULE.COMPAT_SEEDS:
            production_cap50 = production_by_key[(fixture, seed, "cap50")]
            candidate_cap50 = candidate_by_key[(fixture, seed, "cap50")]
            candidate_default = candidate_by_key[(fixture, seed, "default")]
            candidate_selected = candidate_by_key[(fixture, seed, "cap-selected")]
            assert (
                production_cap50.prediction_sha256,
                production_cap50.artifact_sha256,
            ) == (
                candidate_cap50.prediction_sha256,
                candidate_cap50.artifact_sha256,
            )
            assert (
                candidate_default.prediction_sha256,
                candidate_default.artifact_sha256,
            ) == (
                candidate_selected.prediction_sha256,
                candidate_selected.artifact_sha256,
            )
            assert candidate_default.cap == candidate_selected.cap == 5
