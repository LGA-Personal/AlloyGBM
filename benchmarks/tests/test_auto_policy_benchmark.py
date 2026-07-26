"""Contract tests for the deterministic auto-policy calibration benchmark."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = REPO_ROOT / "benchmarks" / "auto_policy_benchmark.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "auto_policy_benchmark_module", BENCHMARK_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load benchmark from {BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark()

BALANCED_SPECS = (
    BENCHMARK.FixtureSpec(
        "small-narrow-reg", "small-narrow", "regression", 512, 8, 10
    ),
    BENCHMARK.FixtureSpec(
        "small-wide-sparse", "small-wide", "sparse_regression", 512, 128, 10
    ),
    BENCHMARK.FixtureSpec(
        "medium-narrow-bin", "medium-narrow", "binary", 2_048, 16, 10
    ),
    BENCHMARK.FixtureSpec(
        "medium-wide-multi", "medium-wide", "multiclass", 2_048, 128, 10
    ),
    BENCHMARK.FixtureSpec(
        "large-narrow-rank", "large-narrow", "ranking", 16_384, 16, 10
    ),
    BENCHMARK.FixtureSpec(
        "large-wide-reg", "large-wide", "regression", 16_384, 128, 10
    ),
)


def sample_resolved_policy(
    *,
    min_split_gain: float = 0.001,
    row_subsample: float = 0.8,
    col_subsample: float = 0.5,
    auto_split_l2_applied: bool = False,
    effective_split_l2: float = 0.0,
) -> dict[str, object]:
    return {
        "requested_mode": "auto",
        "requested_rounds": 40,
        "effective_round_cap": 40,
        "min_rows_per_leaf": 12,
        "min_split_gain": min_split_gain,
        "row_subsample": row_subsample,
        "col_subsample": col_subsample,
        "auto_split_l2_applied": auto_split_l2_applied,
        "effective_split_l2": effective_split_l2,
    }


def record(
    *,
    fixture: str = "fixture",
    shape_stratum: str = "small-narrow",
    objective: str = "regression",
    seed: int = 7,
    arm: str,
    primary_metric: float,
    accuracy: float | None = None,
    ndcg_at_10: float | None = None,
    completed_rounds: int = 10,
    fit_seconds: float = 1.0,
    resolved_policy: dict[str, object] | None = None,
    error: str | None = None,
) -> object:
    return BENCHMARK.BenchmarkRecord(
        fixture=fixture,
        shape_stratum=shape_stratum,
        objective=objective,
        seed=seed,
        arm=arm,
        primary_metric=primary_metric,
        accuracy=accuracy,
        ndcg_at_10=ndcg_at_10,
        completed_rounds=completed_rounds,
        fit_seconds=fit_seconds,
        resolved_policy=(
            sample_resolved_policy()
            if resolved_policy is None
            else resolved_policy
        ),
        error=error,
    )


def paired_records(
    *,
    candidate_arm: str = "quality_first",
    fixture: str = "fixture",
    shape_stratum: str = "small-narrow",
    objective: str = "regression",
    seed: int = 7,
    current_loss: float = 1.0,
    candidate_loss: float = 0.98,
    current_accuracy: float | None = None,
    candidate_accuracy: float | None = None,
    current_ndcg: float | None = None,
    candidate_ndcg: float | None = None,
    candidate_rounds: int = 10,
    candidate_fit_seconds: float = 1.0,
    candidate_error: str | None = None,
) -> list[object]:
    return [
        record(
            fixture=fixture,
            shape_stratum=shape_stratum,
            objective=objective,
            seed=seed,
            arm="current_auto",
            primary_metric=current_loss,
            accuracy=current_accuracy,
            ndcg_at_10=current_ndcg,
        ),
        record(
            fixture=fixture,
            shape_stratum=shape_stratum,
            objective=objective,
            seed=seed,
            arm=candidate_arm,
            primary_metric=candidate_loss,
            accuracy=candidate_accuracy,
            ndcg_at_10=candidate_ndcg,
            completed_rounds=candidate_rounds,
            fit_seconds=candidate_fit_seconds,
            error=candidate_error,
        ),
    ]


def balanced_records(
    *,
    candidate_arm: str = "quality_first",
    candidate_loss_ratio: float = 0.98,
    candidate_fit_seconds: float = 1.0,
) -> list[object]:
    rows: list[object] = []
    cases = (
        ("small-narrow-reg", "small-narrow", "regression", None, None),
        ("small-wide-sparse", "small-wide", "sparse_regression", None, None),
        ("medium-narrow-bin", "medium-narrow", "binary", 0.80, None),
        ("medium-wide-multi", "medium-wide", "multiclass", 0.75, None),
        ("large-narrow-rank", "large-narrow", "ranking", None, 0.72),
        ("large-wide-reg", "large-wide", "regression", None, None),
    )
    for fixture, stratum, objective, current_accuracy, current_ndcg in cases:
        rows.extend(
            paired_records(
                candidate_arm=candidate_arm,
                fixture=fixture,
                shape_stratum=stratum,
                objective=objective,
                current_loss=1.0,
                candidate_loss=candidate_loss_ratio,
                current_accuracy=current_accuracy,
                candidate_accuracy=(
                    current_accuracy if current_accuracy is not None else None
                ),
                current_ndcg=current_ndcg,
                candidate_ndcg=current_ndcg,
                candidate_fit_seconds=candidate_fit_seconds,
            )
        )
    return rows


def test_full_matrix_crosses_rows_and_columns_independently() -> None:
    specs = BENCHMARK.full_specs()
    shapes = {(spec.rows, spec.features) for spec in specs}
    assert shapes == {
        (512, 8),
        (1_023, 16),
        (512, 128),
        (1_023, 256),
        (2_048, 16),
        (8_192, 16),
        (2_048, 128),
        (8_192, 256),
        (16_384, 16),
        (16_384, 256),
    }
    assert {spec.objective for spec in specs} == {
        "regression",
        "sparse_regression",
        "binary",
        "multiclass",
        "ranking",
    }
    assert len(specs) == 10 * 5
    assert all(
        spec.rounds == 300
        for spec in specs
        if spec.shape_stratum == "small-wide"
    )
    assert all(
        spec.rounds < 300
        for spec in specs
        if spec.shape_stratum != "small-wide"
    )
    for shape in shapes:
        assert {
            spec.objective
            for spec in specs
            if (spec.rows, spec.features) == shape
        } == {"regression", "sparse_regression", "binary", "multiclass", "ranking"}


def test_shape_strata_cover_boundary_combinations() -> None:
    assert BENCHMARK.classify_shape(512, 8) == "small-narrow"
    assert BENCHMARK.classify_shape(512, 128) == "small-wide"
    assert BENCHMARK.classify_shape(2_048, 16) == "medium-narrow"
    assert BENCHMARK.classify_shape(2_048, 128) == "medium-wide"
    assert BENCHMARK.classify_shape(16_384, 16) == "large-narrow"
    assert BENCHMARK.classify_shape(16_384, 256) == "large-wide"


def test_quick_matrix_covers_all_strata_and_rotates_all_objectives() -> None:
    specs = BENCHMARK.quick_specs()

    assert len(specs) == 6
    assert {spec.shape_stratum for spec in specs} == {
        "small-narrow",
        "small-wide",
        "medium-narrow",
        "medium-wide",
        "large-narrow",
        "large-wide",
    }
    assert {spec.objective for spec in specs} == {
        "regression",
        "sparse_regression",
        "binary",
        "multiclass",
        "ranking",
    }
    assert all(spec.rounds <= 12 for spec in specs)


@pytest.mark.parametrize(
    "objective",
    ["regression", "sparse_regression", "binary", "multiclass", "ranking"],
)
def test_same_seed_fixtures_are_byte_identical_and_different_seeds_differ(
    objective: str,
) -> None:
    spec = BENCHMARK.FixtureSpec(
        name=f"determinism-{objective}",
        shape_stratum=BENCHMARK.classify_shape(512, 128),
        objective=objective,
        rows=512,
        features=128,
        rounds=5,
    )

    first = BENCHMARK.make_fixture(spec, seed=7)
    second = BENCHMARK.make_fixture(spec, seed=7)
    different = BENCHMARK.make_fixture(spec, seed=13)

    for left, right in zip(first.arrays(), second.arrays(), strict=True):
        np.testing.assert_array_equal(left, right)
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(first.arrays(), different.arrays(), strict=True)
    )


@pytest.mark.parametrize(
    "objective", ["regression", "sparse_regression", "binary", "multiclass"]
)
def test_non_ranking_fixtures_are_contiguous_and_targets_contain_signal(
    objective: str,
) -> None:
    features = 128 if objective == "sparse_regression" else 16
    spec = BENCHMARK.FixtureSpec(
        name=f"signal-{objective}",
        shape_stratum=BENCHMARK.classify_shape(512, features),
        objective=objective,
        rows=512,
        features=features,
        rounds=5,
    )

    fixture = BENCHMARK.make_fixture(spec, seed=7)

    assert fixture.X_train.dtype == np.float32
    assert fixture.X_test.dtype == np.float32
    assert fixture.y_train.dtype == np.float32
    assert fixture.y_test.dtype == np.float32
    assert fixture.X_train.flags.c_contiguous
    assert fixture.X_test.flags.c_contiguous
    assert fixture.y_train.flags.c_contiguous
    assert fixture.y_test.flags.c_contiguous
    assert not np.array_equal(fixture.X_train[: len(fixture.X_test)], fixture.X_test)
    assert fixture.target_signal > 0.10
    if objective in {"binary", "multiclass"}:
        assert len(np.unique(fixture.y_train)) == (2 if objective == "binary" else 4)


def test_sparse_fixture_density_is_below_ten_percent() -> None:
    spec = BENCHMARK.FixtureSpec(
        name="sparse",
        shape_stratum="small-wide",
        objective="sparse_regression",
        rows=512,
        features=128,
        rounds=5,
    )

    fixture = BENCHMARK.make_fixture(spec, seed=7)

    assert np.count_nonzero(fixture.X_train) / fixture.X_train.size < 0.10
    assert np.count_nonzero(fixture.X_test) / fixture.X_test.size < 0.10


def test_ranking_groups_cover_every_row_and_targets_are_query_local() -> None:
    spec = BENCHMARK.FixtureSpec(
        name="ranking",
        shape_stratum="small-narrow",
        objective="ranking",
        rows=512,
        features=16,
        rounds=5,
    )

    fixture = BENCHMARK.make_fixture(spec, seed=7)

    assert fixture.group_train is not None
    assert fixture.group_test is not None
    assert sum(np.unique(fixture.group_train, return_counts=True)[1]) == spec.rows
    assert sum(np.unique(fixture.group_test, return_counts=True)[1]) == len(
        fixture.y_test
    )
    assert fixture.target_signal > 0.10
    for labels in (
        fixture.y_train[fixture.group_train == group]
        for group in np.unique(fixture.group_train)
    ):
        assert labels.min() == 0.0
        assert labels.max() == 4.0


def test_candidate_arms_change_only_declared_controls() -> None:
    resolved = sample_resolved_policy(
        min_split_gain=0.001,
        row_subsample=0.8,
        col_subsample=0.5,
        auto_split_l2_applied=True,
        effective_split_l2=2.0,
    )

    no_gain = BENCHMARK.derive_candidate_params("no_gain_floor", resolved)
    quality = BENCHMARK.derive_candidate_params("quality_first", resolved)

    assert no_gain == {
        "training_policy": "manual",
        "n_estimators": 40,
        "min_data_in_leaf": 12,
        "min_split_gain": 0.0,
        "row_subsample": 0.8,
        "col_subsample": 0.5,
    }
    assert quality == {
        "training_policy": "manual",
        "n_estimators": 40,
        "min_data_in_leaf": 12,
        "min_split_gain": 0.0,
        "row_subsample": 1.0,
        "col_subsample": 1.0,
    }
    assert "lambda_l2" not in no_gain
    assert "lambda_l2" not in quality


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda policy: policy.pop("row_subsample"), "row_subsample"),
        (lambda policy: policy.__setitem__("effective_round_cap", 0), "effective_round_cap"),
        (lambda policy: policy.__setitem__("min_split_gain", float("nan")), "min_split_gain"),
        (lambda policy: policy.__setitem__("requested_mode", "manual"), "requested_mode"),
    ],
)
def test_candidate_derivation_rejects_malformed_current_auto_diagnostics(
    mutation, match: str
) -> None:
    resolved = sample_resolved_policy()
    mutation(resolved)

    with pytest.raises(ValueError, match=match):
        BENCHMARK.derive_candidate_params("quality_first", resolved)


@pytest.mark.parametrize(
    ("field", "bad_value", "params_override"),
    [
        ("requested_rounds", 40.0, {}),
        ("min_split_gain", 0, {}),
        ("row_subsample", True, {"row_subsample": 1.0}),
        ("auto_split_l2_applied", 0, {}),
    ],
)
def test_manual_policy_rejects_wrong_scalar_categories_before_comparison(
    field: str,
    bad_value: object,
    params_override: dict[str, object],
) -> None:
    resolved = {
        **sample_resolved_policy(
            min_split_gain=0.0,
            row_subsample=0.8,
            col_subsample=0.5,
        ),
        "requested_mode": "manual",
        field: bad_value,
    }
    params = {
        "n_estimators": 40,
        "min_data_in_leaf": 12,
        "min_split_gain": 0.0,
        "row_subsample": 0.8,
        "col_subsample": 0.5,
        **params_override,
    }

    with pytest.raises(ValueError, match=field):
        BENCHMARK._validate_manual_policy(
            resolved,
            params=params,
            split_l2=None,
            context="manual-test",
        )


def test_temporary_split_l2_sets_and_deletes_previously_absent_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(BENCHMARK.SPLIT_L2_ENV_VAR, raising=False)
    estimator = SimpleNamespace(lambda_l2=3.5)
    observed: list[tuple[str | None, float]] = []

    def fit_callback() -> None:
        observed.append(
            (os.environ.get(BENCHMARK.SPLIT_L2_ENV_VAR), estimator.lambda_l2)
        )

    with BENCHMARK.temporary_split_l2(2.0):
        fit_callback()

    assert observed == [("2.0", 3.5)]
    assert BENCHMARK.SPLIT_L2_ENV_VAR not in os.environ
    assert estimator.lambda_l2 == 3.5


def test_temporary_split_l2_restores_existing_value_after_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(BENCHMARK.SPLIT_L2_ENV_VAR, "7.25")

    with pytest.raises(RuntimeError, match="fit failed"):
        with BENCHMARK.temporary_split_l2(None):
            assert BENCHMARK.SPLIT_L2_ENV_VAR not in os.environ
            raise RuntimeError("fit failed")

    assert os.environ[BENCHMARK.SPLIT_L2_ENV_VAR] == "7.25"


def test_candidate_is_rejected_by_one_protected_shape_regression() -> None:
    rows = balanced_records(candidate_loss_ratio=0.98)
    rows.extend(
        paired_records(
            fixture="small-wide-regression",
            shape_stratum="small-wide",
            candidate_loss=1.031,
        )
    )

    result = BENCHMARK.evaluate_candidate(rows, "quality_first")

    assert not result.passed
    assert "small-wide" in result.detail
    assert "3%" in result.detail


@pytest.mark.parametrize("objective", ["binary", "multiclass"])
def test_candidate_is_rejected_when_classification_accuracy_drops(
    objective: str,
) -> None:
    rows = paired_records(
        objective=objective,
        current_loss=1.0,
        candidate_loss=0.98,
        current_accuracy=0.80,
        candidate_accuracy=0.779,
    )

    result = BENCHMARK.evaluate_candidate(rows, "quality_first")

    assert not result.passed
    assert "accuracy" in result.detail


def test_candidate_is_rejected_when_ndcg_at_10_drops() -> None:
    rows = paired_records(
        objective="ranking",
        current_loss=0.20,
        candidate_loss=0.19,
        current_ndcg=0.80,
        candidate_ndcg=0.779,
    )

    result = BENCHMARK.evaluate_candidate(rows, "quality_first")

    assert not result.passed
    assert "NDCG@10" in result.detail


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("primary_metric", float("nan")),
        ("accuracy", float("inf")),
        ("ndcg_at_10", float("-inf")),
        ("fit_seconds", float("nan")),
    ],
)
def test_candidate_is_rejected_for_non_finite_values(field: str, value: float) -> None:
    kwargs = {
        "objective": "ranking" if field == "ndcg_at_10" else "binary",
        "current_loss": 1.0,
        "candidate_loss": 0.98,
        "current_accuracy": 0.8,
        "candidate_accuracy": 0.8,
        "current_ndcg": 0.75 if field == "ndcg_at_10" else None,
        "candidate_ndcg": 0.75 if field == "ndcg_at_10" else None,
    }
    rows = paired_records(**kwargs)
    rows[-1] = BENCHMARK.BenchmarkRecord(
        **{**rows[-1].__dict__, field: value}
    )

    result = BENCHMARK.evaluate_candidate(rows, "quality_first")

    assert not result.passed
    assert field in result.detail
    assert "fixture" in result.detail
    assert "seed=7" in result.detail


def test_candidate_is_rejected_for_zero_rounds_or_recorded_error() -> None:
    zero_rounds = paired_records(candidate_rounds=0)
    errored = paired_records(candidate_error="native fit failed")

    zero_result = BENCHMARK.evaluate_candidate(zero_rounds, "quality_first")
    error_result = BENCHMARK.evaluate_candidate(errored, "quality_first")

    assert not zero_result.passed
    assert "completed_rounds" in zero_result.detail
    assert not error_result.passed
    assert "native fit failed" in error_result.detail


def test_candidate_is_rejected_by_a_worse_shape_median() -> None:
    rows = balanced_records(candidate_loss_ratio=0.98)
    for seed in (7, 13, 29):
        rows.extend(
            paired_records(
                fixture=f"small-wide-{seed}",
                shape_stratum="small-wide",
                seed=seed,
                current_loss=1.0,
                candidate_loss=1.001,
            )
        )

    result = BENCHMARK.evaluate_candidate(rows, "quality_first")

    assert not result.passed
    assert "shape median" in result.detail
    assert "small-wide" in result.detail


def test_candidate_requires_at_least_one_percent_overall_improvement() -> None:
    rows = balanced_records(candidate_loss_ratio=0.9901)

    result = BENCHMARK.evaluate_candidate(rows, "quality_first")

    assert not result.passed
    assert "1%" in result.detail


def test_exactly_one_percent_overall_improvement_qualifies() -> None:
    rows = balanced_records(candidate_loss_ratio=0.99)

    result = BENCHMARK.evaluate_candidate(rows, "quality_first")

    assert result.passed
    assert result.overall_loss_ratio == pytest.approx(0.99)


def test_behavioral_distance_precedes_fit_time_within_quality_band() -> None:
    rows = [
        *balanced_records(
            candidate_arm="no_gain_floor",
            candidate_loss_ratio=0.989,
            candidate_fit_seconds=2.0,
        ),
        *[
            row
            for row in balanced_records(
                candidate_arm="quality_first",
                candidate_loss_ratio=0.985,
                candidate_fit_seconds=0.5,
            )
            if row.arm != "current_auto"
        ],
        *[
            row
            for row in balanced_records(
                candidate_arm="manual_default",
                candidate_loss_ratio=0.995,
                candidate_fit_seconds=0.25,
            )
            if row.arm != "current_auto"
        ],
    ]

    result = BENCHMARK.evaluate_gates(rows, specs=BALANCED_SPECS, seeds=(7,))

    assert result.passed
    assert result.selected_arm == "no_gain_floor"


def test_fit_time_breaks_tie_only_after_quality_band_and_behavioral_distance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        BENCHMARK,
        "BEHAVIORAL_DISTANCE",
        {"no_gain_floor": 1, "quality_first": 1, "manual_default": 3},
    )
    rows = [
        *balanced_records(
            candidate_arm="no_gain_floor",
            candidate_loss_ratio=0.987,
            candidate_fit_seconds=2.0,
        ),
        *[
            row
            for row in balanced_records(
                candidate_arm="quality_first",
                candidate_loss_ratio=0.985,
                candidate_fit_seconds=0.5,
            )
            if row.arm != "current_auto"
        ],
        *[
            row
            for row in balanced_records(
                candidate_arm="manual_default",
                candidate_loss_ratio=0.995,
                candidate_fit_seconds=0.1,
            )
            if row.arm != "current_auto"
        ],
    ]

    result = BENCHMARK.evaluate_gates(rows, specs=BALANCED_SPECS, seeds=(7,))

    assert result.selected_arm == "quality_first"


def test_fit_time_cannot_rescue_candidate_outside_quality_band() -> None:
    rows = [
        *balanced_records(
            candidate_arm="quality_first",
            candidate_loss_ratio=0.98,
            candidate_fit_seconds=2.0,
        ),
        *[
            row
            for row in balanced_records(
                candidate_arm="no_gain_floor",
                candidate_loss_ratio=0.989,
                candidate_fit_seconds=0.01,
            )
            if row.arm != "current_auto"
        ],
        *[
            row
            for row in balanced_records(
                candidate_arm="manual_default",
                candidate_loss_ratio=0.995,
                candidate_fit_seconds=0.001,
            )
            if row.arm != "current_auto"
        ],
    ]

    result = BENCHMARK.evaluate_gates(rows, specs=BALANCED_SPECS, seeds=(7,))

    assert result.selected_arm == "quality_first"


def test_fit_time_cannot_rescue_candidate_that_fails_quality_gates() -> None:
    rows = [
        *balanced_records(
            candidate_arm="no_gain_floor",
            candidate_loss_ratio=0.989,
            candidate_fit_seconds=2.0,
        ),
        *[
            row
            for row in balanced_records(
                candidate_arm="quality_first",
                candidate_loss_ratio=0.995,
                candidate_fit_seconds=0.001,
            )
            if row.arm != "current_auto"
        ],
        *[
            row
            for row in balanced_records(
                candidate_arm="manual_default",
                candidate_loss_ratio=0.995,
                candidate_fit_seconds=0.001,
            )
            if row.arm != "current_auto"
        ],
    ]

    result = BENCHMARK.evaluate_gates(rows, specs=BALANCED_SPECS, seeds=(7,))

    assert result.selected_arm == "no_gain_floor"


def test_present_records_cannot_define_a_complete_matrix() -> None:
    rows = balanced_records(
        candidate_arm="quality_first", candidate_loss_ratio=0.995
    )

    result = BENCHMARK.evaluate_gates(rows, specs=BALANCED_SPECS, seeds=(7,))

    assert not result.passed
    assert not result.evidence_valid
    assert "manual_default" in result.detail
    assert "no_gain_floor" in result.detail


def test_evaluate_gates_requires_explicit_expected_matrix_context() -> None:
    rows = balanced_records(candidate_loss_ratio=0.995)

    with pytest.raises(TypeError, match="specs"):
        BENCHMARK.evaluate_gates(rows)


def test_complete_valid_evidence_keeps_current_when_no_candidate_qualifies() -> None:
    rows = [
        *balanced_records(
            candidate_arm="manual_default", candidate_loss_ratio=0.995
        ),
        *[
            row
            for row in balanced_records(
                candidate_arm="no_gain_floor", candidate_loss_ratio=0.995
            )
            if row.arm != "current_auto"
        ],
        *[
            row
            for row in balanced_records(
                candidate_arm="quality_first", candidate_loss_ratio=0.995
            )
            if row.arm != "current_auto"
        ],
    ]

    result = BENCHMARK.evaluate_gates(rows, specs=BALANCED_SPECS, seeds=(7,))

    assert result.passed
    assert result.selected_arm == "current_auto"
    assert "keep current" in result.detail


def test_matrix_completeness_reports_missing_fixture_seed_and_arm() -> None:
    specs = (
        BENCHMARK.FixtureSpec(
            name="expected",
            shape_stratum="small-narrow",
            objective="regression",
            rows=512,
            features=8,
            rounds=5,
        ),
    )
    rows = paired_records(candidate_arm="quality_first")

    result = BENCHMARK.evaluate_gates(rows, specs=specs, seeds=(7,))

    assert not result.passed
    assert "expected" in result.detail
    assert "manual_default" in result.detail
    assert "seed=7" in result.detail


def test_json_and_markdown_outputs_include_outcome_and_records(
    tmp_path: Path,
) -> None:
    rows = [
        *balanced_records(
            candidate_arm="manual_default", candidate_loss_ratio=0.995
        ),
        *[
            row
            for row in balanced_records(
                candidate_arm="no_gain_floor", candidate_loss_ratio=0.995
            )
            if row.arm != "current_auto"
        ],
        *[
            row
            for row in balanced_records(
                candidate_arm="quality_first", candidate_loss_ratio=0.995
            )
            if row.arm != "current_auto"
        ],
    ]
    result = BENCHMARK.evaluate_gates(rows, specs=BALANCED_SPECS, seeds=(7,))
    json_path = tmp_path / "records.json"
    report_path = tmp_path / "report.md"

    BENCHMARK.write_json(json_path, rows, result)
    BENCHMARK.write_report(report_path, rows, result, command="benchmark --gate")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    assert payload["gate"]["selected_arm"] == "current_auto"
    assert len(payload["records"]) == len(rows)
    assert "# Auto-Policy Calibration Benchmark" in report
    assert "benchmark --gate" in report
    assert "## Environment" in report
    assert "Python:" in report
    assert "Timing is descriptive only" in report


def test_markdown_report_records_complete_runtime_environment(
    tmp_path: Path,
) -> None:
    rows = [
        *balanced_records(
            candidate_arm="manual_default", candidate_loss_ratio=0.995
        ),
        *[
            row
            for row in balanced_records(
                candidate_arm="no_gain_floor", candidate_loss_ratio=0.995
            )
            if row.arm != "current_auto"
        ],
        *[
            row
            for row in balanced_records(
                candidate_arm="quality_first", candidate_loss_ratio=0.995
            )
            if row.arm != "current_auto"
        ],
    ]
    result = BENCHMARK.evaluate_gates(rows, specs=BALANCED_SPECS, seeds=(7,))
    report_path = tmp_path / "report.md"

    BENCHMARK.write_report(report_path, rows, result, command="benchmark --gate")

    report = report_path.read_text(encoding="utf-8")
    assert re.search(r"- Git commit: `[0-9a-f]{40}`", report)
    assert "- OS/platform: `" in report
    assert "- Architecture: `" in report
    assert "- Python: `" in report
    assert "- Rust: `rustc " in report
    assert "- NumPy: `" in report
    assert "- AlloyGBM: `" in report


def test_markdown_report_discloses_public_auto_split_l2_observations(
    tmp_path: Path,
) -> None:
    rows = [
        record(
            fixture="first",
            arm="current_auto",
            primary_metric=1.0,
            resolved_policy=sample_resolved_policy(effective_split_l2=0.0),
        ),
        record(
            fixture="second",
            arm="current_auto",
            primary_metric=1.0,
            resolved_policy=sample_resolved_policy(effective_split_l2=2.5),
        ),
    ]
    report_path = tmp_path / "report.md"
    result = BENCHMARK.GateResult(
        name="selection",
        passed=True,
        detail="keep current",
        selected_arm="current_auto",
    )

    BENCHMARK.write_report(report_path, rows, result, command="benchmark --gate")

    report = report_path.read_text(encoding="utf-8")
    assert "## Resolved Policy Observations" in report
    assert (
        "- Current-auto records activating automatic split-L2: `0 of 2`"
        in report
    )
    assert (
        "- Distinct current-auto effective split-L2 values: "
        "`0.000000, 2.500000`"
    ) in report
    assert (
        "Python public current-auto did not activate the engine-only "
        "auto split-L2 rule in this matrix."
    ) in report


def test_markdown_report_exposes_matrix_gates_ratios_decision_and_diagnostics(
    tmp_path: Path,
) -> None:
    rows = [
        *balanced_records(
            candidate_arm="manual_default", candidate_loss_ratio=0.995
        ),
        *[
            row
            for row in balanced_records(
                candidate_arm="no_gain_floor", candidate_loss_ratio=0.995
            )
            if row.arm != "current_auto"
        ],
        *[
            row
            for row in balanced_records(
                candidate_arm="quality_first", candidate_loss_ratio=0.995
            )
            if row.arm != "current_auto"
        ],
    ]
    result = BENCHMARK.evaluate_gates(rows, specs=BALANCED_SPECS, seeds=(7,))
    report_path = tmp_path / "report.md"

    BENCHMARK.write_report(report_path, rows, result, command="benchmark --gate")

    report = report_path.read_text(encoding="utf-8")
    assert "- Complete records: `24`" in report
    assert "- Distinct fixtures: `6`" in report
    assert "- Distinct objectives: `5`" in report
    assert "- Distinct seeds: `1`" in report
    assert "- Distinct arms: `4`" in report
    assert "## Candidate Gate Results" in report
    assert "| manual_default | fail | 0.995000 |" in report
    assert "| no_gain_floor | fail | 0.995000 |" in report
    assert "| quality_first | fail | 0.995000 |" in report
    assert "## Shape/Objective Loss Ratios" in report
    assert "| no_gain_floor | small-narrow | regression | 0.995000 |" in report
    assert "## Decision" in report
    assert "Keep the production auto-policy heuristics unchanged." in report
    assert "## Resolved Policy Diagnostics" in report
    assert "Min split gain" in report
    assert "Effective split-L2" in report


def test_invalid_records_still_write_standards_compliant_json(
    tmp_path: Path,
) -> None:
    rows = paired_records(candidate_error="fit failed")
    rows[-1] = BENCHMARK.BenchmarkRecord(
        **{**rows[-1].__dict__, "primary_metric": float("nan")}
    )
    specs = (
        BENCHMARK.FixtureSpec(
            "fixture", "small-narrow", "regression", 512, 8, 10
        ),
    )
    result = BENCHMARK.evaluate_gates(rows, specs=specs, seeds=(7,))
    json_path = tmp_path / "invalid.json"

    BENCHMARK.write_json(json_path, rows, result)

    payload = json.loads(
        json_path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant: {value}")
        ),
    )
    assert payload["records"][-1]["primary_metric"] is None
    assert payload["records"][-1]["error"] == "fit failed"


def test_ci_workflow_contains_compact_auto_policy_leg() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    condition = "matrix.os == 'ubuntu-latest' && matrix.python-version == '3.13'"

    assert "name: Auto-policy benchmark contract tests" in workflow
    assert "run: python -m pytest benchmarks/tests/test_auto_policy_benchmark.py -q" in workflow
    assert "name: Auto-policy benchmark sentinel gates" in workflow
    assert "run: python benchmarks/auto_policy_benchmark.py --quick --gate" in workflow
    assert workflow.count(f"if: {condition}") >= 4
