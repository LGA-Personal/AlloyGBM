"""Tests for auditable accuracy-at-depth measurement and replay."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from accuracy_depth_sweep import (  # noqa: E402
    DEFAULT_MODELS,
    DEFAULT_SCENARIOS,
    GUARD_SCENARIOS,
    analyse_cells,
    build_runner_command,
    classify_verdict,
    parse_arm_specs,
    parse_seeds,
    normalise_gap,
    _render_analysis_report,
    validate_external_baseline,
    validate_records,
    verify_resume_compatibility,
)


def _record(scenario, model, seed, value=1.0, *, task="regression", **extra):
    record = {
        "scenario": scenario,
        "task_type": task,
        "profile_name": "single",
        "profile_index": 1,
        "run_index": 1,
        "seed": seed,
        "learning_rate": 0.1,
        "max_depth": 6,
        "rounds": 120,
        "model": model,
        "rmse": value,
        "mae": value,
        "log_loss_val": value,
        "ndcg_10": value,
        "status": "PASS",
        "requested_alloy_param_overrides": {},
        "constructed_estimator_params": {"max_depth": 6} if model == "alloygbm" else None,
        "resolved_training_policy": {"mode": "auto"} if model == "alloygbm" else None,
    }
    record.update(extra)
    return record


def _payload(records):
    return {"run_id": "20260919T000000Z", "records": records}


def _manifest(*, config=None, source="src-a", native="native-a", dataset="data-a"):
    return {
        "arms": {"baseline": {"overrides": [], "parsed_overrides": {}}},
        "config": config or {
            "depths": [6],
            "seeds": [11, 12],
            "scenarios": ["dense_numeric"],
            "models": ["alloygbm", "lightgbm"],
            "rounds": 120,
            "learning_rate": 0.1,
            "threads": 1,
            "binning_strategy": "linear",
            "bin_count": 256,
        },
        "source_state": {"source_fingerprint_sha256": source, "git_sha": "abc"},
        "runtime": {"native_module_sha256": native, "package_sha256": "pkg"},
        "dataset_identities": {"dense_numeric": {"sha256": dataset}},
    }


def test_gap_inside_band_is_a_tie():
    assert classify_verdict(-3.0, 9.3) == "tie"
    assert classify_verdict(+3.0, 9.3) == "tie"


def test_gap_outside_band_takes_its_sign():
    assert classify_verdict(-40.0, 9.3) == "WIN"
    assert classify_verdict(+40.0, 9.3) == "loss"


def test_band_boundary_is_a_tie():
    assert classify_verdict(9.3, 9.3) == "tie"
    assert classify_verdict(-9.3, 9.3) == "tie"


def test_gap_sign_is_normalised_for_both_metric_directions():
    assert normalise_gap(0.9, 1.0, lower_is_better=True) < 0
    assert normalise_gap(0.9, 0.8, lower_is_better=False) < 0


def test_zero_reference_has_undefined_relative_gap_and_absolute_difference():
    assert normalise_gap(0.25, 0.0, lower_is_better=True) is None


def test_guard_scenarios_are_explicit_and_default_suite_excludes_known_bad_cell():
    assert GUARD_SCENARIOS == ("histogram_stress", "panel_time_series")
    assert DEFAULT_MODELS == ("alloygbm", "lightgbm", "xgboost", "catboost")
    assert "histogram_stress" in DEFAULT_SCENARIOS
    assert "panel_time_series" in DEFAULT_SCENARIOS
    assert "synthetic_categorical" not in DEFAULT_SCENARIOS


def test_user_selected_synthetic_categorical_fails_truthfully():
    with pytest.raises(ValueError, match="numeric/finite filtering"):
        validate_records(
            _payload([_record("synthetic_categorical", "alloygbm", 11)]),
            scenarios=["synthetic_categorical"],
            models=["alloygbm"],
            seed=11,
            depth=6,
            rounds=120,
            learning_rate=0.1,
        )


def test_duplicate_arm_names_or_keys_are_rejected():
    with pytest.raises(ValueError, match="duplicate arm"):
        parse_arm_specs(["base:", "base:min_child_hessian=1"])
    with pytest.raises(ValueError, match="specified more than once"):
        parse_arm_specs(["probe:min_child_hessian=1,min_child_hessian=2"])
    with pytest.raises(ValueError, match="safe arm name"):
        parse_arm_specs(["../escape:min_child_hessian=1"])


def test_named_empty_arm_and_alloy_overrides_parse():
    arms = parse_arm_specs(["baseline:", "hess1:min_child_hessian=1.0"])
    assert arms["baseline"]["overrides"] == []
    assert arms["hess1"]["overrides"] == ["min_child_hessian=1.0"]
    assert arms["hess1"]["parsed_overrides"] == {"min_child_hessian": 1.0}


def test_seed_count_and_explicit_seed_list_are_unambiguous():
    assert parse_seeds("5", 20260919) == [20260919, 20260920, 20260921, 20260922, 20260923]
    assert parse_seeds("11,17,29", 20260919) == [11, 17, 29]
    with pytest.raises(ValueError, match="duplicate seed"):
        parse_seeds("11,11", 20260919)


def test_runner_command_uses_shared_controls_and_alloy_only_overrides(tmp_path):
    args = {
        "scenarios": ["dense_numeric", "histogram_stress"],
        "models": ["alloygbm", "lightgbm"],
        "threads": 1,
        "rounds": 400,
        "learning_rate": 0.05,
        "binning_strategy": "rank",
        "bin_count": 96,
    }
    command = build_runner_command(
        repo_root=Path("/repo"),
        target_dir=tmp_path,
        overrides=["min_child_hessian=1.0"],
        depth=12,
        seed=17,
        args=args,
    )
    assert command[:3] == [sys.executable, "-B", "benchmarks/run_model_comparison.py"]
    assert command[command.index("--max-depth") + 1] == "12"
    assert command[command.index("--seed") + 1] == "17"
    assert command[command.index("--rounds") + 1] == "400"
    assert command[command.index("--learning-rate") + 1] == "0.05"
    assert command[command.index("--alloy-continuous-binning-strategy") + 1] == "rank"
    assert command[command.index("--alloy-continuous-binning-max-bins") + 1] == "96"
    assert command[command.index("--alloy-param") + 1] == "min_child_hessian=1.0"
    assert command[command.index("--scenarios") + 1 : command.index("--models")] == [
        "dense_numeric", "histogram_stress"
    ]


def test_missing_duplicate_failed_and_nonfinite_cells_are_rejected():
    one = _record("dense_numeric", "alloygbm", 11)
    with pytest.raises(ValueError, match="missing requested result cell"):
        validate_records(_payload([one]), ["dense_numeric"], ["alloygbm", "lightgbm"], 11, 6, 120, 0.1)
    with pytest.raises(ValueError, match="duplicate result cell"):
        validate_records(_payload([one, one]), ["dense_numeric"], ["alloygbm"], 11, 6, 120, 0.1)
    failed = _record("dense_numeric", "alloygbm", 11, status="FAIL")
    with pytest.raises(ValueError, match="status FAIL"):
        validate_records(_payload([failed]), ["dense_numeric"], ["alloygbm"], 11, 6, 120, 0.1)
    nonfinite = _record("dense_numeric", "alloygbm", 11, rmse=float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        validate_records(_payload([nonfinite]), ["dense_numeric"], ["alloygbm"], 11, 6, 120, 0.1)


def test_unknown_tasks_and_profile_mixing_are_rejected():
    unknown = _record("dense_numeric", "alloygbm", 11, task="ranking")
    with pytest.raises(ValueError, match="task_type mismatch"):
        validate_records(_payload([unknown]), ["dense_numeric"], ["alloygbm"], 11, 6, 120, 0.1)
    mixed = _record("dense_numeric", "alloygbm", 11, profile_name="deep_low_lr")
    with pytest.raises(ValueError, match="unrequested profile"):
        validate_records(_payload([mixed]), ["dense_numeric"], ["alloygbm"], 11, 6, 120, 0.1)


def test_paired_seed_analysis_uses_per_scenario_spread_and_detects_guard_regression():
    baseline = {}
    candidate = {}
    peer = {}
    for seed, b, c, p in ((11, 1.0, 1.2, 1.05), (12, 1.1, 1.3, 1.0)):
        baseline[("histogram_stress", "alloygbm", seed)] = {"metric": "rmse", "value": b}
        candidate[("histogram_stress", "alloygbm", seed)] = {"metric": "rmse", "value": c}
        candidate[("histogram_stress", "lightgbm", seed)] = {"metric": "rmse", "value": p}
    analysis = analyse_cells(
        candidate=candidate,
        baseline=baseline,
        scenarios=["histogram_stress"],
        models=["alloygbm", "lightgbm"],
        seeds=[11, 12],
        baseline_arm="baseline",
    )
    comparison = analysis["scenarios"]["histogram_stress"]["candidate_vs_baseline"]
    assert comparison["paired_absolute_deltas"] == [
        {"seed": 11, "candidate": 1.2, "reference": 1.0, "delta": pytest.approx(0.2)},
        {"seed": 12, "candidate": 1.3, "reference": 1.1, "delta": pytest.approx(0.2)},
    ]
    assert comparison["verdict"] == "loss"
    assert analysis["guards"]["histogram_stress"]["status"] == "REGRESSION"
    assert analysis["guards"]["panel_time_series"]["status"] == "NOT_RUN"


def test_analyse_cells_preserves_raw_seed_metrics_and_zero_reference_absolute_gap():
    candidate = {
        ("dense_numeric", "alloygbm", 11): {"metric": "rmse", "value": 0.25},
        ("dense_numeric", "alloygbm", 12): {"metric": "rmse", "value": 0.30},
    }
    baseline = {
        ("dense_numeric", "alloygbm", 11): {"metric": "rmse", "value": 0.0},
        ("dense_numeric", "alloygbm", 12): {"metric": "rmse", "value": 0.0},
    }
    summary = analyse_cells(candidate, baseline, ["dense_numeric"], ["alloygbm"], [11, 12], "baseline")
    row = summary["scenarios"]["dense_numeric"]["candidate_vs_baseline"]
    assert row["candidate"]["values_by_seed"] == {"11": 0.25, "12": 0.3}
    assert row["absolute_delta_median"] == pytest.approx(0.275)
    assert row["relative_gap_pct"] is None
    assert row["verdict"] is None


def test_markdown_report_includes_candidate_peer_and_absolute_seed_spreads():
    baseline = {}
    candidate = {}
    for seed, baseline_value, candidate_value, peer_value in (
        (11, 1.0, 1.2, 1.05),
        (12, 1.1, 1.3, 1.0),
    ):
        baseline[("dense_numeric", "alloygbm", seed)] = {
            "metric": "rmse",
            "value": baseline_value,
        }
        candidate[("dense_numeric", "alloygbm", seed)] = {
            "metric": "rmse",
            "value": candidate_value,
        }
        candidate[("dense_numeric", "lightgbm", seed)] = {
            "metric": "rmse",
            "value": peer_value,
        }
    analysis = analyse_cells(
        candidate,
        baseline,
        ["dense_numeric"],
        ["alloygbm", "lightgbm"],
        [11, 12],
        "baseline",
    )
    report = _render_analysis_report(
        {
            "evidence_class": "smoke",
            "config": {
                "seeds": [11, 12],
                "scenarios": ["dense_numeric"],
                "models": ["alloygbm", "lightgbm"],
                "depths": [6],
                "rounds": 120,
                "learning_rate": 0.1,
                "threads": 1,
            },
        },
        {"d6": {"probe": analysis}},
        None,
    )
    assert "Candidate median / spread" in report
    assert "| dense_numeric | probe | baseline | rmse |" in report
    assert "| dense_numeric | probe | lightgbm | rmse |" in report
    assert "0.1 (8.00%)" in report


def test_resume_requires_identical_config_source_runtime_and_dataset():
    saved = _manifest()
    verify_resume_compatibility(saved, _manifest())
    for changed in (
        _manifest(config={**saved["config"], "rounds": 200}),
        _manifest(source="source-b"),
        _manifest(dataset="data-b"),
    ):
        with pytest.raises(ValueError, match="resume mismatch"):
            verify_resume_compatibility(saved, changed)


def test_resume_rejects_a_changed_external_baseline_reference():
    saved = _manifest()
    current = _manifest()
    saved["external_baseline"] = {
        "path": "/experiments/cap-a",
        "manifest_sha256": "manifest-a",
        "arm": "baseline",
    }
    current["external_baseline"] = {
        "path": "/experiments/cap-b",
        "manifest_sha256": "manifest-b",
        "arm": "baseline",
    }
    with pytest.raises(ValueError, match="resume mismatch.*external baseline"):
        verify_resume_compatibility(saved, current)


def test_external_baseline_requires_matched_configuration_and_datasets_but_allows_treatment_build():
    candidate = _manifest(source="candidate-src", native="candidate-native")
    baseline = _manifest(source="baseline-src", native="baseline-native")
    treatment = validate_external_baseline(candidate, baseline, baseline_arm="baseline")
    assert treatment["source_changed"] is True
    assert treatment["native_module_changed"] is True
    with pytest.raises(ValueError, match="external baseline mismatch.*rounds"):
        bad_config = _manifest(config={**baseline["config"], "rounds": 400})
        validate_external_baseline(candidate, bad_config, baseline_arm="baseline")
    with pytest.raises(ValueError, match="external baseline mismatch.*dataset"):
        validate_external_baseline(candidate, _manifest(dataset="different"), baseline_arm="baseline")
