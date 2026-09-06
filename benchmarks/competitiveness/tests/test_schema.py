"""Tests for the versioned competitiveness benchmark result contract."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from benchmarks.competitiveness.schema import (
    METRIC_DIRECTIONS,
    SCHEMA_VERSION,
    BenchmarkRecordV1,
    BenchmarkSummaryV1,
    ProfileRecordV1,
    load_records,
    validate_record,
    validate_summary,
)


def profile() -> ProfileRecordV1:
    return ProfileRecordV1(
        rows=128,
        features=8,
        rounds=4,
        threads=2,
        loop_wall_ns=10_000,
        untimed_ns=900,
        stage_ns={label: (100 if label == "gradients" else 9_000 if label == "tree_build" else 0)
                  for label in ("gradients", "row_sampling", "feature_tiles", "prediction_copy", "tree_build", "prediction_update", "loss", "validation")},
        tree_stage_ns={label: (4_000 if label == "histogram_build" else 1_000 if label == "split_find" else 0)
                       for label in ("histogram_build", "split_find", "partition")},
    )


def record(**changes: object) -> BenchmarkRecordV1:
    values: dict[str, object] = {
        "schema": SCHEMA_VERSION,
        "run_id": "run-1",
        "repetition": 0,
        "dataset_sha256": "a" * 64,
        "scenario": "dense_regression",
        "task": "regression",
        "library": "alloygbm",
        "library_version": "1.0.0",
        "git_sha": None,
        "seed": 20260904,
        "threads": 2,
        "effective_params": {
            "n_estimators": 4,
            "max_depth": 6,
            "nested": {"search": [1, {"value": "kept"}]},
        },
        "input_representation": "dense",
        "preprocessing_seconds": 0.001,
        "fit_seconds": 0.1,
        "predict_seconds": 0.01,
        "peak_rss_bytes": 100_000,
        "metric_name": "rmse",
        "metric_value": 0.5,
        "rounds_completed": 4,
        "machine": {"hostname": "host-a", "platform": "darwin", "arch": "arm64"},
        "profile": profile(),
    }
    values.update(changes)
    return BenchmarkRecordV1(**values)  # type: ignore[arg-type]


def summary(**changes: object) -> BenchmarkSummaryV1:
    values: dict[str, object] = {
        "schema": SCHEMA_VERSION,
        "run_id": "run-1",
        "scenario": "dense_regression",
        "task": "regression",
        "library": "alloygbm",
        "library_version": "1.0.0",
        "threads": 2,
        "dataset_sha256": "a" * 64,
        "input_representation": "dense",
        "metric_name": "rmse",
        "metric_median": 0.5,
        "metric_mad": 0.01,
        "preprocessing_median_seconds": 0.001,
        "preprocessing_mad_seconds": 0.0001,
        "fit_median_seconds": 0.1,
        "fit_mad_seconds": 0.01,
        "predict_median_seconds": 0.01,
        "predict_mad_seconds": 0.001,
        "peak_rss_median_bytes": 100_000,
        "peak_rss_mad_bytes": 1_000,
        "raw_repetition_ids": (0, 1, 2, 3, 4),
    }
    values.update(changes)
    return BenchmarkSummaryV1(**values)  # type: ignore[arg-type]


def test_metric_registry_has_the_versioned_directions() -> None:
    assert METRIC_DIRECTIONS == {
        "rmse": "minimize",
        "mae": "minimize",
        "log_loss": "minimize",
        "error_rate": "minimize",
        "r2": "maximize",
        "accuracy": "maximize",
        "roc_auc": "maximize",
        "ndcg_at_10": "maximize",
    }


def test_record_round_trips_with_explicit_json_and_is_frozen() -> None:
    original = record()
    decoded = BenchmarkRecordV1.from_json(original.to_json())
    assert decoded == original
    assert decoded.task == "regression"
    assert json.loads(original.to_json())["profile"]["stage_ns"]["gradients"] == 100
    with pytest.raises(FrozenInstanceError):
        original.fit_seconds = 2.0  # type: ignore[misc]


def test_record_nested_values_are_defensively_frozen() -> None:
    original = record()
    with pytest.raises(TypeError):
        original.effective_params["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        original.effective_params["nested"]["search"] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        original.effective_params["nested"]["search"][1]["value"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        original.machine["platform"] = "linux"  # type: ignore[index]
    with pytest.raises(TypeError):
        original.profile.stage_ns["gradients"] = 200  # type: ignore[union-attr,index]
    with pytest.raises(TypeError):
        original.profile.tree_stage_ns["split_find"] = 200  # type: ignore[union-attr,index]


def test_single_record_jsonl_loads_successfully(tmp_path: Path) -> None:
    path = tmp_path / "one-record.jsonl"
    path.write_text(record().to_json() + "\n")
    assert load_records(path) == [record()]


def test_validate_record_rejects_missing_dataset_fingerprint() -> None:
    with pytest.raises(ValueError, match="dataset_sha256"):
        validate_record(replace(record(), dataset_sha256=""))


def test_validate_record_rejects_blank_task() -> None:
    with pytest.raises(ValueError, match="task"):
        validate_record(replace(record(), task="  "))


@pytest.mark.parametrize(
    "machine",
    [{}, {"platform": "darwin"}, {"hostname": "  "}, {"hostname": "host", "platform": ""}, {1: "host"}],
)
def test_validate_record_requires_nonempty_hostname_machine_metadata(machine: object) -> None:
    with pytest.raises(ValueError, match="machine"):
        validate_record(replace(record(), machine=machine))  # type: ignore[arg-type]


@pytest.mark.parametrize("params", [{"set": {1, 2}}, {"bytes": b"x"}, {1: "non-string-key"}])
def test_validate_record_rejects_non_json_like_effective_params(params: object) -> None:
    with pytest.raises(ValueError, match="JSON-like|effective_params"):
        validate_record(replace(record(), effective_params=params))  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["preprocessing_seconds", "fit_seconds", "predict_seconds"])
def test_validate_record_rejects_nonpositive_durations(field: str) -> None:
    with pytest.raises(ValueError, match="duration"):
        validate_record(replace(record(), **{field: 0.0}))


def test_validate_record_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        validate_record(replace(record(), metric_name="made_up"))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_validate_record_rejects_nonfinite_numbers(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_record(replace(record(), metric_value=value))


def test_load_records_rejects_mismatched_library_versions(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                json.loads(record().to_json()),
                json.loads(replace(record(), repetition=1, library_version="1.1.0").to_json()),
            )
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="library_version"):
        load_records(path)


def test_load_records_rejects_duplicate_repetition_keys(tmp_path: Path) -> None:
    path = tmp_path / "records.json"
    path.write_text(json.dumps([json.loads(record().to_json()), json.loads(record().to_json())]))
    with pytest.raises(ValueError, match="duplicate"):
        load_records(path)


def test_same_repetition_at_different_thread_counts_is_distinct(tmp_path: Path) -> None:
    path = tmp_path / "thread-sweep.jsonl"
    path.write_text(
        "\n".join(
            replace(record(), threads=threads).to_json() for threads in (1, 4)
        )
        + "\n"
    )
    assert [item.threads for item in load_records(path)] == [1, 4]


@pytest.mark.parametrize("fingerprint", ["", "a" * 63, "A" * 64, "g" * 64])
def test_validate_record_requires_lowercase_sha256_fingerprint(fingerprint: str) -> None:
    with pytest.raises(ValueError, match="dataset_sha256"):
        validate_record(replace(record(), dataset_sha256=fingerprint))


def test_summary_threads_are_grouping_keys() -> None:
    one = summary(threads=1)
    four = summary(threads=4)
    assert one != four
    assert "threads" in one.grouping_keys
    assert "task" in one.grouping_keys
    assert one.to_dict()["threads"] == 1


def test_summary_round_trip_preserves_task() -> None:
    original = summary(task="ranking")
    decoded = BenchmarkSummaryV1.from_json(original.to_json())
    assert decoded.task == "ranking"
    assert decoded == original


def test_validate_summary_rejects_blank_task() -> None:
    with pytest.raises(ValueError, match="task"):
        validate_summary(replace(summary(), task=""))


def test_validate_summary_rejects_unknown_input_representation() -> None:
    with pytest.raises(ValueError, match="input_representation"):
        validate_summary(replace(summary(), input_representation="unknown"))


@pytest.mark.parametrize(
    "machine",
    [{}, {"platform": "darwin"}, {"hostname": "  "}, {"hostname": "host", "platform": ""}, {1: "host"}],
)
def test_validate_summary_requires_nonempty_hostname_machine_metadata(machine: object) -> None:
    with pytest.raises(ValueError, match="machine"):
        validate_summary(replace(summary(), machine=machine))  # type: ignore[arg-type]


@pytest.mark.parametrize("params", [{"set": {1, 2}}, {"bytes": b"x"}, {1: "non-string-key"}])
def test_validate_summary_rejects_non_json_like_effective_params(params: object) -> None:
    with pytest.raises(ValueError, match="JSON-like|effective_params"):
        validate_summary(replace(summary(), effective_params=params))  # type: ignore[arg-type]


def test_summary_requires_raw_repetition_ids() -> None:
    with pytest.raises(ValueError, match="raw_repetition_ids"):
        validate_summary(replace(summary(), raw_repetition_ids=()))


def test_summary_round_trip_preserves_repetition_ids() -> None:
    original = summary()
    decoded = BenchmarkSummaryV1.from_json(original.to_json())
    assert decoded == original
    assert decoded.raw_repetition_ids == (0, 1, 2, 3, 4)


def test_summary_round_trip_preserves_profiled_provenance() -> None:
    original = summary(profiled=False)
    decoded = BenchmarkSummaryV1.from_json(original.to_json())
    assert decoded.profiled is False
    assert json.loads(original.to_json())["profiled"] is False


@pytest.mark.parametrize("value", ["false", 0, 1, [], {}])
def test_validate_summary_rejects_non_boolean_profiled(value: object) -> None:
    with pytest.raises(ValueError, match="profiled"):
        validate_summary(replace(summary(), profiled=value))  # type: ignore[arg-type]


def test_profile_stage_maps_reject_unknown_labels() -> None:
    bad = replace(profile(), stage_ns={"not_a_stage": 1})
    with pytest.raises(ValueError, match="stage"):
        validate_record(replace(record(), profile=bad))


def test_profile_requires_untimed_ns_and_exact_stage_maps() -> None:
    payload = json.loads(profile().to_json())
    payload.pop("untimed_ns")
    with pytest.raises((KeyError, ValueError), match="untimed_ns"):
        ProfileRecordV1.from_dict(payload)
    missing_stage = replace(profile(), stage_ns={"gradients": 1})
    with pytest.raises(ValueError, match="exact|missing|stage"):
        validate_record(replace(record(), profile=missing_stage))
    missing_tree_stage = replace(profile(), tree_stage_ns={"histogram_build": 1})
    with pytest.raises(ValueError, match="exact|missing|tree_stage"):
        validate_record(replace(record(), profile=missing_tree_stage))


@pytest.mark.parametrize("field", ["untimed_ns", "loop_wall_ns"])
def test_profile_rejects_bool_or_negative_durations(field: str) -> None:
    for value in (True, -1):
        with pytest.raises(ValueError):
            validate_record(replace(record(), profile=replace(profile(), **{field: value})))


def test_smoke_manifest_is_deterministic_and_complete() -> None:
    import yaml

    manifest_path = Path(__file__).parents[1] / "manifests" / "pr_smoke.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    assert manifest["schema"] == SCHEMA_VERSION
    assert manifest["seed"] == 20260904
    assert manifest["warmup_repetitions"] == 1
    assert manifest["timed_repetitions"] == 5
    assert manifest["scenarios"] == [
        {
            "name": "dense_regression",
            "task": "regression",
            "rows": 4096,
            "features": 40,
            "rounds": 40,
            "depth": 6,
            "metric": "rmse",
            "input_representation": "dense",
        },
        {
            "name": "binary",
            "task": "binary_classification",
            "rows": 4096,
            "features": 40,
            "rounds": 40,
            "depth": 6,
            "metric": "log_loss",
            "input_representation": "dense",
        },
        {
            "name": "grouped_ranking",
            "task": "ranking",
            "rows": 4096,
            "features": 30,
            "groups": 128,
            "rounds": 40,
            "depth": 6,
            "metric": "ndcg_at_10",
            "input_representation": "dense",
        },
        {
            "name": "native_categorical",
            "task": "binary_classification",
            "rows": 4096,
            "numeric_features": 20,
            "categorical_cardinalities": [16, 256],
            "rounds": 40,
            "depth": 6,
            "metric": "log_loss",
            "input_representation": "native_categorical",
        },
        {
            "name": "csr_sparse",
            "task": "regression",
            "rows": 4096,
            "features": 1000,
            "density": 0.01,
            "rounds": 40,
            "depth": 6,
            "metric": "rmse",
            "input_representation": "csr",
        },
        {
            "name": "joint_multi_output",
            "task": "multi_output_regression",
            "rows": 4096,
            "features": 30,
            "outputs": 8,
            "rounds": 40,
            "depth": 6,
            "metric": "rmse",
            "input_representation": "dense",
        },
    ]
