from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping

import yaml

from benchmarks.competitiveness.schema import (
    BenchmarkSummaryV1,
    METRIC_DIRECTIONS,
    RunMetadataV1,
    SCHEMA_VERSION,
    load_records,
    load_run_metadata,
)
from benchmarks.competitiveness.datasets import build_dataset_cases
from benchmarks.competitiveness.run import load_manifest
from benchmarks.competitiveness.summarize import summarize_file


ROOT = Path(__file__).parents[3]
BASELINES = ROOT / "benchmarks" / "competitiveness" / "baselines"


def test_committed_baseline_is_complete_traceable_and_round_trips() -> None:
    raw_path = BASELINES / "adfa2c8-pr-smoke.jsonl"
    summary_path = BASELINES / "adfa2c8-pr-smoke.summary.json"
    records = load_records(raw_path)
    manifest = load_manifest(ROOT / "benchmarks" / "competitiveness" / "manifests" / "pr_smoke.yaml")
    specs = [item for item in manifest["scenarios"] if isinstance(item, Mapping)]
    cases = {case.name: case for case in build_dataset_cases(specs, int(manifest["seed"]))}
    expected_scenarios = set(cases)
    expected_libraries = {"alloygbm", "lightgbm", "xgboost", "catboost"}
    expected_versions = {
        "alloygbm": "1.0.0",
        "lightgbm": "4.6.0",
        "xgboost": "3.2.0",
        "catboost": "1.2.10",
    }
    assert len(records) == 120
    assert {record.git_sha for record in records} == {
        "adfa2c8e593cea68b124e7975f3b4fd9f862a148"
    }
    assert len({record.run_id for record in records}) == 1
    assert len({tuple(sorted(record.machine.items())) for record in records}) == 1
    assert {record.repetition for record in records} == {0, 1, 2, 3, 4}
    assert all(record.effective_params for record in records)
    assert {record.scenario for record in records} == expected_scenarios
    assert {record.library for record in records} == expected_libraries
    assert {record.library: record.library_version for record in records} == expected_versions
    populations = Counter((record.scenario, record.library) for record in records)
    assert set(populations) == {
        (scenario, library)
        for scenario in expected_scenarios
        for library in expected_libraries
    }
    assert len(populations) == 24 and set(populations.values()) == {5}
    for record in records:
        case = cases[record.scenario]
        assert record.dataset_sha256 == case.dataset_sha256
        assert record.task == case.task
        assert record.metric_name == case.metric_name
        allowed_inputs = {case.input_representation}
        if record.scenario == "csr_sparse" and record.library == "alloygbm":
            allowed_inputs.add("dense_fallback")
        assert record.input_representation in allowed_inputs

    payload = json.loads(summary_path.read_text())
    assert payload["schema"] == SCHEMA_VERSION
    assert payload["status"] == "insufficient-data"
    assert len(payload["summaries"]) == 24
    raw_lines = raw_path.read_text().splitlines()
    referenced_lines: list[int] = []
    for encoded in payload["summaries"]:
        summary = BenchmarkSummaryV1.from_json(json.dumps(encoded, sort_keys=True))
        assert summary.raw_line_numbers is not None
        assert summary.raw_repetition_ids == (0, 1, 2, 3, 4)
        assert len(summary.raw_line_numbers) == 5
        referenced_lines.extend(summary.raw_line_numbers)
        summary_records = [records[line_number - 1] for line_number in summary.raw_line_numbers]
        assert [record.repetition for record in summary_records] == list(summary.raw_repetition_ids)
        assert len(summary_records) == 5
        for index, record in enumerate(summary_records):
            assert record.run_id == summary.run_id
            assert record.scenario == summary.scenario
            assert record.task == summary.task
            assert record.library == summary.library
            assert record.library_version == summary.library_version
            assert record.threads == summary.threads
            assert record.dataset_sha256 == summary.dataset_sha256
            assert record.input_representation == summary.input_representation
            assert record.metric_name == summary.metric_name
            assert METRIC_DIRECTIONS[record.metric_name] == summary.metric_direction
            assert dict(record.machine) == dict(summary.machine or {})
            assert dict(record.effective_params) == dict(summary.effective_params or {})
            assert json.loads(raw_lines[summary.raw_line_numbers[index] - 1]) == record.to_dict()
    assert sorted(referenced_lines) == list(range(1, 121))
    regenerated = summarize_file(raw_path)
    regenerated_rows = [summary.to_dict() for summary in regenerated]
    assert payload["summaries"] == regenerated_rows
    expected_payload = {
        "schema": SCHEMA_VERSION,
        "status": "insufficient-data",
        "claim": None,
        "summaries": regenerated_rows,
    }
    assert summary_path.read_text() == json.dumps(
        expected_payload, allow_nan=False, sort_keys=True, indent=2
    ) + "\n"

    metadata = load_run_metadata(BASELINES / "adfa2c8-pr-smoke.run-metadata.json")
    assert metadata.run_id == records[0].run_id
    assert metadata.measured_git_sha == "adfa2c8e593cea68b124e7975f3b4fd9f862a148"
    assert metadata.harness_git_sha == "7082301fcd79bac3e1f05e696c376588158eaee3"
    assert metadata.manifest_identifier == "benchmarks/competitiveness/manifests/pr_smoke.yaml"
    assert metadata.harness_source_path == "benchmarks/competitiveness"
    assert metadata.manifest_path == metadata.manifest_identifier
    assert metadata.working_directory == "repository-root"
    assert "/Users/" not in metadata.to_json()
    assert metadata.manifest_sha256 == "02bb62801670d5104aa44a304bafa4d2469a6c66191deb3e872424df89453fce"
    assert metadata.raw_record_count == len(records)
    assert metadata.raw_sha256 == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert metadata.libraries == ("alloygbm", "lightgbm", "xgboost", "catboost")
    assert metadata.scenarios == tuple(cases)


def test_deep_scaling_fixture_identity_is_runnable_and_published_fixture_is_distinct() -> None:
    deep_manifest = load_manifest(ROOT / "benchmarks" / "competitiveness" / "manifests" / "deep_scaling.yaml")
    deep_specs = [item for item in deep_manifest["scenarios"] if isinstance(item, Mapping)]
    assert len(deep_specs) == 27
    assert {item["fixture"] for item in deep_specs} == {"nightly_dense"}
    small_specs = [dict(item, rows=32, rounds=2) for item in deep_specs]
    small_cases = build_dataset_cases(small_specs, int(deep_manifest["seed"]))
    assert len(small_cases) == 27
    assert all(case.input_representation == "dense" for case in small_cases)
    assert all(case.X_train.shape[0] > 0 and case.X_test.shape[0] > 0 for case in small_cases)

    published_manifest = load_manifest(ROOT / "benchmarks" / "competitiveness" / "manifests" / "published_v1_crosscheck.yaml")
    published_spec = [item for item in published_manifest["scenarios"] if isinstance(item, Mapping)][0]
    published_case = build_dataset_cases([published_spec], int(published_manifest["seed"]))[0]
    assert published_spec["fixture"] == "published_deep_scaling_v1"
    assert published_case.X_train.shape == (400000, 40)
    assert published_case.X_test.shape == (100000, 40)
    assert published_case.dataset_sha256 != small_cases[0].dataset_sha256


def test_published_crosscheck_artifacts_bind_to_exact_five_row_capture() -> None:
    raw_path = BASELINES / "adfa2c8-published-v1-crosscheck.jsonl"
    summary_path = BASELINES / "adfa2c8-published-v1-crosscheck.summary.json"
    metadata_path = BASELINES / "adfa2c8-published-v1-crosscheck.run-metadata.json"
    records = load_records(raw_path)
    metadata = load_run_metadata(metadata_path)
    assert len(records) == 5
    assert {record.library for record in records} == {"alloygbm"}
    assert {record.git_sha for record in records} == {"adfa2c8e593cea68b124e7975f3b4fd9f862a148"}
    assert metadata.run_id == records[0].run_id
    assert metadata.harness_git_sha == "88f754c9f3f2d17d8e929842923d6a3760ebbc09"
    assert metadata.manifest_identifier.endswith("published_v1_crosscheck.yaml")
    assert metadata.manifest_path == metadata.manifest_identifier
    assert metadata.harness_source_path == "benchmarks/competitiveness"
    assert metadata.working_directory == "repository-root"
    assert "/Users/" not in metadata.to_json()
    assert metadata.raw_sha256 == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    payload = json.loads(summary_path.read_text())
    assert payload["status"] == "insufficient-data"
    assert len(payload["summaries"]) == 1
    summary = BenchmarkSummaryV1.from_json(json.dumps(payload["summaries"][0], sort_keys=True))
    assert summary.raw_repetition_ids == (0, 1, 2, 3, 4)
    assert summary.raw_line_numbers == (1, 2, 3, 4, 5)
    assert [record.metric_value for record in records] == [0.16377460956573486] * 5


def test_alloy_only_smoke_command_emits_six_records_without_peers(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.competitiveness.run",
            "--manifest",
            "benchmarks/competitiveness/manifests/pr_smoke.yaml",
            "--output-dir",
            str(output),
            "--smoke",
            "--libraries",
            "alloygbm",
            "--threads",
            "1",
            "--repetitions",
            "1",
            "--warmups",
            "0",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    paths = list(output.glob("*/raw.jsonl"))
    assert len(paths) == 1
    records = load_records(paths[0])
    assert len(records) == 6
    assert {record.library for record in records} == {"alloygbm"}
    assert {record.repetition for record in records} == {0}
    assert len({record.scenario for record in records}) == 6


def test_ci_workflow_has_event_scoped_smoke_and_observational_comparator() -> None:
    workflow_path = ROOT / ".github" / "workflows" / "ci.yml"
    workflow_text = workflow_path.read_text()
    workflow = yaml.load(workflow_text, Loader=yaml.BaseLoader)
    assert "on" in workflow
    assert "workflow_dispatch" in workflow["on"]
    assert "schedule" in workflow["on"]
    assert "competitiveness-comparator" in workflow["jobs"]
    comparator = workflow["jobs"]["competitiveness-comparator"]
    assert "workflow_dispatch" in comparator["if"]
    assert "schedule" in comparator["if"]
    assert "actions/upload-artifact@v7" in workflow_text
    assert "actions/upload-artifact@v4" not in workflow_text
    assert "--smoke --libraries alloygbm --threads 1 --repetitions 1 --warmups 0" in workflow_text
    assert "--repetitions 3 --warmups 1" in workflow_text
    assert "evaluate_claim" not in workflow_text
    assert "no timing gate applied" in workflow_text


def test_docs_link_committed_artifacts_and_match_recorded_provenance() -> None:
    benchmark_doc = (ROOT / "docs" / "benchmarks" / "v1.0.0_deep_scaling.md").read_text()
    readme = (ROOT / "benchmarks" / "README.md").read_text()
    for text in (benchmark_doc, readme):
        assert "adfa2c8-pr-smoke.jsonl" in text
        assert "adfa2c8-pr-smoke.summary.json" in text
        assert "run-metadata" in text
    assert "2026-09-04 PDT" in benchmark_doc
    assert "median and unscaled MAD" in benchmark_doc
    assert "numerically comparable" in benchmark_doc
    assert "adfa2c8e593cea68b124e7975f3b4fd9f862a148" in benchmark_doc
    assert "legacy comparison scripts" in readme
    assert "manifest-driven" in readme
    records = load_records(BASELINES / "adfa2c8-pr-smoke.jsonl")
    versions = {record.library_version for record in records}
    assert all(version in benchmark_doc for version in versions)
