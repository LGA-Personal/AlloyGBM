from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping

import yaml
import pytest

from benchmarks.competitiveness.schema import (
    BenchmarkSummaryV1,
    METRIC_DIRECTIONS,
    ProfileRecordV1,
    SCHEMA_VERSION,
    harness_tree_sha256,
    load_records,
    load_run_bundle,
)
from benchmarks.competitiveness.datasets import build_dataset_cases
from benchmarks.competitiveness.run import load_manifest
from benchmarks.competitiveness.summarize import summarize_file


ROOT = Path(__file__).parents[3]
BASELINES = ROOT / "benchmarks" / "competitiveness" / "baselines"


def _historical_harness_digest(sha: str) -> str:
    """Independently hash top-level harness sources from a git revision."""

    paths = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-tree", "-r", "--name-only", sha, "--", "benchmarks/competitiveness"],
        text=True,
    ).splitlines()
    names = sorted(
        path.removeprefix("benchmarks/competitiveness/")
        for path in paths
        if Path(path).parent == Path("benchmarks/competitiveness") and path.endswith(".py")
    )
    digest = hashlib.sha256()
    for name in names:
        content = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{sha}:benchmarks/competitiveness/{name}"],
        )
        name_bytes = name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def test_committed_baseline_is_complete_traceable_and_round_trips() -> None:
    raw_path = BASELINES / "adfa2c8-pr-smoke.jsonl"
    summary_path = BASELINES / "adfa2c8-pr-smoke.summary.json"
    metadata, records = load_run_bundle(raw_path)
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
    expected_dataset_sha256 = {
        "dense_regression": "e0e11ef536a47421f3b03e5bf21b387bdfbbd8be044ee47ef2886acff01ba6be",
        "binary": "04e7aac0fefbe3c2270d852a1222df47a395d72d6721f2219e7f4cab6114947a",
        "grouped_ranking": "d5533db052925250d2bf9f69874f4ddd6135d7a9ca11daa201d5152e19e4df49",
        "native_categorical": "e4fc65f230b5c90edd3829d17ee8a9512f12278cfe76d69b3931cce243d05c31",
        "csr_sparse": "cf6847173f10fda14d7e27e840a8afb577aba9a7bd58ee09d0c4d492436547ff",
        "joint_multi_output": "f07bca7fc46b5b50e992282d920448d3be08bcad080044ca686559d075dcce8e",
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
    assert set(expected_dataset_sha256) == expected_scenarios
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
        assert record.dataset_sha256 == expected_dataset_sha256[record.scenario]
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
    for scenario, expected_hash in expected_dataset_sha256.items():
        assert {record.dataset_sha256 for record in records if record.scenario == scenario} == {
            expected_hash
        }
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

    assert metadata.run_id == records[0].run_id
    assert metadata.measured_git_sha == "adfa2c8e593cea68b124e7975f3b4fd9f862a148"
    assert metadata.harness_git_sha == "7082301fcd79bac3e1f05e696c376588158eaee3"
    assert metadata.harness_tree_sha256 == _historical_harness_digest(metadata.harness_git_sha)
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
    metadata, records = load_run_bundle(raw_path, metadata_path)
    assert len(records) == 5
    assert {record.library for record in records} == {"alloygbm"}
    assert {record.git_sha for record in records} == {"adfa2c8e593cea68b124e7975f3b4fd9f862a148"}
    assert metadata.run_id == records[0].run_id
    assert metadata.harness_git_sha == "88f754c9f3f2d17d8e929842923d6a3760ebbc09"
    assert metadata.harness_tree_sha256 == _historical_harness_digest(metadata.harness_git_sha)
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


def test_harness_tree_digest_detects_source_drift_without_git_sha_drift(tmp_path: Path) -> None:
    source_root = tmp_path / "competitiveness"
    source_root.mkdir()
    source_paths = list((ROOT / "benchmarks" / "competitiveness").glob("*.py"))
    for source_path in source_paths:
        shutil.copyfile(source_path, source_root / source_path.name)
    before_sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    before_digest = harness_tree_sha256(source_root)
    drifted = source_root / "run.py"
    drifted.write_bytes(drifted.read_bytes() + b"\n# local harness drift\n")
    after_digest = harness_tree_sha256(source_root)
    after_sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    assert after_digest != before_digest
    assert after_sha == before_sha


def test_run_bundle_loader_fails_closed_on_metadata_raw_drift(tmp_path: Path) -> None:
    raw_path = BASELINES / "adfa2c8-pr-smoke.jsonl"
    metadata_path = BASELINES / "adfa2c8-pr-smoke.run-metadata.json"
    copied_raw = tmp_path / raw_path.name
    copied_metadata = tmp_path / metadata_path.name
    shutil.copyfile(raw_path, copied_raw)
    shutil.copyfile(metadata_path, copied_metadata)
    metadata, records = load_run_bundle(copied_raw, copied_metadata)
    assert metadata.raw_record_count == len(records) == 120
    copied_raw.write_text(copied_raw.read_text() + "\n")
    with pytest.raises(ValueError, match="raw checksum"):
        load_run_bundle(copied_raw, copied_metadata)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("scenarios", ["bogus-scenario"], "scenarios"),
        ("libraries", ["bogus-library"], "libraries"),
        ("threads", 4, "threads"),
        ("seed", 17, "seed"),
        ("measured_git_sha", "0" * 40, "git_sha"),
    ],
)
def test_run_bundle_rejects_false_metadata_declarations(
    tmp_path: Path, field: str, value: object, error: str
) -> None:
    raw_path = BASELINES / "adfa2c8-pr-smoke.jsonl"
    metadata_path = BASELINES / "adfa2c8-pr-smoke.run-metadata.json"
    copied_raw = tmp_path / raw_path.name
    copied_metadata = tmp_path / metadata_path.name
    shutil.copyfile(raw_path, copied_raw)
    metadata = json.loads(metadata_path.read_text())
    metadata[field] = value
    copied_metadata.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match=error):
        load_run_bundle(copied_raw, copied_metadata)


def test_run_bundle_rejects_alloy_profile_metadata_for_non_alloy_bundle(tmp_path: Path) -> None:
    raw_path = BASELINES / "adfa2c8-pr-smoke.jsonl"
    metadata_path = BASELINES / "adfa2c8-pr-smoke.run-metadata.json"
    copied_raw = tmp_path / raw_path.name
    copied_metadata = tmp_path / metadata_path.name
    shutil.copyfile(raw_path, copied_raw)
    metadata = json.loads(metadata_path.read_text())
    metadata["profile_alloy"] = True
    copied_metadata.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="profile"):
        load_run_bundle(copied_raw, copied_metadata)


def test_run_bundle_rejects_duplicate_alloy_profile_metadata_library(tmp_path: Path) -> None:
    raw_path = BASELINES / "adfa2c8-published-v1-crosscheck.jsonl"
    metadata_path = BASELINES / "adfa2c8-published-v1-crosscheck.run-metadata.json"
    copied_raw = tmp_path / raw_path.name
    copied_metadata = tmp_path / metadata_path.name
    shutil.copyfile(raw_path, copied_raw)
    metadata = json.loads(metadata_path.read_text())
    metadata["libraries"] = ["alloygbm", "alloygbm"]
    metadata["profile_alloy"] = True
    copied_metadata.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="libraries|profile"):
        load_run_bundle(copied_raw, copied_metadata)


def test_run_bundle_rejects_alloy_record_profile_when_metadata_disables_it(tmp_path: Path) -> None:
    raw_path = BASELINES / "adfa2c8-pr-smoke.jsonl"
    metadata_path = BASELINES / "adfa2c8-pr-smoke.run-metadata.json"
    copied_raw = tmp_path / raw_path.name
    copied_metadata = tmp_path / metadata_path.name
    payloads = [json.loads(line) for line in raw_path.read_text().splitlines()]
    payloads[0]["profile"] = ProfileRecordV1(
        rows=1, features=1, rounds=1, threads=1, loop_wall_ns=1,
        untimed_ns=0,
        stage_ns={label: 0 for label in ("gradients", "row_sampling", "feature_tiles", "prediction_copy", "tree_build", "prediction_update", "loss", "validation")},
        tree_stage_ns={label: 0 for label in ("histogram_build", "split_find", "partition")},
    ).to_dict()
    copied_raw.write_text("\n".join(json.dumps(payload, sort_keys=True) for payload in payloads) + "\n")
    metadata = json.loads(metadata_path.read_text())
    metadata["raw_sha256"] = hashlib.sha256(copied_raw.read_bytes()).hexdigest()
    copied_metadata.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="profile_alloy"):
        load_run_bundle(copied_raw, copied_metadata)


def test_run_bundle_rejects_missing_and_extra_cohort_repetitions(tmp_path: Path) -> None:
    raw_path = BASELINES / "adfa2c8-pr-smoke.jsonl"
    metadata_path = BASELINES / "adfa2c8-pr-smoke.run-metadata.json"
    records = load_records(raw_path)

    missing_raw = tmp_path / "missing.jsonl"
    missing_metadata = tmp_path / "missing.run-metadata.json"
    missing_raw.write_text("\n".join(record.to_json() for record in records[:-1]) + "\n")
    missing_payload = json.loads(metadata_path.read_text())
    missing_payload["raw_record_count"] = len(records) - 1
    missing_payload["raw_sha256"] = hashlib.sha256(missing_raw.read_bytes()).hexdigest()
    missing_metadata.write_text(json.dumps(missing_payload, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="repetition IDs"):
        load_run_bundle(missing_raw, missing_metadata)

    extra_raw = tmp_path / "extra.jsonl"
    extra_metadata = tmp_path / "extra.run-metadata.json"
    extra_record = records[-1].to_dict()
    extra_record["repetition"] = 5
    extra_raw.write_text(
        "\n".join(record.to_json() for record in records)
        + "\n"
        + json.dumps(extra_record, sort_keys=True)
        + "\n"
    )
    extra_payload = json.loads(metadata_path.read_text())
    extra_payload["raw_record_count"] = len(records) + 1
    extra_payload["raw_sha256"] = hashlib.sha256(extra_raw.read_bytes()).hexdigest()
    extra_metadata.write_text(json.dumps(extra_payload, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="repetition IDs"):
        load_run_bundle(extra_raw, extra_metadata)


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
    metadata, bound_records = load_run_bundle(paths[0])
    assert bound_records == records
    assert metadata.harness_tree_sha256 == harness_tree_sha256(
        ROOT / "benchmarks" / "competitiveness"
    )


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
    assert "load_run_bundle" in workflow_text
    assert "--smoke --libraries alloygbm --threads 1 --repetitions 1 --warmups 0" in workflow_text
    assert "--repetitions 3 --warmups 1" in workflow_text
    assert "evaluate_claim" not in workflow_text
    assert "no timing gate applied" in workflow_text

    python_smoke = workflow["jobs"]["python-smoke"]
    fetch_steps = [
        step
        for step in python_smoke["steps"]
        if step.get("name") == "Fetch historical harness commits for contracts"
    ]
    assert len(fetch_steps) == 1
    fetch_step = fetch_steps[0]
    assert fetch_step["if"] == "matrix.os == 'ubuntu-latest' && matrix.python-version == '3.13'"
    assert "git fetch --no-tags --depth=1 origin" in fetch_step["run"]
    assert "7082301fcd79bac3e1f05e696c376588158eaee3" in fetch_step["run"]
    assert "88f754c9f3f2d17d8e929842923d6a3760ebbc09" in fetch_step["run"]

    security_workflow = yaml.load(
        (ROOT / ".github" / "workflows" / "security-audit.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    audit_job = security_workflow["jobs"]["cargo-audit"]
    audit_text = "\n".join(str(step.get("run", "")) for step in audit_job["steps"])
    assert "rustsec/audit-check@" not in "\n".join(
        str(step.get("uses", "")) for step in audit_job["steps"]
    )
    assert any(
        step.get("with", {}).get("toolchain") == "1.92.0"
        for step in audit_job["steps"]
    )
    assert "cargo install cargo-audit --version 0.22.2 --locked" in audit_text
    assert "cargo audit" in audit_text


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


def test_deep_scaling_crosscheck_distinguishes_reproducibility_from_timing() -> None:
    benchmark_doc = (ROOT / "docs" / "benchmarks" / "v1.0.0_deep_scaling.md").read_text()
    assert "pre-date `072478d`" in benchmark_doc
    assert "roughly 7% at depth 12" in benchmark_doc
    assert "no measured depth-8 effect" in benchmark_doc
    assert "differ in commit and estimator" in benchmark_doc
    assert "does not explain the 15.97 vs 16.76 delta" in benchmark_doc
    assert "coarse sanity check" in benchmark_doc
    assert "meaningful reproducibility signal" in benchmark_doc
