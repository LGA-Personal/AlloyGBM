from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

from benchmarks.competitiveness.schema import (
    BenchmarkSummaryV1,
    SCHEMA_VERSION,
    load_records,
)


ROOT = Path(__file__).parents[3]
BASELINES = ROOT / "benchmarks" / "competitiveness" / "baselines"


def test_committed_baseline_is_complete_traceable_and_round_trips() -> None:
    raw_path = BASELINES / "adfa2c8-pr-smoke.jsonl"
    summary_path = BASELINES / "adfa2c8-pr-smoke.summary.json"
    records = load_records(raw_path)
    assert len(records) == 120
    assert {record.git_sha for record in records} == {
        "adfa2c8e593cea68b124e7975f3b4fd9f862a148"
    }
    assert len({record.run_id for record in records}) == 1
    assert len({tuple(sorted(record.machine.items())) for record in records}) == 1
    assert {record.repetition for record in records} == {0, 1, 2, 3, 4}
    assert all(record.effective_params for record in records)
    populations = Counter((record.scenario, record.library) for record in records)
    assert len(populations) == 24
    assert set(populations.values()) == {5}
    assert len({record.dataset_sha256 for record in records if record.scenario == "dense_regression"}) == 1
    assert len({record.dataset_sha256 for record in records if record.scenario == "binary"}) == 1
    assert len({record.dataset_sha256 for record in records if record.scenario == "grouped_ranking"}) == 1
    assert len({record.dataset_sha256 for record in records if record.scenario == "native_categorical"}) == 1
    assert len({record.dataset_sha256 for record in records if record.scenario == "csr_sparse"}) == 1
    assert len({record.dataset_sha256 for record in records if record.scenario == "joint_multi_output"}) == 1

    payload = json.loads(summary_path.read_text())
    assert payload["schema"] == SCHEMA_VERSION
    assert payload["status"] == "insufficient-data"
    assert len(payload["summaries"]) == 24
    raw_lines = raw_path.read_text().splitlines()
    for encoded in payload["summaries"]:
        summary = BenchmarkSummaryV1.from_json(json.dumps(encoded, sort_keys=True))
        assert summary.raw_line_numbers is not None
        assert summary.raw_repetition_ids == (0, 1, 2, 3, 4)
        assert len(summary.raw_line_numbers) == 5
        for line_number in summary.raw_line_numbers:
            assert json.loads(raw_lines[line_number - 1])["run_id"] == summary.run_id
            assert json.loads(raw_lines[line_number - 1])["scenario"] == summary.scenario
            assert json.loads(raw_lines[line_number - 1])["library"] == summary.library


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
    assert "actions/upload-artifact@v4" in workflow_text
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
    assert "2026-09-04 PDT" in benchmark_doc
    assert "median and unscaled MAD" in benchmark_doc
    assert "numerically comparable" in benchmark_doc
    assert "adfa2c8e593cea68b124e7975f3b4fd9f862a148" in benchmark_doc
    records = load_records(BASELINES / "adfa2c8-pr-smoke.jsonl")
    versions = {record.library_version for record in records}
    assert all(version in benchmark_doc for version in versions)
