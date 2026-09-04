from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from benchmarks.competitiveness.adapters import AdapterResult
from benchmarks.competitiveness.datasets import build_dataset_cases, fingerprint_case
from benchmarks.competitiveness.run import (
    load_manifest,
    run_benchmark,
    validate_options,
)
from benchmarks.competitiveness.schema import load_records


class FakeAdapter:
    def __init__(self, name: str, values: list[float]) -> None:
        self.name = name
        self.values = iter(values)
        self.calls: list[int] = []

    def fit_predict(self, case, seed: int, threads: int) -> AdapterResult:
        self.calls.append(seed)
        value = next(self.values)
        prediction = np.full(np.asarray(case.y_test).shape, value, dtype=np.float32)
        return AdapterResult(
            predictions=prediction,
            preprocessing_seconds=0.01,
            fit_seconds=value,
            predict_seconds=0.02,
            peak_rss_bytes=123,
            library=self.name,
            library_version="fake-1",
            effective_params={"topology": "fake", "seed": seed},
            input_representation=case.input_representation,
            rounds_completed=case.rounds,
        )


def tiny_manifest(path: Path) -> None:
    path.write_text(
        """schema: alloygbm-competitiveness/v1
seed: 7
warmup_repetitions: 1
timed_repetitions: 3
scenarios:
  - name: dense_regression
    task: regression
    rows: 20
    features: 4
    rounds: 2
    depth: 2
    metric: rmse
    input_representation: dense
"""
    )


def test_runner_excludes_warmups_persists_all_timed_and_uses_raw_median(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    tiny_manifest(manifest)
    first = FakeAdapter("fake-a", [9.0, 1.0, 5.0, 3.0])
    second = FakeAdapter("fake-b", [8.0, 2.0, 6.0, 4.0])

    run_dir = run_benchmark(
        manifest,
        tmp_path / "out",
        adapters={"fake-a": first, "fake-b": second},
        libraries=["fake-a", "fake-b"],
    )

    records = load_records(run_dir / "raw.jsonl")
    assert len(records) == 6
    assert [record.repetition for record in records] == [0, 1, 2, 0, 1, 2]
    assert first.calls == [7, 7, 7, 7]
    assert second.calls == [7, 7, 7, 7]
    # The runner keeps every observation; no minimum/best repetition was selected.
    assert [record.fit_seconds for record in records if record.library == "fake-a"] == [1.0, 5.0, 3.0]
    assert float(np.median([1.0, 5.0, 3.0])) == 3.0


def test_rerun_has_new_path_and_preserves_prior_output(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    tiny_manifest(manifest)
    first_dir = run_benchmark(
        manifest,
        tmp_path / "out",
        adapters={"fake": FakeAdapter("fake", [1, 2, 3, 4])},
        libraries=["fake"],
    )
    first_contents = (first_dir / "raw.jsonl").read_text()
    second_dir = run_benchmark(
        manifest,
        tmp_path / "out",
        adapters={"fake": FakeAdapter("fake", [1, 2, 3, 4])},
        libraries=["fake"],
    )
    assert first_dir != second_dir
    assert first_dir.exists() and second_dir.exists()
    assert (first_dir / "raw.jsonl").read_text() == first_contents


def test_fixture_fingerprints_are_deterministic_and_sensitive(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    tiny_manifest(manifest)
    spec = load_manifest(manifest)["scenarios"][0]
    first = build_dataset_cases([spec], seed=7)[0]
    second = build_dataset_cases([spec], seed=7)[0]
    assert first.dataset_sha256 == second.dataset_sha256 == fingerprint_case(first)
    changed = replace(first, y_train=first.y_train.copy())
    changed.y_train[0] += 1  # type: ignore[misc]
    assert fingerprint_case(changed) != first.dataset_sha256


def test_ranking_split_is_group_safe_and_sparse_stays_csr(tmp_path: Path) -> None:
    import scipy.sparse as sp

    ranking = {
        "name": "grouped_ranking", "task": "ranking", "rows": 20,
        "features": 3, "groups": 5, "rounds": 2, "depth": 2,
        "metric": "ndcg_at_10", "input_representation": "dense",
    }
    sparse = {
        "name": "csr_sparse", "task": "regression", "rows": 20,
        "features": 30, "density": 0.1, "rounds": 2, "depth": 2,
        "metric": "rmse", "input_representation": "csr",
    }
    ranked = build_dataset_cases([ranking], seed=7)[0]
    assert ranked.group_train is not None and ranked.group_test is not None
    assert np.all(np.diff(ranked.group_train) >= 0)
    assert np.all(np.diff(ranked.group_test) >= 0)
    assert set(ranked.train_indices).isdisjoint(ranked.test_indices)
    sparse_case = build_dataset_cases([sparse], seed=7)[0]
    assert sp.isspmatrix_csr(sparse_case.X_train)
    assert sp.isspmatrix_csr(sparse_case.X_test)


@pytest.mark.parametrize("kwargs", [
    {"threads": 0}, {"warmups": -1}, {"repetitions": 2},
])
def test_validate_options_rejects_invalid_values(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        validate_options(["dense_regression"], ["alloygbm"], **kwargs)


def test_validate_options_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="scenario"):
        validate_options(["missing"], ["alloygbm"])
    with pytest.raises(ValueError, match="library"):
        validate_options(["dense_regression"], ["missing"])


def test_modules_do_not_import_optional_competitors() -> None:
    import sys
    import benchmarks.competitiveness.adapters  # noqa: F401

    assert "lightgbm" not in sys.modules
    assert "xgboost" not in sys.modules
    assert "catboost" not in sys.modules

