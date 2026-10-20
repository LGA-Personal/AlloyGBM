from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dev_suite  # noqa: E402
import special_modes  # noqa: E402

TEST = [
    {"task_id": 1, "dataset_id": 10, "name": "california_housing", "rows": 20640, "features": 8},
    {"task_id": 2, "dataset_id": 20, "name": "credit-approval", "rows": 690, "features": 15},
]


def _entry(**overrides) -> dict:
    return {"task_id": 99, "dataset_id": 999, "name": "pol", "rows": 15000, "features": 26} | overrides


def test_overlap_matches_id_name_alias_and_shape():
    assert dev_suite.overlap_reason(_entry(), TEST) is None
    assert "dataset id" in dev_suite.overlap_reason(_entry(dataset_id=10), TEST)
    assert "same name" in dev_suite.overlap_reason(_entry(name="California_Housing"), TEST)
    assert "re-upload" in dev_suite.overlap_reason(_entry(name="houses"), TEST)
    assert "re-upload" in dev_suite.overlap_reason(_entry(name="Australian"), TEST)
    assert "shape" in dev_suite.overlap_reason(_entry(rows=690, features=15), TEST)


def test_committed_manifest_is_disjoint_from_the_test_suites():
    manifest = json.loads(dev_suite.MANIFEST.read_text())
    test = manifest["test_tasks"]
    test_ids = {t["dataset_id"] for t in test}
    test_names = {dev_suite.normalize(t["name"]) for t in test}
    assert len(manifest["tasks"]) >= 50
    for e in manifest["tasks"]:
        name = dev_suite.normalize(e["name"])
        assert e["dataset_id"] not in test_ids
        assert name not in test_names
        assert not set(dev_suite.ALIASES.get(name, ())) & test_names
        assert name not in dev_suite.SUSPECTED_OVERLAP


def test_load_builds_seeded_splits_and_caps_training_rows(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(float)
    np.savez_compressed(tmp_path / "7.npz", X=X, y=y, categorical=np.array([], dtype=np.int64),
                        kind="classification", name="toy", task_id=7, original_rows=200)
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"tasks": [{"task_id": 7, "name": "toy",
                                               "kind": "classification"},
                                              {"task_id": 8, "name": "absent",
                                               "kind": "regression"}]}))
    logs: list[str] = []
    (ds,) = dev_suite.load(tmp_path, path=manifest, log=logs.append, train_cap=50)
    assert ds.spec.name == "dev_7_toy" and ds.task == "binary"
    assert len(ds.folds) == dev_suite.SPLITS and ds.train_cap == 50
    train, test = ds.folds[0]
    assert len(set(train) & set(test)) == 0 and len(test) == 50
    assert any("absent" in line for line in logs)


def _record(arm, library, seed, loss, fit=1.0, dataset="d|defaults", task="regression"):
    return {"dataset": dataset, "arm": arm, "library": library, "seed": seed, "loss": loss,
            "fit_seconds": fit, "task": task, "error": None}


def test_special_modes_pairs_arms_by_seed_and_reports_peers():
    records = []
    for seed in range(3):
        records.append(_record("alloy", "alloygbm", seed, 1.0))
        records.append(_record("alloy_pl", "alloygbm", seed, 0.8, fit=2.0))
        records.append(_record("lightgbm", "lightgbm", seed, 0.9))
    cmp = special_modes.compare_arms(records, "alloy")
    assert set(cmp) == {"alloy_pl"}
    (row,) = cmp["alloy_pl"]
    assert row["outcome"] == "win"
    assert abs(np.exp(row["log_ratio"]) - 0.8) < 1e-9
    assert abs(np.exp(row["time_log_ratio"]) - 2.0) < 1e-9
    assert abs(row["vs_peer"] - 0.8 / 0.9) < 1e-9
    assert "alloy_pl" in special_modes.render(cmp, "alloy")


def test_overlap_tolerates_test_entries_without_shape():
    # The committed manifest's test_tasks predate the rows/features fields.
    bare = [{k: t[k] for k in ("task_id", "dataset_id", "name")} for t in TEST]
    assert dev_suite.overlap_reason(_entry(), bare) is None
    assert "dataset id" in dev_suite.overlap_reason(_entry(dataset_id=20), bare)


def test_load_supports_more_splits_than_the_default(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 2))
    np.savez_compressed(tmp_path / "7.npz", X=X, y=X[:, 0], categorical=np.array([], dtype=np.int64),
                        kind="regression", name="toy", task_id=7, original_rows=120)
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"tasks": [{"task_id": 7, "name": "toy", "kind": "regression"}]}))
    (ds,) = dev_suite.load(tmp_path, splits=5, path=manifest, log=lambda _: None)
    assert len(ds.folds) == 5


def test_subsample_falls_back_when_stratifying_is_impossible():
    # 10 classes but only 5 rows left out: stratified splitting raises.
    labels = np.repeat(np.arange(10), 3)
    keep = dev_suite._subsample(len(labels), 25, labels, seed=0)
    assert len(keep) == 25 and len(set(keep)) == 25


def test_summary_skips_non_positive_peer_ratios():
    rows = [{"outcome": "tie", "log_ratio": 0.0, "time_log_ratio": 0.0,
             "vs_peer": v, "base_vs_peer": None} for v in (0.0, 1.0, 4.0)]
    summary = special_modes._summary(rows)
    assert summary["vs_peer"] == 2.0 and summary["base_vs_peer"] is None
