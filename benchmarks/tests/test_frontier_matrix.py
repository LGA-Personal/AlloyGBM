from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frontier_matrix as fm  # noqa: E402


def _spec(**overrides) -> fm.DataSpec:
    base = {"name": "t", "task": fm.REGRESSION, "axis": "test", "level": "x", "rows": 600, "features": 10}
    return fm.DataSpec(**(base | overrides))


def test_scenario_names_are_unique_and_tiers_valid():
    specs = fm.data_scenarios()
    assert len({s.name for s in specs}) == len(specs)
    assert {s.tier for s in specs} <= set(fm.TIERS)
    assert {s.task for s in specs} == set(fm.TASKS)


def test_every_axis_is_covered_by_the_quick_tier():
    quick = fm.build_cells("quick")
    full = fm.build_cells("full")
    assert {c.axis for c in quick} == {c.axis for c in full}
    assert {c.regime.name for c in quick} >= {r.name for r in fm.REGIMES if r.tier != "full"}


def test_tiers_are_nested():
    smoke = {c.id for c in fm.build_cells("smoke")}
    quick = {c.id for c in fm.build_cells("quick")}
    full = {c.id for c in fm.build_cells("full")}
    assert smoke < quick < full


def test_cell_filters():
    cells = fm.build_cells("quick", axes=["rows"], regimes=["defaults"])
    assert cells and all(c.axis == "rows" and c.regime.name == "defaults" for c in cells)
    panel = fm.build_cells("quick", axes=["hyperparameters"])
    assert {c.regime.name for c in panel} - set(fm.DATA_REGIMES)
    named = fm.build_cells("full", names=["cat_high"])
    assert named and all("cat_high" in c.id for c in named)


def test_datasets_are_deterministic():
    spec = _spec(cat_low=2, missing_rate=0.2)
    a, b = fm.make_dataset(spec), fm.make_dataset(spec)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.y, b.y)
    other = fm.make_dataset(replace(spec, noise=0.9))
    assert not np.array_equal(a.y, other.y)


def test_axis_tags_do_not_change_the_data():
    spec = _spec()
    retagged = replace(spec, axis="other", level="y", tier="full")
    np.testing.assert_array_equal(fm.make_dataset(spec).y, fm.make_dataset(retagged).y)


def test_categorical_columns_hold_integer_codes():
    ds = fm.make_dataset(_spec(rows=5_000, cat_low=2, cat_high=1))
    assert ds.categorical == (7, 8, 9)
    for j in ds.categorical:
        col = ds.X[:, j]
        assert np.all(col == np.round(col))
    assert ds.X[:, 9].max() > 50  # high-cardinality column
    assert ds.X[:, 7].max() <= 7


@pytest.mark.parametrize("mechanism", ["mcar", "mnar"])
def test_missing_rate_is_close_to_requested(mechanism):
    ds = fm.make_dataset(_spec(rows=5_000, missing_rate=0.3, missing_mechanism=mechanism))
    assert abs(np.isnan(ds.X).mean() - 0.3) < 0.05


def test_mnar_missingness_is_informative():
    ds = fm.make_dataset(_spec(rows=20_000, informative=1, signal="linear", noise=0.1,
                               missing_rate=0.3, missing_mechanism="mnar"))
    missing = np.isnan(ds.X[:, 0])
    assert abs(ds.y[missing].mean() - ds.y[~missing].mean()) > 0.3


def test_binary_positive_rate_and_multiclass_classes():
    ds = fm.make_dataset(_spec(task=fm.BINARY, rows=8_000, positive_rate=0.05))
    assert 0.02 < ds.y.mean() < 0.10
    mc = fm.make_dataset(_spec(task=fm.MULTICLASS, rows=8_000, n_classes=6))
    assert set(np.unique(mc.y)) == set(range(6))


def test_ranking_groups_are_contiguous_and_graded():
    ds = fm.make_dataset(_spec(task=fm.RANKING, rows=1_010, group_size=20))
    assert len(ds.y) == 1_000 and ds.groups is not None
    assert np.all(np.diff(ds.groups) >= 0)
    assert set(np.unique(ds.y)) == {0, 1, 2, 3, 4}


def test_feature_kinds():
    sparse = fm.make_dataset(_spec(rows=4_000, feature_kind="sparse"))
    assert 0.85 < (sparse.X == 0).mean() < 0.95
    discrete = fm.make_dataset(_spec(rows=4_000, feature_kind="discrete"))
    assert len(np.unique(discrete.X[:, 0])) <= 9
    skewed = fm.make_dataset(_spec(rows=4_000, feature_kind="skewed"))
    assert skewed.X.min() >= 0.0


def test_split_is_stratified_and_carves_validation():
    ds = fm.make_dataset(_spec(task=fm.BINARY, rows=4_000))
    split = fm.make_split(ds, seed=3, validation_fraction=0.15)
    parts = [split.train, split.test, split.valid]
    assert sum(len(p) for p in parts) == 4_000
    assert len(np.intersect1d(split.train, split.valid)) == 0
    assert abs(ds.y[split.test].mean() - ds.y.mean()) < 0.02
    assert fm.make_split(ds, seed=3, validation_fraction=None).valid is None


def test_ranking_split_keeps_queries_whole():
    ds = fm.make_dataset(_spec(task=fm.RANKING, rows=2_000, group_size=20))
    split = fm.make_split(ds, seed=0, validation_fraction=0.15)
    sides = [set(ds.groups[p]) for p in (split.train, split.test, split.valid)]
    assert not (sides[0] & sides[1]) and not (sides[0] & sides[2]) and not (sides[1] & sides[2])


def test_lightgbm_depth_lifts_the_leaf_cap():
    params, _ = fm.library_params(fm.REGIMES_BY_NAME["deep"], "lightgbm", fm.REGRESSION)
    assert params["max_depth"] == 12 and params["num_leaves"] == 4096


def test_light_regularization_turns_off_alloys_auto_policy():
    params, unmapped = fm.library_params(fm.REGIMES_BY_NAME["light_reg"], "alloygbm", fm.BINARY)
    assert params["training_policy"] == "manual"
    assert params["lambda_l2"] == 0.0 and params["min_child_hessian"] == 0.0
    assert not unmapped


def test_unmapped_knobs_are_reported():
    heavy = fm.REGIMES_BY_NAME["heavy_reg"]
    _, cat_unmapped = fm.library_params(heavy, "catboost", fm.REGRESSION)
    assert set(cat_unmapped) == {"l1", "min_child_weight", "min_data_in_leaf"}
    _, xgb_unmapped = fm.library_params(heavy, "xgboost", fm.REGRESSION)
    assert xgb_unmapped == ["min_data_in_leaf"]
    _, lgb_unmapped = fm.library_params(heavy, "lightgbm", fm.REGRESSION)
    assert lgb_unmapped == []


def test_leafwise_regime_maps_to_each_librarys_best_first_growth():
    regime = fm.REGIMES_BY_NAME["leafwise"]
    assert fm.library_params(regime, "alloygbm", fm.REGRESSION)[0]["tree_growth"] == "leaf"
    assert fm.library_params(regime, "xgboost", fm.REGRESSION)[0]["grow_policy"] == "lossguide"
    assert fm.library_params(regime, "catboost", fm.REGRESSION)[0]["grow_policy"] == "Lossguide"
    assert fm.library_params(regime, "lightgbm", fm.REGRESSION)[0]["num_leaves"] == 63


def test_defaults_regime_passes_nothing():
    for library in ("alloygbm", "lightgbm", "xgboost", "catboost"):
        assert fm.library_params(fm.REGIMES_BY_NAME["defaults"], library, fm.BINARY) == ({}, [])


def test_ndcg():
    y = np.array([3.0, 2.0, 0.0, 1.0])
    groups = np.zeros(4)
    assert fm.ndcg_at_k(y, y, groups) == pytest.approx(1.0)
    assert fm.ndcg_at_k(y, -y, groups) < 0.7
    # A constant scorer earns the worst ordering, not a lucky one.
    assert fm.ndcg_at_k(y, np.zeros(4), groups) == pytest.approx(fm.ndcg_at_k(y, -y, groups))


def test_outcome_respects_noise_and_minimum_gap():
    assert fm.outcome(math.log(0.9), 0.01) == "win"
    assert fm.outcome(math.log(1.1), 0.01) == "loss"
    assert fm.outcome(math.log(0.9), 0.1) == "tie"
    assert fm.outcome(math.log(0.998), 0.0) == "tie"


def _result(losses: dict[str, dict[tuple[str, int], float]], meta: dict[str, dict]) -> dict:
    arms = [{"name": a, "library": {"alloy": "alloygbm"}.get(a, a), "version": "x"} for a in losses]
    return {
        "tier": "quick", "seeds": [0, 1, 2], "threads": 1, "git_commit": "abc",
        "arms": arms,
        "datasets": [{"name": cell, **m} for cell, m in meta.items()],
        "records": [
            {"dataset": cell, "seed": seed, "arm": arm, "loss": loss, "fit_seconds": 1.0, "error": None}
            for arm, by_cell in losses.items() for (cell, seed), loss in by_cell.items()
        ],
    }


def test_summarize_counts_wins_ties_and_losses():
    meta = {
        "a|defaults": {"scenario": "a", "regime": "defaults", "axis": "rows", "level": "300", "task": "regression"},
        "b|defaults": {"scenario": "b", "regime": "defaults", "axis": "rows", "level": "300", "task": "regression"},
        "regression_base|deep": {"scenario": "regression_base", "regime": "deep", "axis": "base",
                                 "level": "base", "task": "regression"},
    }
    seeds = range(3)
    losses = {
        "alloy": {**{("a|defaults", s): 0.9 + 0.001 * s for s in seeds},
                  **{("b|defaults", s): 1.2 for s in seeds},
                  **{("regression_base|deep", s): 1.0 for s in seeds}},
        "lightgbm": {(c, s): 1.0 for c in meta for s in seeds},
        "catboost": {(c, s): 1.1 for c in meta for s in seeds},
    }
    summary = fm.summarize(_result(losses, meta))
    rows = summary["by_axis"]["defaults"]["rows = 300"]
    assert rows["scored"] == 2
    assert rows["vs_lightgbm"] == {"win": 1, "tie": 0, "loss": 1}
    assert rows["best_or_tied"] == 1
    assert summary["by_regime"]["deep"]["vs_lightgbm"] == {"win": 0, "tie": 1, "loss": 0}
    report = fm.render_report(_result(losses, meta))
    assert "rows = 300" in report and "Largest gaps" in report


def test_failed_fits_are_reported():
    meta = {"a|standard": {"scenario": "a", "regime": "standard", "axis": "rows", "level": "2k",
                           "task": "binary"}}
    result = _result({"alloy": {}, "lightgbm": {("a|standard", 0): 0.5}}, meta)
    result["records"].append({"dataset": "a|standard", "seed": 0, "arm": "alloy", "loss": None,
                              "fit_seconds": None, "error": "ValueError: boom"})
    summary = fm.summarize(result)
    assert summary["overall"]["errors"] == 1
    assert "ValueError: boom" in fm.render_report(result)


def test_records_feed_the_paired_gate():
    meta = {f"c{i}|defaults": {"scenario": f"c{i}", "regime": "defaults", "axis": "rows",
                               "level": "x", "task": "regression"} for i in range(8)}
    base = _result({"alloy": {(c, s): 1.0 for c in meta for s in range(3)}}, meta)
    cand = _result({"alloy": {(c, s): 0.95 + 0.001 * s for c in meta for s in range(3)}}, meta)
    cmp = fm.paired_compare(base, cand, "alloy", "alloy", 0.0, 0.03)
    assert cmp["gate_passed"], cmp["gate_reasons"]


def test_list_command(capsys):
    assert fm.main(["list", "--tier", "smoke"]) == 0
    out = capsys.readouterr().out
    assert "regression_base|defaults" in out and "8 cells" in out


def test_alloy_runs_a_tiny_cell_of_every_task(tmp_path):
    pytest.importorskip("alloygbm")
    cells = [
        fm.Cell(_spec(name=f"tiny_{task}", task=task, rows=400, features=6, informative=3,
                      n_classes=3, group_size=10, cat_low=1 if task == fm.BINARY else 0,
                      missing_rate=0.1 if task == fm.REGRESSION else 0.0),
                fm.REGIMES_BY_NAME[regime])
        for task in fm.TASKS for regime in ("defaults", "early_stopping")
    ]
    arms = [fm.parse_arm("alloy")]
    result = fm.run(arms, cells, [0], threads=1, tier="smoke", log=lambda _: None)
    errors = [r["error"] for r in result["records"] if r["error"]]
    assert not errors, errors
    assert all(r["loss"] > 0 for r in result["records"])
    out = tmp_path / "r.json"
    out.write_text(json.dumps(result))
    assert fm.main(["report", str(out)]) == 0


@pytest.mark.parametrize("library", ["lightgbm", "xgboost", "catboost"])
def test_peer_adapters_on_tiny_cells(library):
    pytest.importorskip(library)
    cells = [
        fm.Cell(_spec(name=f"tiny_{task}", task=task, rows=400, features=6, informative=3,
                      n_classes=3, group_size=10, cat_low=1, missing_rate=0.1),
                fm.REGIMES_BY_NAME[regime])
        for task in fm.TASKS for regime in ("standard", "early_stopping", "leafwise")
    ]
    result = fm.run([fm.parse_arm(library)], cells, [0], threads=1, tier="smoke", log=lambda _: None)
    errors = [(r["dataset"], r["error"]) for r in result["records"] if r["error"]]
    assert not errors, errors


def test_ratio_vs_best_peer_is_paired_over_shared_seeds():
    meta = {"a|defaults": {"scenario": "a", "regime": "defaults", "axis": "rows", "level": "300",
                           "task": "regression"}}
    # Seed 2 is missing for alloy: the ratio must use seeds 0 and 1 only.
    losses = {"alloy": {("a|defaults", 0): 1.0, ("a|defaults", 1): 2.0},
              "lightgbm": {("a|defaults", 0): 1.0, ("a|defaults", 1): 2.0, ("a|defaults", 2): 100.0}}
    row = fm.summarize(_result(losses, meta))["cells"][0]
    assert row["ratio_vs_best_peer"] == pytest.approx(1.0)
    assert row["vs_best_peer"] == "tie"


def test_dataset_from_frame_cleans_class_labels():
    pd = pytest.importorskip("pandas")
    y = pd.Series(pd.Categorical(["a"] * 10 + ["b"] * 10 + ["rare"] + [None] * 3,
                                 categories=["a", "b", "gone", "rare"]))
    X = pd.DataFrame({"x": np.arange(len(y), dtype=float)})
    ds = fm.dataset_from_frame("t", fm.BINARY, X, y, [False], max_rows=1_000)
    assert len(ds.y) == 20
    assert set(np.unique(ds.y)) == {0.0, 1.0}
    assert ds.spec.n_classes == 2


def _fake_openml(monkeypatch, n_rows=40, repeats=1, n_folds=4, regression=False):
    """A stand-in ``openml`` module: one suite, one task, fixed folds."""
    import types

    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "num": rng.normal(size=n_rows),
        "cat": pd.Categorical(rng.choice(["u", "v", "w"], size=n_rows)),
    })
    if regression:
        y = pd.Series(X["num"] * 2.0)
    else:
        labels = np.where(X["num"] > 0, "yes", "no").astype(object)
        labels[0] = None  # a missing label: dropped and re-indexed away
        y = pd.Series(pd.Categorical(labels))
    order = np.arange(n_rows)
    fold_of = order % n_folds

    TaskType = types.SimpleNamespace(SUPERVISED_REGRESSION="reg", SUPERVISED_CLASSIFICATION="clf")

    class Task:
        task_type_id = TaskType.SUPERVISED_REGRESSION if regression else TaskType.SUPERVISED_CLASSIFICATION
        target_name = "y"

        def get_dataset(self):
            return types.SimpleNamespace(
                name="toy", get_data=lambda target, dataset_format: (X, y, [False, True], ["num", "cat"]))

        def get_split_dimensions(self):
            return repeats, n_folds, 1

        def get_train_test_split_indices(self, repeat, fold, sample):
            return order[fold_of != fold], order[fold_of == fold]

    fake = types.SimpleNamespace(
        study=types.SimpleNamespace(get_suite=lambda sid: types.SimpleNamespace(tasks=[7])),
        tasks=types.SimpleNamespace(get_task=lambda tid, download_splits: Task(), TaskType=TaskType),
    )
    monkeypatch.setitem(sys.modules, "openml", fake)
    return order, fold_of


def test_openml_cells_use_official_folds(monkeypatch):
    order, fold_of = _fake_openml(monkeypatch)
    pairs = fm.openml_cells(["cc18"], max_rows=1_000, regimes=["defaults"], log=lambda _: None)
    assert len(pairs) == 1
    cell, ds = pairs[0]
    assert cell.source == "openml:cc18:7"
    assert ds.task == fm.BINARY and ds.categorical == (1,)
    assert len(ds.folds) == 4
    # Row 0 (missing label, test fold 0) is gone; the rest shift down by one.
    tr0, te0 = ds.folds[0]
    assert te0.tolist() == [i - 1 for i in order[fold_of == 0] if i != 0]
    assert tr0.tolist() == [i - 1 for i in order[fold_of != 0]]
    for f, (tr, te) in enumerate(ds.folds):
        assert not set(tr.tolist()) & set(te.tolist())
        split = fm.make_split(ds, f, None)
        assert split.test.tolist() == sorted(te.tolist())
        assert split.train.tolist() == sorted(tr.tolist())
    with pytest.raises(IndexError):
        fm.make_split(ds, 4, None)


def test_openml_train_cap_and_validation_keep_test_fold(monkeypatch):
    _fake_openml(monkeypatch, n_rows=80, regression=True)
    (_, ds), = fm.openml_cells(["99"], max_rows=1_000, regimes=["defaults"], train_cap=20,
                               max_folds=2, log=lambda _: None)
    assert ds.task == fm.REGRESSION and len(ds.folds) == 2
    split = fm.make_split(ds, 1, 0.25)
    assert len(split.train) + len(split.valid) == 20
    assert split.test.tolist() == sorted(ds.folds[1][1].tolist())
    assert not set(split.train.tolist()) & set(split.test.tolist())
    assert fm.make_split(ds, 1, 0.25).train.tolist() == split.train.tolist()


def test_openml_skips_large_tasks_and_dedupes(monkeypatch):
    _fake_openml(monkeypatch)
    skipped = []
    assert fm.openml_cells(["cc18"], max_rows=10, regimes=["defaults"], log=skipped.append) == []
    assert skipped and "skip openml task 7" in skipped[0]
    pairs = fm.openml_cells(["cc18", "ctr23"], max_rows=1_000, regimes=["defaults", "standard"],
                            log=lambda _: None)
    assert len(pairs) == 2  # one task, two regimes; the second suite repeats it


def test_run_uses_every_official_fold_as_the_seed_axis(monkeypatch):
    pytest.importorskip("alloygbm")
    _fake_openml(monkeypatch, n_rows=60, n_folds=3)
    (cell, ds), = fm.openml_cells(["cc18"], max_rows=1_000, regimes=["defaults"], log=lambda _: None)
    arm = fm.parse_arm("alloy=alloygbm:n_estimators=5")
    result = fm.run([arm], [cell], seeds=[0], threads=1, tier="smoke",
                    datasets={ds.spec.name: ds}, log=lambda _: None)
    assert sorted(r["seed"] for r in result["records"]) == [0, 1, 2]
    assert all(r["error"] is None for r in result["records"]), result["records"]
    assert result["datasets"][0]["folds"] == 3
