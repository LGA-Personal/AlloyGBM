import pandas as pd

from benchmarks.tabarena.report import write_coverage


def test_published_only_report_keeps_failed_tasks_without_local_rf(
    tmp_path, monkeypatch
):
    import json
    from types import SimpleNamespace

    from benchmarks.tabarena import protocol, report
    from benchmarks.tabarena.run import write_event

    manifest = {
        "spec": {"profile": "lite-hpo", "execution": "local", "models": ["AlloyGBM"]},
        "fingerprint": "fit",
        "expected_jobs": 2,
        "tasks": ["easy", "hard"],
        "provenance": {"source_commit": "source"},
        "upstream": "pin",
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    write_event(tmp_path, "a", "completed")
    write_event(tmp_path, "b", "failed", error="examined timeout")
    methods = [SimpleNamespace(method="[Local] AlloyGBM")]
    monkeypatch.setattr(report, "process_standard_results", lambda _: methods)
    planned = object()
    monkeypatch.setattr(
        protocol,
        "resolve_run",
        lambda _: SimpleNamespace(
            jobs=[],
            context=SimpleNamespace(
                task_metadata_collection=SimpleNamespace(
                    subset_to_jobs=lambda _: planned
                )
            ),
        ),
    )

    class Context:
        calibration_method = fillna_method = "RF (default)"
        method_metadata_collection = SimpleNamespace(method_metadata_lst=[])

        def compare(self, *, output_dir):
            output_dir.mkdir()
            frame = pd.DataFrame({"method": ["[Local] AlloyGBM"], "elo": [1000]})
            frame.to_csv(output_dir / "tabarena_leaderboard.csv", index=False)
            return frame

    def comparison(methods, backend, *, published=False, task_metadata=None, **kwargs):
        assert published and task_metadata is planned
        return Context()

    monkeypatch.setattr(report, "comparison_context", comparison)
    output = report.write_report(tmp_path, published_only=True)
    assert output == tmp_path / "report-published"
    assert not (tmp_path / "report-local").exists()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["partial"] and summary["outcomes"] == {"completed": 1, "failed": 1}
    sources = json.loads((output / "sources.json").read_text())
    assert sources["fillna_method"] == sources["calibration_method"] == "RF (default)"
    assert sources["local_entrants"] == ["[Local] AlloyGBM"]


def test_comparison_uses_local_rf_and_only_adds_alloy_to_published_pool(monkeypatch):
    from types import SimpleNamespace

    import tabarena.contexts

    from benchmarks.tabarena.report import comparison_context

    monkeypatch.setattr(tabarena.contexts, "TabArenaContext", lambda **kwargs: kwargs)
    methods = [
        SimpleNamespace(method="[Local] " + name)
        for name in ("AlloyGBM", "LightGBM", "RandomForest")
    ]
    local = comparison_context(methods, "native", calibrated=True)
    assert local["methods"] == []
    assert (
        local["fillna_method"] == local["calibration_method"] == "[Local] RF (default)"
    )
    planned_tasks = object()
    published = comparison_context(
        methods, "native", published=True, task_metadata=planned_tasks
    )
    assert [m.method for m in published["extra_methods"]] == ["[Local] AlloyGBM"]
    assert "fillna_method" not in published
    assert published["task_metadata"] is planned_tasks
    assert published["only_valid_tasks"] is False


def test_failures_cannot_escape_local_calibration():
    import pytest

    from benchmarks.tabarena.report import validate_local_coverage

    manifest = {
        "spec": {"models": ["AlloyGBM", "RandomForest"]},
        "tasks": ["task"],
        "expected_jobs": 2,
    }
    latest = {
        "a": {"experiment": "TA-AlloyGBM_c1", "status": "failed"},
        "rf": {"experiment": "RandomForest_c1", "status": "completed"},
    }
    assert validate_local_coverage(manifest, latest)
    latest["rf"]["status"] = "failed"
    with pytest.raises(ValueError, match="calibration"):
        validate_local_coverage(manifest, latest)
    manifest["spec"]["models"] = ["AlloyGBM"]
    with pytest.raises(ValueError, match="complete coverage"):
        validate_local_coverage(manifest, latest)


def test_upstream_rf_fallback_penalizes_a_missing_task():
    from tabarena.contexts import TabArenaContext

    rf = pd.DataFrame(
        [
            {
                "dataset": "easy",
                "fold": 0,
                "method": "[Local] RF (default)",
                "metric_error": 0.3,
            },
            {
                "dataset": "hard",
                "fold": 0,
                "method": "[Local] RF (default)",
                "metric_error": 0.8,
            },
        ]
    )
    alloy = pd.DataFrame(
        [
            {
                "dataset": "easy",
                "fold": 0,
                "method": "[Local] ALLOYGBM (default)",
                "metric_error": 0.1,
            }
        ]
    )
    filled = TabArenaContext.fillna_metrics(pd.concat([rf, alloy]), rf)
    hard = filled[
        (filled.method == "[Local] ALLOYGBM (default)") & (filled.dataset == "hard")
    ].iloc[0]
    assert hard.imputed
    assert hard.metric_error == 0.8


def test_coverage_never_turns_failure_into_measurement(tmp_path):
    import json

    from benchmarks.tabarena.run import write_event

    (tmp_path / "run_manifest.json").write_text(json.dumps({"expected_jobs": 3}))
    write_event(tmp_path, "a", "completed", experiment="a", task=["d", 0, 0])
    write_event(
        tmp_path, "b", "failed", experiment="b", task=["d", 0, 0], error="timeout"
    )
    output = write_coverage(tmp_path)
    rows = pd.read_csv(output)
    assert len(rows) == 3
    assert set(rows.status) == {"completed", "failed", "pending"}
    assert len(rows[rows.status == "completed"]) == 1


def test_upstream_selection_is_independent_of_test_scores(tmp_path):
    import numpy as np
    from tabarena.benchmark.result import BaselineResult
    from tabarena.repository.generate_repo import generate_repo_from_results_lst
    from tabarena.simulation.repo_simulator import RepoSimulator

    def evaluate(test_target):
        y = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        raw = []
        for name, val, test in [("Dummy_c1", y, y + 2), ("Dummy_r1", y + 2, y)]:
            raw.append(
                BaselineResult.from_dict(
                    {
                        "framework": name,
                        "metric_error": float(
                            np.sqrt(np.mean((test - test_target) ** 2))
                        ),
                        "metric": "rmse",
                        "problem_type": "regression",
                        "time_train_s": 1.0,
                        "time_infer_s": 0.01,
                        "metric_error_val": float(np.sqrt(np.mean((val - y) ** 2))),
                        "task_metadata": {
                            "fold": 0,
                            "repeat": 0,
                            "sample": 0,
                            "split_idx": 0,
                            "tid": 1,
                            "name": "dummy",
                        },
                        "simulation_artifacts": {
                            "y_val": y,
                            "y_val_idx": np.arange(4),
                            "y_test": test_target,
                            "y_test_idx": np.arange(4, 8),
                            "pred_val": val,
                            "pred_test": test,
                            "num_classes": None,
                            "label": "target",
                        },
                        "method_metadata": {
                            "model_hyperparameters": {},
                            "model_cls": "DummyModel",
                            "model_type": "DUMMY",
                            "ag_key": "DUMMY",
                            "name_prefix": "Dummy",
                        },
                    }
                )
            )
        repo = generate_repo_from_results_lst(
            results_lst=raw,
            task_metadata=pd.DataFrame({"dataset": ["dummy"], "tid": [1]}),
        )
        # One iteration is exactly the standard tuned variant. Weights, rather than
        # headline test error, reveal which config validation selected.
        _errors, weights = repo.evaluate_ensembles(
            datasets=["dummy"],
            configs=["Dummy_c1", "Dummy_r1"],
            ensemble_size=1,
            backend="native",
        )
        variants = RepoSimulator(repo, backend="native").run_minimal_single("DUMMY")
        assert set(variants.method_subtype) == {"default", "tuned", "tuned_ensemble"}
        from types import SimpleNamespace

        from benchmarks.tabarena.report import write_selection_weights

        method = SimpleNamespace(
            hpo_results=variants,
            method_metadata=SimpleNamespace(
                method="Dummy", load_processed=lambda **kwargs: repo
            ),
        )
        write_selection_weights(method, tmp_path)
        exported = pd.read_csv(tmp_path / "Dummy-tuned-weights.csv")
        pd.testing.assert_frame_equal(
            exported, weights.reset_index(), check_dtype=False
        )
        assert (tmp_path / "Dummy-tuned_ensemble-weights.csv").exists()
        return weights

    a = evaluate(np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32))
    b = evaluate(np.array([9.0, 8.0, 9.0, 8.0], dtype=np.float32))
    pd.testing.assert_frame_equal(a, b)
    assert a.iloc[0]["Dummy_c1"] == 1
    assert a.iloc[0]["Dummy_r1"] == 0
