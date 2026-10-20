import pandas as pd

from benchmarks.tabarena.report import write_coverage


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
