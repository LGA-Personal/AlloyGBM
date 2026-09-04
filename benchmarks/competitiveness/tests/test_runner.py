from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from benchmarks.competitiveness.adapters import AdapterResult, _prepare_input
from benchmarks.competitiveness.schema import INPUT_REPRESENTATIONS, ProfileRecordV1, validate_record
from benchmarks.competitiveness.datasets import build_dataset_cases, fingerprint_case
from benchmarks.competitiveness.run import (
    _record_from_measurement,
    _subprocess_measurement,
    load_manifest,
    run_benchmark,
    run_subprocess_benchmark,
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


@pytest.mark.parametrize("library", ["lightgbm", "xgboost"])
def test_prepared_categorical_columns_keep_native_dtype_and_training_levels(library: str) -> None:
    spec = {"name": "native_categorical", "task": "binary_classification", "rows": 20,
            "numeric_features": 20, "categorical_cardinalities": [3, 4], "rounds": 2,
            "depth": 2, "metric": "log_loss", "input_representation": "native_categorical"}
    case = build_dataset_cases([spec], seed=7)[0]
    train, test, representation, params, _, fit_kwargs = _prepare_input(case, library)
    assert representation == "native_categorical"
    assert str(train.iloc[:, 20].dtype) == "category"
    assert str(test.iloc[:, 20].dtype) == "category"
    assert train.iloc[:, 20].cat.categories.dtype.kind in "iu"
    assert train.iloc[:, 20].cat.categories.equals(test.iloc[:, 20].cat.categories)
    assert fit_kwargs == {}


def test_alloy_categorical_values_are_preprocessing_payload() -> None:
    spec = {"name": "native_categorical", "task": "binary_classification", "rows": 20,
            "numeric_features": 20, "categorical_cardinalities": [3, 4], "rounds": 2,
            "depth": 2, "metric": "log_loss", "input_representation": "native_categorical"}
    case = build_dataset_cases([spec], seed=7)[0]
    _, _, _, _, preprocessing_seconds, fit_kwargs = _prepare_input(case, "alloygbm")
    assert preprocessing_seconds > 0
    assert len(fit_kwargs["categorical_feature_values_list"]) == 2


def test_catboost_keeps_csr_input_without_dense_fallback() -> None:
    spec = {"name": "csr_sparse", "task": "regression", "rows": 20,
            "features": 30, "density": 0.1, "rounds": 2, "depth": 2,
            "metric": "rmse", "input_representation": "csr"}
    case = build_dataset_cases([spec], seed=7)[0]
    train, test, representation, params, _, _ = _prepare_input(case, "catboost")
    import scipy.sparse as sp
    assert sp.isspmatrix_csr(train) and sp.isspmatrix_csr(test)
    assert representation == "csr"
    assert params["sparse_fallback"] == "none"


def test_schema_rejects_unknown_input_representation() -> None:
    assert INPUT_REPRESENTATIONS == frozenset({"dense", "native_categorical", "csr", "csc", "dense_fallback"})
    with pytest.raises(ValueError, match="input_representation"):
        validate_record(__import__("dataclasses").replace(_record_fixture(), input_representation="unknown"))


def _record_fixture():
    from benchmarks.competitiveness.tests.test_schema import record
    return record()


def test_record_rejects_shape_mismatch_and_nonfinite_predictions(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    tiny_manifest(manifest)

    class BadAdapter:
        def fit_predict(self, case, seed, threads):
            return AdapterResult(np.zeros(1), 0.01, 0.01, 0.01, 123, "bad", "1", {}, "dense", 2)

    with pytest.raises(ValueError, match="shape"):
        run_benchmark(manifest, tmp_path / "out", libraries=["bad"], adapters={"bad": BadAdapter()})

    class NonfiniteAdapter(BadAdapter):
        def fit_predict(self, case, seed, threads):
            return AdapterResult(np.full(case.y_test.shape, np.nan), 0.01, 0.01, 0.01, 123, "bad", "1", {}, "dense", 2)
    with pytest.raises(ValueError, match="finite"):
        run_benchmark(manifest, tmp_path / "out-finite", libraries=["bad"], adapters={"bad": NonfiniteAdapter()})


def test_binary_fixture_requires_both_classes_in_each_split() -> None:
    spec = {"name": "binary", "task": "binary_classification", "rows": 2,
            "features": 4, "rounds": 2, "depth": 2, "metric": "log_loss",
            "input_representation": "dense"}
    with pytest.raises(ValueError, match="both classes"):
        build_dataset_cases([spec], seed=7)


def test_native_multioutput_adapters_use_supported_vector_objectives() -> None:
    xgb_case = {"name": "joint_multi_output", "task": "multi_output_regression", "rows": 100,
                "features": 5, "outputs": 2, "rounds": 2, "depth": 2,
                "metric": "rmse", "input_representation": "dense"}
    case = build_dataset_cases([xgb_case], seed=7)[0]
    xgb = pytest.importorskip("xgboost")
    xgb_result = __import__("benchmarks.competitiveness.adapters", fromlist=["load_adapters"]).load_adapters(["xgboost"])["xgboost"].fit_predict(case, 7, 1)
    assert xgb_result.predictions.shape == case.y_test.shape
    assert xgb_result.effective_params["multi_output_strategy"] == "multi_output_tree"
    assert xgb_result.effective_params["objective"] == "reg:squarederror"
    cat = pytest.importorskip("catboost")
    cat_result = __import__("benchmarks.competitiveness.adapters", fromlist=["load_adapters"]).load_adapters(["catboost"])["catboost"].fit_predict(case, 7, 1)
    assert cat_result.predictions.shape == case.y_test.shape
    assert cat_result.effective_params["multi_output_strategy"] == "native_multi_rmse"
    assert cat_result.effective_params["objective"] == "MultiRMSE"


def test_xgboost_native_categorical_fit_accepts_integer_category_ids() -> None:
    pytest.importorskip("xgboost")
    spec = {"name": "native_categorical", "task": "binary_classification", "rows": 100,
            "numeric_features": 20, "categorical_cardinalities": [3, 4], "rounds": 2,
            "depth": 2, "metric": "log_loss", "input_representation": "native_categorical"}
    case = build_dataset_cases([spec], seed=7)[0]
    from benchmarks.competitiveness.adapters import load_adapters
    result = load_adapters(["xgboost"])["xgboost"].fit_predict(case, 7, 1)
    assert result.input_representation == "native_categorical"
    assert result.predictions.shape == case.y_test.shape


def test_missing_optional_dependency_is_library_named(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmarks.competitiveness import adapters
    real_import = adapters.importlib.import_module
    def fail(name):
        if name == "not-installed-lib":
            raise ImportError("missing")
        return real_import(name)
    monkeypatch.setattr(adapters.importlib, "import_module", fail)
    with pytest.raises(RuntimeError, match="lightgbm optional dependency unavailable"):
        adapters._import_optional("not-installed-lib", "lightgbm")


def test_worker_environment_contains_all_thread_controls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import subprocess
    captured = {}
    class Completed:
        returncode = 0
        stdout = json.dumps({"dataset_sha256": "a" * 64, "scenario": "dense_regression", "task": "regression", "metric_name": "rmse", "metric_value": 1.0, "preprocessing_seconds": 0.01, "fit_seconds": 0.01, "predict_seconds": 0.01, "peak_rss_bytes": 123, "library": "lightgbm", "library_version": "1", "effective_params": {}, "input_representation": "dense", "rounds_completed": 1})
        stderr = ""
    def fake_run(command, **kwargs):
        captured.update(kwargs["env"])
        return Completed()
    monkeypatch.setattr(subprocess, "run", fake_run)
    spec = {"name": "dense_regression", "task": "regression", "rows": 20, "features": 4, "rounds": 2, "depth": 2, "metric": "rmse", "input_representation": "dense"}
    case = build_dataset_cases([spec], 7)[0]
    _subprocess_measurement("manifest.yaml", "dense_regression", "lightgbm", 7, 3)
    assert all(captured[name] == "3" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS"))


def test_subprocess_orchestration_does_not_build_parent_fixtures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    tiny_manifest(manifest)
    import benchmarks.competitiveness.run as runner
    monkeypatch.setattr(runner, "build_dataset_cases", lambda *args, **kwargs: pytest.fail("parent must not materialize fixtures"))
    payload = {"dataset_sha256": "a" * 64, "scenario": "dense_regression", "task": "regression", "metric_name": "rmse", "metric_value": 1.0, "preprocessing_seconds": 0.01, "fit_seconds": 0.02, "predict_seconds": 0.01, "peak_rss_bytes": 123, "library": "alloygbm", "library_version": "1", "effective_params": {"topology": "levelwise", "objective": "squared_error"}, "input_representation": "dense", "rounds_completed": 2}
    monkeypatch.setattr(runner, "_subprocess_measurement", lambda *args, **kwargs: payload)
    path = runner.run_subprocess_benchmark(manifest, tmp_path / "out", libraries=["alloygbm"], repetitions=1, warmups=0, smoke=True)
    assert len(load_records(path / "raw.jsonl")) == 1


def _measurement_payload(**changes):
    value = {"dataset_sha256": "a" * 64, "scenario": "dense_regression", "task": "regression", "metric_name": "rmse", "metric_value": 1.0, "preprocessing_seconds": 0.01, "fit_seconds": 0.02, "predict_seconds": 0.01, "peak_rss_bytes": 123, "library": "alloygbm", "library_version": "1", "effective_params": {"topology": "levelwise", "objective": "squared_error"}, "input_representation": "dense", "rounds_completed": 2}
    value.update(changes)
    return value


def _profile_payload(**changes):
    value = {
        "rows": 20,
        "features": 4,
        "rounds": 2,
        "threads": 1,
        "loop_wall_ns": 100,
        "untimed_ns": 10,
        "stage_ns": {label: (20 if label == "tree_build" else 0) for label in (
            "gradients", "row_sampling", "feature_tiles", "prediction_copy",
            "tree_build", "prediction_update", "loss", "validation")},
        "tree_stage_ns": {label: 0 for label in ("histogram_build", "split_find", "partition")},
    }
    value.update(changes)
    return value


@pytest.mark.parametrize("stderr, message", [
    ("", "exactly one"),
    ("[alloygbm profile json] {}\n[alloygbm profile json] {}\n", "exactly one"),
    ("[alloygbm profile json] not-json\n", "JSON"),
])
def test_profiled_alloy_requires_exactly_one_valid_json_record(monkeypatch: pytest.MonkeyPatch, stderr: str, message: str) -> None:
    import subprocess
    class Completed:
        returncode = 0
        stdout = json.dumps(_measurement_payload())
    Completed.stderr = stderr
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: Completed())
    with pytest.raises((RuntimeError, ValueError), match=message):
        _subprocess_measurement("manifest.yaml", "dense_regression", "alloygbm", 7, 1, profile_alloy=True)


def test_valid_profile_is_attached_to_compact_alloy_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    class Completed:
        returncode = 0
        stdout = json.dumps(_measurement_payload())
        stderr = "[alloygbm profile json] " + json.dumps(_profile_payload()) + "\n"
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: Completed())
    value = _subprocess_measurement("manifest.yaml", "dense_regression", "alloygbm", 7, 1, profile_alloy=True)
    assert isinstance(value["profile"], ProfileRecordV1)
    assert value["profile"].threads == 1


def test_profile_flag_sets_environment_only_for_alloy(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    captured: list[dict[str, str]] = []
    class Completed:
        returncode = 0
        stdout = json.dumps(_measurement_payload())
        stderr = "[alloygbm profile json] " + json.dumps(_profile_payload()) + "\n"
    def fake_run(*args, **kwargs):
        captured.append(kwargs["env"])
        return Completed()
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("ALLOYGBM_PROFILE", "json")
    _subprocess_measurement("manifest.yaml", "dense_regression", "alloygbm", 7, 1, profile_alloy=True)
    _subprocess_measurement("manifest.yaml", "dense_regression", "lightgbm", 7, 1, profile_alloy=True)
    _subprocess_measurement("manifest.yaml", "dense_regression", "alloygbm", 7, 1, profile_alloy=False)
    assert captured[0]["ALLOYGBM_PROFILE"] == "json"
    assert "ALLOYGBM_PROFILE" not in captured[1]
    assert "ALLOYGBM_PROFILE" not in captured[2]


def test_warmup_profile_is_not_persisted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    tiny_manifest(manifest)
    payload = _measurement_payload(profile=_profile_payload())
    calls: list[bool] = []
    import benchmarks.competitiveness.run as runner
    def fake_measurement(*args, **kwargs):
        calls.append(kwargs.get("profile_alloy", False))
        return payload
    monkeypatch.setattr(runner, "_subprocess_measurement", fake_measurement)
    run_dir = runner.run_subprocess_benchmark(manifest, tmp_path / "out", libraries=["alloygbm"], repetitions=1, warmups=1, smoke=True, profile_alloy=True)
    records = load_records(run_dir / "raw.jsonl")
    assert len(records) == 1
    assert records[0].profile is not None
    assert calls == [True, True]


def test_profile_allows_honest_early_termination_but_rejects_impossible_rounds() -> None:
    early = _measurement_payload(profile=_profile_payload(rounds=1), rounds_completed=1)
    record = _record_from_measurement(early, "run", 0, 7, 1, None,
                                      expected_scenario="dense_regression", expected_task="regression",
                                      expected_metric="rmse", requested_library="alloygbm")
    assert record.profile is not None and record.profile.rounds == 1
    with pytest.raises(ValueError, match="fewer than completed"):
        _record_from_measurement(_measurement_payload(profile=_profile_payload(rounds=1), rounds_completed=2), "run", 0, 7, 1, None)


def test_real_joint_alloy_smoke_emits_one_structured_profile(tmp_path: Path) -> None:
    manifest = tmp_path / "joint.yaml"
    manifest.write_text(
        """schema: alloygbm-competitiveness/v1
seed: 7
warmup_repetitions: 0
timed_repetitions: 1
scenarios:
  - name: joint_multi_output
    task: multi_output_regression
    rows: 40
    features: 4
    outputs: 2
    rounds: 2
    depth: 2
    metric: rmse
    input_representation: dense
"""
    )
    run_dir = run_subprocess_benchmark(
        manifest,
        tmp_path / "out",
        libraries=["alloygbm"],
        repetitions=1,
        warmups=0,
        smoke=True,
        profile_alloy=True,
    )
    records = load_records(run_dir / "raw.jsonl")
    assert len(records) == 1
    assert records[0].profile is not None
    assert set(records[0].profile.stage_ns) == {
        "gradients", "row_sampling", "feature_tiles", "prediction_copy",
        "tree_build", "prediction_update", "loss", "validation",
    }
    assert set(records[0].profile.tree_stage_ns) == {"histogram_build", "split_find", "partition"}


@pytest.mark.parametrize("changes, message", [
    ({"scenario": "binary"}, "scenario"),
    ({"library": "xgboost"}, "library"),
    ({"task": "binary_classification"}, "task"),
    ({"metric_name": "log_loss"}, "metric"),
])
def test_compact_worker_identity_is_checked_before_record(tmp_path: Path, changes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _record_from_measurement(_measurement_payload(**changes), "run", 0, 7, 1, None, expected_scenario="dense_regression", expected_task="regression", expected_metric="rmse", requested_library="alloygbm")


def test_compact_worker_identity_valid_payload_passes() -> None:
    record = _record_from_measurement(_measurement_payload(), "run", 0, 7, 1, None, expected_scenario="dense_regression", expected_task="regression", expected_metric="rmse", requested_library="alloygbm")
    assert record.scenario == "dense_regression"


def test_compact_worker_rejects_non_mapping_payload() -> None:
    with pytest.raises(ValueError, match="mapping"):
        _record_from_measurement([], "run", 0, 7, 1, None)  # type: ignore[arg-type]


def test_alloy_joint_metadata_does_not_claim_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    import types
    import benchmarks.competitiveness.adapters as adapters
    spec = {"name": "joint_multi_output", "task": "multi_output_regression", "rows": 20,
            "features": 4, "outputs": 2, "rounds": 2, "depth": 2,
            "metric": "rmse", "input_representation": "dense"}
    case = build_dataset_cases([spec], seed=7)[0]

    class FakeJoint:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def fit(self, X, y):
            self.resolved_training_policy_ = None
        def predict(self, X):
            return np.zeros((len(X), 2), dtype=np.float32)

    fake_module = types.SimpleNamespace(MultiLabelGBMRanker=FakeJoint)
    monkeypatch.setattr(adapters.importlib, "import_module", lambda name: fake_module if name == "alloygbm" else __import__(name))
    result = adapters._fit_alloy(case, 7, 1)
    assert result.predictions.shape == case.y_test.shape
    assert result.effective_params["deterministic_applied"] is False
    assert result.effective_params["policy_verification"] == "unavailable_joint_bridge"
    assert result.effective_params.get("deterministic") is not True


def test_mismatched_later_measurement_is_not_appended(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    tiny_manifest(manifest)
    import benchmarks.competitiveness.run as runner
    payloads = iter([_measurement_payload(), _measurement_payload(scenario="binary")])
    monkeypatch.setattr(runner, "_subprocess_measurement", lambda *args, **kwargs: next(payloads))
    with pytest.raises(ValueError, match="scenario"):
        runner.run_subprocess_benchmark(manifest, tmp_path / "out", libraries=["alloygbm"], repetitions=2, warmups=0, smoke=True)
    raw = next((tmp_path / "out").glob("*/raw.jsonl"))
    assert len(raw.read_text().splitlines()) == 1
