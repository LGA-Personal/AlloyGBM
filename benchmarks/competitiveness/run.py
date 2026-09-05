"""Manifest-driven, repetition-preserving competitiveness benchmark runner."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml

from .adapters import Adapter, AdapterResult, load_adapters
from .datasets import DatasetCase, build_dataset_cases
from .schema import (
    BenchmarkRecordV1,
    ProfileRecordV1,
    RunMetadataV1,
    SCHEMA_VERSION,
    harness_tree_sha256,
    load_records,
    validate_record,
    validate_run_metadata,
)

DEFAULT_LIBRARIES = ("alloygbm", "lightgbm", "xgboost", "catboost")
KNOWN_LIBRARIES = frozenset(DEFAULT_LIBRARIES)
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS",
)
PROFILE_JSON_PREFIX = "[alloygbm profile json] "


def load_manifest(path: str | Path) -> dict[str, object]:
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("manifest must contain a mapping")
    if value.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema: {value.get('schema')!r}")
    scenarios = value.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("manifest scenarios must be a nonempty list")
    names: set[str] = set()
    for scenario in scenarios:
        if not isinstance(scenario, Mapping) or not scenario.get("name"):
            raise ValueError("each scenario must have a name")
        name = str(scenario["name"])
        if name in names:
            raise ValueError(f"duplicate scenario: {name}")
        names.add(name)
    return value


def validate_options(
    scenarios: Sequence[str],
    libraries: Sequence[str],
    *,
    threads: int = 1,
    repetitions: int | None = None,
    warmups: int | None = None,
    smoke: bool = False,
    profile_alloy: bool = False,
    known_scenarios: Sequence[str] | None = None,
    known_libraries: Sequence[str] | None = None,
) -> None:
    if threads <= 0:
        raise ValueError("threads must be positive")
    if repetitions is not None and repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if warmups is not None and warmups < 0:
        raise ValueError("warmups must be nonnegative")
    if repetitions is not None and repetitions < 3 and not smoke:
        raise ValueError("at least three timed repetitions are required unless --smoke is used")
    known = set(known_scenarios or ("dense_regression", "binary", "grouped_ranking", "native_categorical", "csr_sparse", "joint_multi_output"))
    if known:
        unknown = set(scenarios) - known
        if unknown:
            raise ValueError(f"unknown scenario(s): {sorted(unknown)}")
    unknown_libraries = set(libraries) - set(known_libraries or KNOWN_LIBRARIES)
    if unknown_libraries:
        raise ValueError(f"unknown library(ies): {sorted(unknown_libraries)}")
    if not libraries:
        raise ValueError("at least one library is required")
    if profile_alloy and tuple(libraries) != ("alloygbm",):
        raise ValueError("profile_alloy requires the selected library set to be exactly AlloyGBM")


def _metric(case: DatasetCase, predictions: np.ndarray) -> float:
    truth = np.asarray(case.y_test)
    prediction = np.asarray(predictions)
    if prediction.shape != truth.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} does not match target shape {truth.shape}"
        )
    if not np.all(np.isfinite(prediction)):
        raise ValueError("predictions must be finite")
    def finite_metric(value: float) -> float:
        if not np.isfinite(value):
            raise ValueError("metric result must be finite")
        return float(value)
    if case.metric_name == "rmse":
        return finite_metric(float(np.sqrt(np.mean((truth - prediction) ** 2))))
    if case.metric_name == "mae":
        return finite_metric(float(np.mean(np.abs(truth - prediction))))
    if case.metric_name == "log_loss":
        probability = np.clip(prediction.astype(float), 1e-15, 1 - 1e-15)
        return finite_metric(float(-np.mean(truth * np.log(probability) + (1 - truth) * np.log(1 - probability))))
    if case.metric_name == "ndcg_at_10":
        if case.group_test is None:
            raise ValueError("ranking metrics require test groups")
        scores: list[float] = []
        start = 0
        groups = case.group_test
        while start < len(groups):
            end = start + 1
            while end < len(groups) and groups[end] == groups[start]:
                end += 1
            relevance = truth[start:end]
            order = np.argsort(-prediction[start:end], kind="stable")[:10]
            ideal = np.sort(relevance)[::-1][:10]
            discounts = np.log2(np.arange(2, len(order) + 2))
            dcg = float(np.sum((2.0 ** relevance[order] - 1.0) / discounts))
            idcg = float(np.sum((2.0 ** ideal - 1.0) / discounts[: len(ideal)]))
            scores.append(dcg / idcg if idcg else 0.0)
            start = end
        result = float(np.mean(scores)) if scores else 0.0
        return finite_metric(result)
    raise ValueError(f"unsupported metric: {case.metric_name}")


def _validate_predictions(case: DatasetCase, predictions: np.ndarray) -> np.ndarray:
    value = np.asarray(predictions)
    target = np.asarray(case.y_test)
    if value.shape != target.shape:
        raise ValueError(f"prediction shape {value.shape} does not match target shape {target.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("predictions must be finite")
    return value


def _machine() -> dict[str, str]:
    return {
        "platform": platform.system().lower(), "architecture": platform.machine(),
        "logical_cpu_count": str(os.cpu_count() or 1), "python_version": platform.python_version(),
        "hostname": socket.gethostname(),
    }


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_sha_at(path: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _harness_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest_identifier(path: Path, harness_root: Path) -> str:
    try:
        return path.resolve().relative_to(harness_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _write_run_metadata(
    run_path: Path,
    manifest: str | Path,
    *,
    run_id: str,
    measured_git_sha: str | None,
    scenario_names: Sequence[str],
    libraries: Sequence[str],
    seed: int,
    threads: int,
    repetitions: int,
    warmups: int,
    smoke: bool,
    profile_alloy: bool = False,
) -> Path:
    raw_path = run_path / "raw.jsonl"
    manifest_path = Path(manifest).resolve()
    harness_root = _harness_root()
    metadata = RunMetadataV1(
        schema=SCHEMA_VERSION,
        run_id=run_id,
        measured_git_sha=measured_git_sha,
        git_sha_semantics="runner working-directory source commit; for AlloyGBM this is the measured library commit",
        harness_git_sha=_git_sha_at(harness_root),
        harness_tree_sha256=harness_tree_sha256(
            harness_root / "benchmarks" / "competitiveness"
        ),
        harness_source_path="benchmarks/competitiveness",
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        manifest_identifier=_manifest_identifier(manifest_path, harness_root),
        manifest_path=_manifest_identifier(manifest_path, harness_root),
        libraries=tuple(libraries),
        scenarios=tuple(scenario_names),
        seed=seed,
        threads=threads,
        repetitions=repetitions,
        warmups=warmups,
        smoke=smoke,
        profile_alloy=profile_alloy,
        raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        raw_record_count=len(load_records(raw_path)),
        created_at_utc=datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat(),
        working_directory="repository-root",
    )
    validate_run_metadata(metadata)
    metadata_path = run_path / "run-metadata.json"
    metadata_path.write_text(json.dumps(metadata.to_dict(), allow_nan=False, sort_keys=True, indent=2) + "\n")
    return metadata_path


def _record(case: DatasetCase, result: AdapterResult, run_id: str, repetition: int, seed: int, threads: int, git_sha: str | None = None) -> BenchmarkRecordV1:
    record = BenchmarkRecordV1(
        schema=SCHEMA_VERSION, run_id=run_id, repetition=repetition,
        dataset_sha256=case.dataset_sha256, scenario=case.name, task=case.task,
        library=result.library, library_version=result.library_version,
        git_sha=git_sha, seed=seed, threads=threads,
        effective_params=dict(result.effective_params),
        input_representation=result.input_representation,
        preprocessing_seconds=result.preprocessing_seconds,
        fit_seconds=result.fit_seconds, predict_seconds=result.predict_seconds,
        peak_rss_bytes=result.peak_rss_bytes, metric_name=case.metric_name,
        metric_value=_metric(case, _validate_predictions(case, result.predictions)),
        rounds_completed=result.rounds_completed, machine=_machine(), profile=None,
    )
    validate_record(record)
    return record


def run_benchmark(
    manifest: str | Path,
    output_dir: str | Path,
    *,
    scenario: str | None = None,
    libraries: Sequence[str] | None = None,
    threads: int = 1,
    repetitions: int | None = None,
    warmups: int | None = None,
    smoke: bool = False,
    adapters: Mapping[str, Adapter] | None = None,
    capture_git_sha: bool | None = None,
) -> Path:
    config = load_manifest(manifest)
    all_specs = [item for item in config["scenarios"] if isinstance(item, Mapping)]  # type: ignore[index]
    names = [str(item["name"]) for item in all_specs]
    selected_names = [scenario] if scenario else names
    selected_specs = [item for item in all_specs if str(item["name"]) in selected_names]
    selected_libraries = list(libraries or DEFAULT_LIBRARIES)
    validate_options(selected_names, selected_libraries, threads=threads, repetitions=repetitions, warmups=warmups, smoke=smoke, known_scenarios=names, known_libraries=(list(adapters) if adapters is not None else None))
    if len(selected_specs) != len(selected_names):
        raise ValueError(f"unknown scenario(s): {sorted(set(selected_names) - set(names))}")
    timed = int(repetitions if repetitions is not None else config.get("timed_repetitions", 0))
    warmup_count = int(warmups if warmups is not None else config.get("warmup_repetitions", 0))
    validate_options(selected_names, selected_libraries, threads=threads, repetitions=timed, warmups=warmup_count, smoke=smoke, known_scenarios=names, known_libraries=(list(adapters) if adapters is not None else None))
    seed = int(config.get("seed", 0))
    cases = {case.name: case for case in build_dataset_cases(selected_specs, seed)}
    adapter_map: Mapping[str, Adapter] = adapters if adapters is not None else load_adapters(selected_libraries)
    # Injected adapters intentionally stay in-process and subprocess-free for
    # unit tests; real CLI runs opt into git metadata collection.
    git_sha = _git_sha() if (capture_git_sha if capture_git_sha is not None else adapters is None) else None
    missing = set(selected_libraries) - set(adapter_map)
    if missing:
        raise ValueError(f"no adapter registered for library(ies): {sorted(missing)}")
    run_id = str(uuid.uuid4())
    run_path = Path(output_dir) / run_id
    run_path.mkdir(parents=True, exist_ok=False)
    raw_path = run_path / "raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as output:
        for case in cases.values():
            for library in selected_libraries:
                adapter = adapter_map[library]
                for _ in range(warmup_count):
                    adapter.fit_predict(case, seed, threads)
                for repetition in range(timed):
                    result = adapter.fit_predict(case, seed, threads)
                    output.write(_record(case, result, run_id, repetition, seed, threads, git_sha).to_json() + "\n")
    load_records(raw_path)
    _write_run_metadata(
        run_path, manifest, run_id=run_id, measured_git_sha=git_sha,
        scenario_names=selected_names, libraries=selected_libraries, seed=seed,
        threads=threads, repetitions=timed, warmups=warmup_count, smoke=smoke,
        profile_alloy=False,
    )
    return run_path


def _worker_result(manifest: str, scenario: str, library: str, seed: int, threads: int, measurement: bool = False) -> None:
    # This function runs in an isolated process, after thread variables were set.
    config = load_manifest(manifest)
    specs = [item for item in config["scenarios"] if isinstance(item, Mapping) and item["name"] == scenario]  # type: ignore[index]
    case = build_dataset_cases(specs, seed)[0]
    result = load_adapters([library])[library].fit_predict(case, seed, threads)
    predictions = _validate_predictions(case, result.predictions)
    value = {"dataset_sha256": case.dataset_sha256, "scenario": case.name, "task": case.task, "metric_name": case.metric_name, "metric_value": _metric(case, predictions), "preprocessing_seconds": result.preprocessing_seconds, "fit_seconds": result.fit_seconds, "predict_seconds": result.predict_seconds, "peak_rss_bytes": result.peak_rss_bytes, "library": result.library, "library_version": result.library_version, "effective_params": dict(result.effective_params), "input_representation": result.input_representation, "rounds_completed": result.rounds_completed}
    if not measurement:
        value["predictions"] = predictions.tolist()
    print(json.dumps(value))


def _subprocess_adapter(manifest: str | Path, case: DatasetCase, library: str, seed: int, threads: int) -> AdapterResult:
    env = os.environ.copy()
    for variable in THREAD_ENVIRONMENT:
        env[variable] = str(threads)
    env.pop("ALLOYGBM_PROFILE", None)
    command = [sys.executable, "-m", "benchmarks.competitiveness.run", "--worker", "--manifest", str(manifest), "--scenario", case.name, "--library", library, "--seed", str(seed), "--threads", str(threads)]
    completed = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"{library} worker failed: {completed.stderr.strip() or completed.stdout.strip()}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{library} worker returned invalid JSON: {completed.stdout!r}") from exc
    return AdapterResult(np.asarray(value["predictions"]), value["preprocessing_seconds"], value["fit_seconds"], value["predict_seconds"], value["peak_rss_bytes"], value["library"], value["library_version"], value["effective_params"], value["input_representation"], value["rounds_completed"])


def _profile_from_stderr(
    stderr: str,
    *,
    required: bool,
    manifest: str | Path | None = None,
    scenario: str | None = None,
    threads: int | None = None,
) -> ProfileRecordV1 | None:
    lines = [line[len(PROFILE_JSON_PREFIX):] for line in stderr.splitlines() if line.startswith(PROFILE_JSON_PREFIX)]
    if not required:
        return None
    if len(lines) != 1:
        raise RuntimeError(f"profile JSON must contain exactly one record, found {len(lines)}")
    try:
        profile = ProfileRecordV1.from_json(lines[0])
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid AlloyGBM profile JSON: {exc}") from exc
    if threads is not None and profile.threads != threads:
        raise RuntimeError(f"profile threads {profile.threads} do not match requested {threads}")
    if manifest is not None and scenario is not None and Path(manifest).exists():
        config = load_manifest(manifest)
        spec = next((item for item in config["scenarios"] if isinstance(item, Mapping) and item.get("name") == scenario), None)  # type: ignore[index]
        if spec is not None:
            expected_rows = int(spec["rows"]) * 4 // 5
            if "groups" in spec:
                group_size = int(spec["rows"]) // int(spec["groups"])
                expected_rows = (expected_rows // group_size) * group_size
            expected_features = int(spec.get("features", int(spec.get("numeric_features", 0)) + len(spec.get("categorical_cardinalities", []))))
            expected_rounds = int(spec["rounds"])
            if (profile.rows, profile.features) != (expected_rows, expected_features):
                raise RuntimeError(
                    "profile dimensions do not match manifest: "
                    f"got {(profile.rows, profile.features)}, "
                    f"expected {(expected_rows, expected_features)}"
                )
            if profile.rounds > expected_rounds:
                raise RuntimeError(
                    "profile dimensions do not match manifest: "
                    f"profile rounds {profile.rounds} exceeds configured {expected_rounds}"
                )
    return profile


def _subprocess_measurement(
    manifest: str | Path,
    scenario: str,
    library: str,
    seed: int,
    threads: int,
    *,
    profile_alloy: bool = False,
) -> dict[str, object]:
    env = os.environ.copy()
    for variable in THREAD_ENVIRONMENT:
        env[variable] = str(threads)
    env.pop("ALLOYGBM_PROFILE", None)
    if profile_alloy and library == "alloygbm":
        env["ALLOYGBM_PROFILE"] = "json"
    command = [sys.executable, "-m", "benchmarks.competitiveness.run", "--worker", "--measurement", "--manifest", str(manifest), "--scenario", scenario, "--library", library, "--seed", str(seed), "--threads", str(threads)]
    completed = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"{library} worker failed: {completed.stderr.strip() or completed.stdout.strip()}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{library} worker returned invalid JSON: {completed.stdout!r}") from exc
    if "predictions" in value:
        raise RuntimeError("worker measurement payload must not contain predictions")
    profile = _profile_from_stderr(
        completed.stderr,
        required=profile_alloy and library == "alloygbm",
        manifest=manifest,
        scenario=scenario,
        threads=threads,
    )
    if profile is not None:
        value["profile"] = profile
    return value


def _record_from_measurement(value: Mapping[str, object], run_id: str, repetition: int, seed: int, threads: int, git_sha: str | None, *, expected_scenario: str | None = None, expected_task: str | None = None, expected_metric: str | None = None, requested_library: str | None = None) -> BenchmarkRecordV1:
    if not isinstance(value, Mapping):
        raise ValueError("worker measurement payload must be a mapping")
    required = {"dataset_sha256", "scenario", "task", "metric_name", "metric_value", "preprocessing_seconds", "fit_seconds", "predict_seconds", "peak_rss_bytes", "library", "library_version", "effective_params", "input_representation", "rounds_completed"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"worker measurement payload missing fields: {sorted(missing)}")
    if not isinstance(value["effective_params"], Mapping):
        raise ValueError("worker measurement effective_params must be a mapping")
    checks = (("scenario", expected_scenario), ("task", expected_task), ("metric_name", expected_metric), ("library", requested_library))
    for field, expected in checks:
        if expected is not None and value.get(field) != expected:
            raise ValueError(f"worker payload {field} does not match requested value {expected!r}")
    profile_value = value.get("profile")
    if profile_value is not None and not isinstance(profile_value, ProfileRecordV1 | Mapping):
        raise ValueError("worker profile must be a profile object")
    profile = profile_value if isinstance(profile_value, ProfileRecordV1) else ProfileRecordV1.from_dict(profile_value) if profile_value is not None else None
    if profile is not None and profile.threads != threads:
        raise ValueError(f"worker profile threads {profile.threads} do not match requested {threads}")
    record = BenchmarkRecordV1(
        schema=SCHEMA_VERSION, run_id=run_id, repetition=repetition,
        dataset_sha256=str(value["dataset_sha256"]), scenario=str(value["scenario"]),
        task=str(value["task"]), library=str(value["library"]),
        library_version=str(value["library_version"]), git_sha=git_sha,
        seed=seed, threads=threads, effective_params=dict(value["effective_params"]),
        input_representation=str(value["input_representation"]),
        preprocessing_seconds=float(value["preprocessing_seconds"]),
        fit_seconds=float(value["fit_seconds"]), predict_seconds=float(value["predict_seconds"]),
        peak_rss_bytes=int(value["peak_rss_bytes"]), metric_name=str(value["metric_name"]),
        metric_value=float(value["metric_value"]), rounds_completed=int(value["rounds_completed"]),
        machine=_machine(), profile=profile,
    )
    validate_record(record)
    return record


def run_subprocess_benchmark(
    manifest: str | Path, output_dir: str | Path, *, scenario: str | None = None,
    libraries: Sequence[str] | None = None, threads: int = 1,
    repetitions: int | None = None, warmups: int | None = None, smoke: bool = False,
    profile_alloy: bool = False,
) -> Path:
    """Run the real CLI path without retaining feature matrices in the parent."""
    config = load_manifest(manifest)
    specs = [item for item in config["scenarios"] if isinstance(item, Mapping)]  # type: ignore[index]
    names = [str(item["name"]) for item in specs]
    spec_by_name = {str(item["name"]): item for item in specs}
    selected_names = [scenario] if scenario else names
    selected_libraries = list(libraries or DEFAULT_LIBRARIES)
    validate_options(selected_names, selected_libraries, threads=threads, repetitions=repetitions, warmups=warmups, smoke=smoke, profile_alloy=profile_alloy, known_scenarios=names)
    if any(name not in names for name in selected_names):
        raise ValueError(f"unknown scenario(s): {sorted(set(selected_names) - set(names))}")
    timed = int(repetitions if repetitions is not None else config.get("timed_repetitions", 0))
    warmup_count = int(warmups if warmups is not None else config.get("warmup_repetitions", 0))
    validate_options(selected_names, selected_libraries, threads=threads, repetitions=timed, warmups=warmup_count, smoke=smoke, profile_alloy=profile_alloy, known_scenarios=names)
    seed = int(config.get("seed", 0))
    run_id = str(uuid.uuid4())
    run_path = Path(output_dir) / run_id
    run_path.mkdir(parents=True, exist_ok=False)
    git_sha = _git_sha()
    with (run_path / "raw.jsonl").open("w", encoding="utf-8") as output:
        for name in selected_names:
            for library in selected_libraries:
                for _ in range(warmup_count):
                    _subprocess_measurement(manifest, name, library, seed, threads, profile_alloy=False)
                for repetition in range(timed):
                    value = _subprocess_measurement(manifest, name, library, seed, threads, profile_alloy=profile_alloy)
                    spec = spec_by_name[name]
                    output.write(_record_from_measurement(value, run_id, repetition, seed, threads, git_sha, expected_scenario=name, expected_task=str(spec["task"]), expected_metric=str(spec["metric"]), requested_library=library).to_json() + "\n")
    load_records(run_path / "raw.jsonl")
    _write_run_metadata(
        run_path, manifest, run_id=run_id, measured_git_sha=git_sha,
        scenario_names=selected_names, libraries=selected_libraries, seed=seed,
        threads=threads, repetitions=timed, warmups=warmup_count, smoke=smoke,
        profile_alloy=profile_alloy,
    )
    return run_path


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=False, default=".")
    parser.add_argument("--scenario")
    parser.add_argument("--libraries", nargs="+", default=list(DEFAULT_LIBRARIES))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--profile-alloy", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--measurement", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--library", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        _worker_result(args.manifest, args.scenario, args.library, args.seed, args.threads, args.measurement)
        return 0
    config = load_manifest(args.manifest)
    names = [str(item["name"]) for item in config["scenarios"] if isinstance(item, Mapping)]  # type: ignore[index]
    selected = [args.scenario] if args.scenario else names
    validate_options(selected, args.libraries, threads=args.threads, repetitions=args.repetitions, warmups=args.warmups, smoke=args.smoke, profile_alloy=args.profile_alloy, known_scenarios=names)
    run_path = run_subprocess_benchmark(args.manifest, args.output_dir, scenario=args.scenario, libraries=args.libraries, threads=args.threads, repetitions=args.repetitions, warmups=args.warmups, smoke=args.smoke, profile_alloy=args.profile_alloy)
    print(run_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
