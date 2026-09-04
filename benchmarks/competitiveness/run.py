"""Manifest-driven, repetition-preserving competitiveness benchmark runner."""

from __future__ import annotations

import argparse
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
from .schema import BenchmarkRecordV1, SCHEMA_VERSION, load_records, validate_record

DEFAULT_LIBRARIES = ("alloygbm", "lightgbm", "xgboost", "catboost")
KNOWN_LIBRARIES = frozenset(DEFAULT_LIBRARIES)
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS",
)


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


def _metric(case: DatasetCase, predictions: np.ndarray) -> float:
    truth = np.asarray(case.y_test)
    prediction = np.asarray(predictions)
    if case.metric_name == "rmse":
        return float(np.sqrt(np.mean((truth - prediction) ** 2)))
    if case.metric_name == "mae":
        return float(np.mean(np.abs(truth - prediction)))
    if case.metric_name == "log_loss":
        probability = np.clip(prediction.astype(float), 1e-15, 1 - 1e-15)
        return float(-np.mean(truth * np.log(probability) + (1 - truth) * np.log(1 - probability)))
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
        return float(np.mean(scores)) if scores else 0.0
    raise ValueError(f"unsupported metric: {case.metric_name}")


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
        metric_value=_metric(case, result.predictions),
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
    return run_path


def _worker_result(manifest: str, scenario: str, library: str, seed: int, threads: int) -> None:
    # This function runs in an isolated process, after thread variables were set.
    config = load_manifest(manifest)
    specs = [item for item in config["scenarios"] if isinstance(item, Mapping) and item["name"] == scenario]  # type: ignore[index]
    case = build_dataset_cases(specs, seed)[0]
    result = load_adapters([library])[library].fit_predict(case, seed, threads)
    print(json.dumps({"predictions": np.asarray(result.predictions).tolist(), "preprocessing_seconds": result.preprocessing_seconds, "fit_seconds": result.fit_seconds, "predict_seconds": result.predict_seconds, "peak_rss_bytes": result.peak_rss_bytes, "library": result.library, "library_version": result.library_version, "effective_params": dict(result.effective_params), "input_representation": result.input_representation, "rounds_completed": result.rounds_completed}))


def _subprocess_adapter(manifest: str | Path, case: DatasetCase, library: str, seed: int, threads: int) -> AdapterResult:
    env = os.environ.copy()
    for variable in THREAD_ENVIRONMENT:
        env[variable] = str(threads)
    command = [sys.executable, "-m", "benchmarks.competitiveness.run", "--worker", "--manifest", str(manifest), "--scenario", case.name, "--library", library, "--seed", str(seed), "--threads", str(threads)]
    completed = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"{library} worker failed: {completed.stderr.strip() or completed.stdout.strip()}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{library} worker returned invalid JSON: {completed.stdout!r}") from exc
    return AdapterResult(np.asarray(value["predictions"]), value["preprocessing_seconds"], value["fit_seconds"], value["predict_seconds"], value["peak_rss_bytes"], value["library"], value["library_version"], value["effective_params"], value["input_representation"], value["rounds_completed"])


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
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--library", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        _worker_result(args.manifest, args.scenario, args.library, args.seed, args.threads)
        return 0
    config = load_manifest(args.manifest)
    names = [str(item["name"]) for item in config["scenarios"] if isinstance(item, Mapping)]  # type: ignore[index]
    selected = [args.scenario] if args.scenario else names
    validate_options(selected, args.libraries, threads=args.threads, repetitions=args.repetitions, warmups=args.warmups, smoke=args.smoke, known_scenarios=names)
    # Real comparator runs use one fresh process per repetition.  The injected
    # core above remains direct and cheap for unit tests.
    class ProcessAdapter:
        def __init__(self, name: str) -> None:
            self.name = name
        def fit_predict(self, case: DatasetCase, seed: int, threads: int) -> AdapterResult:
            return _subprocess_adapter(args.manifest, case, self.name, seed, threads)
    adapter_map = {name: ProcessAdapter(name) for name in args.libraries}
    run_path = run_benchmark(args.manifest, args.output_dir, scenario=args.scenario, libraries=args.libraries, threads=args.threads, repetitions=args.repetitions, warmups=args.warmups, smoke=args.smoke, adapters=adapter_map, capture_git_sha=True)
    print(run_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
