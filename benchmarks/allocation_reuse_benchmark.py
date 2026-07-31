#!/usr/bin/env python3
"""Paired allocation-reuse benchmark with isolated AlloyGBM runtimes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
QUICK_REPETITIONS = 1
FULL_REPETITIONS = 5
QUICK_ESTIMATORS = 3
FULL_ESTIMATORS = 24
TIMING_SLOWDOWN_LIMIT = 1.03
RSS_INCREASE_LIMIT = 1.05
MISSING_RATE = 0.03
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()


@dataclass(frozen=True)
class CaseSpec:
    name: str
    shape: str
    n_rows: int
    n_features: int
    n_eval_rows: int
    max_depth: int
    tree_growth: str
    seed: int


@dataclass(frozen=True)
class Fixture:
    X_train: np.ndarray
    y_train: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray

    def arrays(self) -> tuple[np.ndarray, ...]:
        return self.X_train, self.y_train, self.X_eval, self.y_eval


@dataclass(frozen=True)
class RuntimeSpec:
    name: str
    python: Path
    workdir: Path
    source_commit: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "python": str(self.python),
            "workdir": str(self.workdir),
            "source_commit": self.source_commit,
        }


@dataclass(frozen=True)
class WorkerInvocation:
    command: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]

    def value_after(self, flag: str) -> str:
        index = self.command.index(flag)
        return self.command[index + 1]


@dataclass(frozen=True)
class CaseResult:
    artifact_sha256: str
    prediction_sha256: str
    native_seconds: float
    rss_mib: float
    rmse: float = 0.0
    case: str = ""
    shape: str = ""
    tree_growth: str = ""
    repetition: int = 0
    source_commit: str = ""
    extension_sha256: str = ""
    runtime_name: str = "baseline"
    fit_seconds: float = 0.0
    dimensions: dict[str, int] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    python_executable: str = ""
    package_path: str = ""
    extension_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CaseResult:
        required = {
            "artifact_sha256",
            "prediction_sha256",
            "native_seconds",
            "rss_mib",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"worker record missing fields: {missing}")
        return cls(
            artifact_sha256=str(payload["artifact_sha256"]),
            prediction_sha256=str(payload["prediction_sha256"]),
            native_seconds=float(payload["native_seconds"]),
            rss_mib=float(payload["rss_mib"]),
            rmse=float(payload.get("rmse", 0.0)),
            case=str(payload.get("case", "")),
            shape=str(payload.get("shape", "")),
            tree_growth=str(payload.get("tree_growth", "")),
            repetition=int(payload.get("repetition", 0)),
            source_commit=str(payload.get("source_commit", "")),
            extension_sha256=str(payload.get("extension_sha256", "")),
            runtime_name=str(payload.get("runtime_name", "baseline")),
            fit_seconds=float(payload.get("fit_seconds", 0.0)),
            dimensions={
                str(key): int(value)
                for key, value in dict(payload.get("dimensions", {})).items()
            },
            parameters=dict(payload.get("parameters", {})),
            python_executable=str(payload.get("python_executable", "")),
            package_path=str(payload.get("package_path", "")),
            extension_path=str(payload.get("extension_path", "")),
        )


@dataclass(frozen=True)
class PairedResult:
    baseline: CaseResult
    candidate: CaseResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
        }


@dataclass(frozen=True)
class PairEvaluation:
    equivalent: bool
    failures: tuple[str, ...]
    timing_ratio: float
    rss_ratio: float


@dataclass(frozen=True)
class CaseSummary:
    case: str
    shape: str
    tree_growth: str
    repetitions: int
    baseline_native_seconds: float
    candidate_native_seconds: float
    timing_ratio: float
    baseline_rss_mib: float
    candidate_rss_mib: float
    rss_ratio: float


@dataclass(frozen=True)
class GateEvaluation:
    failures: tuple[str, ...]
    aggregate_timing_ratio: float
    aggregate_rss_ratio: float
    performance_gated: bool
    case_summaries: tuple[CaseSummary, ...]


_QUICK_CASES = (
    CaseSpec("tall_deep-level", "tall_deep", 4_096, 12, 1_024, 6, "level", 1101),
    CaseSpec("wide_deep-leaf", "wide_deep", 1_024, 96, 512, 6, "leaf", 1201),
    CaseSpec("short_wide-level", "short_wide", 384, 192, 256, 4, "level", 1301),
    CaseSpec("shallow_tall-leaf", "shallow_tall", 8_192, 8, 1_024, 2, "leaf", 1401),
)

_FULL_SHAPES = (
    ("tall_deep", 32_768, 16, 4_096, 8, 2101),
    ("wide_deep", 8_192, 192, 2_048, 8, 2201),
    ("short_wide", 1_024, 512, 512, 6, 2301),
    ("shallow_tall", 65_536, 8, 4_096, 3, 2401),
)


def quick_cases() -> tuple[CaseSpec, ...]:
    return _QUICK_CASES


def full_cases() -> tuple[CaseSpec, ...]:
    return tuple(
        CaseSpec(
            f"{shape}-{growth}",
            shape,
            rows,
            features,
            eval_rows,
            depth,
            growth,
            seed,
        )
        for shape, rows, features, eval_rows, depth, seed in _FULL_SHAPES
        for growth in ("level", "leaf")
    )


def profile_cases(profile: str) -> tuple[CaseSpec, ...]:
    if profile == "quick":
        return quick_cases()
    if profile == "full":
        return full_cases()
    raise ValueError(f"unknown profile {profile!r}")


def profile_repetitions(profile: str) -> int:
    if profile == "quick":
        return QUICK_REPETITIONS
    if profile == "full":
        return FULL_REPETITIONS
    raise ValueError(f"unknown profile {profile!r}")


def _partition(
    X: np.ndarray, coefficients: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n_features = X.shape[1]
    signal = X @ coefficients
    signal += 0.8 * np.sin(X[:, 0] * 1.7)
    signal += 0.35 * X[:, min(1, n_features - 1)] * X[:, min(2, n_features - 1)]
    signal += 0.2 * np.square(X[:, min(3, n_features - 1)])
    target = signal + rng.normal(0.0, 0.12, size=len(X))
    missing = rng.random(X.shape) < MISSING_RATE
    missing[0, 0] = True
    X[missing] = np.nan
    return np.ascontiguousarray(X, dtype=np.float32), np.ascontiguousarray(
        target, dtype=np.float32
    )


def make_fixture(case: CaseSpec) -> Fixture:
    coefficient_seed, data_seed = np.random.SeedSequence(case.seed).spawn(2)
    coefficient_rng = np.random.default_rng(coefficient_seed)
    data_rng = np.random.default_rng(data_seed)
    coefficients = coefficient_rng.normal(size=case.n_features).astype(np.float32)
    coefficients /= max(float(np.linalg.norm(coefficients)), 1e-6)
    total_rows = case.n_rows + case.n_eval_rows
    X = data_rng.normal(size=(total_rows, case.n_features)).astype(np.float32)
    X_train, y_train = _partition(
        X[: case.n_rows].copy(), coefficients, data_rng
    )
    X_eval, y_eval = _partition(
        X[case.n_rows :].copy(), coefficients, data_rng
    )
    return Fixture(X_train, y_train, X_eval, y_eval)


def _git_commit(workdir: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(workdir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot resolve source commit for {workdir}") from error
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError(f"invalid source commit for {workdir}: {commit!r}")
    return commit


def resolve_runtime(name: str, python: Path, workdir: Path) -> RuntimeSpec:
    python = python.expanduser().absolute()
    workdir = workdir.expanduser().resolve()
    if not python.is_file():
        raise ValueError(f"{name} Python executable does not exist: {python}")
    if not workdir.is_dir():
        raise ValueError(f"{name} workdir does not exist: {workdir}")
    return RuntimeSpec(name, python, workdir, _git_commit(workdir))


def build_worker_invocation(
    runtime: RuntimeSpec,
    case: CaseSpec,
    *,
    profile: str,
    repetition: int,
    n_jobs: int,
) -> WorkerInvocation:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    command = (
        str(runtime.python),
        "-I",
        str(SCRIPT_PATH),
        "--worker",
        "--runtime-name",
        runtime.name,
        "--runtime-workdir",
        str(runtime.workdir),
        "--expected-source-commit",
        runtime.source_commit,
        "--profile",
        profile,
        "--case",
        case.name,
        "--repetition",
        str(repetition),
        "--n-jobs",
        str(n_jobs),
    )
    return WorkerInvocation(command, runtime.workdir, env)


def _finite_nonnegative(value: float) -> bool:
    return math.isfinite(value) and value >= 0.0


def parse_worker_output(
    stdout: str,
    *,
    runtime_name: str,
    expected_case: str,
    expected_repetition: int,
    expected_source_commit: str,
) -> CaseResult:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise ValueError("worker must emit a single JSON object") from error
    if not isinstance(payload, dict):
        raise ValueError("worker must emit a single JSON object")
    try:
        result = CaseResult.from_dict(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid worker record: {error}") from error
    if result.runtime_name != runtime_name:
        raise ValueError(
            f"worker runtime name mismatch: expected {runtime_name}, got {result.runtime_name}"
        )
    if result.case != expected_case or result.repetition != expected_repetition:
        raise ValueError("worker case or repetition does not match invocation")
    if result.source_commit != expected_source_commit:
        raise ValueError(
            "worker source commit mismatch: "
            f"expected {expected_source_commit}, got {result.source_commit}"
        )
    for name, value in (
        ("native_seconds", result.native_seconds),
        ("fit_seconds", result.fit_seconds),
        ("rss_mib", result.rss_mib),
        ("rmse", result.rmse),
    ):
        if not _finite_nonnegative(value):
            raise ValueError(f"worker {name} must be finite and non-negative")
    if not result.artifact_sha256 or not result.prediction_sha256:
        raise ValueError("worker digests must be non-empty")
    return result


def _peak_rss_mib() -> float:
    try:
        import resource

        raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, AttributeError):
        return 0.0
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return raw / divisor


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worker_result(args: argparse.Namespace) -> CaseResult:
    source_commit = _git_commit(args.runtime_workdir)
    if source_commit != args.expected_source_commit:
        raise ValueError(
            "runtime worktree moved after orchestration: "
            f"expected {args.expected_source_commit}, got {source_commit}"
        )
    cases = {case.name: case for case in profile_cases(args.profile)}
    try:
        case = cases[args.case]
    except KeyError as error:
        raise ValueError(f"unknown {args.profile} case {args.case!r}") from error

    # Runtime imports are deliberately worker-only. The orchestrator must not
    # resolve AlloyGBM from its own interpreter or checkout.
    import alloygbm
    from alloygbm import GBMRegressor, _alloygbm

    fixture = make_fixture(case)
    n_estimators = QUICK_ESTIMATORS if args.profile == "quick" else FULL_ESTIMATORS
    parameters: dict[str, Any] = {
        "n_estimators": n_estimators,
        "learning_rate": 0.08,
        "max_depth": case.max_depth,
        "min_data_in_leaf": 8,
        "min_split_gain": 0.0,
        "lambda_l2": 1.0,
        "training_policy": "manual",
        "continuous_binning_max_bins": 64,
        "tree_growth": case.tree_growth,
        "seed": case.seed,
        "deterministic": True,
        "n_jobs": args.n_jobs,
    }
    if case.tree_growth == "leaf":
        parameters["max_leaves"] = min(64, 2**case.max_depth)
    estimator = GBMRegressor(**parameters)
    rss_before = _peak_rss_mib()
    started = time.perf_counter()
    estimator.fit(fixture.X_train, fixture.y_train)
    fit_seconds = time.perf_counter() - started
    rss_after = _peak_rss_mib()
    predictions = np.ascontiguousarray(
        estimator.predict(fixture.X_eval), dtype="<f8"
    )
    residuals = predictions - fixture.y_eval.astype(np.float64)
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    timing = dict(getattr(estimator, "fit_timing_", None) or {})
    native_seconds = float(timing.get("native_train_seconds", fit_seconds))
    package_path = Path(alloygbm.__file__).resolve()
    extension_path = Path(_alloygbm.__file__).resolve()
    return CaseResult(
        artifact_sha256=hashlib.sha256(bytes(estimator.artifact_bytes)).hexdigest(),
        prediction_sha256=hashlib.sha256(predictions.tobytes()).hexdigest(),
        native_seconds=native_seconds,
        rss_mib=max(0.0, rss_after - rss_before),
        rmse=rmse,
        case=case.name,
        shape=case.shape,
        tree_growth=case.tree_growth,
        repetition=args.repetition,
        source_commit=source_commit,
        extension_sha256=_sha256_file(extension_path),
        runtime_name=args.runtime_name,
        fit_seconds=fit_seconds,
        dimensions={
            "n_rows": case.n_rows,
            "n_features": case.n_features,
            "n_eval_rows": case.n_eval_rows,
        },
        parameters=parameters,
        python_executable=str(Path(sys.executable).absolute()),
        package_path=str(package_path),
        extension_path=str(extension_path),
    )


def run_worker(
    runtime: RuntimeSpec,
    case: CaseSpec,
    *,
    profile: str,
    repetition: int,
    n_jobs: int,
    timeout_seconds: float,
) -> CaseResult:
    invocation = build_worker_invocation(
        runtime,
        case,
        profile=profile,
        repetition=repetition,
        n_jobs=n_jobs,
    )
    try:
        completed = subprocess.run(
            invocation.command,
            cwd=invocation.cwd,
            env=invocation.env,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "no worker output").strip()
        raise RuntimeError(
            f"{runtime.name} worker failed for {case.name}: {detail}"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"{runtime.name} worker timed out for {case.name}"
        ) from error
    return parse_worker_output(
        completed.stdout,
        runtime_name=runtime.name,
        expected_case=case.name,
        expected_repetition=repetition,
        expected_source_commit=runtime.source_commit,
    )


def _ratio(candidate: float, baseline: float) -> float:
    if baseline == 0.0:
        return 1.0 if candidate == 0.0 else math.inf
    return candidate / baseline


def evaluate_pair(baseline: CaseResult, candidate: CaseResult) -> PairEvaluation:
    failures: list[str] = []
    if baseline.case != candidate.case or baseline.repetition != candidate.repetition:
        failures.append("case/repetition identity mismatch")
    if baseline.dimensions != candidate.dimensions:
        failures.append("dimensions differ")
    if baseline.parameters != candidate.parameters:
        failures.append("parameters differ")
    if baseline.artifact_sha256 != candidate.artifact_sha256:
        failures.append("artifact SHA-256 differs")
    if baseline.prediction_sha256 != candidate.prediction_sha256:
        failures.append("prediction SHA-256 differs")
    if baseline.rmse.hex() != candidate.rmse.hex():
        failures.append("RMSE differs")
    return PairEvaluation(
        equivalent=not failures,
        failures=tuple(failures),
        timing_ratio=_ratio(candidate.native_seconds, baseline.native_seconds),
        rss_ratio=_ratio(candidate.rss_mib, baseline.rss_mib),
    )


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    if any(value < 0.0 for value in values):
        return math.nan
    if any(math.isinf(value) for value in values):
        return math.inf
    if any(value == 0.0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _case_summaries(pairs: Sequence[PairedResult]) -> tuple[CaseSummary, ...]:
    grouped: dict[str, list[PairedResult]] = {}
    for pair in pairs:
        grouped.setdefault(pair.baseline.case, []).append(pair)
    summaries = []
    for case, group in sorted(grouped.items()):
        baseline_native = median(pair.baseline.native_seconds for pair in group)
        candidate_native = median(pair.candidate.native_seconds for pair in group)
        baseline_rss = median(pair.baseline.rss_mib for pair in group)
        candidate_rss = median(pair.candidate.rss_mib for pair in group)
        first = group[0].baseline
        summaries.append(
            CaseSummary(
                case=case,
                shape=first.shape,
                tree_growth=first.tree_growth,
                repetitions=len(group),
                baseline_native_seconds=baseline_native,
                candidate_native_seconds=candidate_native,
                timing_ratio=_ratio(candidate_native, baseline_native),
                baseline_rss_mib=baseline_rss,
                candidate_rss_mib=candidate_rss,
                rss_ratio=_ratio(candidate_rss, baseline_rss),
            )
        )
    return tuple(summaries)


def evaluate_gates(
    pairs: Sequence[PairedResult], *, profile: str
) -> GateEvaluation:
    if profile not in {"quick", "full"}:
        raise ValueError(f"unknown profile {profile!r}")
    failures: list[str] = []
    for pair in pairs:
        evaluation = evaluate_pair(pair.baseline, pair.candidate)
        failures.extend(
            f"{pair.baseline.case}/rep-{pair.baseline.repetition}: {failure}"
            for failure in evaluation.failures
        )
    summaries = _case_summaries(pairs)
    aggregate_timing = _geometric_mean(
        [summary.timing_ratio for summary in summaries]
    )
    aggregate_rss = _geometric_mean([summary.rss_ratio for summary in summaries])
    performance_gated = profile == "full"
    if performance_gated:
        if not math.isfinite(aggregate_timing) or aggregate_timing > TIMING_SLOWDOWN_LIMIT:
            failures.append(
                "aggregate timing slowdown exceeds 3% "
                f"(ratio={aggregate_timing:.4f})"
            )
        if not math.isfinite(aggregate_rss) or aggregate_rss > RSS_INCREASE_LIMIT:
            failures.append(
                "aggregate RSS increase exceeds 5% "
                f"(ratio={aggregate_rss:.4f})"
            )
        deep = [
            summary
            for summary in summaries
            if summary.shape in {"tall_deep", "wide_deep"}
        ]
        if not any(
            summary.timing_ratio < 1.0 or summary.rss_ratio < 1.0
            for summary in deep
        ):
            failures.append("no median timing or RSS improvement in any deep pressure case")
    return GateEvaluation(
        tuple(failures),
        aggregate_timing,
        aggregate_rss,
        performance_gated,
        summaries,
    )


def _is_within(path_text: str, root: Path) -> bool:
    if not path_text:
        return False
    try:
        Path(path_text).resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def validate_runtime_pair(
    baseline: RuntimeSpec,
    candidate: RuntimeSpec,
    pairs: Sequence[PairedResult],
) -> None:
    if not pairs:
        raise ValueError("benchmark produced no paired results")
    baseline_hashes = {pair.baseline.extension_sha256 for pair in pairs}
    candidate_hashes = {pair.candidate.extension_sha256 for pair in pairs}
    if len(baseline_hashes) != 1 or len(candidate_hashes) != 1:
        raise ValueError("a runtime loaded inconsistent native extensions across workers")
    if (
        baseline.source_commit != candidate.source_commit
        and baseline_hashes == candidate_hashes
    ):
        raise ValueError("distinct source commits loaded the same native extension")
    if baseline.workdir != candidate.workdir:
        for pair in pairs:
            if _is_within(pair.baseline.package_path, candidate.workdir) or _is_within(
                pair.baseline.extension_path, candidate.workdir
            ):
                raise ValueError("baseline worker imported the candidate worktree runtime")
            if _is_within(pair.candidate.package_path, baseline.workdir) or _is_within(
                pair.candidate.extension_path, baseline.workdir
            ):
                raise ValueError("candidate worker imported the baseline worktree runtime")


def build_report(
    *,
    profile: str,
    baseline: RuntimeSpec,
    candidate: RuntimeSpec,
    pairs: Sequence[PairedResult],
    n_jobs: int,
) -> dict[str, Any]:
    gates = evaluate_gates(pairs, profile=profile)
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": profile,
        "repetitions": profile_repetitions(profile),
        "warmup_runs_per_case": 1,
        "n_jobs": n_jobs,
        "host": {
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
        },
        "runtimes": {
            "baseline": baseline.to_dict(),
            "candidate": candidate.to_dict(),
        },
        "cases": [asdict(case) for case in profile_cases(profile)],
        "pairs": [pair.to_dict() for pair in pairs],
        "gate": {
            "failures": list(gates.failures),
            "performance_gated": gates.performance_gated,
            "aggregate_timing_ratio": gates.aggregate_timing_ratio,
            "aggregate_rss_ratio": gates.aggregate_rss_ratio,
            "case_summaries": [asdict(summary) for summary in gates.case_summaries],
        },
    }


def run_benchmark(
    *,
    profile: str,
    baseline: RuntimeSpec,
    candidate: RuntimeSpec,
    n_jobs: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    cases = profile_cases(profile)
    pairs: list[PairedResult] = []
    for case_index, case in enumerate(cases):
        warmup_order = (
            (baseline, candidate) if case_index % 2 == 0 else (candidate, baseline)
        )
        for runtime in warmup_order:
            run_worker(
                runtime,
                case,
                profile=profile,
                repetition=-1,
                n_jobs=n_jobs,
                timeout_seconds=timeout_seconds,
            )
        for repetition in range(profile_repetitions(profile)):
            order = (
                (baseline, candidate)
                if (case_index + repetition) % 2 == 0
                else (candidate, baseline)
            )
            measured = {
                runtime.name: run_worker(
                    runtime,
                    case,
                    profile=profile,
                    repetition=repetition,
                    n_jobs=n_jobs,
                    timeout_seconds=timeout_seconds,
                )
                for runtime in order
            }
            pairs.append(PairedResult(measured["baseline"], measured["candidate"]))
    validate_runtime_pair(baseline, candidate, pairs)
    return build_report(
        profile=profile,
        baseline=baseline,
        candidate=candidate,
        pairs=pairs,
        n_jobs=n_jobs,
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    runtimes = report["runtimes"]
    gate = report["gate"]
    lines = [
        "# Allocation Reuse Benchmark",
        "",
        "## Runtime Identity",
        "",
        "| Arm | Source commit | Python | Workdir |",
        "| --- | --- | --- | --- |",
    ]
    for arm in ("baseline", "candidate"):
        runtime = runtimes[arm]
        lines.append(
            f"| {arm} | `{runtime['source_commit']}` | `{runtime['python']}` | "
            f"`{runtime['workdir']}` |"
        )
    lines.extend(
        [
            "",
            "## Aggregate Gate",
            "",
            f"- Aggregate timing ratio: {gate['aggregate_timing_ratio']:.4f}",
            f"- Aggregate RSS ratio: {gate['aggregate_rss_ratio']:.4f}",
            f"- Performance gated: {gate['performance_gated']}",
            f"- Failures: {len(gate['failures'])}",
        ]
    )
    for failure in gate["failures"]:
        lines.append(f"- FAIL: {failure}")
    lines.extend(
        [
            "",
            "Timing and RSS use per-case medians and aggregate geometric ratios; per-case values are descriptive.",
            "Warmup subprocesses are excluded from every recorded repetition.",
            "",
            "## Case Medians",
            "",
            "| Case | Reps | Baseline native s | Candidate native s | Ratio | Baseline RSS MiB | Candidate RSS MiB | Ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for summary in gate["case_summaries"]:
        lines.append(
            f"| `{summary['case']}` | {summary['repetitions']} | "
            f"{summary['baseline_native_seconds']:.6f} | "
            f"{summary['candidate_native_seconds']:.6f} | "
            f"{summary['timing_ratio']:.4f} | "
            f"{summary['baseline_rss_mib']:.2f} | "
            f"{summary['candidate_rss_mib']:.2f} | {summary['rss_ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Exact Equivalence",
            "",
            "| Case | Rep | RMSE | Artifact SHA-256 | Prediction SHA-256 |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    for pair in report["pairs"]:
        baseline_result = pair["baseline"]
        lines.append(
            f"| `{baseline_result['case']}` | {baseline_result['repetition']} | "
            f"{baseline_result['rmse']:.12g} | "
            f"`{baseline_result['artifact_sha256']}` | "
            f"`{baseline_result['prediction_sha256']}` |"
        )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--quick", action="store_true")
    profile.add_argument("--full", action="store_true")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--baseline-python", type=Path)
    parser.add_argument("--baseline-workdir", type=Path)
    parser.add_argument("--candidate-python", type=Path)
    parser.add_argument("--candidate-workdir", type=Path)
    parser.add_argument("--n-jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--worker-timeout", type=float, default=900.0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--runtime-name", help=argparse.SUPPRESS)
    parser.add_argument("--runtime-workdir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expected-source-commit", help=argparse.SUPPRESS)
    parser.add_argument("--profile", choices=("quick", "full"), help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--repetition", type=int, help=argparse.SUPPRESS)
    return parser


def _require_worker_args(args: argparse.Namespace) -> None:
    required = (
        "runtime_name",
        "runtime_workdir",
        "expected_source_commit",
        "profile",
        "case",
        "repetition",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"worker missing arguments: {', '.join(missing)}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be at least 1")
    if args.worker:
        _require_worker_args(args)
        print(json.dumps(_worker_result(args).to_dict(), sort_keys=True))
        return 0

    profile = "full" if args.full else "quick"
    candidate_python = args.candidate_python or Path(sys.executable)
    candidate_workdir = args.candidate_workdir or REPO_ROOT
    baseline_python = args.baseline_python or candidate_python
    baseline_workdir = args.baseline_workdir or candidate_workdir
    baseline = resolve_runtime("baseline", baseline_python, baseline_workdir)
    candidate = resolve_runtime("candidate", candidate_python, candidate_workdir)
    report = run_benchmark(
        profile=profile,
        baseline=baseline,
        candidate=candidate,
        n_jobs=args.n_jobs,
        timeout_seconds=args.worker_timeout,
    )
    markdown = render_markdown(report)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")
    failures = report["gate"]["failures"]
    if args.gate and failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
