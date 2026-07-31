#!/usr/bin/env python3
"""Paired allocation-reuse benchmark with isolated AlloyGBM runtimes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
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


SCHEMA_VERSION = 3
RUNTIME_MANIFEST_SCHEMA_VERSION = 1
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
class RuntimeManifest:
    schema_version: int
    runtime_name: str
    source_commit: str
    python_executable: str
    package_path: str
    extension_path: str
    extension_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeManifest:
        expected = {
            "schema_version",
            "runtime_name",
            "source_commit",
            "python_executable",
            "package_path",
            "extension_path",
            "extension_sha256",
        }
        if set(payload) != expected:
            raise ValueError(
                "runtime manifest fields must exactly match the schema: "
                f"expected={sorted(expected)}, actual={sorted(payload)}"
            )
        string_fields = expected - {"schema_version"}
        if type(payload["schema_version"]) is not int or any(
            not isinstance(payload[field_name], str)
            for field_name in string_fields
        ):
            raise ValueError("runtime manifest field types do not match the schema")
        manifest = cls(
            schema_version=payload["schema_version"],
            runtime_name=payload["runtime_name"],
            source_commit=payload["source_commit"],
            python_executable=payload["python_executable"],
            package_path=payload["package_path"],
            extension_path=payload["extension_path"],
            extension_sha256=payload["extension_sha256"],
        )
        if manifest.schema_version != RUNTIME_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported runtime manifest schema {manifest.schema_version}"
            )
        return manifest


@dataclass(frozen=True)
class RuntimeSpec:
    name: str
    python: Path
    workdir: Path
    source_commit: str
    manifest_path: Path | None = None
    attestation: RuntimeManifest | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "python": str(self.python),
            "workdir": str(self.workdir),
            "source_commit": self.source_commit,
            "manifest_path": (
                None if self.manifest_path is None else str(self.manifest_path)
            ),
        }
        if self.attestation is not None:
            payload["attestation"] = self.attestation.to_dict()
        return payload


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
    rss_mib: float | None
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
            rss_mib=(
                None
                if payload["rss_mib"] is None
                else float(payload["rss_mib"])
            ),
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
    rss_ratio: float | None


@dataclass(frozen=True)
class CaseSummary:
    case: str
    shape: str
    tree_growth: str
    repetitions: int
    baseline_native_seconds: float
    candidate_native_seconds: float
    timing_ratio: float
    baseline_rss_mib: float | None
    candidate_rss_mib: float | None
    rss_ratio: float | None


@dataclass(frozen=True)
class GateEvaluation:
    failures: tuple[str, ...]
    aggregate_timing_ratio: float
    aggregate_rss_ratio: float | None
    rss_cases_available: int
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


def resolve_runtime(
    name: str,
    python: Path,
    workdir: Path,
    manifest_path: Path | None = None,
) -> RuntimeSpec:
    python = Path(os.path.abspath(python.expanduser()))
    workdir = workdir.expanduser().resolve()
    if not python.is_file():
        raise ValueError(f"{name} Python executable does not exist: {python}")
    if not workdir.is_dir():
        raise ValueError(f"{name} workdir does not exist: {workdir}")
    runtime = RuntimeSpec(name, python, workdir, _git_commit(workdir))
    if manifest_path is None:
        return runtime
    manifest_path = manifest_path.expanduser().resolve()
    manifest = load_runtime_manifest(
        manifest_path, runtime=runtime, expected_name=name
    )
    return replace(
        runtime,
        manifest_path=manifest_path,
        attestation=manifest,
    )


def build_worker_invocation(
    runtime: RuntimeSpec,
    case: CaseSpec,
    *,
    profile: str,
    repetition: int,
    n_jobs: int,
) -> WorkerInvocation:
    require_runtime_attestation(runtime)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    command_prefix = (
        str(runtime.python),
        "-I",
        str(SCRIPT_PATH),
        f"--{profile}",
        "--worker",
        "--runtime-name",
        runtime.name,
        "--runtime-workdir",
        str(runtime.workdir),
        "--expected-python",
        str(runtime.python),
        "--expected-source-commit",
        runtime.source_commit,
    )
    command = command_prefix + (
        "--runtime-manifest",
        str(runtime.manifest_path),
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


def strict_json_dumps(payload: Any, **kwargs: Any) -> str:
    return json.dumps(payload, allow_nan=False, **kwargs)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _reject_duplicate_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def strict_json_loads(payload: str) -> Any:
    return json.loads(
        payload,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )


def _normalized_path(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _valid_hex(value: str, length: int) -> bool:
    return len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _validate_manifest_against_runtime(
    manifest: RuntimeManifest,
    runtime: RuntimeSpec,
    *,
    expected_name: str | None,
) -> None:
    if expected_name is not None and manifest.runtime_name != expected_name:
        raise ValueError(
            "runtime manifest name mismatch: "
            f"expected {expected_name}, got {manifest.runtime_name}"
        )
    if manifest.source_commit != runtime.source_commit:
        raise ValueError(
            "runtime manifest commit mismatch: "
            f"expected {runtime.source_commit}, got {manifest.source_commit}"
        )
    if not _valid_hex(manifest.source_commit, 40):
        raise ValueError("runtime manifest commit must be a lowercase Git SHA")
    if _normalized_path(manifest.python_executable) != _normalized_path(
        runtime.python
    ):
        raise ValueError("runtime manifest executable does not match declared Python")
    if not runtime.python.is_file():
        raise ValueError("runtime manifest executable is missing")
    try:
        package_path = Path(manifest.package_path).resolve(strict=True)
    except OSError as error:
        raise ValueError("runtime manifest package path is missing") from error
    if not package_path.is_file():
        raise ValueError("runtime manifest package path is not a file")
    try:
        extension_path = Path(manifest.extension_path).resolve(strict=True)
    except OSError as error:
        raise ValueError("runtime manifest extension path is missing") from error
    if not extension_path.is_file():
        raise ValueError("runtime manifest extension path is not a file")
    if not _valid_hex(manifest.extension_sha256, 64):
        raise ValueError("runtime manifest extension digest must be lowercase SHA-256")
    actual_digest = _sha256_file(extension_path)
    if actual_digest != manifest.extension_sha256:
        raise ValueError(
            "runtime manifest extension digest is stale: "
            f"expected {manifest.extension_sha256}, got {actual_digest}"
        )


def write_runtime_manifest(path: Path, manifest: RuntimeManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        strict_json_dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_runtime_manifest(
    path: Path,
    *,
    runtime: RuntimeSpec,
    expected_name: str | None,
) -> RuntimeManifest:
    try:
        payload = strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"runtime manifest is not strict JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("runtime manifest must contain a JSON object")
    manifest = RuntimeManifest.from_dict(payload)
    _validate_manifest_against_runtime(
        manifest, runtime, expected_name=expected_name
    )
    return manifest


def loaded_runtime_attestation(runtime: RuntimeSpec) -> RuntimeManifest:
    if runtime.manifest_path is None or runtime.attestation is None:
        raise ValueError(
            f"{runtime.name} requires a loaded, verified runtime manifest"
        )
    return runtime.attestation


def require_runtime_attestation(runtime: RuntimeSpec) -> RuntimeManifest:
    attestation = loaded_runtime_attestation(runtime)
    _validate_manifest_against_runtime(
        attestation,
        runtime,
        expected_name=None,
    )
    return attestation


def validate_result_binding(runtime: RuntimeSpec, result: CaseResult) -> None:
    manifest = require_runtime_attestation(runtime)
    try:
        package_matches = Path(result.package_path).resolve(
            strict=True
        ) == Path(manifest.package_path).resolve(strict=True)
        extension_matches = Path(result.extension_path).resolve(
            strict=True
        ) == Path(manifest.extension_path).resolve(strict=True)
    except OSError:
        package_matches = False
        extension_matches = False
    manifest_matches = (
        result.source_commit == manifest.source_commit
        and _normalized_path(result.python_executable)
        == _normalized_path(manifest.python_executable)
        and package_matches
        and extension_matches
        and result.extension_sha256 == manifest.extension_sha256
    )
    if not manifest_matches:
        raise ValueError(
            f"{runtime.name} live worker record does not match runtime manifest"
        )


def require_native_train_seconds(timing: Mapping[str, Any]) -> float:
    value = timing.get("native_train_seconds")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError("fit_timing_ must contain finite positive native_train_seconds")
    return float(value)


def parse_worker_output(
    stdout: str,
    *,
    runtime: RuntimeSpec,
    expected_case: str,
    expected_repetition: int,
) -> CaseResult:
    try:
        payload = json.loads(stdout, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError("worker must emit a single JSON object") from error
    if not isinstance(payload, dict):
        raise ValueError("worker must emit a single JSON object")
    try:
        result = CaseResult.from_dict(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid worker record: {error}") from error
    if result.runtime_name != runtime.name:
        raise ValueError(
            f"worker runtime name mismatch: expected {runtime.name}, got {result.runtime_name}"
        )
    if result.case != expected_case or result.repetition != expected_repetition:
        raise ValueError("worker case or repetition does not match invocation")
    if result.source_commit != runtime.source_commit:
        raise ValueError(
            "worker source commit mismatch: "
            f"expected {runtime.source_commit}, got {result.source_commit}"
        )
    for name, value in (
        ("native_seconds", result.native_seconds),
        ("fit_seconds", result.fit_seconds),
        ("rmse", result.rmse),
    ):
        if not _finite_nonnegative(value):
            raise ValueError(f"worker {name} must be finite and non-negative")
    if result.native_seconds <= 0.0:
        raise ValueError("worker native_seconds must be positive")
    if result.rss_mib is not None and not _finite_nonnegative(result.rss_mib):
        raise ValueError("worker rss_mib must be null or finite and non-negative")
    if not result.artifact_sha256 or not result.prediction_sha256:
        raise ValueError("worker digests must be non-empty")
    if len(result.extension_sha256) != 64:
        raise ValueError("worker extension SHA-256 must contain 64 hex characters")
    validate_result_binding(runtime, result)
    return result


def _peak_rss_mib() -> float | None:
    try:
        import resource

        raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, AttributeError):
        return None
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return raw / divisor


def _rss_delta_mib(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    if not math.isfinite(before) or not math.isfinite(after):
        return None
    return max(0.0, after - before)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worker_result(args: argparse.Namespace) -> CaseResult:
    profile = profile_from_args(args)
    source_commit = _git_commit(args.runtime_workdir)
    if source_commit != args.expected_source_commit:
        raise ValueError(
            "runtime worktree moved after orchestration: "
            f"expected {args.expected_source_commit}, got {source_commit}"
        )
    cases = {case.name: case for case in profile_cases(profile)}
    try:
        case = cases[args.case]
    except KeyError as error:
        raise ValueError(f"unknown {profile} case {args.case!r}") from error

    # Runtime imports are deliberately worker-only. The orchestrator must not
    # resolve AlloyGBM from its own interpreter or checkout.
    import alloygbm
    from alloygbm import GBMRegressor, _alloygbm

    package_path = Path(alloygbm.__file__).resolve()
    extension_path = Path(_alloygbm.__file__).resolve()
    worker_runtime = RuntimeSpec(
        args.runtime_name,
        _normalized_path(args.expected_python),
        args.runtime_workdir.resolve(),
        source_commit,
    )
    manifest_path = args.runtime_manifest.resolve()
    worker_runtime = replace(
        worker_runtime,
        manifest_path=manifest_path,
        attestation=load_runtime_manifest(
            manifest_path,
            runtime=worker_runtime,
            expected_name=None,
        ),
    )
    extension_sha256 = _sha256_file(extension_path)
    binding_record = CaseResult(
        artifact_sha256="binding-only",
        prediction_sha256="binding-only",
        native_seconds=1.0,
        rss_mib=None,
        source_commit=source_commit,
        extension_sha256=extension_sha256,
        runtime_name=args.runtime_name,
        python_executable=str(_normalized_path(sys.executable)),
        package_path=str(package_path),
        extension_path=str(extension_path),
    )
    validate_result_binding(worker_runtime, binding_record)
    fixture = make_fixture(case)
    n_estimators = QUICK_ESTIMATORS if profile == "quick" else FULL_ESTIMATORS
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
    native_seconds = require_native_train_seconds(timing)
    return CaseResult(
        artifact_sha256=hashlib.sha256(bytes(estimator.artifact_bytes)).hexdigest(),
        prediction_sha256=hashlib.sha256(predictions.tobytes()).hexdigest(),
        native_seconds=native_seconds,
        rss_mib=_rss_delta_mib(rss_before, rss_after),
        rmse=rmse,
        case=case.name,
        shape=case.shape,
        tree_growth=case.tree_growth,
        repetition=args.repetition,
        source_commit=source_commit,
        extension_sha256=extension_sha256,
        runtime_name=args.runtime_name,
        fit_seconds=fit_seconds,
        dimensions={
            "n_rows": case.n_rows,
            "n_features": case.n_features,
            "n_eval_rows": case.n_eval_rows,
        },
        parameters=parameters,
        python_executable=str(_normalized_path(sys.executable)),
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
    try:
        return parse_worker_output(
            completed.stdout,
            runtime=runtime,
            expected_case=case.name,
            expected_repetition=repetition,
        )
    except ValueError as error:
        raise RuntimeError(
            f"{runtime.name} worker produced invalid output for {case.name}: {error}"
        ) from error


def _required_ratio(candidate: float, baseline: float) -> float:
    if (
        not math.isfinite(candidate)
        or not math.isfinite(baseline)
        or candidate <= 0.0
        or baseline <= 0.0
    ):
        raise ValueError("timing ratios require finite positive values")
    ratio = candidate / baseline
    if not math.isfinite(ratio):
        raise ValueError("timing ratio must be finite")
    return ratio


def _rss_ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    if (
        not math.isfinite(candidate)
        or not math.isfinite(baseline)
        or candidate <= 0.0
        or baseline <= 0.0
    ):
        return None
    ratio = candidate / baseline
    return ratio if math.isfinite(ratio) else None


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
        timing_ratio=_required_ratio(
            candidate.native_seconds, baseline.native_seconds
        ),
        rss_ratio=_rss_ratio(candidate.rss_mib, baseline.rss_mib),
    )


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("geometric mean requires at least one value")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _case_summaries(pairs: Sequence[PairedResult]) -> tuple[CaseSummary, ...]:
    grouped: dict[str, list[PairedResult]] = {}
    for pair in pairs:
        grouped.setdefault(pair.baseline.case, []).append(pair)
    summaries = []
    for case, group in sorted(grouped.items()):
        baseline_native = median(pair.baseline.native_seconds for pair in group)
        candidate_native = median(pair.candidate.native_seconds for pair in group)
        baseline_rss_values = [pair.baseline.rss_mib for pair in group]
        candidate_rss_values = [pair.candidate.rss_mib for pair in group]
        rss_available = all(
            value is not None
            for value in (*baseline_rss_values, *candidate_rss_values)
        )
        baseline_rss = (
            median(float(value) for value in baseline_rss_values)
            if rss_available
            else None
        )
        candidate_rss = (
            median(float(value) for value in candidate_rss_values)
            if rss_available
            else None
        )
        first = group[0].baseline
        summaries.append(
            CaseSummary(
                case=case,
                shape=first.shape,
                tree_growth=first.tree_growth,
                repetitions=len(group),
                baseline_native_seconds=baseline_native,
                candidate_native_seconds=candidate_native,
                timing_ratio=_required_ratio(candidate_native, baseline_native),
                baseline_rss_mib=baseline_rss,
                candidate_rss_mib=candidate_rss,
                rss_ratio=_rss_ratio(candidate_rss, baseline_rss),
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
    rss_ratios = [
        summary.rss_ratio
        for summary in summaries
        if summary.rss_ratio is not None
    ]
    aggregate_rss = _geometric_mean(rss_ratios) if rss_ratios else None
    performance_gated = profile == "full"
    if performance_gated:
        if not math.isfinite(aggregate_timing) or aggregate_timing > TIMING_SLOWDOWN_LIMIT:
            failures.append(
                "aggregate timing slowdown exceeds 3% "
                f"(ratio={aggregate_timing:.4f})"
            )
        if aggregate_rss is None:
            failures.append("aggregate RSS unavailable: no case has positive deltas")
        elif aggregate_rss > RSS_INCREASE_LIMIT:
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
            summary.timing_ratio < 1.0
            or (summary.rss_ratio is not None and summary.rss_ratio < 1.0)
            for summary in deep
        ):
            failures.append("no median timing or RSS improvement in any deep pressure case")
    return GateEvaluation(
        tuple(failures),
        aggregate_timing,
        aggregate_rss,
        len(rss_ratios),
        performance_gated,
        summaries,
    )


def validate_run_configuration(
    *,
    profile: str,
    baseline: RuntimeSpec,
    candidate: RuntimeSpec,
    repetitions: int,
) -> None:
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    if profile not in {"quick", "full"}:
        raise ValueError(f"unknown profile {profile!r}")
    baseline_attestation = loaded_runtime_attestation(baseline)
    candidate_attestation = loaded_runtime_attestation(candidate)
    python_same = _normalized_path(baseline.python) == _normalized_path(
        candidate.python
    )
    workdir_same = baseline.workdir.resolve() == candidate.workdir.resolve()
    commit_same = baseline.source_commit == candidate.source_commit
    if profile == "quick":
        if not (python_same and workdir_same and commit_same):
            raise ValueError("quick mode requires candidate self-comparison")
        if (
            baseline.manifest_path.resolve()
            != candidate.manifest_path.resolve()
            or baseline_attestation != candidate_attestation
        ):
            raise ValueError(
                "quick mode requires one shared candidate runtime manifest"
            )
        _validate_manifest_against_runtime(
            candidate_attestation,
            candidate,
            expected_name="candidate",
        )
        return
    if repetitions < 3:
        raise ValueError("full mode requires at least 3 repetitions")
    if python_same and workdir_same and commit_same:
        raise ValueError("full mode rejects self-comparison")
    if python_same or workdir_same or commit_same:
        raise ValueError(
            "full mode requires distinct Python executables, workdirs, and commits"
        )
    if baseline.manifest_path.resolve() == candidate.manifest_path.resolve():
        raise ValueError("full mode requires distinct runtime manifest files")
    _validate_manifest_against_runtime(
        baseline_attestation, baseline, expected_name="baseline"
    )
    _validate_manifest_against_runtime(
        candidate_attestation, candidate, expected_name="candidate"
    )
    same_package = Path(baseline_attestation.package_path).resolve() == Path(
        candidate_attestation.package_path
    ).resolve()
    same_extension = Path(baseline_attestation.extension_path).resolve() == Path(
        candidate_attestation.extension_path
    ).resolve()
    same_digest = (
        baseline_attestation.extension_sha256
        == candidate_attestation.extension_sha256
    )
    if same_package or same_extension or same_digest:
        raise ValueError("full mode manifests attest the same runtime")


def quick_baseline_runtime(candidate: RuntimeSpec) -> RuntimeSpec:
    return replace(candidate, name="baseline")


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
    for pair in pairs:
        validate_result_binding(baseline, pair.baseline)
        validate_result_binding(candidate, pair.candidate)


def build_report(
    *,
    profile: str,
    baseline: RuntimeSpec,
    candidate: RuntimeSpec,
    pairs: Sequence[PairedResult],
    repetitions: int,
    n_jobs: int,
) -> dict[str, Any]:
    gates = evaluate_gates(pairs, profile=profile)
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": profile,
        "repetitions": repetitions,
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
            "rss_cases_available": gates.rss_cases_available,
            "case_summaries": [asdict(summary) for summary in gates.case_summaries],
        },
    }


def run_benchmark(
    *,
    profile: str,
    baseline: RuntimeSpec,
    candidate: RuntimeSpec,
    repetitions: int,
    n_jobs: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    validate_run_configuration(
        profile=profile,
        baseline=baseline,
        candidate=candidate,
        repetitions=repetitions,
    )
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
        for repetition in range(repetitions):
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
        repetitions=repetitions,
        n_jobs=n_jobs,
    )


def _format_optional(value: float | None, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


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
    first_pair = report["pairs"][0]
    lines.append("")
    for arm in ("baseline", "candidate"):
        result = first_pair[arm]
        title = arm.capitalize()
        lines.extend(
            [
                f"- {title} package path: `{result['package_path']}`",
                f"- {title} extension path: `{result['extension_path']}`",
                f"- {title} extension SHA-256: `{result['extension_sha256']}`",
            ]
        )
    aggregate_rss = _format_optional(gate["aggregate_rss_ratio"], 4)
    lines.extend(
        [
            "",
            "## Aggregate Gate",
            "",
            f"- Aggregate timing ratio: {gate['aggregate_timing_ratio']:.4f}",
            f"- Aggregate RSS ratio: {aggregate_rss}",
            f"- RSS cases available: {gate['rss_cases_available']}",
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
        baseline_rss = _format_optional(summary["baseline_rss_mib"], 2)
        candidate_rss = _format_optional(summary["candidate_rss_mib"], 2)
        rss_ratio = _format_optional(summary["rss_ratio"], 4)
        lines.append(
            f"| `{summary['case']}` | {summary['repetitions']} | "
            f"{summary['baseline_native_seconds']:.6f} | "
            f"{summary['candidate_native_seconds']:.6f} | "
            f"{summary['timing_ratio']:.4f} | "
            f"{baseline_rss} | {candidate_rss} | {rss_ratio} |"
        )
    lines.extend(
        [
            "",
            "## Exact Equivalence",
            "",
            "| Case | Rep | Baseline RMSE | Candidate RMSE | Baseline Artifact SHA-256 | Candidate Artifact SHA-256 | Baseline Prediction SHA-256 | Candidate Prediction SHA-256 |",
            "| --- | ---: | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for pair in report["pairs"]:
        baseline_result = pair["baseline"]
        candidate_result = pair["candidate"]
        lines.append(
            f"| `{baseline_result['case']}` | {baseline_result['repetition']} | "
            f"{baseline_result['rmse']:.12g} | "
            f"{candidate_result['rmse']:.12g} | "
            f"`{baseline_result['artifact_sha256']}` | "
            f"`{candidate_result['artifact_sha256']}` | "
            f"`{baseline_result['prediction_sha256']}` | "
            f"`{candidate_result['prediction_sha256']}` |"
        )
    return "\n".join(lines) + "\n"


def create_runtime_manifest(runtime: RuntimeSpec) -> RuntimeManifest:
    if _normalized_path(sys.executable) != _normalized_path(runtime.python):
        raise ValueError(
            "runtime manifest must be written by the declared Python executable"
        )

    import alloygbm
    from alloygbm import _alloygbm

    package_path = Path(alloygbm.__file__).resolve(strict=True)
    extension_path = Path(_alloygbm.__file__).resolve(strict=True)
    return RuntimeManifest(
        schema_version=RUNTIME_MANIFEST_SCHEMA_VERSION,
        runtime_name=runtime.name,
        source_commit=runtime.source_commit,
        python_executable=str(_normalized_path(sys.executable)),
        package_path=str(package_path),
        extension_path=str(extension_path),
        extension_sha256=_sha256_file(extension_path),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--write-runtime-manifest", type=Path)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--baseline-python", type=Path)
    parser.add_argument("--baseline-workdir", type=Path)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--candidate-python", type=Path)
    parser.add_argument("--candidate-workdir", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--n-jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--worker-timeout", type=float, default=900.0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--runtime-name")
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--runtime-workdir", type=Path)
    parser.add_argument("--runtime-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expected-python", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expected-source-commit", help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--repetition", type=int, help=argparse.SUPPRESS)
    return parser


def _require_worker_args(args: argparse.Namespace) -> None:
    required = (
        "runtime_name",
        "runtime_workdir",
        "runtime_manifest",
        "expected_python",
        "expected_source_commit",
        "case",
        "repetition",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"worker missing arguments: {', '.join(missing)}")


def profile_from_args(args: argparse.Namespace) -> str:
    if args.quick:
        return "quick"
    if args.full:
        return "full"
    raise ValueError("an explicit --quick or --full profile is required")


def repetitions_from_args(args: argparse.Namespace) -> int:
    return (
        args.repetitions
        if args.repetitions is not None
        else profile_repetitions(profile_from_args(args))
    )


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.write_runtime_manifest is not None:
        if (
            args.runtime_name is None
            or args.runtime_python is None
            or args.runtime_workdir is None
        ):
            raise ValueError(
                "manifest mode requires --runtime-name, --runtime-python, "
                "and --runtime-workdir"
            )
        benchmark_only_values = (
            args.gate,
            args.baseline_python,
            args.baseline_workdir,
            args.baseline_manifest,
            args.candidate_python,
            args.candidate_workdir,
            args.candidate_manifest,
            args.repetitions,
            args.output_json,
            args.output_markdown,
        )
        if any(
            value is not None and value is not False
            for value in benchmark_only_values
        ):
            raise ValueError(
                "manifest mode does not accept benchmark runtime, gate, repetition, "
                "or output arguments"
            )
        return
    profile = profile_from_args(args)
    repetitions = repetitions_from_args(args)
    if repetitions < 1:
        raise ValueError("--repetitions must be at least 1")
    runtime_values = (
        args.baseline_python,
        args.baseline_workdir,
        args.baseline_manifest,
        args.candidate_python,
        args.candidate_workdir,
        args.candidate_manifest,
    )
    if profile == "full":
        if any(value is None for value in runtime_values):
            raise ValueError(
                "full mode requires explicit baseline/candidate Python executables "
                "workdirs, and manifests"
            )
        if repetitions < 3:
            raise ValueError("full mode requires at least 3 repetitions")
    else:
        if (
            args.baseline_python is not None
            or args.baseline_workdir is not None
            or args.baseline_manifest is not None
        ):
            raise ValueError("quick mode does not accept baseline runtime arguments")
        if args.candidate_manifest is None:
            raise ValueError("quick mode requires --candidate-manifest")
        if (args.candidate_python is None) != (args.candidate_workdir is None):
            raise ValueError(
                "quick mode candidate Python and workdir must be supplied together"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be at least 1")
    if args.worker:
        _require_worker_args(args)
        print(strict_json_dumps(_worker_result(args).to_dict(), sort_keys=True))
        return 0

    validate_cli_args(args)
    if args.write_runtime_manifest is not None:
        runtime = resolve_runtime(
            args.runtime_name,
            args.runtime_python,
            args.runtime_workdir,
        )
        manifest = create_runtime_manifest(runtime)
        write_runtime_manifest(args.write_runtime_manifest, manifest)
        print(args.write_runtime_manifest)
        return 0
    profile = profile_from_args(args)
    repetitions = repetitions_from_args(args)
    candidate_python = args.candidate_python or Path(sys.executable)
    candidate_workdir = args.candidate_workdir or REPO_ROOT
    candidate = resolve_runtime(
        "candidate",
        candidate_python,
        candidate_workdir,
        args.candidate_manifest,
    )
    if profile == "quick":
        baseline = quick_baseline_runtime(candidate)
    else:
        baseline = resolve_runtime(
            "baseline",
            args.baseline_python,
            args.baseline_workdir,
            args.baseline_manifest,
        )
    report = run_benchmark(
        profile=profile,
        baseline=baseline,
        candidate=candidate,
        repetitions=repetitions,
        n_jobs=args.n_jobs,
        timeout_seconds=args.worker_timeout,
    )
    markdown = render_markdown(report)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            strict_json_dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
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
