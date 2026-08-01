#!/usr/bin/env python3
"""Manifest-attested sampled-prediction-delta benchmark harness."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
_ALLOCATION_SUPPORT_PATH = SCRIPT_PATH.with_name("allocation_reuse_benchmark.py")


def _load_allocation_support():
    module_name = "_alloygbm_allocation_reuse_benchmark_support"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, _ALLOCATION_SUPPORT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"failed to load allocation benchmark support from {_ALLOCATION_SUPPORT_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_SUPPORT = _load_allocation_support()
RuntimeManifest = _SUPPORT.RuntimeManifest
RuntimeSpec = _SUPPORT.RuntimeSpec
WorkerInvocation = _SUPPORT.WorkerInvocation
RUNTIME_MANIFEST_SCHEMA_VERSION = _SUPPORT.RUNTIME_MANIFEST_SCHEMA_VERSION
strict_json_dumps = _SUPPORT.strict_json_dumps
strict_json_loads = _SUPPORT.strict_json_loads
write_runtime_manifest = _SUPPORT.write_runtime_manifest
load_runtime_manifest = _SUPPORT.load_runtime_manifest
create_runtime_manifest = _SUPPORT.create_runtime_manifest
resolve_runtime = _SUPPORT.resolve_runtime
quick_baseline_runtime = _SUPPORT.quick_baseline_runtime
require_runtime_attestation = _SUPPORT.require_runtime_attestation
require_native_train_seconds = _SUPPORT.require_native_train_seconds
validate_result_binding = _SUPPORT.validate_result_binding
_git_commit = _SUPPORT._git_commit
_require_clean_worktree = _SUPPORT._require_clean_worktree
_normalized_path = _SUPPORT._normalized_path
_peak_rss_mib = _SUPPORT._peak_rss_mib
_rss_delta_mib = _SUPPORT._rss_delta_mib
_sha256_file = _SUPPORT._sha256_file


SCHEMA_VERSION = 1
QUICK_REPETITIONS = 1
FULL_REPETITIONS = 5
QUICK_ESTIMATORS = 3
FULL_ESTIMATORS = 24
MISSING_RATE = 0.03

DELTA_SENSITIVE_TIME_RATIO_LIMIT = 0.98
ALL_ELIGIBLE_TIME_RATIO_LIMIT = 1.03
PER_CASE_TIME_RATIO_LIMIT = 1.08
AGGREGATE_RSS_RATIO_LIMIT = 1.05
NATIVE_TIME_NOISE_FLOOR_SECONDS = 0.05

QUICK_CASE_NAMES = {
    "scalar_tall_narrow_level_subsample_050",
    "multiclass_tall_narrow_level_subsample_050",
    "fallback_scalar_dart_subsample_050",
    "fallback_scalar_quantile_subsample_050",
}
FULL_CASE_NAMES = {
    "scalar_tall_narrow_level_full",
    "scalar_tall_narrow_level_subsample_080",
    "scalar_tall_narrow_level_subsample_050",
    "scalar_shallow_tall_leaf_subsample_050",
    "scalar_medium_wide_level_goss",
    "scalar_small_wide_leaf_subsample_050",
    "multiclass_tall_narrow_level_subsample_050",
    "multiclass_medium_wide_leaf_goss",
    "fallback_scalar_dart_subsample_050",
    "fallback_scalar_quantile_subsample_050",
}


@dataclass(frozen=True)
class CaseSpec:
    name: str
    shape: str
    task: str
    growth: str
    sampling: str
    n_rows: int
    n_features: int
    n_eval_rows: int
    max_depth: int
    seed: int
    row_subsample: float = 1.0
    boosting_mode: str = "standard"
    objective: str = "squared_error"
    fallback_sentinel: str | None = None
    delta_sensitive: bool = True
    performance_eligible: bool = True


@dataclass(frozen=True)
class Fixture:
    X_train: np.ndarray
    y_train: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray

    def arrays(self) -> tuple[np.ndarray, ...]:
        return self.X_train, self.y_train, self.X_eval, self.y_eval


@dataclass(frozen=True)
class CaseResult:
    case: str
    shape: str
    task: str
    growth: str
    sampling: str
    repetition: int
    runtime_name: str
    source_commit: str
    python_executable: str
    package_path: str
    extension_path: str
    extension_sha256: str
    artifact_sha256: str
    prediction_sha256: str
    native_seconds: float
    rss_mib: float | None
    completed_rounds: int
    stop_reason: str
    quality_metric: str
    quality_value: float
    fallback_sentinel: str | None
    dimensions: dict[str, int] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CaseResult:
        expected = {
            "case",
            "shape",
            "task",
            "growth",
            "sampling",
            "repetition",
            "runtime_name",
            "source_commit",
            "python_executable",
            "package_path",
            "extension_path",
            "extension_sha256",
            "artifact_sha256",
            "prediction_sha256",
            "native_seconds",
            "rss_mib",
            "completed_rounds",
            "stop_reason",
            "quality_metric",
            "quality_value",
            "fallback_sentinel",
            "dimensions",
            "parameters",
        }
        if set(payload) != expected:
            raise ValueError(
                "worker record fields must exactly match the schema: "
                f"expected={sorted(expected)}, actual={sorted(payload)}"
            )
        string_fields = {
            "case",
            "shape",
            "task",
            "growth",
            "sampling",
            "runtime_name",
            "source_commit",
            "python_executable",
            "package_path",
            "extension_path",
            "extension_sha256",
            "artifact_sha256",
            "prediction_sha256",
            "stop_reason",
            "quality_metric",
        }
        if any(not isinstance(payload[name], str) for name in string_fields):
            raise ValueError("worker record string fields have invalid types")
        if type(payload["repetition"]) is not int:
            raise ValueError("worker repetition must be an integer")
        if type(payload["completed_rounds"]) is not int:
            raise ValueError("worker completed_rounds must be an integer")
        if payload["fallback_sentinel"] is not None and not isinstance(
            payload["fallback_sentinel"], str
        ):
            raise ValueError("worker fallback_sentinel must be a string or null")
        if not isinstance(payload["dimensions"], dict) or not isinstance(
            payload["parameters"], dict
        ):
            raise ValueError("worker dimensions and parameters must be objects")
        for name in ("native_seconds", "quality_value"):
            value = payload[name]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"worker {name} must be numeric")
        rss = payload["rss_mib"]
        if rss is not None and (
            not isinstance(rss, (int, float)) or isinstance(rss, bool)
        ):
            raise ValueError("worker rss_mib must be numeric or null")
        if any(
            type(key) is not str or type(value) is not int or value < 0
            for key, value in payload["dimensions"].items()
        ):
            raise ValueError("worker dimensions must contain non-negative integers")
        dimensions = dict(payload["dimensions"])
        return cls(
            case=payload["case"],
            shape=payload["shape"],
            task=payload["task"],
            growth=payload["growth"],
            sampling=payload["sampling"],
            repetition=payload["repetition"],
            runtime_name=payload["runtime_name"],
            source_commit=payload["source_commit"],
            python_executable=payload["python_executable"],
            package_path=payload["package_path"],
            extension_path=payload["extension_path"],
            extension_sha256=payload["extension_sha256"],
            artifact_sha256=payload["artifact_sha256"],
            prediction_sha256=payload["prediction_sha256"],
            native_seconds=float(payload["native_seconds"]),
            rss_mib=None if rss is None else float(rss),
            completed_rounds=payload["completed_rounds"],
            stop_reason=payload["stop_reason"],
            quality_metric=payload["quality_metric"],
            quality_value=float(payload["quality_value"]),
            fallback_sentinel=payload["fallback_sentinel"],
            dimensions=dimensions,
            parameters=dict(payload["parameters"]),
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
    time_ratio: float
    rss_ratio: float | None


@dataclass(frozen=True)
class CaseSummary:
    case: str
    shape: str
    task: str
    growth: str
    sampling: str
    fallback_sentinel: str | None
    delta_sensitive: bool
    performance_eligible: bool
    repetitions: int
    baseline_native_seconds: float
    candidate_native_seconds: float
    time_ratio: float
    baseline_rss_mib: float | None
    candidate_rss_mib: float | None
    rss_ratio: float | None
    noise_floor_waived: bool


@dataclass(frozen=True)
class GateEvaluation:
    failures: tuple[str, ...]
    performance_gated: bool
    delta_sensitive_time_ratio: float
    all_eligible_time_ratio: float
    aggregate_rss_ratio: float | None
    case_summaries: tuple[CaseSummary, ...]


def _case(
    name: str,
    shape: str,
    task: str,
    growth: str,
    sampling: str,
    dimensions: tuple[int, int, int],
    depth: int,
    seed: int,
    **changes: Any,
) -> CaseSpec:
    return CaseSpec(
        name,
        shape,
        task,
        growth,
        sampling,
        *dimensions,
        depth,
        seed,
        **changes,
    )


_FULL_CASES = (
    _case(
        "scalar_tall_narrow_level_full", "tall_narrow", "scalar", "level",
        "full", (65_536, 12, 4_096), 6, 5101, delta_sensitive=False,
    ),
    _case(
        "scalar_tall_narrow_level_subsample_080", "tall_narrow", "scalar",
        "level", "subsample_080", (65_536, 12, 4_096), 6, 5102,
        row_subsample=0.8,
    ),
    _case(
        "scalar_tall_narrow_level_subsample_050", "tall_narrow", "scalar",
        "level", "subsample_050", (65_536, 12, 4_096), 6, 5103,
        row_subsample=0.5,
    ),
    _case(
        "scalar_shallow_tall_leaf_subsample_050", "shallow_tall", "scalar",
        "leaf", "subsample_050", (131_072, 8, 4_096), 3, 5201,
        row_subsample=0.5,
    ),
    _case(
        "scalar_medium_wide_level_goss", "medium_wide", "scalar", "level",
        "goss", (16_384, 96, 2_048), 6, 5301, boosting_mode="goss",
    ),
    _case(
        "scalar_small_wide_leaf_subsample_050", "small_wide", "scalar", "leaf",
        "subsample_050", (4_096, 256, 1_024), 5, 5401, row_subsample=0.5,
        delta_sensitive=False,
    ),
    _case(
        "multiclass_tall_narrow_level_subsample_050", "tall_narrow",
        "multiclass", "level", "subsample_050", (49_152, 12, 4_096), 5,
        5501, row_subsample=0.5,
    ),
    _case(
        "multiclass_medium_wide_leaf_goss", "medium_wide", "multiclass", "leaf",
        "goss", (12_288, 96, 2_048), 5, 5601, boosting_mode="goss",
    ),
    _case(
        "fallback_scalar_dart_subsample_050", "tall_narrow", "scalar", "level",
        "subsample_050", (49_152, 12, 4_096), 5, 5701, row_subsample=0.5,
        boosting_mode="dart", fallback_sentinel="dart_full_replay",
        delta_sensitive=False, performance_eligible=False,
    ),
    _case(
        "fallback_scalar_quantile_subsample_050", "tall_narrow", "scalar",
        "level", "subsample_050", (49_152, 12, 4_096), 5, 5801,
        row_subsample=0.5, objective="quantile",
        fallback_sentinel="quantile_full_replay", delta_sensitive=False,
        performance_eligible=False,
    ),
)


def full_cases() -> tuple[CaseSpec, ...]:
    return _FULL_CASES


def quick_cases() -> tuple[CaseSpec, ...]:
    quick_dimensions = {
        "scalar_tall_narrow_level_subsample_050": (1_024, 12, 256, 3),
        "multiclass_tall_narrow_level_subsample_050": (1_200, 12, 256, 3),
        "fallback_scalar_dart_subsample_050": (1_024, 12, 256, 3),
        "fallback_scalar_quantile_subsample_050": (1_024, 12, 256, 3),
    }
    return tuple(
        replace(
            case,
            n_rows=quick_dimensions[case.name][0],
            n_features=quick_dimensions[case.name][1],
            n_eval_rows=quick_dimensions[case.name][2],
            max_depth=quick_dimensions[case.name][3],
        )
        for case in _FULL_CASES
        if case.name in QUICK_CASE_NAMES
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


def make_fixture(case: CaseSpec) -> Fixture:
    coefficient_seed, data_seed = np.random.SeedSequence(case.seed).spawn(2)
    coefficient_rng = np.random.default_rng(coefficient_seed)
    data_rng = np.random.default_rng(data_seed)
    total_rows = case.n_rows + case.n_eval_rows
    X = data_rng.normal(size=(total_rows, case.n_features)).astype(np.float32)
    if case.task == "scalar":
        coefficients = coefficient_rng.normal(size=case.n_features).astype(np.float32)
        coefficients /= max(float(np.linalg.norm(coefficients)), 1e-6)
        target = X @ coefficients
        target += 0.7 * np.sin(X[:, 0] * 1.5)
        target += 0.3 * X[:, 1] * X[:, 2]
        target += data_rng.normal(0.0, 0.1, size=total_rows)
        y: np.ndarray = np.ascontiguousarray(target, dtype=np.float32)
    elif case.task == "multiclass":
        coefficients = coefficient_rng.normal(size=(case.n_features, 3)).astype(
            np.float32
        )
        logits = X @ coefficients
        logits[:, 0] += 0.6 * np.sin(X[:, 0] * 1.3)
        logits[:, 1] += 0.35 * X[:, 1] * X[:, 2]
        logits[:, 2] -= 0.25 * np.square(X[:, 3])
        logits += data_rng.normal(0.0, 0.15, size=logits.shape)
        y = np.ascontiguousarray(np.argmax(logits, axis=1), dtype=np.int64)
        y[:3] = np.arange(3, dtype=np.int64)
    else:
        raise ValueError(f"unknown task {case.task!r}")
    missing = data_rng.random(X.shape) < MISSING_RATE
    missing[0, 0] = True
    missing[case.n_rows, 0] = True
    X[missing] = np.nan
    return Fixture(
        np.ascontiguousarray(X[: case.n_rows], dtype=np.float32),
        np.ascontiguousarray(y[: case.n_rows]),
        np.ascontiguousarray(X[case.n_rows :], dtype=np.float32),
        np.ascontiguousarray(y[case.n_rows :]),
    )


def _runtime_from_manifest(name: str, manifest_path: Path) -> RuntimeSpec:
    manifest_path = manifest_path.expanduser().resolve()
    try:
        payload = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"runtime manifest is not strict JSON: {manifest_path}") from error
    if not isinstance(payload, dict):
        raise ValueError("runtime manifest must contain a JSON object")
    manifest = RuntimeManifest.from_dict(payload)
    if manifest.runtime_name != name:
        raise ValueError(
            f"runtime manifest name mismatch: expected {name}, got {manifest.runtime_name}"
        )
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(Path(manifest.package_path).resolve().parent),
                "rev-parse",
                "--show-toplevel",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            f"cannot derive runtime worktree from manifest package path: {manifest.package_path}"
        ) from error
    return resolve_runtime(
        name,
        Path(manifest.python_executable),
        Path(completed.stdout.strip()),
        manifest_path,
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
    profile_args = ("--quick",) if profile == "quick" else ()
    command = (
        str(runtime.python),
        "-I",
        str(SCRIPT_PATH),
        *profile_args,
        "--worker",
        "--runtime-name",
        runtime.name,
        "--runtime-workdir",
        str(runtime.workdir),
        "--runtime-manifest",
        str(runtime.manifest_path),
        "--expected-python",
        str(runtime.python),
        "--expected-source-commit",
        runtime.source_commit,
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


def _valid_hex(value: str, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


def _same_typed_mapping(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    return set(actual) == set(expected) and all(
        type(actual[key]) is type(expected[key]) and actual[key] == expected[key]
        for key in expected
    )


def _validate_case_metadata(
    result: CaseResult, *, profile: str, n_jobs: int
) -> None:
    cases = {case.name: case for case in profile_cases(profile)}
    try:
        case = cases[result.case]
    except KeyError as error:
        raise ValueError(f"worker case metadata names unknown case {result.case!r}")
    expected_metric = "rmse" if case.task == "scalar" else "log_loss"
    metadata_matches = (
        (
            result.shape,
            result.task,
            result.growth,
            result.sampling,
            result.fallback_sentinel,
            result.quality_metric,
            result.dimensions,
        )
        == (
            case.shape,
            case.task,
            case.growth,
            case.sampling,
            case.fallback_sentinel,
            expected_metric,
            {
                "n_rows": case.n_rows,
                "n_features": case.n_features,
                "n_eval_rows": case.n_eval_rows,
            },
        )
        and _same_typed_mapping(
            result.parameters,
            _estimator_parameters(case, profile=profile, n_jobs=n_jobs),
        )
    )
    if not metadata_matches:
        raise ValueError(f"worker case metadata does not match catalog for {result.case}")


def parse_worker_output(
    stdout: str,
    *,
    runtime: RuntimeSpec,
    expected_case: str,
    expected_repetition: int,
    expected_profile: str,
    expected_n_jobs: int,
) -> CaseResult:
    try:
        payload = strict_json_loads(stdout)
    except (ValueError, TypeError) as error:
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
    _validate_case_metadata(
        result, profile=expected_profile, n_jobs=expected_n_jobs
    )
    if result.source_commit != runtime.source_commit:
        raise ValueError(
            "worker source commit mismatch: "
            f"expected {runtime.source_commit}, got {result.source_commit}"
        )
    if not math.isfinite(result.native_seconds) or result.native_seconds <= 0.0:
        raise ValueError("worker native_seconds must be finite and positive")
    if not _finite_nonnegative(result.quality_value):
        raise ValueError("worker quality_value must be finite and non-negative")
    if result.rss_mib is not None and not _finite_nonnegative(result.rss_mib):
        raise ValueError("worker rss_mib must be null or finite and non-negative")
    if result.completed_rounds < 0:
        raise ValueError("worker completed_rounds must be non-negative")
    if not result.stop_reason:
        raise ValueError("worker stop_reason must be non-empty")
    for name, digest in (
        ("artifact", result.artifact_sha256),
        ("prediction", result.prediction_sha256),
        ("extension", result.extension_sha256),
    ):
        if not _valid_hex(digest, 64):
            raise ValueError(f"worker {name} SHA-256 must contain 64 lowercase hex characters")
    validate_result_binding(runtime, result)
    return result


def _estimator_parameters(case: CaseSpec, *, profile: str, n_jobs: int) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "n_estimators": QUICK_ESTIMATORS if profile == "quick" else FULL_ESTIMATORS,
        "learning_rate": 0.08,
        "max_depth": case.max_depth,
        "min_data_in_leaf": 8,
        "min_split_gain": 0.0,
        "lambda_l2": 1.0,
        "training_policy": "manual",
        "continuous_binning_max_bins": 64,
        "tree_growth": case.growth,
        "row_subsample": case.row_subsample,
        "boosting_mode": case.boosting_mode,
        "seed": case.seed,
        "deterministic": True,
        "n_jobs": n_jobs,
    }
    if case.growth == "leaf":
        parameters["max_leaves"] = min(64, 2**case.max_depth)
    if case.boosting_mode == "goss":
        parameters.update(goss_top_rate=0.2, goss_other_rate=0.1)
    if case.boosting_mode == "dart":
        parameters.update(dart_drop_rate=0.1, dart_max_drop=5)
    if case.objective == "quantile":
        parameters.update(objective="quantile", quantile_alpha=0.5)
    return parameters


def require_completion_diagnostics(estimator: Any) -> tuple[int, str]:
    rounds = getattr(estimator, "rounds_completed_", None)
    stop_reason = getattr(estimator, "stop_reason_", None)
    if (
        type(rounds) is not int
        or rounds < 0
        or not isinstance(stop_reason, str)
        or not stop_reason
    ):
        raise ValueError(
            "fitted estimator must expose valid native completion diagnostics"
        )
    return rounds, stop_reason


def _worker_result(args: argparse.Namespace) -> CaseResult:
    profile = profile_from_args(args)
    _require_clean_worktree(args.runtime_workdir)
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

    live_python = _normalized_path(sys.executable)
    declared_python = _normalized_path(args.expected_python)
    if live_python != declared_python:
        raise ValueError(
            "live Python interpreter does not match declared runtime: "
            f"live={live_python}, declared={declared_python}"
        )
    worker_runtime = RuntimeSpec(
        args.runtime_name,
        live_python,
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
            expected_name=("candidate" if profile == "quick" else args.runtime_name),
        ),
    )

    # AlloyGBM is imported only after the worker has validated its attestation.
    import alloygbm
    from alloygbm import GBMClassifier, GBMRegressor, _alloygbm

    package_path = Path(alloygbm.__file__).resolve()
    extension_path = Path(_alloygbm.__file__).resolve()
    extension_sha256 = _sha256_file(extension_path)
    binding_record = _SUPPORT.CaseResult(
        artifact_sha256="binding-only",
        prediction_sha256="binding-only",
        native_seconds=1.0,
        rss_mib=None,
        source_commit=source_commit,
        extension_sha256=extension_sha256,
        runtime_name=args.runtime_name,
        python_executable=str(live_python),
        package_path=str(package_path),
        extension_path=str(extension_path),
    )
    validate_result_binding(worker_runtime, binding_record)
    fixture = make_fixture(case)
    parameters = _estimator_parameters(case, profile=profile, n_jobs=args.n_jobs)
    estimator = (
        GBMRegressor(**parameters)
        if case.task == "scalar"
        else GBMClassifier(**parameters)
    )
    rss_before = _peak_rss_mib()
    estimator.fit(fixture.X_train, fixture.y_train)
    rss_after = _peak_rss_mib()
    if case.task == "scalar":
        predictions = np.ascontiguousarray(
            estimator.predict(fixture.X_eval), dtype="<f4"
        )
        residuals = predictions.astype(np.float64) - fixture.y_eval.astype(np.float64)
        quality_metric = "rmse"
        quality_value = float(np.sqrt(np.mean(np.square(residuals))))
    else:
        predictions = np.ascontiguousarray(
            estimator.predict_proba(fixture.X_eval), dtype="<f4"
        )
        labels = fixture.y_eval.astype(np.int64)
        true_probabilities = predictions[np.arange(len(labels)), labels].astype(np.float64)
        quality_metric = "log_loss"
        quality_value = float(-np.mean(np.log(np.clip(true_probabilities, 1e-15, 1.0))))
    timing = dict(getattr(estimator, "fit_timing_", None) or {})
    native_seconds = require_native_train_seconds(timing)
    completed_rounds, stop_reason = require_completion_diagnostics(estimator)
    result = CaseResult(
        case=case.name,
        shape=case.shape,
        task=case.task,
        growth=case.growth,
        sampling=case.sampling,
        repetition=args.repetition,
        runtime_name=args.runtime_name,
        source_commit=source_commit,
        python_executable=str(_normalized_path(sys.executable)),
        package_path=str(package_path),
        extension_path=str(extension_path),
        extension_sha256=extension_sha256,
        artifact_sha256=hashlib.sha256(bytes(estimator.artifact_bytes)).hexdigest(),
        prediction_sha256=hashlib.sha256(predictions.tobytes()).hexdigest(),
        native_seconds=native_seconds,
        rss_mib=_rss_delta_mib(rss_before, rss_after),
        completed_rounds=completed_rounds,
        stop_reason=stop_reason,
        quality_metric=quality_metric,
        quality_value=quality_value,
        fallback_sentinel=case.fallback_sentinel,
        dimensions={
            "n_rows": case.n_rows,
            "n_features": case.n_features,
            "n_eval_rows": case.n_eval_rows,
        },
        parameters=parameters,
    )
    validate_result_binding(worker_runtime, result)
    return result


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
        raise RuntimeError(f"{runtime.name} worker timed out for {case.name}") from error
    try:
        return parse_worker_output(
            completed.stdout,
            runtime=runtime,
            expected_case=case.name,
            expected_repetition=repetition,
            expected_profile=profile,
            expected_n_jobs=n_jobs,
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
        raise ValueError("ratios require finite positive values")
    ratio = candidate / baseline
    if not math.isfinite(ratio):
        raise ValueError("ratio must be finite")
    return ratio


def _rss_ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    try:
        return _required_ratio(candidate, baseline)
    except ValueError:
        return None


def evaluate_pair(baseline: CaseResult, candidate: CaseResult) -> PairEvaluation:
    failures: list[str] = []
    identity_fields = (
        "case",
        "shape",
        "task",
        "growth",
        "sampling",
        "repetition",
        "fallback_sentinel",
        "dimensions",
        "parameters",
    )
    for name in identity_fields:
        if getattr(baseline, name) != getattr(candidate, name):
            failures.append(f"{name} differs")
    if baseline.artifact_sha256 != candidate.artifact_sha256:
        failures.append("artifact SHA-256 differs")
    if baseline.prediction_sha256 != candidate.prediction_sha256:
        failures.append("prediction SHA-256 differs")
    if baseline.completed_rounds != candidate.completed_rounds:
        failures.append("completed rounds differ")
    if baseline.stop_reason != candidate.stop_reason:
        failures.append("stop reason differs")
    if baseline.quality_metric != candidate.quality_metric:
        failures.append("quality metric differs")
    if not math.isfinite(baseline.quality_value) or not math.isfinite(
        candidate.quality_value
    ):
        failures.append("quality metric is non-finite")
    elif baseline.quality_value.hex() != candidate.quality_value.hex():
        failures.append("quality value differs")
    return PairEvaluation(
        equivalent=not failures,
        failures=tuple(failures),
        time_ratio=_required_ratio(candidate.native_seconds, baseline.native_seconds),
        rss_ratio=_rss_ratio(candidate.rss_mib, baseline.rss_mib),
    )


def _geometric_mean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _case_summaries(
    pairs: Sequence[PairedResult], *, profile: str
) -> tuple[CaseSummary, ...]:
    catalog = {case.name: case for case in profile_cases(profile)}
    grouped: dict[str, list[PairedResult]] = {}
    for pair in pairs:
        if pair.baseline.case in catalog:
            grouped.setdefault(pair.baseline.case, []).append(pair)
    summaries: list[CaseSummary] = []
    for case_name, group in sorted(grouped.items()):
        case = catalog[case_name]
        baseline_native = median(pair.baseline.native_seconds for pair in group)
        candidate_native = median(pair.candidate.native_seconds for pair in group)
        baseline_rss_values = [pair.baseline.rss_mib for pair in group]
        candidate_rss_values = [pair.candidate.rss_mib for pair in group]
        rss_available = all(
            value is not None and math.isfinite(value) and value > 0.0
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
        summaries.append(
            CaseSummary(
                case=case_name,
                shape=case.shape,
                task=case.task,
                growth=case.growth,
                sampling=case.sampling,
                fallback_sentinel=case.fallback_sentinel,
                delta_sensitive=case.delta_sensitive,
                performance_eligible=case.performance_eligible,
                repetitions=len(group),
                baseline_native_seconds=baseline_native,
                candidate_native_seconds=candidate_native,
                time_ratio=_required_ratio(candidate_native, baseline_native),
                baseline_rss_mib=baseline_rss,
                candidate_rss_mib=candidate_rss,
                rss_ratio=_rss_ratio(candidate_rss, baseline_rss),
                noise_floor_waived=(
                    case.performance_eligible
                    and baseline_native < NATIVE_TIME_NOISE_FLOOR_SECONDS
                ),
            )
        )
    return tuple(summaries)


def _full_matrix_failures(pairs: Sequence[PairedResult]) -> tuple[str, ...]:
    expected = {
        (case.name, repetition)
        for case in full_cases()
        for repetition in range(FULL_REPETITIONS)
    }
    observed = [(pair.baseline.case, pair.baseline.repetition) for pair in pairs]
    counts: dict[tuple[str, int], int] = {}
    for identity in observed:
        counts[identity] = counts.get(identity, 0) + 1
    observed_set = set(counts)
    duplicates = sorted(
        identity for identity, count in counts.items() if count != 1
    )
    missing = sorted(expected - observed_set)
    unexpected = sorted(observed_set - expected)
    if len(observed) == len(expected) and not duplicates and not missing and not unexpected:
        return ()
    return (
        "full mode requires the exact 10-case x 5-repetition matrix "
        f"(records={len(observed)}, missing={missing}, unexpected={unexpected}, "
        f"duplicates={duplicates})",
    )


def evaluate_gates(pairs: Sequence[PairedResult], *, profile: str) -> GateEvaluation:
    if profile not in {"quick", "full"}:
        raise ValueError(f"unknown profile {profile!r}")
    if not pairs:
        raise ValueError("benchmark produced no paired results")
    failures: list[str] = []
    if profile == "full":
        failures.extend(_full_matrix_failures(pairs))
    for pair in pairs:
        evaluation = evaluate_pair(pair.baseline, pair.candidate)
        failures.extend(
            f"{pair.baseline.case}/rep-{pair.baseline.repetition}: {failure}"
            for failure in evaluation.failures
        )
    summaries = _case_summaries(pairs, profile=profile)
    delta_ratios = [
        summary.time_ratio
        for summary in summaries
        if summary.performance_eligible and summary.delta_sensitive
    ]
    eligible_ratios = [
        summary.time_ratio for summary in summaries if summary.performance_eligible
    ]
    delta_aggregate = _geometric_mean(delta_ratios) if delta_ratios else 1.0
    eligible_aggregate = _geometric_mean(eligible_ratios) if eligible_ratios else 1.0
    rss_ratios = [
        summary.rss_ratio for summary in summaries if summary.rss_ratio is not None
    ]
    aggregate_rss = _geometric_mean(rss_ratios) if rss_ratios else None
    performance_gated = profile == "full"
    if performance_gated:
        if delta_aggregate > DELTA_SENSITIVE_TIME_RATIO_LIMIT:
            failures.append(
                "delta-sensitive aggregate native-time ratio exceeds 0.98 "
                f"(ratio={delta_aggregate:.4f})"
            )
        if eligible_aggregate > ALL_ELIGIBLE_TIME_RATIO_LIMIT:
            failures.append(
                "all-eligible aggregate native-time ratio exceeds 1.03 "
                f"(ratio={eligible_aggregate:.4f})"
            )
        for summary in summaries:
            if (
                summary.performance_eligible
                and not summary.noise_floor_waived
                and summary.time_ratio > PER_CASE_TIME_RATIO_LIMIT
            ):
                failures.append(
                    f"{summary.case}: per-case native-time ratio exceeds 1.08 "
                    f"(ratio={summary.time_ratio:.4f})"
                )
        for pair in pairs:
            for arm, result in (("baseline", pair.baseline), ("candidate", pair.candidate)):
                if (
                    result.rss_mib is None
                    or not math.isfinite(result.rss_mib)
                    or result.rss_mib <= 0.0
                ):
                    failures.append(
                        f"{result.case}/rep-{result.repetition}/{arm}: "
                        "full mode requires measurable positive RSS"
                    )
        if aggregate_rss is None:
            failures.append("aggregate RSS ratio is unavailable")
        elif aggregate_rss > AGGREGATE_RSS_RATIO_LIMIT:
            failures.append(
                "aggregate RSS ratio exceeds 1.05 "
                f"(ratio={aggregate_rss:.4f})"
            )
    return GateEvaluation(
        failures=tuple(failures),
        performance_gated=performance_gated,
        delta_sensitive_time_ratio=delta_aggregate,
        all_eligible_time_ratio=eligible_aggregate,
        aggregate_rss_ratio=aggregate_rss,
        case_summaries=summaries,
    )


def validate_run_configuration(
    *,
    profile: str,
    baseline: RuntimeSpec,
    candidate: RuntimeSpec,
    repetitions: int,
) -> None:
    if profile == "full" and repetitions != FULL_REPETITIONS:
        raise ValueError("full mode requires exactly 5 repetitions")
    _SUPPORT.validate_run_configuration(
        profile=profile,
        baseline=baseline,
        candidate=candidate,
        repetitions=repetitions,
    )


def validate_runtime_pair(
    baseline: RuntimeSpec,
    candidate: RuntimeSpec,
    pairs: Sequence[PairedResult],
) -> None:
    _SUPPORT.validate_runtime_pair(baseline, candidate, pairs)


def build_report(
    *,
    profile: str,
    baseline: RuntimeSpec,
    candidate: RuntimeSpec,
    pairs: Sequence[PairedResult],
    repetitions: int,
    n_jobs: int,
) -> dict[str, Any]:
    if profile == "full" and repetitions != FULL_REPETITIONS:
        raise ValueError("full mode requires exactly 5 repetitions")
    gates = evaluate_gates(pairs, profile=profile)
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": profile,
        "repetitions": repetitions,
        "warmup_runs_per_case_per_arm": 1,
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
            "delta_sensitive_time_ratio": gates.delta_sensitive_time_ratio,
            "all_eligible_time_ratio": gates.all_eligible_time_ratio,
            "aggregate_rss_ratio": gates.aggregate_rss_ratio,
            "limits": {
                "delta_sensitive_time_ratio": DELTA_SENSITIVE_TIME_RATIO_LIMIT,
                "all_eligible_time_ratio": ALL_ELIGIBLE_TIME_RATIO_LIMIT,
                "per_case_time_ratio": PER_CASE_TIME_RATIO_LIMIT,
                "aggregate_rss_ratio": AGGREGATE_RSS_RATIO_LIMIT,
                "native_time_noise_floor_seconds": NATIVE_TIME_NOISE_FLOOR_SECONDS,
            },
            "fallback_timing_exclusions": [
                case.name
                for case in profile_cases(profile)
                if case.fallback_sentinel is not None
            ],
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
    pairs: list[PairedResult] = []
    for case_index, case in enumerate(profile_cases(profile)):
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
    gate = report["gate"]
    lines = [
        "# Sampled Prediction Delta Benchmark",
        "",
        "## Runtime Identity",
        "",
        "| Arm | Source commit | Python | Workdir |",
        "| --- | --- | --- | --- |",
    ]
    for arm in ("baseline", "candidate"):
        runtime = report["runtimes"][arm]
        lines.append(
            f"| {arm} | `{runtime['source_commit']}` | `{runtime['python']}` | "
            f"`{runtime['workdir']}` |"
        )
    lines.extend(
        [
            "",
            "## Gates",
            "",
            f"- Performance gated: {gate['performance_gated']}",
            f"- Delta-sensitive native-time ratio: {gate['delta_sensitive_time_ratio']:.4f}",
            f"- All-eligible native-time ratio: {gate['all_eligible_time_ratio']:.4f}",
            f"- Aggregate RSS ratio: {_format_optional(gate['aggregate_rss_ratio'], 4)}",
            f"- Failures: {len(gate['failures'])}",
            "- DART and quantile timing are descriptive fallback sentinels.",
        ]
    )
    lines.extend(f"- FAIL: {failure}" for failure in gate["failures"])
    lines.extend(
        [
            "",
            "Quick mode is a candidate self-comparison and proves harness consistency, not speed.",
            "Every arm receives one unmeasured warmup subprocess per case.",
            "",
            "## Case Medians",
            "",
            "| Case | Reps | Baseline native s | Candidate native s | Ratio | Baseline RSS MiB | Candidate RSS MiB | Ratio | Fallback |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for summary in gate["case_summaries"]:
        lines.append(
            f"| `{summary['case']}` | {summary['repetitions']} | "
            f"{summary['baseline_native_seconds']:.6f} | "
            f"{summary['candidate_native_seconds']:.6f} | "
            f"{summary['time_ratio']:.4f} | "
            f"{_format_optional(summary['baseline_rss_mib'], 2)} | "
            f"{_format_optional(summary['candidate_rss_mib'], 2)} | "
            f"{_format_optional(summary['rss_ratio'], 4)} | "
            f"{summary['fallback_sentinel'] or ''} |"
        )
    lines.extend(
        [
            "",
            "## Exact Equivalence",
            "",
            "| Case | Rep | Metric | Baseline quality | Candidate quality | Artifact SHA-256 | Prediction SHA-256 | Rounds | Stop reason |",
            "| --- | ---: | --- | ---: | ---: | --- | --- | ---: | --- |",
        ]
    )
    for pair in report["pairs"]:
        baseline = pair["baseline"]
        candidate = pair["candidate"]
        artifact = (
            baseline["artifact_sha256"]
            if baseline["artifact_sha256"] == candidate["artifact_sha256"]
            else "MISMATCH"
        )
        prediction = (
            baseline["prediction_sha256"]
            if baseline["prediction_sha256"] == candidate["prediction_sha256"]
            else "MISMATCH"
        )
        lines.append(
            f"| `{baseline['case']}` | {baseline['repetition']} | "
            f"{baseline['quality_metric']} | {baseline['quality_value']:.12g} | "
            f"{candidate['quality_value']:.12g} | `{artifact}` | `{prediction}` | "
            f"{baseline['completed_rounds']} | {baseline['stop_reason']} |"
        )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--write-runtime-manifest", type=Path)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--baseline-manifest", type=Path)
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
    return "quick" if args.quick else "full"


def repetitions_from_args(args: argparse.Namespace) -> int:
    if args.repetitions is not None:
        return args.repetitions
    return profile_repetitions(profile_from_args(args))


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.write_runtime_manifest is not None:
        if any(
            value is None
            for value in (args.runtime_name, args.runtime_python, args.runtime_workdir)
        ):
            raise ValueError(
                "manifest mode requires --runtime-name, --runtime-python, and --runtime-workdir"
            )
        benchmark_values = (
            args.gate,
            args.baseline_manifest,
            args.candidate_manifest,
            args.repetitions,
            args.output_json,
            args.output_markdown,
        )
        if any(value is not None and value is not False for value in benchmark_values):
            raise ValueError("manifest mode does not accept benchmark arguments")
        return
    repetitions = repetitions_from_args(args)
    if repetitions < 1:
        raise ValueError("--repetitions must be at least 1")
    if args.quick:
        if args.baseline_manifest is not None:
            raise ValueError("quick mode does not accept a baseline manifest")
        if args.candidate_manifest is None:
            raise ValueError("quick mode requires --candidate-manifest")
    else:
        if args.baseline_manifest is None or args.candidate_manifest is None:
            raise ValueError("full mode requires baseline and candidate manifests")
        if repetitions != FULL_REPETITIONS:
            raise ValueError("full mode requires exactly 5 repetitions")


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
    candidate = _runtime_from_manifest("candidate", args.candidate_manifest)
    baseline = (
        quick_baseline_runtime(candidate)
        if profile == "quick"
        else _runtime_from_manifest("baseline", args.baseline_manifest)
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
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            strict_json_dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown, encoding="utf-8")
    if args.output_markdown is None:
        print(markdown, end="")
    return 1 if args.gate and report["gate"]["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
