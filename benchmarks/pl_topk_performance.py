#!/usr/bin/env python3
"""Acceptance benchmark for top-k piecewise-linear split construction."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time
from typing import Iterable, Mapping, Sequence

import numpy as np


RESULT_SCHEMA_VERSION = 1
PRODUCTION_BASE = "ea4df36"
ARMS = ("legacy", "k0", "k1", "k8", "all")
PL_FRIENDLY_VARIANTS = frozenset({"local-linear", "raw-scale"})
MAX_QUALITY_REGRESSION = 0.01
MIN_PL_FRIENDLY_IMPROVEMENT = 0.05
MAX_DEFAULT_MEDIAN_TIME_RATIO = 3.0
MAX_DEFAULT_RECORD_TIME_RATIO = 5.0
MAX_WIDE_EXHAUSTIVE_TIME_RATIO = 0.5
MAX_RSS_RELATIVE_GROWTH = 0.15
MAX_RSS_ABSOLUTE_GROWTH = 32 * 1024 * 1024
MAX_CONVERGENCE_ROUND_RATIO = 0.75
REJECTED_TRIALS: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    shape: str
    task_family: str
    rows: int
    features: int
    rounds: int
    variant: str
    classes: int = 0
    queries: int = 0


@dataclass(frozen=True)
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    group_train: list[int] | None = None
    group_ids_test: np.ndarray | None = None


@dataclass(frozen=True)
class PLTopKRecord:
    arm: str
    dataset: str
    task_family: str
    shape: str
    variant: str
    features: int
    seed: int
    rounds: int
    primary_metric: str
    primary_value: float
    higher_is_better: bool
    secondary_metrics: dict[str, float]
    fit_seconds: float
    peak_rss_bytes: int
    rounds_completed: int
    prediction_sha256: str
    artifact_sha256: str


@dataclass(frozen=True)
class PLTopKComparison:
    passed: bool
    key_coverage_exact: bool
    k0_artifact_parity: bool
    k0_prediction_parity: bool
    default_quality_passed: bool
    pl_friendly_improvement: float
    convergence_passed: bool
    median_fixed_round_ratio: float
    worst_fixed_round_ratio: float
    wide_exhaustive_ratios: dict[str, float]
    worst_rss_growth_bytes: int
    reasons: tuple[str, ...]
    rejected_trials: tuple[dict[str, object], ...] = ()


def full_specs() -> tuple[FixtureSpec, ...]:
    convergence_rounds = (5, 10, 20, 40)
    specs = [
        FixtureSpec("reg-small-narrow", "small-narrow", "regression", 512, 8, 40, "nonlinear"),
        FixtureSpec("reg-small-wide", "small-wide", "regression", 512, 64, 30, "sparse"),
        FixtureSpec("reg-tall-narrow", "tall-narrow", "regression", 4096, 12, 30, "nonlinear"),
        FixtureSpec("reg-tall-wide", "tall-wide", "regression", 3072, 64, 20, "sparse"),
        FixtureSpec("binary-medium", "medium", "binary", 2048, 24, 30, "binary"),
        FixtureSpec("multiclass-small-wide", "small-wide", "multiclass", 1024, 48, 25, "multiclass", classes=4),
        FixtureSpec("ranking-tall-narrow", "tall-narrow", "ranking", 2400, 16, 25, "ranking", queries=80),
    ]
    for rounds in convergence_rounds:
        specs.append(FixtureSpec("local-linear", "medium", "regression", 1536, 16, rounds, "local-linear"))
        specs.append(FixtureSpec("raw-scale", "medium", "regression", 1536, 16, rounds, "raw-scale"))
    return tuple(specs)


def quick_specs() -> tuple[FixtureSpec, ...]:
    return tuple(
        replace(
            spec,
            rows=min(spec.rows, 512),
            features=min(spec.features, 32),
            rounds=min(spec.rounds, 10),
            queries=min(spec.queries, 16) if spec.queries else 0,
        )
        for spec in full_specs()
        if spec.rounds in {10, 20, 30, 40}
    )


def _split(X: np.ndarray, y: np.ndarray) -> DatasetBundle:
    boundary = int(0.8 * len(X))
    return DatasetBundle(X[:boundary], y[:boundary], X[boundary:], y[boundary:])


def _regression_data(spec: FixtureSpec, seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(spec.rows, spec.features)).astype(np.float32)
    if spec.variant == "local-linear":
        region = X[:, 0] >= 0.0
        left = 2.0 * X[:, 1] - 1.5 * X[:, 2] + 0.6 * X[:, 3]
        right = -2.5 * X[:, 1] + 1.2 * X[:, 2] - 0.8 * X[:, 4] + 2.0
        y = np.where(region, right, left)
        y += 0.08 * rng.normal(size=spec.rows)
    elif spec.variant == "raw-scale":
        scales = np.geomspace(1e-3, 1e3, spec.features).astype(np.float32)
        X *= scales
        split_feature = X[:, 0] / scales[0]
        y = np.where(
            split_feature >= 0.0,
            1.8 * X[:, -1] / scales[-1] - X[:, -2] / scales[-2],
            -1.4 * X[:, -1] / scales[-1] + 0.7 * X[:, -3] / scales[-3],
        )
        y += 0.08 * rng.normal(size=spec.rows)
    elif spec.variant == "sparse":
        y = X[:, :6] @ np.array([2.0, -1.7, 1.2, 0.8, -0.5, 0.3], dtype=np.float32)
        y += 0.3 * rng.normal(size=spec.rows)
    else:
        y = 1.8 * np.sin(X[:, 0]) + X[:, 1] * X[:, 2] + 0.5 * X[:, 3] ** 2
        y += 0.5 * rng.normal(size=spec.rows)
    return _split(X, np.asarray(y, dtype=np.float32))


def _classification_data(spec: FixtureSpec, seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(spec.rows, spec.features)).astype(np.float32)
    if spec.task_family == "binary":
        score = 1.6 * X[:, 0] - X[:, 1] + X[:, 2] * X[:, 3]
        y = (score + 0.5 * rng.normal(size=spec.rows) > 0.0).astype(np.int32)
    else:
        weights = rng.normal(size=(spec.features, spec.classes)).astype(np.float32)
        logits = X @ weights + 0.5 * rng.normal(size=(spec.rows, spec.classes))
        y = np.argmax(logits, axis=1).astype(np.int32)
    return _split(X, y)


def _group_sizes(rows: int, queries: int) -> list[int]:
    base, remainder = divmod(rows, queries)
    return [base + (index < remainder) for index in range(queries)]


def _ranking_data(spec: FixtureSpec, seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    train_rows = int(0.8 * spec.rows)
    test_rows = spec.rows - train_rows
    train_queries = max(2, int(0.8 * spec.queries))
    test_queries = max(2, spec.queries - train_queries)
    train_groups = _group_sizes(train_rows, train_queries)
    test_groups = _group_sizes(test_rows, test_queries)

    def make(groups: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = sum(groups)
        X = rng.normal(size=(count, spec.features)).astype(np.float32)
        raw = 1.5 * X[:, 0] - X[:, 1] + X[:, 2] * X[:, 3]
        labels = np.empty(count, dtype=np.float32)
        ids = np.repeat(np.arange(len(groups), dtype=np.int32), groups)
        offset = 0
        for size in groups:
            order = np.argsort(np.argsort(raw[offset : offset + size]))
            labels[offset : offset + size] = np.floor(4.99 * order / max(size - 1, 1))
            offset += size
        return X, labels, ids

    X_train, y_train, _ = make(train_groups)
    X_test, y_test, test_ids = make(test_groups)
    return DatasetBundle(X_train, y_train, X_test, y_test, train_groups, test_ids)


def make_dataset(spec: FixtureSpec, seed: int) -> DatasetBundle:
    if spec.task_family == "regression":
        return _regression_data(spec, seed)
    if spec.task_family in {"binary", "multiclass"}:
        return _classification_data(spec, seed)
    if spec.task_family == "ranking":
        return _ranking_data(spec, seed)
    raise ValueError(f"unknown task family: {spec.task_family}")


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def _log_loss(actual: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(actual, dtype=np.int64)
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim == 1:
        values = np.column_stack((1.0 - values, values))
    selected = np.clip(values[np.arange(len(labels)), labels], 1e-15, 1.0)
    return float(-np.mean(np.log(selected)))


def _hash_array(values: object) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _artifact_bytes(model: object) -> bytes:
    value = getattr(model, "artifact_bytes")
    return bytes(value() if callable(value) else value)


def normalize_peak_rss_bytes(value: int, *, platform_name: str | None = None) -> int:
    system = sys.platform if platform_name is None else platform_name.lower()
    return int(value) if system.startswith("darwin") else int(value) * 1024


def _fit_record(spec: FixtureSpec, arm: str, seed: int) -> PLTopKRecord:
    from alloygbm import GBMClassifier, GBMRanker, GBMRegressor
    from alloygbm.evaluation import ndcg

    bundle = make_dataset(spec, seed)
    kwargs: dict[str, object] = {
        "n_estimators": spec.rounds,
        "max_depth": 4,
        "learning_rate": 0.08,
        "lambda_l2": 1.0,
        "leaf_model": "linear",
        "training_policy": "manual",
        "row_subsample": 1.0,
        "col_subsample": 1.0,
        "deterministic": True,
        "n_jobs": 1,
        "seed": seed,
    }
    if arm != "legacy":
        kwargs["pl_split_candidates"] = {
            "k0": 0,
            "k1": 1,
            "k8": 8,
            "all": spec.features,
        }[arm]

    started = time.perf_counter()
    if spec.task_family == "regression":
        model = GBMRegressor(**kwargs).fit(bundle.X_train, bundle.y_train)
        prediction = np.asarray(model.predict(bundle.X_test), dtype=np.float64)
        metric = "rmse"
        value = _rmse(bundle.y_test, prediction)
        secondary = {"mae": float(np.mean(np.abs(bundle.y_test - prediction)))}
        higher = False
    elif spec.task_family in {"binary", "multiclass"}:
        model = GBMClassifier(**kwargs).fit(bundle.X_train, bundle.y_train)
        probabilities = np.asarray(model.predict_proba(bundle.X_test), dtype=np.float64)
        prediction = np.asarray(model.predict(bundle.X_test))
        metric = "log_loss"
        value = _log_loss(bundle.y_test, probabilities)
        secondary = {"accuracy": float(np.mean(prediction == bundle.y_test))}
        higher = False
    else:
        model = GBMRanker(**kwargs).fit(bundle.X_train, bundle.y_train, group=bundle.group_train)
        prediction = np.asarray(model.predict(bundle.X_test), dtype=np.float64)
        metric = "ndcg_at_10"
        value = float(ndcg(bundle.y_test, prediction, group=bundle.group_ids_test, k=10))
        secondary = {}
        higher = True
    fit_seconds = time.perf_counter() - started
    peak_rss = normalize_peak_rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    artifact = _artifact_bytes(model)
    return PLTopKRecord(
        arm=arm,
        dataset=spec.name,
        task_family=spec.task_family,
        shape=spec.shape,
        variant=spec.variant,
        features=spec.features,
        seed=seed,
        rounds=spec.rounds,
        primary_metric=metric,
        primary_value=value,
        higher_is_better=higher,
        secondary_metrics=secondary,
        fit_seconds=fit_seconds,
        peak_rss_bytes=peak_rss,
        rounds_completed=spec.rounds,
        prediction_sha256=_hash_array(prediction),
        artifact_sha256=hashlib.sha256(artifact).hexdigest(),
    )


def run_record_subprocess(spec: FixtureSpec, arm: str, seed: int) -> PLTopKRecord:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "record",
        "--spec",
        json.dumps(asdict(spec), sort_keys=True),
        "--arm",
        arm,
        "--seed",
        str(seed),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    return PLTopKRecord(**payload)


def record_sort_key(record: PLTopKRecord) -> tuple[object, ...]:
    return (
        record.dataset,
        record.task_family,
        record.shape,
        record.seed,
        record.rounds,
        record.arm,
    )


def _case_key(record: PLTopKRecord) -> tuple[object, ...]:
    return (
        record.dataset,
        record.task_family,
        record.shape,
        record.variant,
        record.features,
        record.seed,
        record.rounds,
        record.primary_metric,
    )


def _validated_map(records: Sequence[PLTopKRecord], label: str) -> tuple[dict[tuple[object, ...], PLTopKRecord], list[str]]:
    result: dict[tuple[object, ...], PLTopKRecord] = {}
    reasons: list[str] = []
    for record in records:
        key = _case_key(record)
        if key in result:
            reasons.append(f"duplicate {label} key: {key}")
            continue
        numeric = [record.primary_value, record.fit_seconds, *record.secondary_metrics.values()]
        if any(not math.isfinite(float(value)) for value in numeric):
            reasons.append(f"non-finite {label} value for key {key}")
        if record.fit_seconds <= 0.0 or record.peak_rss_bytes <= 0:
            reasons.append(f"non-positive timing or RSS for {label} key {key}")
        result[key] = record
    return result, reasons


def _quality_ratio(reference: PLTopKRecord, trial: PLTopKRecord) -> float:
    if reference.higher_is_better != trial.higher_is_better:
        return math.inf
    if reference.primary_value == 0.0:
        return 1.0 if trial.primary_value == 0.0 else math.inf
    if reference.higher_is_better:
        return reference.primary_value / trial.primary_value
    return trial.primary_value / reference.primary_value


def _quality_reaches(trial: PLTopKRecord, target: PLTopKRecord) -> bool:
    if target.higher_is_better:
        return trial.primary_value >= target.primary_value * (1.0 - 1e-12)
    return trial.primary_value <= target.primary_value * (1.0 + 1e-12)


def compare_results(
    baseline: Sequence[PLTopKRecord],
    candidate: Sequence[PLTopKRecord],
    *,
    rejected_trials: Sequence[Mapping[str, object]] = (),
) -> PLTopKComparison:
    base_map, reasons = _validated_map(baseline, "baseline")
    by_arm: dict[str, dict[tuple[object, ...], PLTopKRecord]] = {}
    for arm in ("k0", "k8", "all"):
        arm_map, arm_reasons = _validated_map([record for record in candidate if record.arm == arm], arm)
        by_arm[arm] = arm_map
        reasons.extend(arm_reasons)

    base_keys = set(base_map)
    key_coverage_exact = bool(base_keys) and all(set(records) == base_keys for records in by_arm.values())
    if not key_coverage_exact:
        reasons.append("candidate coverage does not exactly match baseline coverage")

    k0_artifact_parity = key_coverage_exact
    k0_prediction_parity = key_coverage_exact
    quality_ratios: list[float] = []
    friendly_improvements: list[float] = []
    fixed_round_ratios: list[float] = []
    rss_growth: list[int] = []
    wide_exhaustive_ratios: dict[str, float] = {}

    for key in sorted(base_keys & set(by_arm["k0"]) & set(by_arm["k8"]) & set(by_arm["all"])):
        base = base_map[key]
        k0 = by_arm["k0"][key]
        k8 = by_arm["k8"][key]
        exhaustive = by_arm["all"][key]
        if base.artifact_sha256 != k0.artifact_sha256:
            k0_artifact_parity = False
        if base.prediction_sha256 != k0.prediction_sha256:
            k0_prediction_parity = False
        ratio = _quality_ratio(k0, k8)
        quality_ratios.append(ratio)
        if ratio > 1.0 + MAX_QUALITY_REGRESSION:
            reasons.append(f"default quality regression exceeded one percent for {key}: {ratio:.6f}")
        for metric, reference_value in k0.secondary_metrics.items():
            trial_value = k8.secondary_metrics.get(metric)
            if trial_value is None:
                reasons.append(f"secondary quality metric coverage differs for {key}")
            elif reference_value != 0.0 and abs(trial_value - reference_value) / abs(reference_value) > MAX_QUALITY_REGRESSION:
                reasons.append(f"secondary quality regression exceeded one percent for {key} metric={metric}")
        if k0.variant in PL_FRIENDLY_VARIANTS and k0.rounds == max(
            record.rounds
            for record in by_arm["k0"].values()
            if (record.dataset, record.seed) == (k0.dataset, k0.seed)
        ):
            friendly_improvements.append(1.0 - ratio)
        time_ratio = k8.fit_seconds / k0.fit_seconds
        fixed_round_ratios.append(time_ratio)
        if time_ratio > MAX_DEFAULT_RECORD_TIME_RATIO:
            reasons.append(f"default fixed-round fit cost exceeded five times k0 for {key}: {time_ratio:.6f}")
        growth = k8.peak_rss_bytes - k0.peak_rss_bytes
        rss_growth.append(growth)
        allowed_growth = max(int(MAX_RSS_RELATIVE_GROWTH * k0.peak_rss_bytes), MAX_RSS_ABSOLUTE_GROWTH)
        if growth > allowed_growth:
            reasons.append(f"default peak RSS exceeded growth allowance for {key}: RSS growth={growth}")
        if k0.features >= 32:
            exhaustive_ratio = k8.fit_seconds / exhaustive.fit_seconds
            wide_exhaustive_ratios[str(key)] = exhaustive_ratio
            if exhaustive_ratio > MAX_WIDE_EXHAUSTIVE_TIME_RATIO:
                reasons.append(f"wide default fit cost exceeded half exhaustive cost for {key}: {exhaustive_ratio:.6f}")
            if _quality_ratio(exhaustive, k8) > 1.0 + MAX_QUALITY_REGRESSION:
                reasons.append(f"wide default quality regressed over one percent versus exhaustive for {key}")

    if not k0_artifact_parity:
        reasons.append("k0 artifact parity with production failed")
    if not k0_prediction_parity:
        reasons.append("k0 prediction parity with production failed")

    pl_friendly_improvement = max(friendly_improvements, default=-math.inf)
    if pl_friendly_improvement < MIN_PL_FRIENDLY_IMPROVEMENT:
        reasons.append("no PL-friendly case improved by five percent")

    convergence_passed = True
    grouped: dict[tuple[str, int], list[PLTopKRecord]] = {}
    for record in by_arm["k8"].values():
        if record.variant in PL_FRIENDLY_VARIANTS:
            grouped.setdefault((record.dataset, record.seed), []).append(record)
    for group_key, trials in grouped.items():
        reference_records = [
            record for key, record in by_arm["k0"].items()
            if (record.dataset, record.seed) == group_key
        ]
        if not reference_records:
            continue
        target = max(reference_records, key=lambda record: record.rounds)
        reaching = [record.rounds_completed for record in trials if _quality_reaches(record, target)]
        limit = MAX_CONVERGENCE_ROUND_RATIO * target.rounds_completed
        if not reaching or min(reaching) > limit:
            convergence_passed = False
            reasons.append(f"convergence gate failed for {group_key}: limit={limit:g}")

    median_fixed_round_ratio = statistics.median(fixed_round_ratios) if fixed_round_ratios else math.inf
    worst_fixed_round_ratio = max(fixed_round_ratios, default=math.inf)
    if median_fixed_round_ratio > MAX_DEFAULT_MEDIAN_TIME_RATIO:
        reasons.append(f"median fixed-round default cost exceeded three times k0: {median_fixed_round_ratio:.6f}")

    validated_trials = tuple(validate_rejected_trial(trial) for trial in rejected_trials)
    return PLTopKComparison(
        passed=not reasons,
        key_coverage_exact=key_coverage_exact,
        k0_artifact_parity=k0_artifact_parity,
        k0_prediction_parity=k0_prediction_parity,
        default_quality_passed=all(ratio <= 1.0 + MAX_QUALITY_REGRESSION for ratio in quality_ratios),
        pl_friendly_improvement=pl_friendly_improvement,
        convergence_passed=convergence_passed,
        median_fixed_round_ratio=float(median_fixed_round_ratio),
        worst_fixed_round_ratio=float(worst_fixed_round_ratio),
        wide_exhaustive_ratios=wide_exhaustive_ratios,
        worst_rss_growth_bytes=max(rss_growth, default=0),
        reasons=tuple(reasons),
        rejected_trials=validated_trials,
    )


def validate_rejected_trial(trial: Mapping[str, object]) -> dict[str, object]:
    required = {"label", "reason", "metrics"}
    if not required <= trial.keys() or not isinstance(trial["metrics"], Mapping):
        raise ValueError("rejected trial requires label, reason, and metrics")
    metrics = dict(trial["metrics"])
    if any(not math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError("rejected trial metrics must be finite")
    return {key: (metrics if key == "metrics" else value) for key, value in trial.items()}


def synthetic_result_pair(
    *,
    k8_quality_ratio: float = 0.9,
    k8_time_ratio: float = 1.0,
    all_time_ratio: float = 3.0,
    convergence_ratio: float = 0.5,
    task_family: str = "regression",
    primary_metric: str = "rmse",
    higher_is_better: bool = False,
    features: int = 64,
    k0_peak_rss_bytes: int = 100 * 1024 * 1024,
    k8_peak_rss_bytes: int = 110 * 1024 * 1024,
) -> tuple[list[PLTopKRecord], list[PLTopKRecord]]:
    baseline: list[PLTopKRecord] = []
    candidate: list[PLTopKRecord] = []
    for seed in range(3):
        for rounds in (5, 10, 20, 40):
            k0_value = 1.0 + (40 - rounds) / 40
            effective = min(40.0, rounds / max(convergence_ratio, 1e-9))
            accelerated_value = 1.0 + (40.0 - effective) / 40.0
            k8_value = accelerated_value
            if rounds == 40:
                k8_value = k0_value * k8_quality_ratio
            if higher_is_better:
                k0_value = 2.0 - k0_value / 2.0
                k8_value = k0_value * k8_quality_ratio if rounds == 40 else 2.0 - accelerated_value / 2.0
            common = dict(
                dataset="synthetic-local-linear",
                task_family=task_family,
                shape="small-wide",
                variant="local-linear",
                features=features,
                seed=seed,
                rounds=rounds,
                primary_metric=primary_metric,
                higher_is_better=higher_is_better,
                secondary_metrics={},
                rounds_completed=rounds,
                prediction_sha256=f"prediction-{seed}-{rounds}",
                artifact_sha256=f"artifact-{seed}-{rounds}",
            )
            legacy = PLTopKRecord(arm="legacy", primary_value=k0_value, fit_seconds=1.0, peak_rss_bytes=k0_peak_rss_bytes, **common)
            baseline.append(legacy)
            candidate.extend(
                [
                    replace(legacy, arm="k0"),
                    replace(legacy, arm="k8", primary_value=k8_value, fit_seconds=k8_time_ratio, peak_rss_bytes=k8_peak_rss_bytes),
                    replace(legacy, arm="all", primary_value=k8_value, fit_seconds=all_time_ratio, peak_rss_bytes=k8_peak_rss_bytes),
                ]
            )
    return baseline, candidate


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("alloygbm", "numpy", "scikit-learn"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "unknown"
    return result


def write_results(path: str | Path, records: Sequence[PLTopKRecord], *, argv: Sequence[str]) -> None:
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "production_base": PRODUCTION_BASE,
        "git_head": _git_head(),
        "argv": list(argv),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "records": [asdict(record) for record in sorted(records, key=record_sort_key)],
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_results(path: str | Path) -> list[PLTopKRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION or payload.get("production_base") != PRODUCTION_BASE:
        raise ValueError("unsupported or mismatched PL top-k benchmark result")
    records = sorted((PLTopKRecord(**record) for record in payload.get("records", [])), key=record_sort_key)
    if not records:
        raise ValueError("PL top-k benchmark result contains no records")
    return records


def _run(args: argparse.Namespace) -> int:
    specs = quick_specs() if args.quick else full_specs()
    records: list[PLTopKRecord] = []
    for spec in specs:
        for seed in args.seeds:
            for arm in args.arms:
                record = run_record_subprocess(spec, arm, seed)
                records.append(record)
                print(f"{spec.name} rounds={spec.rounds} seed={seed} arm={arm} {record.primary_metric}={record.primary_value:.8g} fit={record.fit_seconds:.4f}s rss={record.peak_rss_bytes}", flush=True)
    write_results(args.output, records, argv=sys.argv[1:])
    return 0


def _compare(args: argparse.Namespace) -> int:
    summary = compare_results(read_results(args.baseline), read_results(args.candidate), rejected_trials=REJECTED_TRIALS)
    payload = json.dumps(asdict(summary), indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if summary.passed else 1


def _record(args: argparse.Namespace) -> int:
    spec = FixtureSpec(**json.loads(args.spec))
    print(json.dumps(asdict(_fit_record(spec, args.arm, args.seed)), sort_keys=True, allow_nan=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--arms", nargs="+", choices=ARMS, required=True)
    run_parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    run_parser.add_argument("--quick", action="store_true")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.set_defaults(handler=_run)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("baseline", type=Path)
    compare_parser.add_argument("candidate", type=Path)
    compare_parser.add_argument("--output", type=Path)
    compare_parser.set_defaults(handler=_compare)
    record_parser = subparsers.add_parser("record", help=argparse.SUPPRESS)
    record_parser.add_argument("--spec", required=True)
    record_parser.add_argument("--arm", choices=ARMS, required=True)
    record_parser.add_argument("--seed", type=int, required=True)
    record_parser.set_defaults(handler=_record)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
