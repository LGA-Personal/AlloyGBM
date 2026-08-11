#!/usr/bin/env python3
"""Deterministic end-to-end performance matrix for the scalar DRO leaf solver."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


RESULT_SCHEMA_VERSION = 1
PRODUCTION_BASE = "2b2e3ef"
ARMS = ("standard", "dro")
MAX_SHAPE_REGRESSION = 0.05
MAX_STANDARD_REGRESSION = 0.03
MIN_FIT_IMPROVEMENT = 0.15
QUALITY_TOLERANCE_ABS = 1e-7
QUALITY_TOLERANCE_REL = 1e-7


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    shape: str
    task_family: str
    rows: int
    features: int
    rounds: int
    variant: str
    query_count: int = 0


@dataclass(frozen=True)
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    group_train: list[int] | None = None
    group_test: list[int] | None = None
    group_ids_test: np.ndarray | None = None


@dataclass(frozen=True)
class DroPerfRecord:
    arm: str
    dataset: str
    task_family: str
    shape: str
    seed: int
    primary_metric: str
    primary_value: float
    higher_is_better: bool
    secondary_metrics: dict[str, float]
    fit_seconds: float
    input_adaptation_seconds: float | None = None
    native_bridge_prepare_seconds: float | None = None
    native_train_seconds: float | None = None
    predict_seconds: float | None = None


@dataclass(frozen=True)
class ComparisonSummary:
    passed: bool
    quality_equivalent: bool
    key_coverage_exact: bool
    baseline_key_count: int
    candidate_key_count: int
    dro_shape_time_ratios: dict[str, float]
    dro_fit_median_ratio: float
    dro_fit_improvement: float
    standard_fit_median_ratio: float
    worst_dro_shape_ratio: float
    worst_standard_ratio: float
    scanner_gate_passed: bool
    reasons: tuple[str, ...]
    rejected_trials: tuple[dict[str, object], ...] = ()


def full_specs() -> tuple[FixtureSpec, ...]:
    return (
        FixtureSpec("reg-small-narrow", "small-narrow", "regression", 640, 8, 80, "linear"),
        FixtureSpec("reg-small-wide", "small-wide", "regression", 640, 128, 80, "sparse"),
        FixtureSpec("reg-tall-narrow", "tall-narrow", "regression", 8192, 16, 80, "linear"),
        FixtureSpec("reg-tall-wide", "tall-wide", "regression", 8192, 128, 60, "sparse"),
        FixtureSpec("reg-noisy", "medium", "regression", 2048, 32, 100, "noisy"),
        FixtureSpec("binary-imbalanced", "medium", "binary", 4096, 32, 100, "imbalanced"),
        FixtureSpec("multiclass-wide", "small-wide", "multiclass", 2048, 96, 80, "multiclass"),
        FixtureSpec(
            "rank-small-query",
            "tall-narrow",
            "ranking",
            2400,
            24,
            60,
            "small-query",
            120,
        ),
        FixtureSpec(
            "rank-large-query",
            "tall-narrow",
            "ranking",
            4096,
            24,
            60,
            "large-query",
            8,
        ),
    )


def quick_specs() -> tuple[FixtureSpec, ...]:
    return tuple(
        replace(
            spec,
            rows=min(spec.rows, 768),
            features=min(spec.features, 48),
            rounds=min(spec.rounds, 12),
            query_count=min(spec.query_count, 12) if spec.query_count else 0,
        )
        for spec in full_specs()
    )


def _split_rows(X: np.ndarray, y: np.ndarray) -> DatasetBundle:
    split = max(1, min(len(X) - 1, int(0.8 * len(X))))
    return DatasetBundle(X[:split], y[:split], X[split:], y[split:])


def _make_regression(spec: FixtureSpec, seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((spec.rows, spec.features), dtype=np.float32)
    if spec.variant == "sparse":
        signal_features = min(8, spec.features)
        coefficients = rng.normal(size=signal_features).astype(np.float32)
        y = X[:, :signal_features] @ coefficients
        y += 0.2 * rng.standard_normal(spec.rows).astype(np.float32)
    elif spec.variant == "noisy":
        y = (
            1.5 * np.sin(X[:, 0])
            + X[:, 1] * X[:, 2]
            + 0.45 * X[:, 3] ** 2
            - 0.35 * X[:, 4]
            + 1.0 * rng.standard_normal(spec.rows)
        ).astype(np.float32)
    else:
        coefficients = rng.normal(size=spec.features).astype(np.float32)
        y = X @ coefficients + 0.2 * rng.standard_normal(spec.rows).astype(np.float32)
    return _split_rows(X, np.asarray(y, dtype=np.float32))


def _make_binary(spec: FixtureSpec, seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((spec.rows, spec.features), dtype=np.float32)
    score = 1.8 * X[:, 0] - 1.2 * X[:, 1] + X[:, 2] * X[:, 3]
    score += 0.5 * rng.standard_normal(spec.rows)
    threshold = float(np.quantile(score, 0.88))
    y = (score >= threshold).astype(np.int32)
    return _split_rows(X, y)


def _make_multiclass(spec: FixtureSpec, seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((spec.rows, spec.features), dtype=np.float32)
    weights = rng.normal(size=(spec.features, 4)).astype(np.float32)
    logits = X @ weights
    logits[:, 0] += 0.5 * X[:, 0] * X[:, 1]
    logits += 0.8 * rng.standard_normal(logits.shape).astype(np.float32)
    y = np.argmax(logits, axis=1).astype(np.int32)
    return _split_rows(X, y)


def _query_sizes(rows: int, query_count: int) -> list[int]:
    query_count = max(2, min(query_count, rows // 2))
    base, remainder = divmod(rows, query_count)
    return [base + (1 if index < remainder else 0) for index in range(query_count)]


def _make_ranking(spec: FixtureSpec, seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    train_query_count = max(2, int(spec.query_count * 0.8))
    test_query_count = max(2, spec.query_count - train_query_count)
    train_rows = max(train_query_count * 2, int(spec.rows * 0.8))
    test_rows = max(test_query_count * 2, spec.rows - train_rows)
    group_train = _query_sizes(train_rows, train_query_count)
    group_test = _query_sizes(test_rows, test_query_count)

    def build(sizes: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        row_count = sum(sizes)
        X = rng.standard_normal((row_count, spec.features), dtype=np.float32)
        raw = 1.5 * X[:, 0] - X[:, 1] + 0.8 * X[:, 2] * X[:, 3]
        ids = np.repeat(np.arange(len(sizes), dtype=np.int32), sizes)
        y = np.empty(row_count, dtype=np.float32)
        offset = 0
        for size in sizes:
            scores = raw[offset : offset + size] + 0.4 * rng.standard_normal(size)
            order = np.argsort(np.argsort(scores))
            y[offset : offset + size] = np.floor(4.99 * order / max(size - 1, 1))
            offset += size
        return X, y, ids

    X_train, y_train, _ = build(group_train)
    X_test, y_test, group_ids_test = build(group_test)
    return DatasetBundle(
        X_train,
        y_train,
        X_test,
        y_test,
        group_train,
        group_test,
        group_ids_test,
    )


def make_dataset(spec: FixtureSpec, seed: int) -> DatasetBundle:
    if spec.task_family == "regression":
        return _make_regression(spec, seed)
    if spec.task_family == "binary":
        return _make_binary(spec, seed)
    if spec.task_family == "multiclass":
        return _make_multiclass(spec, seed)
    if spec.task_family == "ranking":
        return _make_ranking(spec, seed)
    raise ValueError(f"unknown task family: {spec.task_family}")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _multiclass_log_loss(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(y_true, dtype=np.int64)
    selected = np.clip(values[np.arange(len(labels)), labels], 1e-15, 1.0)
    return float(-np.mean(np.log(selected)))


def _binary_log_loss(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    positive = values[:, 1] if values.ndim == 2 else values
    positive = np.clip(positive, 1e-15, 1.0 - 1e-15)
    labels = np.asarray(y_true, dtype=np.float64)
    return float(-np.mean(labels * np.log(positive) + (1.0 - labels) * np.log1p(-positive)))


def _rotated_arms(arms: Sequence[str], case_index: int, seed: int) -> list[str]:
    if not arms:
        return []
    offset = (case_index + seed) % len(arms)
    return [*arms[offset:], *arms[:offset]]


def _fit_record(
    spec: FixtureSpec, bundle: DatasetBundle, arm: str, seed: int
) -> DroPerfRecord:
    from alloygbm import GBMClassifier, GBMRanker, GBMRegressor
    from alloygbm.evaluation import ndcg

    common: dict[str, object] = {
        "n_estimators": spec.rounds,
        "max_depth": 5,
        "learning_rate": 0.05,
        "seed": seed,
        "deterministic": True,
        "n_jobs": 1,
        "lambda_l2": 1.0,
        "leaf_solver": arm,
    }
    if arm == "dro":
        common.update(
            dro_radius=0.05,
            dro_metric="wasserstein",
        )

    fit_started = time.perf_counter()
    if spec.task_family == "regression":
        model = GBMRegressor(**common)
        model.fit(bundle.X_train, bundle.y_train)
        prediction = np.asarray(model.predict(bundle.X_test), dtype=np.float64)
        primary_metric = "rmse"
        primary_value = _rmse(bundle.y_test, prediction)
        secondary = {"mae": _mae(bundle.y_test, prediction)}
        higher_is_better = False
    elif spec.task_family in {"binary", "multiclass"}:
        model = GBMClassifier(**common)
        model.fit(bundle.X_train, bundle.y_train)
        probabilities = np.asarray(model.predict_proba(bundle.X_test), dtype=np.float64)
        prediction = np.asarray(model.predict(bundle.X_test))
        if spec.task_family == "binary":
            primary_metric = "log_loss"
            primary_value = _binary_log_loss(bundle.y_test, probabilities)
        else:
            primary_metric = "log_loss"
            primary_value = _multiclass_log_loss(bundle.y_test, probabilities)
        secondary = {"accuracy": float(np.mean(prediction == bundle.y_test))}
        higher_is_better = False
    else:
        model = GBMRanker(**common)
        model.fit(bundle.X_train, bundle.y_train, group=bundle.group_train)
        prediction = np.asarray(model.predict(bundle.X_test), dtype=np.float64)
        primary_metric = "ndcg_at_10"
        primary_value = float(
            ndcg(bundle.y_test, prediction, group=bundle.group_ids_test, k=10)
        )
        secondary = {}
        higher_is_better = True
    fit_seconds = time.perf_counter() - fit_started

    timing = getattr(model, "fit_timing_", {})
    return DroPerfRecord(
        arm=arm,
        dataset=spec.name,
        task_family=spec.task_family,
        shape=spec.shape,
        seed=seed,
        primary_metric=primary_metric,
        primary_value=primary_value,
        higher_is_better=higher_is_better,
        secondary_metrics=secondary,
        fit_seconds=fit_seconds,
        input_adaptation_seconds=_optional_timing(timing, "input_adaptation_seconds"),
        native_bridge_prepare_seconds=_optional_timing(
            timing, "native_bridge_prepare_seconds"
        ),
        native_train_seconds=_optional_timing(timing, "native_train_seconds"),
        predict_seconds=None,
    )


def _optional_timing(timing: object, key: str) -> float | None:
    if not isinstance(timing, dict) or key not in timing:
        return None
    value = float(timing[key])
    return value if math.isfinite(value) else None


def run_matrix(
    arms: Sequence[str],
    seeds: Sequence[int],
    *,
    quick: bool = False,
) -> list[DroPerfRecord]:
    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        raise ValueError(f"unknown benchmark arms: {unknown}")
    if not arms:
        raise ValueError("at least one benchmark arm is required")
    if not seeds:
        raise ValueError("at least one benchmark seed is required")

    specs = quick_specs() if quick else full_specs()
    records: list[DroPerfRecord] = []
    for case_index, spec in enumerate(specs):
        for seed in seeds:
            bundle = make_dataset(spec, seed)
            for arm in _rotated_arms(list(arms), case_index, seed):
                record = _fit_record(spec, bundle, arm, seed)
                records.append(record)
                print(
                    f"{spec.name} seed={seed} arm={arm} "
                    f"{record.primary_metric}={record.primary_value:.9f} "
                    f"fit_seconds={record.fit_seconds:.6f}",
                    flush=True,
                )
    return records


def _record_key(record: DroPerfRecord) -> tuple[str, str, str, str, int, str]:
    return (
        record.arm,
        record.dataset,
        record.task_family,
        record.shape,
        record.seed,
        record.primary_metric,
    )


def _sorted_records(records: Iterable[DroPerfRecord]) -> list[DroPerfRecord]:
    return sorted(records, key=_record_key)


def _record_map(
    records: Sequence[DroPerfRecord], label: str
) -> dict[tuple[str, str, str, str, int, str], DroPerfRecord]:
    result: dict[tuple[str, str, str, str, int, str], DroPerfRecord] = {}
    for record in records:
        key = _record_key(record)
        if key in result:
            raise ValueError(f"duplicate {label} benchmark key: {key}")
        values = [record.primary_value, record.fit_seconds]
        values.extend(record.secondary_metrics.values())
        for value in values:
            if not math.isfinite(float(value)):
                raise ValueError(f"non-finite {label} benchmark value for key {key}")
        if record.fit_seconds <= 0.0:
            raise ValueError(f"fit_seconds must be positive for key {key}")
        result[key] = record
    return result


def _quality_close(left: float, right: float) -> bool:
    return abs(left - right) <= max(
        QUALITY_TOLERANCE_ABS,
        QUALITY_TOLERANCE_REL * abs(left),
        QUALITY_TOLERANCE_REL * abs(right),
    )


def compare_results(
    baseline: Sequence[DroPerfRecord],
    candidate: Sequence[DroPerfRecord],
    *,
    scanner_gate: bool = False,
    rejected_trials: Sequence[dict[str, object]] = (),
) -> ComparisonSummary:
    baseline_map = _record_map(baseline, "baseline")
    candidate_map = _record_map(candidate, "candidate")
    if baseline_map.keys() != candidate_map.keys():
        missing_candidate = sorted(baseline_map.keys() - candidate_map.keys())
        missing_baseline = sorted(candidate_map.keys() - baseline_map.keys())
        raise ValueError(
            "baseline and candidate keys do not match: "
            f"missing_candidate={missing_candidate}, missing_baseline={missing_baseline}"
        )

    quality_errors: list[str] = []
    for key in sorted(baseline_map):
        base = baseline_map[key]
        trial = candidate_map[key]
        if base.higher_is_better != trial.higher_is_better:
            raise ValueError(f"metric direction differs for key {key}")
        if base.secondary_metrics.keys() != trial.secondary_metrics.keys():
            raise ValueError(f"quality equivalence metric keys differ for key {key}")
        if not _quality_close(base.primary_value, trial.primary_value):
            quality_errors.append(
                f"primary quality drift for {key}: "
                f"baseline={base.primary_value:.12g} candidate={trial.primary_value:.12g}"
            )
        for metric_name in sorted(base.secondary_metrics):
            base_value = base.secondary_metrics[metric_name]
            trial_value = trial.secondary_metrics[metric_name]
            if not _quality_close(base_value, trial_value):
                quality_errors.append(
                    f"secondary quality drift for {key} metric={metric_name}: "
                    f"baseline={base_value:.12g} candidate={trial_value:.12g}"
                )
    if quality_errors:
        raise ValueError("quality equivalence failed: " + "; ".join(quality_errors[:8]))

    def ratios_for_arm(arm: str) -> list[tuple[DroPerfRecord, float]]:
        return [
            (base, candidate_map[key].fit_seconds / base.fit_seconds)
            for key, base in sorted(baseline_map.items())
            if arm == key[0]
        ]

    dro_pairs = ratios_for_arm("dro")
    standard_pairs = ratios_for_arm("standard")
    if not dro_pairs or not standard_pairs:
        raise ValueError("baseline and candidate must contain both standard and dro arms")

    per_shape: dict[str, list[float]] = {}
    for record, ratio in dro_pairs:
        per_shape.setdefault(record.shape, []).append(ratio)
    dro_shape_time_ratios = {
        shape: float(statistics.median(values))
        for shape, values in sorted(per_shape.items())
    }
    dro_fit_median_ratio = float(statistics.median(dro_shape_time_ratios.values()))
    standard_fit_median_ratio = float(
        statistics.median(ratio for _, ratio in standard_pairs)
    )
    worst_dro_shape_ratio = max(dro_shape_time_ratios.values())
    worst_standard_ratio = max(ratio for _, ratio in standard_pairs)

    reasons: list[str] = []
    regressed_shapes = [
        shape
        for shape, ratio in dro_shape_time_ratios.items()
        if ratio > 1.0 + MAX_SHAPE_REGRESSION
    ]
    if regressed_shapes:
        reasons.append(
            "shape regression exceeded 5%: " + ", ".join(regressed_shapes)
        )
    if worst_standard_ratio > 1.0 + MAX_STANDARD_REGRESSION:
        reasons.append(
            f"standard-arm regression exceeded 3%: {worst_standard_ratio:.6f}"
        )

    dro_fit_improvement = 1.0 - dro_fit_median_ratio
    performance_passed = scanner_gate or dro_fit_improvement >= MIN_FIT_IMPROVEMENT
    if not performance_passed:
        reasons.append(
            "scanner gate did not pass and median DRO fit-time improvement "
            f"{dro_fit_improvement:.6f} missed 15%"
        )

    return ComparisonSummary(
        passed=not reasons,
        quality_equivalent=True,
        key_coverage_exact=True,
        baseline_key_count=len(baseline_map),
        candidate_key_count=len(candidate_map),
        dro_shape_time_ratios=dro_shape_time_ratios,
        dro_fit_median_ratio=dro_fit_median_ratio,
        dro_fit_improvement=dro_fit_improvement,
        standard_fit_median_ratio=standard_fit_median_ratio,
        worst_dro_shape_ratio=worst_dro_shape_ratio,
        worst_standard_ratio=worst_standard_ratio,
        scanner_gate_passed=scanner_gate,
        reasons=tuple(reasons),
        rejected_trials=tuple(rejected_trials),
    )


def synthetic_paired_records(
    *,
    metric_delta: float,
    time_ratio: float,
    standard_time_ratio: float = 1.0,
) -> list[DroPerfRecord]:
    """Return a small deterministic baseline/candidate-shaped record set for tests."""
    if not math.isfinite(metric_delta) or not math.isfinite(time_ratio):
        raise ValueError("synthetic benchmark inputs must be finite")
    shapes = ("small-narrow", "tall-narrow", "medium")
    records: list[DroPerfRecord] = []
    for shape_index, shape in enumerate(shapes):
        for seed in range(5):
            dataset = f"synthetic-{shape}-{seed}"
            for arm in ARMS:
                base_value = 1.0 + 0.01 * shape_index
                records.append(
                    DroPerfRecord(
                        arm=arm,
                        dataset=dataset,
                        task_family="regression",
                        shape=shape,
                        seed=seed,
                        primary_metric="rmse",
                        primary_value=base_value + (metric_delta if arm == "dro" else 0.0),
                        higher_is_better=False,
                        secondary_metrics={"mae": base_value / 2.0},
                        fit_seconds=(
                            time_ratio if arm == "dro" else standard_time_ratio
                        ),
                    )
                )
    return records


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("numpy", "scikit-learn", "alloygbm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def write_results(
    path: str | Path,
    records: Sequence[DroPerfRecord],
    arguments: dict[str, object],
    *,
    git_head: str | None = None,
) -> None:
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "git_head": _git_head() if git_head is None else git_head,
        "production_base": PRODUCTION_BASE,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "arguments": arguments,
        "records": [asdict(record) for record in _sorted_records(records)],
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_results(path: str | Path) -> list[DroPerfRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(f"unsupported DRO benchmark schema: {payload.get('schema_version')!r}")
    if payload.get("production_base") != PRODUCTION_BASE:
        raise ValueError("DRO benchmark production_base does not match PR #133")
    records = [DroPerfRecord(**record) for record in payload.get("records", [])]
    if not records:
        raise ValueError("DRO benchmark result contains no records")
    return records


def write_comparison(path: str | Path, summary: ComparisonSummary) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run_command(args: argparse.Namespace) -> int:
    records = run_matrix(args.arms, args.seeds, quick=args.quick)
    write_results(
        args.output,
        records,
        {"arms": list(args.arms), "seeds": list(args.seeds), "quick": args.quick},
    )
    return 0


def _compare_command(args: argparse.Namespace) -> int:
    baseline = read_results(args.baseline)
    candidate = read_results(args.candidate)
    summary = compare_results(baseline, candidate)
    write_comparison(args.output, summary)
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0 if summary.passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the paired fit-time matrix")
    run_parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    run_parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    run_parser.add_argument("--quick", action="store_true")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.set_defaults(handler=_run_command)

    compare_parser = subparsers.add_parser("compare", help="compare two result files")
    compare_parser.add_argument("baseline", type=Path)
    compare_parser.add_argument("candidate", type=Path)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.set_defaults(handler=_compare_command)

    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
