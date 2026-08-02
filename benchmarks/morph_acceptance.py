#!/usr/bin/env python3
"""Deterministic MorphBoost performance and quality acceptance matrix."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


RESULT_SCHEMA_VERSION = 1
PRODUCTION_BASE = "77dbf6d"
PRACTICAL_TIE = 0.001
MIN_MEAN_IMPROVEMENT = 0.0025
MAX_FAMILY_REGRESSION = 0.005
MAX_PAIR_REGRESSION = 0.03
MIN_WIN_OR_TIE_FRACTION = 0.60
MIN_BOOTSTRAP_LOW = -0.0025
BOOTSTRAP_RESAMPLES = 10_000


@dataclass(frozen=True)
class MorphBenchmarkRecord:
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


@dataclass(frozen=True)
class CandidateGate:
    candidate_arm: str
    passed: bool
    mean_improvement: float
    median_improvement: float
    win_or_tie_fraction: float
    bootstrap_low: float
    worst_paired_change: float
    family_means: dict[str, float]
    reasons: tuple[str, ...]


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
    lambda_l1: float = 0.0
    leaf_solver: str = "standard"


@dataclass(frozen=True)
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    group_train: list[int] | None = None
    group_test: list[int] | None = None
    group_ids_test: np.ndarray | None = None


ARMS: dict[str, dict[str, object]] = {
    "auto": {},
    "morph_current": {"training_mode": "morph"},
    "morph_no_balance": {"training_mode": "morph", "balance_penalty": False},
    "morph_info_005": {"training_mode": "morph", "info_score_weight": 0.05},
    "morph_info_0075": {"training_mode": "morph", "info_score_weight": 0.075},
    "morph_info_010": {"training_mode": "morph", "info_score_weight": 0.1},
    "morph_info_015": {"training_mode": "morph", "info_score_weight": 0.15},
}


def full_specs() -> tuple[FixtureSpec, ...]:
    return (
        FixtureSpec("reg-small-narrow", "small-narrow", "regression", 640, 8, 80, "linear"),
        FixtureSpec("reg-small-wide", "small-wide", "regression", 640, 128, 80, "sparse"),
        FixtureSpec("reg-tall-narrow", "tall-narrow", "regression", 8_192, 16, 80, "linear"),
        FixtureSpec("reg-tall-wide", "tall-wide", "regression", 8_192, 128, 60, "sparse"),
        FixtureSpec("reg-noisy-nonlinear", "medium", "regression", 2_048, 32, 100, "nonlinear"),
        FixtureSpec("binary-imbalanced", "medium", "binary", 4_096, 32, 100, "imbalanced"),
        FixtureSpec("multiclass-wide", "small-wide", "multiclass", 2_048, 96, 80, "multiclass"),
        FixtureSpec("rank-small-query", "tall-narrow", "ranking", 2_400, 24, 60, "ranking", 120),
        FixtureSpec("rank-large-query", "tall-narrow", "ranking", 4_096, 24, 60, "ranking", 8),
    )


def quick_specs() -> tuple[FixtureSpec, ...]:
    quick: list[FixtureSpec] = []
    for spec in full_specs():
        query_count = spec.query_count
        rows = min(spec.rows, 768)
        if spec.task_family == "ranking":
            query_count = min(query_count, 12)
            rows = max(query_count * 16, min(rows, spec.rows))
        quick.append(
            replace(
                spec,
                rows=rows,
                features=min(spec.features, 48),
                rounds=12,
                query_count=query_count,
            )
        )
    return tuple(quick)


def regularized_specs() -> tuple[FixtureSpec, ...]:
    base = full_specs()
    dro_names = {
        "reg-small-narrow",
        "binary-imbalanced",
        "multiclass-wide",
        "rank-small-query",
    }
    specs: list[FixtureSpec] = []
    for lambda_l1 in (0.1, 0.5):
        suffix = str(lambda_l1).replace(".", "")
        specs.extend(
            replace(
                spec,
                name=f"{spec.name}-l1-{suffix}",
                lambda_l1=lambda_l1,
            )
            for spec in base
        )
        specs.extend(
            replace(
                spec,
                name=f"{spec.name}-l1-{suffix}-dro",
                lambda_l1=lambda_l1,
                leaf_solver="dro",
            )
            for spec in base
            if spec.name in dro_names
        )
    return tuple(specs)


def normalized_improvement(control: float, candidate: float, higher_is_better: bool) -> float:
    denominator = max(abs(control), 1e-12)
    if higher_is_better:
        return (candidate - control) / denominator
    return (control - candidate) / denominator


def _record_key(record: MorphBenchmarkRecord) -> tuple[str, str, str, str, int, str]:
    return (
        record.arm,
        record.dataset,
        record.task_family,
        record.shape,
        record.seed,
        record.primary_metric,
    )


def merge_record_sets(*record_sets: Iterable[MorphBenchmarkRecord]) -> list[MorphBenchmarkRecord]:
    merged: list[MorphBenchmarkRecord] = []
    seen: set[tuple[str, str, str, str, int, str]] = set()
    for records in record_sets:
        for record in records:
            key = _record_key(record)
            if key in seen:
                raise ValueError(f"duplicate benchmark record: {key}")
            seen.add(key)
            merged.append(record)
    return merged


def relabel_arm(
    records: Iterable[MorphBenchmarkRecord], old: str, new: str
) -> list[MorphBenchmarkRecord]:
    return [replace(record, arm=new) if record.arm == old else record for record in records]


def _paired_improvements(
    records: Sequence[MorphBenchmarkRecord], control_arm: str, candidate_arm: str
) -> list[tuple[MorphBenchmarkRecord, MorphBenchmarkRecord, float]]:
    controls = {
        _record_key(record)[1:]: record for record in records if record.arm == control_arm
    }
    candidates = {
        _record_key(record)[1:]: record for record in records if record.arm == candidate_arm
    }
    if not controls or not candidates:
        raise ValueError("control and candidate records are both required")
    if controls.keys() != candidates.keys():
        missing_candidate = sorted(controls.keys() - candidates.keys())
        missing_control = sorted(candidates.keys() - controls.keys())
        raise ValueError(
            "benchmark pairs do not match: "
            f"missing_candidate={missing_candidate}, missing_control={missing_control}"
        )
    pairs = []
    for key in sorted(controls):
        control = controls[key]
        candidate = candidates[key]
        if control.higher_is_better != candidate.higher_is_better:
            raise ValueError(f"metric direction differs for pair {key}")
        values = [control.primary_value, candidate.primary_value, candidate.fit_seconds]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite benchmark value for pair {key}")
        improvement = normalized_improvement(
            control.primary_value, candidate.primary_value, control.higher_is_better
        )
        pairs.append((control, candidate, improvement))
    return pairs


def _equal_dataset_values(
    pairs: Sequence[tuple[MorphBenchmarkRecord, MorphBenchmarkRecord, float]],
) -> np.ndarray:
    by_dataset: dict[tuple[str, str, str], list[float]] = {}
    for control, _, improvement in pairs:
        key = (control.dataset, control.task_family, control.shape)
        by_dataset.setdefault(key, []).append(improvement)
    return np.asarray(
        [float(np.mean(by_dataset[key])) for key in sorted(by_dataset)], dtype=np.float64
    )


def _bootstrap_low(values: np.ndarray, seed: int) -> float:
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty benchmark")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_RESAMPLES, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025))


def evaluate_candidate(
    records: Sequence[MorphBenchmarkRecord],
    *,
    control_arm: str,
    candidate_arm: str,
    bootstrap_seed: int = 132,
) -> CandidateGate:
    pairs = _paired_improvements(records, control_arm, candidate_arm)
    improvements = np.asarray([pair[2] for pair in pairs], dtype=np.float64)
    dataset_values = _equal_dataset_values(pairs)
    family_values: dict[str, list[float]] = {}
    for control, _, improvement in pairs:
        family_values.setdefault(control.task_family, []).append(improvement)
    family_means = {
        family: float(np.mean(values)) for family, values in sorted(family_values.items())
    }
    mean_improvement = float(np.mean(dataset_values))
    median_improvement = float(np.median(improvements))
    win_or_tie_fraction = float(np.mean(improvements >= -PRACTICAL_TIE))
    bootstrap_low = _bootstrap_low(dataset_values, bootstrap_seed)
    worst_paired_change = float(np.min(improvements))
    reasons: list[str] = []
    if mean_improvement < MIN_MEAN_IMPROVEMENT:
        reasons.append(
            f"mean improvement {mean_improvement:.6f} is below {MIN_MEAN_IMPROVEMENT:.6f}"
        )
    if median_improvement < 0.0:
        reasons.append(f"median improvement {median_improvement:.6f} is negative")
    if win_or_tie_fraction < MIN_WIN_OR_TIE_FRACTION:
        reasons.append(
            f"win/tie fraction {win_or_tie_fraction:.3f} is below "
            f"{MIN_WIN_OR_TIE_FRACTION:.3f}"
        )
    for family, mean in family_means.items():
        if mean < -MAX_FAMILY_REGRESSION:
            reasons.append(
                f"task family {family} mean {mean:.6f} is below {-MAX_FAMILY_REGRESSION:.6f}"
            )
    if worst_paired_change < -MAX_PAIR_REGRESSION:
        reasons.append(
            f"worst paired change {worst_paired_change:.6f} is below {-MAX_PAIR_REGRESSION:.6f}"
        )
    if bootstrap_low <= MIN_BOOTSTRAP_LOW:
        reasons.append(
            f"bootstrap lower bound {bootstrap_low:.6f} is not above {MIN_BOOTSTRAP_LOW:.6f}"
        )
    return CandidateGate(
        candidate_arm=candidate_arm,
        passed=not reasons,
        mean_improvement=mean_improvement,
        median_improvement=median_improvement,
        win_or_tie_fraction=win_or_tie_fraction,
        bootstrap_low=bootstrap_low,
        worst_paired_change=worst_paired_change,
        family_means=family_means,
        reasons=tuple(reasons),
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
    elif spec.variant == "nonlinear":
        y = (
            1.5 * np.sin(X[:, 0])
            + X[:, 1] * X[:, 2]
            + 0.5 * X[:, 3] ** 2
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
    train_query_count = max(1, int(spec.query_count * 0.8))
    test_query_count = max(1, spec.query_count - train_query_count)
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
            denominator = max(size - 1, 1)
            y[offset : offset + size] = np.floor(4.99 * order / denominator)
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


def _log_loss(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(probabilities[np.arange(len(y_true)), y_true.astype(int)], 1e-15, 1.0)
    return float(-np.mean(np.log(clipped)))


def _rotated_arms(arms: Sequence[str], case_index: int, seed: int) -> list[str]:
    if not arms:
        return []
    offset = (case_index + seed) % len(arms)
    return [*arms[offset:], *arms[:offset]]


def _fit_record(
    spec: FixtureSpec, bundle: DatasetBundle, arm: str, seed: int
) -> MorphBenchmarkRecord:
    from alloygbm import GBMClassifier, GBMRanker, GBMRegressor
    from alloygbm.evaluation import ndcg

    common: dict[str, object] = {
        "n_estimators": spec.rounds,
        "max_depth": 5,
        "learning_rate": 0.05,
        "seed": seed,
        "deterministic": True,
        "n_jobs": 1,
        "lambda_l1": spec.lambda_l1,
        "leaf_solver": spec.leaf_solver,
        "dro_radius": 0.05,
        **ARMS[arm],
    }
    started = time.perf_counter()
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
        primary_metric = "log_loss"
        primary_value = _log_loss(bundle.y_test, probabilities)
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
    elapsed = time.perf_counter() - started
    return MorphBenchmarkRecord(
        arm=arm,
        dataset=spec.name,
        task_family=spec.task_family,
        shape=spec.shape,
        seed=seed,
        primary_metric=primary_metric,
        primary_value=primary_value,
        higher_is_better=higher_is_better,
        secondary_metrics=secondary,
        fit_seconds=elapsed,
    )


def run_matrix(
    arms: Sequence[str],
    seeds: Sequence[int],
    *,
    quick: bool = False,
    profile: str = "default",
) -> list[MorphBenchmarkRecord]:
    unknown = sorted(set(arms) - ARMS.keys())
    if unknown:
        raise ValueError(f"unknown benchmark arms: {unknown}")
    if profile == "regularized":
        specs = regularized_specs()
        if quick:
            specs = tuple(
                replace(spec, rows=min(spec.rows, 768), features=min(spec.features, 48), rounds=12)
                for spec in specs
            )
    elif profile == "default":
        specs = quick_specs() if quick else full_specs()
    else:
        raise ValueError(f"unknown benchmark profile: {profile}")
    records: list[MorphBenchmarkRecord] = []
    for case_index, spec in enumerate(specs):
        for seed in seeds:
            bundle = make_dataset(spec, seed)
            for arm in _rotated_arms(list(arms), case_index, seed):
                record = _fit_record(spec, bundle, arm, seed)
                records.append(record)
                print(
                    f"{spec.name} seed={seed} arm={arm} "
                    f"{record.primary_metric}={record.primary_value:.6f} "
                    f"fit_seconds={record.fit_seconds:.6f}",
                    flush=True,
                )
    return records


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_results(
    path: str | Path, records: Sequence[MorphBenchmarkRecord], arguments: dict[str, object]
) -> None:
    from alloygbm import __version__

    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "git_head": _git_head(),
        "production_base": PRODUCTION_BASE,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "alloygbm": __version__,
        "arguments": arguments,
        "records": [asdict(record) for record in records],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_results(path: str | Path) -> list[MorphBenchmarkRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(f"unsupported Morph benchmark schema: {payload.get('schema_version')!r}")
    return [MorphBenchmarkRecord(**record) for record in payload["records"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["auto", "morph_current"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--profile", choices=["default", "regularized"], default="default")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = run_matrix(args.arms, args.seeds, quick=args.quick, profile=args.profile)
    write_results(
        args.output,
        records,
        {
            "arms": args.arms,
            "seeds": args.seeds,
            "quick": args.quick,
            "profile": args.profile,
        },
    )
    if "morph_current" in args.arms and "auto" in args.arms:
        gate = evaluate_candidate(
            records, control_arm="auto", candidate_arm="morph_current"
        )
        print(json.dumps(asdict(gate), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
