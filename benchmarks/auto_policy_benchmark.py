#!/usr/bin/env python3
"""Deterministic shape-matrix benchmark for AlloyGBM's auto training policy."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Iterator, Sequence

import numpy as np


ARMS = ("current_auto", "manual_default", "no_gain_floor", "quality_first")
SEEDS = (7, 13, 29)
OBJECTIVES = (
    "regression",
    "sparse_regression",
    "binary",
    "multiclass",
    "ranking",
)
SPLIT_L2_ENV_VAR = "ALLOYGBM_EXPERIMENT_SPLIT_L2"
POLICY_KEYS = {
    "requested_mode",
    "requested_rounds",
    "effective_round_cap",
    "min_rows_per_leaf",
    "min_split_gain",
    "row_subsample",
    "col_subsample",
    "auto_split_l2_applied",
    "effective_split_l2",
}
BEHAVIORAL_DISTANCE = {
    "no_gain_floor": 1,
    "quality_first": 3,
    "manual_default": 4,
}
_FULL_SHAPES = (
    (512, 8),
    (1_023, 16),
    (512, 128),
    (1_023, 256),
    (2_048, 16),
    (8_192, 16),
    (2_048, 128),
    (8_192, 256),
    (16_384, 16),
    (16_384, 256),
)
_QUICK_CASES = (
    (512, 8, "regression"),
    (512, 128, "sparse_regression"),
    (2_048, 16, "binary"),
    (2_048, 128, "multiclass"),
    (16_384, 16, "ranking"),
    (16_384, 128, "regression"),
)


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    shape_stratum: str
    objective: str
    rows: int
    features: int
    rounds: int


@dataclass(frozen=True)
class BenchmarkRecord:
    fixture: str
    shape_stratum: str
    objective: str
    seed: int
    arm: str
    primary_metric: float
    accuracy: float | None
    ndcg_at_10: float | None
    completed_rounds: int
    fit_seconds: float
    resolved_policy: dict[str, object]
    error: str | None = None


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str
    selected_arm: str | None = None
    overall_loss_ratio: float | None = None
    evidence_valid: bool = True


@dataclass(frozen=True)
class BenchmarkFixture:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    group_train: np.ndarray | None
    group_test: np.ndarray | None
    target_signal: float

    def arrays(self) -> tuple[np.ndarray, ...]:
        arrays = (self.X_train, self.y_train, self.X_test, self.y_test)
        groups = tuple(
            value
            for value in (self.group_train, self.group_test)
            if value is not None
        )
        return (*arrays, *groups)


def classify_shape(rows: int, features: int) -> str:
    """Map a training shape to one of the six protected strata."""
    if rows < 2_048:
        size = "small"
    elif rows < 16_384:
        size = "medium"
    else:
        size = "large"
    width = "wide" if features >= 128 else "narrow"
    return f"{size}-{width}"


def _fixture_name(rows: int, features: int, objective: str) -> str:
    return f"{classify_shape(rows, features)}-{rows}x{features}-{objective}"


def full_specs() -> tuple[FixtureSpec, ...]:
    """Return the full ten-shape by five-objective calibration matrix."""
    specs = []
    for rows, features in _FULL_SHAPES:
        stratum = classify_shape(rows, features)
        rounds = 300 if stratum == "small-wide" else 40
        for objective in OBJECTIVES:
            specs.append(
                FixtureSpec(
                    name=_fixture_name(rows, features, objective),
                    shape_stratum=stratum,
                    objective=objective,
                    rows=rows,
                    features=features,
                    rounds=rounds,
                )
            )
    return tuple(specs)


def quick_specs() -> tuple[FixtureSpec, ...]:
    """Return the six-stratum, five-objective CI sentinel matrix."""
    return tuple(
        FixtureSpec(
            name=f"quick-{_fixture_name(rows, features, objective)}",
            shape_stratum=classify_shape(rows, features),
            objective=objective,
            rows=rows,
            features=features,
            rounds=8,
        )
        for rows, features, objective in _QUICK_CASES
    )


def _dense_features(
    rng: np.random.Generator, rows: int, features: int
) -> np.ndarray:
    return np.ascontiguousarray(
        rng.standard_normal((rows, features), dtype=np.float32)
    )


def _sparse_features(
    rng: np.random.Generator, rows: int, features: int
) -> np.ndarray:
    values = rng.standard_normal((rows, features), dtype=np.float32)
    mask = rng.random((rows, features)) < 0.06
    return np.ascontiguousarray(np.where(mask, values, 0.0), dtype=np.float32)


def _regression_target(
    rng: np.random.Generator, X: np.ndarray, *, sparse: bool
) -> tuple[np.ndarray, float]:
    x0 = X[:, 0]
    x1 = X[:, min(1, X.shape[1] - 1)]
    x2 = X[:, min(2, X.shape[1] - 1)]
    x3 = X[:, min(3, X.shape[1] - 1)]
    if sparse:
        latent = 8.0 * x0 - 6.0 * x1 + 5.0 * x2 + 4.0 * x0 * x3
        noise_scale = 0.8 + 0.4 * np.abs(x1)
    else:
        latent = (
            3.0 * x0
            - 2.0 * x1
            + 1.5 * np.sin(x2)
            + 1.25 * x0 * x3
        )
        noise_scale = 0.45 + 0.25 * np.abs(x2)
    target = latent + rng.normal(size=len(X)) * noise_scale
    signal = abs(float(np.corrcoef(latent, target)[0, 1]))
    return np.ascontiguousarray(target, dtype=np.float32), signal


def _binary_target(
    rng: np.random.Generator, X: np.ndarray
) -> tuple[np.ndarray, float]:
    logits = (
        1.8 * X[:, 0]
        - 1.2 * X[:, min(1, X.shape[1] - 1)]
        + 0.9 * X[:, 0] * X[:, min(2, X.shape[1] - 1)]
        + 0.6 * np.sin(X[:, min(3, X.shape[1] - 1)])
    )
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
    target = (rng.random(len(X)) < probability).astype(np.float32)
    if np.unique(target).size < 2:
        target[:2] = (0.0, 1.0)
    signal = abs(float(np.corrcoef(logits, target)[0, 1]))
    return np.ascontiguousarray(target), signal


def _multiclass_target(
    rng: np.random.Generator, X: np.ndarray
) -> tuple[np.ndarray, float]:
    x0 = X[:, 0]
    x1 = X[:, min(1, X.shape[1] - 1)]
    x2 = X[:, min(2, X.shape[1] - 1)]
    x3 = X[:, min(3, X.shape[1] - 1)]
    logits = np.column_stack(
        (
            1.8 * x0 - 0.7 * x1,
            -1.4 * x0 + 1.2 * x2,
            1.1 * x1 - 1.3 * x3 + 0.6 * x0 * x2,
            -0.8 * x1 + 1.0 * x3 - 0.5 * x0 * x2,
        )
    )
    logits -= np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    draws = rng.random(len(X))
    target = (draws[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1)
    target = np.minimum(target, probabilities.shape[1] - 1).astype(np.float32)
    if np.unique(target).size < probabilities.shape[1]:
        target[: probabilities.shape[1]] = np.arange(
            probabilities.shape[1], dtype=np.float32
        )
    oracle_accuracy = float(np.mean(np.argmax(probabilities, axis=1) == target))
    signal = oracle_accuracy - (1.0 / probabilities.shape[1])
    return np.ascontiguousarray(target), signal


def _group_ids(rows: int, query_size: int = 32) -> np.ndarray:
    query_count = max(1, rows // query_size)
    sizes = np.full(query_count, rows // query_count, dtype=np.int32)
    sizes[: rows % query_count] += 1
    return np.repeat(np.arange(query_count, dtype=np.int32), sizes)


def _ranking_partition(
    rng: np.random.Generator, rows: int, features: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    X = _dense_features(rng, rows, features)
    groups = _group_ids(rows)
    target = np.empty(rows, dtype=np.float32)
    latent_all = np.empty(rows, dtype=np.float32)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        local = X[indices]
        query_bias = rng.normal(scale=0.5)
        latent = (
            1.8 * local[:, 0]
            - 1.1 * local[:, min(1, features - 1)]
            + 0.8 * local[:, 0] * local[:, min(2, features - 1)]
            + 0.5 * np.sin(local[:, min(3, features - 1)])
            + query_bias
            + rng.normal(scale=0.45, size=len(indices))
        )
        order = np.empty(len(indices), dtype=np.intp)
        order[np.argsort(latent, kind="mergesort")] = np.arange(len(indices))
        target[indices] = np.minimum(4, (order * 5) // len(indices))
        latent_all[indices] = latent
    signal = abs(float(np.corrcoef(latent_all, target)[0, 1]))
    return X, np.ascontiguousarray(target), groups, signal


def make_fixture(spec: FixtureSpec, seed: int) -> BenchmarkFixture:
    """Generate independent deterministic train and held-out partitions."""
    if spec.objective not in OBJECTIVES:
        raise ValueError(f"unsupported objective: {spec.objective}")
    if spec.rows < 2 or spec.features < 1:
        raise ValueError("fixture rows and features must be positive")

    rng = np.random.default_rng(seed)
    test_rows = min(1_024, max(128, spec.rows // 4))
    if spec.objective == "ranking":
        X_train, y_train, group_train, train_signal = _ranking_partition(
            rng, spec.rows, spec.features
        )
        X_test, y_test, group_test, test_signal = _ranking_partition(
            rng, test_rows, spec.features
        )
    else:
        sparse = spec.objective == "sparse_regression"
        feature_builder = _sparse_features if sparse else _dense_features
        X_train = feature_builder(rng, spec.rows, spec.features)
        X_test = feature_builder(rng, test_rows, spec.features)
        if spec.objective in {"regression", "sparse_regression"}:
            y_train, train_signal = _regression_target(rng, X_train, sparse=sparse)
            y_test, test_signal = _regression_target(rng, X_test, sparse=sparse)
        elif spec.objective == "binary":
            y_train, train_signal = _binary_target(rng, X_train)
            y_test, test_signal = _binary_target(rng, X_test)
        else:
            y_train, train_signal = _multiclass_target(rng, X_train)
            y_test, test_signal = _multiclass_target(rng, X_test)
        group_train = None
        group_test = None

    return BenchmarkFixture(
        X_train=np.ascontiguousarray(X_train, dtype=np.float32),
        y_train=np.ascontiguousarray(y_train, dtype=np.float32),
        X_test=np.ascontiguousarray(X_test, dtype=np.float32),
        y_test=np.ascontiguousarray(y_test, dtype=np.float32),
        group_train=group_train,
        group_test=group_test,
        target_signal=min(train_signal, test_signal),
    )


def _validated_policy_fields(
    resolved: object,
    *,
    expected_mode: str,
    context: str,
) -> dict[str, object]:
    if not isinstance(resolved, dict):
        raise ValueError(f"{context}: resolved_training_policy_ must be a dictionary")
    missing = POLICY_KEYS - set(resolved)
    if missing:
        raise ValueError(
            f"{context}: resolved_training_policy_ missing keys: {sorted(missing)}"
        )
    if resolved["requested_mode"] != expected_mode:
        raise ValueError(f"{context}: requested_mode must be {expected_mode!r}")
    for key in ("requested_rounds", "effective_round_cap", "min_rows_per_leaf"):
        value = resolved[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{context}: {key} must be an integer")
        if int(value) <= 0:
            raise ValueError(f"{context}: {key} must be greater than zero")
    for key in (
        "min_split_gain",
        "row_subsample",
        "col_subsample",
        "effective_split_l2",
    ):
        value = resolved[key]
        if not isinstance(value, (float, np.floating)):
            raise ValueError(f"{context}: {key} must be a floating-point value")
        if not math.isfinite(float(value)):
            raise ValueError(f"{context}: {key} must be finite")
    if float(resolved["min_split_gain"]) < 0.0:
        raise ValueError(f"{context}: min_split_gain must be non-negative")
    for key in ("row_subsample", "col_subsample"):
        if not 0.0 < float(resolved[key]) <= 1.0:
            raise ValueError(f"{context}: {key} must be in (0, 1]")
    if float(resolved["effective_split_l2"]) < 0.0:
        raise ValueError(f"{context}: effective_split_l2 must be non-negative")
    if not isinstance(resolved["auto_split_l2_applied"], (bool, np.bool_)):
        raise ValueError(f"{context}: auto_split_l2_applied must be boolean")
    return dict(resolved)


def _validated_current_policy(
    resolved: object, *, context: str = "current_auto"
) -> dict[str, object]:
    return _validated_policy_fields(
        resolved,
        expected_mode="auto",
        context=context,
    )


def derive_candidate_params(
    arm: str, resolved_policy: dict[str, object]
) -> dict[str, object]:
    """Derive one manual arm from the exact current-auto diagnostic."""
    resolved = _validated_current_policy(resolved_policy)
    if arm == "manual_default":
        return {
            "training_policy": "manual",
            "n_estimators": int(resolved["requested_rounds"]),
        }
    if arm not in {"no_gain_floor", "quality_first"}:
        raise ValueError(f"unsupported candidate arm: {arm}")
    return {
        "training_policy": "manual",
        "n_estimators": int(resolved["effective_round_cap"]),
        "min_data_in_leaf": int(resolved["min_rows_per_leaf"]),
        "min_split_gain": 0.0,
        "row_subsample": (
            1.0 if arm == "quality_first" else float(resolved["row_subsample"])
        ),
        "col_subsample": (
            1.0 if arm == "quality_first" else float(resolved["col_subsample"])
        ),
    }


@contextmanager
def temporary_split_l2(value: float | None) -> Iterator[None]:
    """Temporarily set split-only L2 and restore the exact prior environment."""
    previous = os.environ.get(SPLIT_L2_ENV_VAR)
    try:
        if value is None:
            os.environ.pop(SPLIT_L2_ENV_VAR, None)
        else:
            os.environ[SPLIT_L2_ENV_VAR] = str(float(value))
        yield
    finally:
        if previous is None:
            os.environ.pop(SPLIT_L2_ENV_VAR, None)
        else:
            os.environ[SPLIT_L2_ENV_VAR] = previous


def _manual_split_l2(
    arm: str, current_policy: dict[str, object]
) -> float | None:
    if arm in {"no_gain_floor", "quality_first"} and bool(
        current_policy["auto_split_l2_applied"]
    ):
        return float(current_policy["effective_split_l2"])
    return None


def _common_estimator_params(spec: FixtureSpec, seed: int) -> dict[str, object]:
    return {
        "n_estimators": spec.rounds,
        "learning_rate": 0.08,
        "max_depth": 5,
        "deterministic": True,
        "seed": seed,
    }


def _validate_manual_policy(
    resolved: object,
    *,
    params: dict[str, object],
    split_l2: float | None,
    context: str,
) -> dict[str, object]:
    validated = _validated_policy_fields(
        resolved,
        expected_mode="manual",
        context=context,
    )
    expected = {
        "requested_mode": "manual",
        "requested_rounds": int(params["n_estimators"]),
        "effective_round_cap": int(params["n_estimators"]),
        "min_rows_per_leaf": int(params.get("min_data_in_leaf", 1)),
        "min_split_gain": float(params.get("min_split_gain", 0.0)),
        "row_subsample": float(params.get("row_subsample", 1.0)),
        "col_subsample": float(params.get("col_subsample", 1.0)),
        "auto_split_l2_applied": False,
        "effective_split_l2": 0.0 if split_l2 is None else float(split_l2),
    }
    for key, expected_value in expected.items():
        actual = validated[key]
        matches = (
            math.isclose(float(actual), float(expected_value), rel_tol=1e-6, abs_tol=1e-7)
            if isinstance(expected_value, float)
            else actual == expected_value
        )
        if not matches:
            raise ValueError(
                f"{context}: resolved {key}={actual!r}, expected {expected_value!r}"
            )
    return validated


def _fit_arm(
    spec: FixtureSpec,
    fixture: BenchmarkFixture,
    *,
    seed: int,
    arm: str,
    current_policy: dict[str, object] | None,
) -> BenchmarkRecord:
    from alloygbm import (
        GBMClassifier,
        GBMRanker,
        GBMRegressor,
        accuracy,
        log_loss,
        multiclass_log_loss,
        ndcg,
        rmse,
    )

    params = _common_estimator_params(spec, seed)
    split_l2 = None
    if arm == "current_auto":
        params["training_policy"] = "auto"
    else:
        if current_policy is None:
            raise ValueError("candidate arm requires current-auto diagnostics")
        params.update(derive_candidate_params(arm, current_policy))
        split_l2 = _manual_split_l2(arm, current_policy)

    if spec.objective in {"regression", "sparse_regression"}:
        estimator = GBMRegressor(**params)
    elif spec.objective in {"binary", "multiclass"}:
        estimator = GBMClassifier(**params)
    else:
        estimator = GBMRanker(ranking_objective="rank:ndcg", **params)

    started = time.perf_counter()
    with temporary_split_l2(split_l2):
        if spec.objective == "ranking":
            estimator.fit(
                fixture.X_train,
                fixture.y_train,
                group=fixture.group_train,
            )
        else:
            estimator.fit(fixture.X_train, fixture.y_train)
    fit_seconds = time.perf_counter() - started

    context = f"{spec.name} seed={seed} arm={arm}"
    if arm == "current_auto":
        resolved = _validated_current_policy(
            estimator.resolved_training_policy_, context=context
        )
    else:
        resolved = _validate_manual_policy(
            estimator.resolved_training_policy_,
            params=params,
            split_l2=split_l2,
            context=context,
        )

    accuracy_value = None
    ndcg_value = None
    if spec.objective in {"regression", "sparse_regression"}:
        prediction = estimator.predict(fixture.X_test)
        primary_metric = rmse(fixture.y_test, prediction)
    elif spec.objective == "binary":
        probability = estimator.predict_proba(fixture.X_test)
        prediction = estimator.predict(fixture.X_test)
        primary_metric = log_loss(fixture.y_test, probability[:, 1])
        accuracy_value = accuracy(fixture.y_test, prediction)
    elif spec.objective == "multiclass":
        probability = estimator.predict_proba(fixture.X_test)
        prediction = estimator.predict(fixture.X_test)
        primary_metric = multiclass_log_loss(fixture.y_test, probability)
        accuracy_value = accuracy(fixture.y_test, prediction)
    else:
        prediction = estimator.predict(fixture.X_test)
        ndcg_value = ndcg(
            fixture.y_test,
            prediction,
            group=fixture.group_test,
            k=10,
        )
        primary_metric = 1.0 - ndcg_value

    return BenchmarkRecord(
        fixture=spec.name,
        shape_stratum=spec.shape_stratum,
        objective=spec.objective,
        seed=seed,
        arm=arm,
        primary_metric=float(primary_metric),
        accuracy=None if accuracy_value is None else float(accuracy_value),
        ndcg_at_10=None if ndcg_value is None else float(ndcg_value),
        completed_rounds=int(estimator.rounds_completed_),
        fit_seconds=float(fit_seconds),
        resolved_policy=resolved,
    )


def _error_record(
    spec: FixtureSpec, seed: int, arm: str, error: BaseException | str
) -> BenchmarkRecord:
    return BenchmarkRecord(
        fixture=spec.name,
        shape_stratum=spec.shape_stratum,
        objective=spec.objective,
        seed=seed,
        arm=arm,
        primary_metric=float("nan"),
        accuracy=None,
        ndcg_at_10=None,
        completed_rounds=0,
        fit_seconds=0.0,
        resolved_policy={},
        error=str(error),
    )


def run_matrix(
    specs: Sequence[FixtureSpec] | None = None,
    *,
    seeds: Sequence[int] = SEEDS,
    quick: bool = False,
) -> list[BenchmarkRecord]:
    """Run every arm serially, with current auto first for each fixture/seed."""
    selected_specs = tuple(specs) if specs is not None else (
        quick_specs() if quick else full_specs()
    )
    records: list[BenchmarkRecord] = []
    for spec in selected_specs:
        if spec.shape_stratum != classify_shape(spec.rows, spec.features):
            raise ValueError(
                f"{spec.name}: declared stratum {spec.shape_stratum!r} does not "
                f"match shape {spec.rows}x{spec.features}"
            )
        for seed in seeds:
            fixture = make_fixture(spec, int(seed))
            current_policy = None
            try:
                current = _fit_arm(
                    spec,
                    fixture,
                    seed=int(seed),
                    arm="current_auto",
                    current_policy=None,
                )
                records.append(current)
                current_policy = current.resolved_policy
            except Exception as exc:  # benchmark evidence records fit failures
                records.append(_error_record(spec, int(seed), "current_auto", exc))
            for arm in ARMS[1:]:
                if current_policy is None:
                    records.append(
                        _error_record(
                            spec,
                            int(seed),
                            arm,
                            "current_auto prerequisite failed",
                        )
                    )
                    continue
                try:
                    records.append(
                        _fit_arm(
                            spec,
                            fixture,
                            seed=int(seed),
                            arm=arm,
                            current_policy=current_policy,
                        )
                    )
                except Exception as exc:  # benchmark evidence records fit failures
                    records.append(_error_record(spec, int(seed), arm, exc))
    return records


def _record_key(record: BenchmarkRecord) -> tuple[str, str, str, int]:
    return (
        record.fixture,
        record.shape_stratum,
        record.objective,
        record.seed,
    )


def _record_issue(record: BenchmarkRecord) -> str | None:
    context = f"{record.fixture} seed={record.seed} arm={record.arm}"
    if record.error is not None:
        return f"{context}: {record.error}"
    if record.completed_rounds <= 0:
        return f"{context}: completed_rounds must be greater than zero"
    for field in ("primary_metric", "fit_seconds"):
        if not math.isfinite(float(getattr(record, field))):
            return f"{context}: {field} must be finite"
    if record.objective in {"binary", "multiclass"}:
        if record.accuracy is None or not math.isfinite(record.accuracy):
            return f"{context}: accuracy must be finite"
    if record.objective == "ranking":
        if record.ndcg_at_10 is None or not math.isfinite(record.ndcg_at_10):
            return f"{context}: ndcg_at_10 must be finite"
    return None


def _loss_ratio(candidate: float, current: float) -> float:
    if current == 0.0:
        return 1.0 if candidate <= 0.0 else math.inf
    return candidate / current


def evaluate_candidate(
    records: Sequence[BenchmarkRecord], candidate_arm: str
) -> GateResult:
    """Apply protected-stratum and aggregate quality gates to one candidate."""
    if candidate_arm == "current_auto":
        raise ValueError("candidate_arm must not be current_auto")
    current = {
        _record_key(row): row for row in records if row.arm == "current_auto"
    }
    candidate = {
        _record_key(row): row for row in records if row.arm == candidate_arm
    }
    invalid: list[str] = []
    for row in (*current.values(), *candidate.values()):
        issue = _record_issue(row)
        if issue is not None:
            invalid.append(issue)
    missing_current = sorted(set(candidate) - set(current))
    missing_candidate = sorted(set(current) - set(candidate))
    invalid.extend(f"{candidate_arm}: missing current_auto pair for {key}" for key in missing_current)
    invalid.extend(f"{candidate_arm}: missing candidate pair for {key}" for key in missing_candidate)
    if invalid:
        return GateResult(
            name=candidate_arm,
            passed=False,
            detail="\n".join(invalid),
            evidence_valid=False,
        )
    if not candidate:
        return GateResult(
            name=candidate_arm,
            passed=False,
            detail=f"{candidate_arm}: no records",
            evidence_valid=False,
        )

    group_ratios: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    group_accuracy_delta: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    group_ndcg_delta: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for key, candidate_row in candidate.items():
        current_row = current[key]
        group = (
            candidate_row.shape_stratum,
            candidate_row.objective,
            candidate_row.seed,
        )
        group_ratios[group].append(
            _loss_ratio(candidate_row.primary_metric, current_row.primary_metric)
        )
        if candidate_row.accuracy is not None and current_row.accuracy is not None:
            group_accuracy_delta[group].append(
                candidate_row.accuracy - current_row.accuracy
            )
        if (
            candidate_row.ndcg_at_10 is not None
            and current_row.ndcg_at_10 is not None
        ):
            group_ndcg_delta[group].append(
                candidate_row.ndcg_at_10 - current_row.ndcg_at_10
            )

    normalized = {key: median(values) for key, values in group_ratios.items()}
    reasons: list[str] = []
    for (stratum, objective, seed), ratio in sorted(normalized.items()):
        if objective != "ranking" and ratio > 1.03 + 1e-12:
            reasons.append(
                f"{stratum}/{objective}/seed={seed} exceeds the 3% "
                f"primary-loss limit ({ratio:.4f}x)"
            )
    for (stratum, objective, seed), values in sorted(group_accuracy_delta.items()):
        delta = median(values)
        if delta < -0.02 - 1e-12:
            reasons.append(
                f"{stratum}/{objective}/seed={seed} accuracy drops by "
                f"{abs(delta):.4f}"
            )
    for (stratum, objective, seed), values in sorted(group_ndcg_delta.items()):
        delta = median(values)
        if delta < -0.02 - 1e-12:
            reasons.append(
                f"{stratum}/{objective}/seed={seed} NDCG@10 drops by "
                f"{abs(delta):.4f}"
            )

    shape_values: dict[str, list[float]] = defaultdict(list)
    for (stratum, _objective, _seed), ratio in normalized.items():
        shape_values[stratum].append(ratio)
    for stratum, ratios in sorted(shape_values.items()):
        shape_ratio = median(ratios)
        if shape_ratio > 1.0 + 1e-12:
            reasons.append(
                f"{stratum} shape median is worse than current auto "
                f"({shape_ratio:.4f}x)"
            )

    overall_ratio = median(normalized.values())
    if overall_ratio > 0.99 + 1e-12:
        reasons.append(
            "overall median primary loss does not reach the required 1% "
            f"improvement ({overall_ratio:.4f}x)"
        )
    return GateResult(
        name=candidate_arm,
        passed=not reasons,
        detail=(
            f"{candidate_arm} qualifies at {overall_ratio:.4f}x current-auto loss"
            if not reasons
            else "\n".join(reasons)
        ),
        selected_arm=candidate_arm if not reasons else None,
        overall_loss_ratio=overall_ratio,
    )


def _expected_record_keys(
    specs: Sequence[FixtureSpec], seeds: Sequence[int]
) -> set[tuple[str, str, str, int, str]]:
    return {
        (spec.name, spec.shape_stratum, spec.objective, int(seed), arm)
        for spec in specs
        for seed in seeds
        for arm in ARMS
    }


def _format_matrix_key(key: tuple[str, str, str, int, str]) -> str:
    fixture, _stratum, _objective, seed, arm = key
    return f"{fixture} seed={seed} arm={arm}"


def evaluate_gates(
    records: Sequence[BenchmarkRecord],
    *,
    specs: Sequence[FixtureSpec],
    seeds: Sequence[int],
) -> GateResult:
    """Validate a declared matrix and select a candidate or current auto."""
    if specs is None or seeds is None:
        raise ValueError("specs and seeds must declare the expected matrix")
    observed = [
        (
            row.fixture,
            row.shape_stratum,
            row.objective,
            row.seed,
            row.arm,
        )
        for row in records
    ]
    expected = _expected_record_keys(specs, seeds)
    observed_set = set(observed)
    missing = sorted(expected - observed_set)
    unexpected = sorted(observed_set - expected)
    duplicates = sorted(key for key, count in Counter(observed).items() if count > 1)
    if missing or unexpected or duplicates:
        details = [
            *[f"missing {_format_matrix_key(key)}" for key in missing],
            *[f"unexpected {_format_matrix_key(key)}" for key in unexpected],
            *[f"duplicate {_format_matrix_key(key)}" for key in duplicates],
        ]
        return GateResult(
            name="matrix",
            passed=False,
            detail="\n".join(details),
            evidence_valid=False,
        )

    current_rows = [row for row in records if row.arm == "current_auto"]
    current_issues = [
        issue for row in current_rows if (issue := _record_issue(row)) is not None
    ]
    if not current_rows:
        current_issues.append("no current_auto records")
    if current_issues:
        return GateResult(
            name="current_auto",
            passed=False,
            detail="\n".join(current_issues),
            evidence_valid=False,
        )

    candidate_arms = [arm for arm in ARMS[1:] if any(row.arm == arm for row in records)]
    results = [evaluate_candidate(records, arm) for arm in candidate_arms]
    invalid = [result for result in results if not result.evidence_valid]
    if invalid:
        return GateResult(
            name="candidate_evidence",
            passed=False,
            detail="\n".join(
                f"{result.name}: {result.detail}" for result in invalid
            ),
            evidence_valid=False,
        )

    quality_qualified = [result for result in results if result.passed]
    rejection_detail = "\n".join(
        f"{result.name} rejected: {result.detail}"
        for result in results
        if not result.passed
    )
    if not quality_qualified:
        detail = "keep current: no candidate meets every quality gate"
        if rejection_detail:
            detail = f"{detail}\n{rejection_detail}"
        return GateResult(
            name="selection",
            passed=True,
            detail=detail,
            selected_arm="current_auto",
        )

    best_ratio = min(
        result.overall_loss_ratio
        for result in quality_qualified
        if result.overall_loss_ratio is not None
    )
    quality_equivalent = [
        result
        for result in quality_qualified
        if result.overall_loss_ratio is not None
        and result.overall_loss_ratio <= best_ratio + 0.005 + 1e-12
    ]
    median_fit_seconds_for_final_tie_break = {
        arm: median(row.fit_seconds for row in records if row.arm == arm)
        for arm in candidate_arms
    }
    # Timing is descriptive until this final ordering: every remaining arm has
    # passed all quality gates and lies inside the 0.5% quality-equivalence band.
    selected = min(
        quality_equivalent,
        key=lambda result: (
            BEHAVIORAL_DISTANCE[result.name],
            median_fit_seconds_for_final_tie_break[result.name],
            result.name,
        ),
    )
    detail = f"selected {selected.name}: {selected.detail}"
    if rejection_detail:
        detail = f"{detail}\n{rejection_detail}"
    return GateResult(
        name="selection",
        passed=True,
        detail=detail,
        selected_arm=selected.name,
        overall_loss_ratio=selected.overall_loss_ratio,
    )


def write_json(
    path: Path,
    records: Sequence[BenchmarkRecord],
    gate: GateResult,
) -> None:
    """Write machine-readable records and gate outcome."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gate": asdict(gate),
        "records": [asdict(record) for record in records],
    }
    path.write_text(
        json.dumps(
            _json_compatible(payload),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _json_compatible(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _format_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _command_output(command: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip() or "unavailable"


def _environment_lines(*, source_commit: str | None = None) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        from alloygbm import __version__ as alloygbm_version
    except ImportError:
        alloygbm_version = "unavailable"
    git_commit = source_commit or _command_output(
        ("git", "rev-parse", "HEAD"), cwd=repo_root
    )
    return [
        f"- Git commit: `{git_commit}`",
        f"- OS/platform: `{platform.platform()}`",
        f"- Architecture: `{platform.machine()}`",
        f"- Python: `{platform.python_version()}`",
        f"- Rust: `{_command_output(('rustc', '--version'))}`",
        f"- NumPy: `{np.__version__}`",
        f"- AlloyGBM: `{alloygbm_version}`",
    ]


def _candidate_gate_results(
    records: Sequence[BenchmarkRecord],
) -> list[GateResult]:
    return [evaluate_candidate(records, arm) for arm in ARMS[1:]]


def _shape_objective_ratios(
    records: Sequence[BenchmarkRecord],
    specs: Sequence[FixtureSpec],
) -> list[tuple[str, int, int, str, float]]:
    current = {
        _record_key(row): row for row in records if row.arm == "current_auto"
    }
    specs_by_record_identity = {
        (spec.name, spec.shape_stratum, spec.objective): spec for spec in specs
    }
    grouped: dict[tuple[str, int, int, str], list[float]] = defaultdict(list)
    for row in records:
        if row.arm == "current_auto":
            continue
        current_row = current.get(_record_key(row))
        if current_row is None:
            continue
        if _record_issue(row) is not None or _record_issue(current_row) is not None:
            continue
        spec = specs_by_record_identity[
            (row.fixture, row.shape_stratum, row.objective)
        ]
        grouped[(row.arm, spec.rows, spec.features, row.objective)].append(
            _loss_ratio(row.primary_metric, current_row.primary_metric)
        )
    return [(*key, median(values)) for key, values in sorted(grouped.items())]


def _policy_observation_lines(
    records: Sequence[BenchmarkRecord],
) -> list[str]:
    current_policies = [
        row.resolved_policy
        for row in records
        if row.arm == "current_auto"
        and "auto_split_l2_applied" in row.resolved_policy
        and "effective_split_l2" in row.resolved_policy
    ]
    activation_count = sum(
        bool(policy["auto_split_l2_applied"]) for policy in current_policies
    )
    effective_values = sorted(
        {float(policy["effective_split_l2"]) for policy in current_policies}
    )
    formatted_values = (
        ", ".join(f"{value:.6f}" for value in effective_values)
        if effective_values
        else "none"
    )
    lines = [
        "## Resolved Policy Observations",
        "",
        "- Current-auto records activating automatic split-L2: "
        f"`{activation_count} of {len(current_policies)}`",
        "- Distinct current-auto effective split-L2 values: "
        f"`{formatted_values}`",
    ]
    if activation_count == 0:
        lines.extend(
            [
                "",
                "Python public current-auto did not activate the engine-only "
                "auto split-L2 rule in this matrix.",
            ]
        )
    return lines


def write_report(
    path: Path,
    records: Sequence[BenchmarkRecord],
    *,
    specs: Sequence[FixtureSpec],
    seeds: Sequence[int],
    command: str,
    source_commit: str | None = None,
) -> None:
    """Write a reproducible Markdown evidence report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    gate = evaluate_gates(records, specs=specs, seeds=seeds)
    complete_records = sum(_record_issue(row) is None for row in records)
    candidate_results = (
        _candidate_gate_results(records) if gate.evidence_valid else []
    )
    ratio_rows = (
        _shape_objective_ratios(records, specs) if gate.evidence_valid else []
    )
    exact_shapes = {(spec.rows, spec.features) for spec in specs}
    lines = [
        "# Auto-Policy Calibration Benchmark",
        "",
        f"- Command: `{command}`",
        f"- Selected outcome: `{gate.selected_arm or 'gate failure'}`",
        f"- Gate passed: `{str(gate.passed).lower()}`",
        "- Timing is descriptive only and is not used as a quality gate.",
        "",
        "## Environment",
        "",
        *_environment_lines(source_commit=source_commit),
        "",
        "## Matrix Completeness",
        "",
        f"- Matrix evidence complete: `{str(gate.evidence_valid).lower()}`",
        f"- Expected records: `{len(_expected_record_keys(specs, seeds))}`",
        f"- Complete records: `{complete_records}`",
        f"- Total records: `{len(records)}`",
        f"- Distinct fixtures: `{len({row.fixture for row in records})}`",
        f"- Declared exact shapes: `{len(exact_shapes)}`",
        f"- Distinct objectives: `{len({row.objective for row in records})}`",
        f"- Distinct seeds: `{len({row.seed for row in records})}`",
        f"- Distinct arms: `{len({row.arm for row in records})}`",
        "",
        "## Gate Detail",
        "",
        "```text",
        gate.detail,
        "```",
        "",
        "## Candidate Gate Results",
        "",
    ]
    if gate.evidence_valid:
        lines.extend(
            [
                "| Arm | Result | Overall loss ratio |",
                "|---|---|---:|",
            ]
        )
        for result in candidate_results:
            ratio = _format_optional(result.overall_loss_ratio)
            status = "pass" if result.passed else "fail"
            lines.append(f"| {result.name} | {status} | {ratio} |")
        for result in candidate_results:
            lines.extend(
                [
                    "",
                    f"### {result.name}",
                    "",
                    "```text",
                    result.detail,
                    "```",
                ]
            )
    else:
        lines.append(
            "Candidate gate results were not computed because matrix evidence "
            "is incomplete or invalid."
        )
    lines.extend(
        [
            "",
            "## Exact-Shape/Objective Loss Ratios",
            "",
        ]
    )
    if gate.evidence_valid:
        lines.extend(
            [
                "| Arm | Rows | Features | Objective | Median normalized loss |",
                "|---|---:|---:|---|---:|",
            ]
        )
        for arm, rows, features, objective, ratio in ratio_rows:
            lines.append(
                f"| {arm} | {rows} | {features} | {objective} | "
                f"{ratio:.6f} |"
            )
    else:
        lines.append(
            "Exact-shape/objective loss ratios were not computed because "
            "matrix evidence is incomplete or invalid."
        )
    lines.extend(
        [
            "",
            *_policy_observation_lines(records),
            "",
            "## Decision",
            "",
        ]
    )
    if not gate.evidence_valid:
        lines.append(
            "No production decision is supported because matrix evidence is "
            "incomplete or invalid."
        )
    elif gate.selected_arm == "current_auto":
        lines.extend(
            [
                "Keep the production auto-policy heuristics unchanged. Neither "
                "experimental candidate met the predeclared 1% overall improvement "
                "requirement without a protected shape/objective regression.",
            ]
        )
    elif gate.selected_arm in {"no_gain_floor", "quality_first"}:
        lines.append(
            f"Apply the production policy change represented by "
            f"`{gate.selected_arm}`."
        )
    elif gate.selected_arm == "manual_default":
        lines.append(
            "The selected `manual_default` comparison arm does not map to a "
            "predeclared production heuristic edit; production behavior requires "
            "an explicit follow-up decision."
        )
    else:
        lines.append(
            "The evidence gate failed, so no production heuristic decision is "
            "supported."
        )
    lines.extend(
        [
            "",
            "## Resolved Policy Diagnostics",
            "",
            "| Fixture | Seed | Arm | Mode | Requested rounds | Round cap | "
            "Min rows | Min split gain | Row sample | Col sample | "
            "Auto split-L2 | Effective split-L2 |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for row in records:
        policy = row.resolved_policy
        lines.append(
            f"| {row.fixture} | {row.seed} | {row.arm} | "
            f"{policy.get('requested_mode', '')} | "
            f"{policy.get('requested_rounds', '')} | "
            f"{policy.get('effective_round_cap', '')} | "
            f"{policy.get('min_rows_per_leaf', '')} | "
            f"{policy.get('min_split_gain', '')} | "
            f"{policy.get('row_subsample', '')} | "
            f"{policy.get('col_subsample', '')} | "
            f"{policy.get('auto_split_l2_applied', '')} | "
            f"{policy.get('effective_split_l2', '')} |"
        )
    lines.extend(
        [
            "",
            "## Records",
            "",
            "| Fixture | Stratum | Objective | Seed | Arm | Primary loss | "
            "Accuracy | NDCG@10 | Rounds | Fit seconds | Error |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in records:
        lines.append(
            f"| {row.fixture} | {row.shape_stratum} | {row.objective} | "
            f"{row.seed} | {row.arm} | {row.primary_metric:.6f} | "
            f"{_format_optional(row.accuracy)} | "
            f"{_format_optional(row.ndcg_at_10)} | {row.completed_rounds} | "
            f"{row.fit_seconds:.6f} | {row.error or ''} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run the compact six-stratum sentinel matrix",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="return nonzero for incomplete or invalid gate evidence",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    specs = quick_specs() if args.quick else full_specs()
    seeds = (SEEDS[0],) if args.quick else SEEDS
    records = run_matrix(specs, seeds=seeds)
    gate = evaluate_gates(records, specs=specs, seeds=seeds)
    command = shlex.join([sys.executable, *sys.argv])
    if args.output_json is not None:
        write_json(args.output_json, records, gate)
    if args.output_report is not None:
        write_report(
            args.output_report,
            records,
            specs=specs,
            seeds=seeds,
            command=command,
        )

    print(f"Selected outcome: {gate.selected_arm or 'gate failure'}")
    print(gate.detail)
    print(f"Records: {len(records)}")
    if args.gate and not gate.passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
