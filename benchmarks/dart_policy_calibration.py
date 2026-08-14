#!/usr/bin/env python3
"""Fixed-matrix calibration harness for the public DART drop cap.

The matrix deliberately passes an explicit ``dart_max_drop`` to every arm.
Only the compatibility capture exercises the installed constructor default.
Workers run in fresh processes so fit time and peak RSS are measured per fit,
and the comparison layer refuses incomplete or ambiguous evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()

MATRIX_SCHEMA_VERSION = 1
COMPAT_SCHEMA_VERSION = 1
DECISION_SCHEMA_VERSION = 1

CANDIDATE_CAPS = (2, 5, 10, 20)
INCUMBENT_CAP = 50
FULL_CAPS = (*CANDIDATE_CAPS, INCUMBENT_CAP)
FULL_SEEDS = (0, 1, 2, 3, 4)
COMPAT_SEEDS = (0, 1, 2)

QUALITY_MEDIAN_LIMIT = 1.02
QUALITY_SEED_LIMIT = 1.10
ACCURACY_LOSS_LIMIT = 0.02
NDCG_LOSS_LIMIT = 0.01
PRESSURE_RATIO_LIMIT = 0.50
TIME_RATIO_LIMIT = 0.85
RSS_RELATIVE_LIMIT = 0.15
RSS_ABSOLUTE_LIMIT_BYTES = 32 * 1024 * 1024

GATE_NAMES = (
    "complete_finite",
    "median_quality",
    "individual_seed_quality",
    "accuracy_ndcg",
    "stress_pressure",
    "stress_time",
    "peak_rss",
    "compatibility",
)

COMPAT_FIXTURE_NAMES = (
    "reg-small-wide",
    "binary-medium",
    "multiclass-four",
    "ranking-groups",
)


@dataclass(frozen=True)
class FixtureSpec:
    """One predeclared DART calibration fixture."""

    name: str
    task: str
    n_rows: int
    n_features: int
    n_estimators: int
    drop_rate: float
    sample_type: str
    normalize_type: str
    tree_growth: str
    stress: bool
    trees_per_round: int = 1
    n_classes: int | None = None
    n_groups: int | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FixtureSpec":
        expected = set(asdict(cls(  # type: ignore[call-arg]
            "", "", 1, 1, 1, 0.1, "", "", "", False
        )))
        _require_exact_fields(payload, expected, "fixture specification")
        for name in (
            "name",
            "task",
            "sample_type",
            "normalize_type",
            "tree_growth",
        ):
            _require_str(payload, name, "fixture specification")
        for name in (
            "n_rows",
            "n_features",
            "n_estimators",
            "trees_per_round",
        ):
            _require_int(payload, name, "fixture specification")
        if type(payload["stress"]) is not bool:
            raise ValueError("fixture specification stress must be a boolean")
        _require_float(payload, "drop_rate", "fixture specification")
        for name in ("n_classes", "n_groups"):
            value = payload[name]
            if value is not None:
                _require_int(payload, name, "fixture specification")
        return cls(
            name=payload["name"],
            task=payload["task"],
            n_rows=payload["n_rows"],
            n_features=payload["n_features"],
            n_estimators=payload["n_estimators"],
            drop_rate=float(payload["drop_rate"]),
            sample_type=payload["sample_type"],
            normalize_type=payload["normalize_type"],
            tree_growth=payload["tree_growth"],
            stress=payload["stress"],
            trees_per_round=payload["trees_per_round"],
            n_classes=payload["n_classes"],
            n_groups=payload["n_groups"],
        )


@dataclass(frozen=True)
class Fixture:
    """Deterministic train/test arrays and group metadata."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_group_sizes: tuple[int, ...] = ()
    test_group_sizes: tuple[int, ...] = ()
    train_group_ids: tuple[int, ...] = ()
    test_group_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class DartPolicyRecord:
    """One isolated fit result."""

    fixture: str
    seed: int
    cap: int
    arm: str
    task: str
    n_rows: int
    n_features: int
    n_estimators: int
    drop_rate: float
    sample_type: str
    normalize_type: str
    tree_growth: str
    stress: bool
    trees_per_round: int
    requested_rounds: int
    completed_rounds: int
    primary_metric: str
    primary_value: float
    secondary_metric: str
    secondary_value: float
    fit_seconds: float
    peak_rss_bytes: int | None
    configured_pressure: float
    prediction_sha256: str
    artifact_sha256: str
    source_commit: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DartPolicyRecord":
        expected = set(asdict(cls(  # type: ignore[call-arg]
            "", 0, 1, "", "", 1, 1, 1, 0.1, "", "", "", False,
            1, 1, 1, "", 1.0, "", 1.0, 0.0, None, 1.0, "a" * 64,
            "b" * 64, "c" * 40
        )))
        _require_exact_fields(payload, expected, "DART policy record")
        for name in (
            "fixture",
            "arm",
            "task",
            "sample_type",
            "normalize_type",
            "tree_growth",
            "primary_metric",
            "secondary_metric",
            "prediction_sha256",
            "artifact_sha256",
            "source_commit",
        ):
            _require_str(payload, name, "DART policy record")
        for name in (
            "seed",
            "cap",
            "n_rows",
            "n_features",
            "n_estimators",
            "trees_per_round",
            "requested_rounds",
            "completed_rounds",
        ):
            _require_int(payload, name, "DART policy record")
        if type(payload["stress"]) is not bool:
            raise ValueError("DART policy record stress must be a boolean")
        for name in (
            "drop_rate",
            "primary_value",
            "secondary_value",
            "fit_seconds",
            "configured_pressure",
        ):
            _require_float(payload, name, "DART policy record")
        rss = payload["peak_rss_bytes"]
        if rss is not None:
            _require_int(payload, "peak_rss_bytes", "DART policy record")
        for name in ("prediction_sha256", "artifact_sha256"):
            if not _valid_hex(payload[name], 64):
                raise ValueError(f"DART policy record {name} must be 64 lowercase hex characters")
        if not payload["source_commit"] or any(
            character not in "0123456789abcdef" for character in payload["source_commit"].lower()
        ):
            raise ValueError("DART policy record source_commit must be hexadecimal")
        return cls(
            fixture=payload["fixture"],
            seed=payload["seed"],
            cap=payload["cap"],
            arm=payload["arm"],
            task=payload["task"],
            n_rows=payload["n_rows"],
            n_features=payload["n_features"],
            n_estimators=payload["n_estimators"],
            drop_rate=float(payload["drop_rate"]),
            sample_type=payload["sample_type"],
            normalize_type=payload["normalize_type"],
            tree_growth=payload["tree_growth"],
            stress=payload["stress"],
            trees_per_round=payload["trees_per_round"],
            requested_rounds=payload["requested_rounds"],
            completed_rounds=payload["completed_rounds"],
            primary_metric=payload["primary_metric"],
            primary_value=float(payload["primary_value"]),
            secondary_metric=payload["secondary_metric"],
            secondary_value=float(payload["secondary_value"]),
            fit_seconds=float(payload["fit_seconds"]),
            peak_rss_bytes=rss,
            configured_pressure=float(payload["configured_pressure"]),
            prediction_sha256=payload["prediction_sha256"],
            artifact_sha256=payload["artifact_sha256"],
            source_commit=payload["source_commit"],
        )


@dataclass(frozen=True)
class CandidateAssessment:
    """All eight predeclared gate results for one candidate cap."""

    cap: int
    passed: bool
    gates: tuple[tuple[str, bool], ...]
    reasons: tuple[str, ...]
    fixture_quality_ratios: tuple[tuple[str, float], ...]
    seed_quality_ratios: tuple[tuple[str, int, float], ...]
    fixture_accuracy_deltas: tuple[tuple[str, float], ...]
    fixture_ndcg_deltas: tuple[tuple[str, float], ...]
    stress_pressure_ratio: float
    stress_time_ratio: float
    rss_ratio: float | None
    candidate_peak_rss_bytes: int | None
    incumbent_peak_rss_bytes: int | None
    allowed_peak_rss_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DartPolicyDecision:
    """Machine-readable result consumed by the public-default task."""

    candidate_caps: tuple[int, ...]
    incumbent_cap: int
    selected_cap: int
    fallback: bool
    matrix_valid: bool
    compatibility_passed: bool
    selection_reason: str
    assessments: tuple[CandidateAssessment, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = DECISION_SCHEMA_VERSION
        payload["kind"] = "dart-policy-decision"
        return payload


def _require_exact_fields(
    payload: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{label} fields do not match schema: expected={sorted(expected)}, "
            f"actual={sorted(actual)}"
        )


def _require_str(payload: Mapping[str, Any], name: str, label: str) -> None:
    if type(payload[name]) is not str:
        raise ValueError(f"{label} {name} must be a string")


def _require_int(payload: Mapping[str, Any], name: str, label: str) -> None:
    if type(payload[name]) is not int:
        raise ValueError(f"{label} {name} must be an integer")


def _require_float(payload: Mapping[str, Any], name: str, label: str) -> None:
    value = payload[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} {name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} {name} must be finite")


def _valid_hex(value: str, length: int) -> bool:
    return len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _json_load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON evidence file {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"evidence file {path} must contain a JSON object")
    return payload


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"evidence payload is not strict JSON: {error}") from error
    path.write_text(encoded + "\n", encoding="utf-8")


def _full_specs() -> tuple[FixtureSpec, ...]:
    return (
        FixtureSpec("reg-small-narrow", "regression", 640, 8, 100, 0.10, "uniform", "tree", "level", False),
        FixtureSpec("reg-small-wide", "regression", 640, 64, 100, 0.10, "uniform", "tree", "level", False),
        FixtureSpec("reg-tall-narrow", "regression", 4096, 12, 200, 0.10, "uniform", "tree", "level", True),
        FixtureSpec("reg-tall-wide-leaf", "regression", 3072, 64, 200, 0.10, "uniform", "tree", "leaf", True),
        FixtureSpec("reg-long-stress", "regression", 2048, 24, 300, 0.20, "uniform", "tree", "level", True),
        FixtureSpec("binary-medium", "binary", 2048, 24, 150, 0.10, "uniform", "tree", "level", False),
        FixtureSpec("multiclass-four", "multiclass", 1600, 20, 100, 0.10, "uniform", "tree", "level", True, 4, 4, None),
        FixtureSpec("ranking-groups", "ranking", 2400, 16, 120, 0.10, "uniform", "tree", "level", False, 1, None, 80),
        FixtureSpec("reg-weighted", "regression", 1536, 16, 200, 0.10, "weighted", "tree", "level", True),
        FixtureSpec("reg-forest", "regression", 1536, 16, 200, 0.10, "uniform", "forest", "level", True),
    )


def full_specs() -> tuple[FixtureSpec, ...]:
    """Return the immutable full calibration catalog."""
    return _full_specs()


def quick_specs() -> tuple[FixtureSpec, ...]:
    """Return a smaller catalog for harness development smoke tests."""
    quick_rows = {
        "reg-small-narrow": (256, 24),
        "reg-small-wide": (256, 24),
        "reg-tall-narrow": (512, 32),
        "reg-tall-wide-leaf": (512, 32),
        "reg-long-stress": (512, 40),
        "binary-medium": (512, 32),
        "multiclass-four": (400, 24),
        "ranking-groups": (400, 24),
        "reg-weighted": (400, 32),
        "reg-forest": (400, 32),
    }
    return tuple(
        replace(spec, n_rows=quick_rows[spec.name][0], n_estimators=quick_rows[spec.name][1])
        for spec in full_specs()
    )


def configured_dropout_pressure(
    n_estimators: int,
    drop_rate: float,
    max_drop: int,
    trees_per_round: int = 1,
) -> float:
    """Return expected selected tree count across a fit horizon.

    ``dart_max_drop`` applies to the existing tree pool.  Multiclass DART
    therefore advances the pool by four trees per completed round, rather than
    treating one logical round as one unit of dropout work.
    """
    if type(n_estimators) is not int or n_estimators < 1:
        raise ValueError("n_estimators must be a positive integer")
    if not _finite(drop_rate) or not 0.0 < float(drop_rate) < 1.0:
        raise ValueError("drop_rate must be finite and in (0.0, 1.0)")
    if type(max_drop) is not int or max_drop < 1:
        raise ValueError("max_drop must be a positive integer")
    if type(trees_per_round) is not int or trees_per_round < 1:
        raise ValueError("trees_per_round must be a positive integer")
    pressure = 0.0
    for completed_rounds in range(1, n_estimators):
        existing_tree_count = completed_rounds * trees_per_round
        pressure += min(
            float(max_drop),
            max(1.0, float(drop_rate) * float(existing_tree_count)),
        )
    return float(pressure)


def _task_seed(seed: int, task: str) -> int:
    offsets = {"regression": 11, "binary": 23, "multiclass": 37, "ranking": 53}
    return seed * 1009 + offsets[task] + 137


def make_fixture(spec: FixtureSpec, seed: int) -> Fixture:
    """Create the deterministic local 80/20 train/test fixture."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if spec.n_rows < 10:
        raise ValueError("fixture must contain at least ten rows")
    rng = np.random.default_rng(_task_seed(seed, spec.task))
    X = np.ascontiguousarray(
        rng.normal(size=(spec.n_rows, spec.n_features)).astype(np.float32)
    )

    if spec.task == "regression":
        coefficients = rng.normal(size=spec.n_features).astype(np.float32)
        coefficients /= max(float(np.linalg.norm(coefficients)), 1e-6)
        y = (
            X @ coefficients
            + 0.55 * np.sin(1.4 * X[:, 0])
            + 0.20 * X[:, 1] * X[:, min(2, spec.n_features - 1)]
            + rng.normal(scale=0.12, size=spec.n_rows)
        ).astype(np.float32)
    elif spec.task == "binary":
        logits = (
            1.3 * X[:, 0]
            - 0.8 * X[:, 1]
            + 0.45 * X[:, 2] * X[:, 3]
            + 0.20 * np.sin(X[:, 4])
        )
        y = (logits > float(np.median(logits))).astype(np.int64)
    elif spec.task == "multiclass":
        class_count = spec.n_classes or 4
        coefficients = rng.normal(size=(class_count, spec.n_features))
        logits = X @ coefficients.T
        logits[:, 0] += 0.35 * np.sin(X[:, 0])
        logits[:, 1] += 0.25 * X[:, 1] * X[:, 2]
        logits[:, 2] -= 0.20 * X[:, 3] ** 2
        y = np.argmax(logits, axis=1).astype(np.int64)
        y[:class_count] = np.arange(class_count, dtype=np.int64)
    elif spec.task == "ranking":
        if spec.n_groups is None or spec.n_rows % spec.n_groups != 0:
            raise ValueError("ranking fixture rows must divide evenly into groups")
        group_size = spec.n_rows // spec.n_groups
        group_ids = np.repeat(np.arange(spec.n_groups, dtype=np.int64), group_size)
        score = (
            1.4 * X[:, 0]
            - 0.7 * X[:, 1]
            + 0.35 * X[:, 2] * X[:, 3]
            + 0.15 * rng.normal(size=spec.n_rows)
        )
        y = np.clip(
            np.floor(2.0 + score + rng.normal(scale=0.35, size=spec.n_rows)),
            0,
            4,
        ).astype(np.int64)
        train_group_count = int(spec.n_groups * 0.8)
        train_rows = train_group_count * group_size
        return Fixture(
            X_train=np.ascontiguousarray(X[:train_rows]),
            y_train=np.ascontiguousarray(y[:train_rows]),
            X_test=np.ascontiguousarray(X[train_rows:]),
            y_test=np.ascontiguousarray(y[train_rows:]),
            train_group_sizes=(group_size,) * train_group_count,
            test_group_sizes=(group_size,) * (spec.n_groups - train_group_count),
            train_group_ids=tuple(int(value) for value in group_ids[:train_rows]),
            test_group_ids=tuple(int(value) for value in group_ids[train_rows:]),
        )
    else:
        raise ValueError(f"unsupported fixture task {spec.task!r}")

    permutation = rng.permutation(spec.n_rows)
    split = int(spec.n_rows * 0.8)
    train_indices = permutation[:split]
    test_indices = permutation[split:]
    return Fixture(
        X_train=np.ascontiguousarray(X[train_indices]),
        y_train=np.ascontiguousarray(y[train_indices]),
        X_test=np.ascontiguousarray(X[test_indices]),
        y_test=np.ascontiguousarray(y[test_indices]),
    )


def _expected_metrics(task: str) -> tuple[str, str]:
    if task == "regression":
        return "rmse", "mae"
    if task in {"binary", "multiclass"}:
        return "log_loss", "accuracy"
    if task == "ranking":
        return "ndcg@10", "ndcg@5"
    raise ValueError(f"unsupported task {task!r}")


def _record_metadata_matches(record: DartPolicyRecord, spec: FixtureSpec) -> bool:
    return (
        record.fixture == spec.name
        and record.task == spec.task
        and record.n_rows == spec.n_rows
        and record.n_features == spec.n_features
        and record.n_estimators == spec.n_estimators
        and record.drop_rate == spec.drop_rate
        and record.sample_type == spec.sample_type
        and record.normalize_type == spec.normalize_type
        and record.tree_growth == spec.tree_growth
        and record.stress is spec.stress
        and record.trees_per_round == spec.trees_per_round
        and record.requested_rounds == spec.n_estimators
        and (record.primary_metric, record.secondary_metric) == _expected_metrics(spec.task)
    )


def record_sort_key(record: DartPolicyRecord | Mapping[str, Any]) -> tuple[str, int, int, str]:
    if isinstance(record, Mapping):
        return (
            record["fixture"],
            record["seed"],
            record["cap"],
            record["arm"],
        )
    return (record.fixture, record.seed, record.cap, record.arm)


def _validate_matrix_payload(payload: Mapping[str, Any]) -> None:
    _require_exact_fields(
        payload,
        {"schema_version", "kind", "caps", "seeds", "specs", "records"},
        "matrix evidence",
    )
    if payload["schema_version"] != MATRIX_SCHEMA_VERSION:
        raise ValueError("matrix evidence schema version is unsupported")
    if payload["kind"] != "dart-policy-matrix":
        raise ValueError("matrix evidence kind is unsupported")
    if not isinstance(payload["caps"], list) or not isinstance(payload["seeds"], list):
        raise ValueError("matrix evidence caps and seeds must be arrays")
    if not isinstance(payload["specs"], list) or not isinstance(payload["records"], list):
        raise ValueError("matrix evidence specs and records must be arrays")
    for value in (*payload["caps"], *payload["seeds"]):
        if type(value) is not int:
            raise ValueError("matrix evidence caps and seeds must contain integers")


def write_matrix(
    path: Path | str,
    specs: Sequence[FixtureSpec],
    records: Sequence[DartPolicyRecord],
    *,
    caps: Sequence[int],
    seeds: Sequence[int],
) -> None:
    payload = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "kind": "dart-policy-matrix",
        "caps": list(caps),
        "seeds": list(seeds),
        "specs": [asdict(spec) for spec in specs],
        "records": [record.to_dict() for record in sorted(records, key=record_sort_key)],
    }
    _json_write(Path(path), payload)


def read_matrix(path: Path | str) -> tuple[tuple[FixtureSpec, ...], tuple[DartPolicyRecord, ...]]:
    payload = _json_load(Path(path))
    _validate_matrix_payload(payload)
    specs = tuple(FixtureSpec.from_dict(item) for item in payload["specs"])
    records = tuple(DartPolicyRecord.from_dict(item) for item in payload["records"])
    return specs, tuple(sorted(records, key=record_sort_key))


def write_compat(
    path: Path | str,
    specs: Sequence[FixtureSpec],
    records: Sequence[DartPolicyRecord],
    *,
    arms: Sequence[str],
    seeds: Sequence[int],
    checks: Mapping[str, Any],
) -> None:
    payload = {
        "schema_version": COMPAT_SCHEMA_VERSION,
        "kind": "dart-policy-compatibility",
        "arms": list(arms),
        "seeds": list(seeds),
        "fixtures": [spec.name for spec in specs],
        "records": [record.to_dict() for record in sorted(records, key=record_sort_key)],
        "checks": dict(checks),
    }
    _json_write(Path(path), payload)


def read_compat(path: Path | str) -> dict[str, Any]:
    payload = _json_load(Path(path))
    _require_exact_fields(
        payload,
        {"schema_version", "kind", "arms", "seeds", "fixtures", "records", "checks"},
        "compatibility evidence",
    )
    if payload["schema_version"] != COMPAT_SCHEMA_VERSION:
        raise ValueError("compatibility evidence schema version is unsupported")
    if payload["kind"] != "dart-policy-compatibility":
        raise ValueError("compatibility evidence kind is unsupported")
    if not all(isinstance(value, list) for value in (payload["arms"], payload["seeds"], payload["fixtures"], payload["records"])):
        raise ValueError("compatibility evidence arrays have invalid types")
    if not isinstance(payload["checks"], dict):
        raise ValueError("compatibility evidence checks must be an object")
    payload = dict(payload)
    payload["records"] = [
        record.to_dict()
        for record in sorted(
            (DartPolicyRecord.from_dict(item) for item in payload["records"]),
            key=record_sort_key,
        )
    ]
    return payload


def _matrix_key(record: DartPolicyRecord) -> tuple[str, int, int]:
    return record.fixture, record.seed, record.cap


def _validate_matrix_records(
    records: Sequence[DartPolicyRecord],
    specs: Sequence[FixtureSpec],
    *,
    candidate_caps: Sequence[int],
    incumbent_cap: int,
    seeds: Sequence[int],
) -> tuple[bool, tuple[str, ...], dict[tuple[str, int, int], DartPolicyRecord]]:
    expected_keys = {
        (spec.name, seed, cap)
        for spec in specs
        for seed in seeds
        for cap in (*candidate_caps, incumbent_cap)
    }
    by_key: dict[tuple[str, int, int], DartPolicyRecord] = {}
    reasons: list[str] = []
    for record in records:
        key = _matrix_key(record)
        if key in by_key:
            reasons.append(f"duplicate matrix key {key!r}")
        else:
            by_key[key] = record
        if not _finite(record.primary_value) or not _finite(record.secondary_value):
            reasons.append(f"non-finite metric value for matrix key {key!r}")
        if not _finite(record.fit_seconds) or not _finite(record.configured_pressure):
            reasons.append(f"non-finite timing/pressure value for matrix key {key!r}")
        if record.peak_rss_bytes is not None and record.peak_rss_bytes < 0:
            reasons.append(f"negative peak RSS for matrix key {key!r}")
        if record.completed_rounds != record.requested_rounds:
            reasons.append(f"completed rounds mismatch for matrix key {key!r}")
        if not _valid_hex(record.prediction_sha256, 64) or not _valid_hex(record.artifact_sha256, 64):
            reasons.append(f"invalid prediction/artifact hash for matrix key {key!r}")
        spec = next((item for item in specs if item.name == record.fixture), None)
        if spec is None:
            reasons.append(f"unknown fixture in matrix key {key!r}")
        elif not _record_metadata_matches(record, spec):
            reasons.append(f"fixture metadata mismatch for matrix key {key!r}")
        if record.arm != f"cap-{record.cap}":
            reasons.append(f"matrix arm does not name explicit cap for key {key!r}")
    actual_keys = set(by_key)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing:
        reasons.append(f"matrix coverage missing {len(missing)} key(s): {missing[:3]!r}")
    if extra:
        reasons.append(f"matrix coverage has {len(extra)} unexpected key(s): {extra[:3]!r}")
    return not reasons, tuple(dict.fromkeys(reasons)), by_key


def _oriented_ratio(candidate: float, incumbent: float, metric: str) -> float:
    if not _finite(candidate) or not _finite(incumbent):
        raise ValueError("metric ratio requires finite values")
    if candidate <= 0.0 or incumbent <= 0.0:
        raise ValueError("metric ratio requires positive values")
    if metric.startswith("ndcg"):
        return float(incumbent / candidate)
    return float(candidate / incumbent)


def _compatibility_status(compatibility: object | None) -> tuple[bool, tuple[str, ...]]:
    if compatibility is None:
        return True, ()
    if isinstance(compatibility, bool):
        return compatibility, () if compatibility else ("compatibility gate failed",)
    if not isinstance(compatibility, Mapping):
        raise ValueError("compatibility must be a boolean or mapping")
    passed = compatibility.get("passed")
    if type(passed) is not bool:
        raise ValueError("compatibility mapping must contain boolean 'passed'")
    reasons = compatibility.get("reasons", ())
    if not isinstance(reasons, (list, tuple)) or not all(isinstance(item, str) for item in reasons):
        raise ValueError("compatibility reasons must be strings")
    return passed, tuple(reasons)


def evaluate_candidate_caps(
    records: Sequence[DartPolicyRecord],
    *,
    specs: Sequence[FixtureSpec] | None = None,
    candidate_caps: Sequence[int] = CANDIDATE_CAPS,
    incumbent_cap: int = INCUMBENT_CAP,
    seeds: Sequence[int] = FULL_SEEDS,
    compatibility: object | None = None,
) -> DartPolicyDecision:
    """Apply the predeclared eight-gate selection rule without rounding."""
    specs = tuple(full_specs() if specs is None else specs)
    candidate_caps = tuple(candidate_caps)
    seeds = tuple(seeds)
    if tuple(candidate_caps) != CANDIDATE_CAPS:
        raise ValueError(f"candidate caps are fixed at {CANDIDATE_CAPS!r}")
    if incumbent_cap != INCUMBENT_CAP:
        raise ValueError(f"incumbent cap is fixed at {INCUMBENT_CAP}")
    compatibility_passed, compatibility_reasons = _compatibility_status(compatibility)
    valid, matrix_reasons, by_key = _validate_matrix_records(
        records,
        specs,
        candidate_caps=candidate_caps,
        incumbent_cap=incumbent_cap,
        seeds=seeds,
    )
    assessments: list[CandidateAssessment] = []
    for cap in candidate_caps:
        if not valid:
            gates = tuple((name, False) for name in GATE_NAMES)
            assessments.append(
                CandidateAssessment(
                    cap=cap,
                    passed=False,
                    gates=gates,
                    reasons=tuple(matrix_reasons),
                    fixture_quality_ratios=(),
                    seed_quality_ratios=(),
                    fixture_accuracy_deltas=(),
                    fixture_ndcg_deltas=(),
                    stress_pressure_ratio=float("nan"),
                    stress_time_ratio=float("nan"),
                    rss_ratio=None,
                    candidate_peak_rss_bytes=None,
                    incumbent_peak_rss_bytes=None,
                    allowed_peak_rss_bytes=None,
                )
            )
            continue

        reasons: list[str] = []
        fixture_quality_ratios: list[tuple[str, float]] = []
        seed_quality_ratios: list[tuple[str, int, float]] = []
        fixture_accuracy_deltas: list[tuple[str, float]] = []
        fixture_ndcg_deltas: list[tuple[str, float]] = []
        for spec in specs:
            per_seed_ratios: list[float] = []
            for seed in seeds:
                candidate = by_key[(spec.name, seed, cap)]
                incumbent = by_key[(spec.name, seed, incumbent_cap)]
                ratio = _oriented_ratio(
                    candidate.primary_value,
                    incumbent.primary_value,
                    candidate.primary_metric,
                )
                per_seed_ratios.append(ratio)
                seed_quality_ratios.append((spec.name, seed, ratio))
            fixture_median = float(np.median(np.asarray(per_seed_ratios, dtype=np.float64)))
            fixture_quality_ratios.append((spec.name, fixture_median))
            if fixture_median > QUALITY_MEDIAN_LIMIT:
                reasons.append(
                    f"median quality for {spec.name} is {fixture_median:.12g} "
                    f"> {QUALITY_MEDIAN_LIMIT:.12g}"
                )

            if spec.task in {"binary", "multiclass"}:
                deltas = [
                    by_key[(spec.name, seed, cap)].secondary_value
                    - by_key[(spec.name, seed, incumbent_cap)].secondary_value
                    for seed in seeds
                ]
                delta = float(np.median(np.asarray(deltas, dtype=np.float64)))
                fixture_accuracy_deltas.append((spec.name, delta))
                if delta < -ACCURACY_LOSS_LIMIT:
                    reasons.append(
                        f"accuracy loss for {spec.name} is {-delta:.12g} "
                        f"> {ACCURACY_LOSS_LIMIT:.12g}"
                    )
            if spec.task == "ranking":
                deltas = [
                    by_key[(spec.name, seed, cap)].primary_value
                    - by_key[(spec.name, seed, incumbent_cap)].primary_value
                    for seed in seeds
                ]
                delta = float(np.median(np.asarray(deltas, dtype=np.float64)))
                fixture_ndcg_deltas.append((spec.name, delta))
                if delta < -NDCG_LOSS_LIMIT:
                    reasons.append(
                        f"NDCG loss for {spec.name} is {-delta:.12g} "
                        f"> {NDCG_LOSS_LIMIT:.12g}"
                    )

        max_seed_ratio = max(ratio for _, _, ratio in seed_quality_ratios)
        if max_seed_ratio > QUALITY_SEED_LIMIT:
            reasons.append(
                f"individual-seed quality is {max_seed_ratio:.12g} "
                f"> {QUALITY_SEED_LIMIT:.12g}"
            )

        stress_specs = [spec for spec in specs if spec.stress]
        stress_candidate_pressure = [
            by_key[(spec.name, seed, cap)].configured_pressure
            for spec in stress_specs
            for seed in seeds
        ]
        stress_incumbent_pressure = [
            by_key[(spec.name, seed, incumbent_cap)].configured_pressure
            for spec in stress_specs
            for seed in seeds
        ]
        stress_candidate_time = [
            by_key[(spec.name, seed, cap)].fit_seconds
            for spec in stress_specs
            for seed in seeds
        ]
        stress_incumbent_time = [
            by_key[(spec.name, seed, incumbent_cap)].fit_seconds
            for spec in stress_specs
            for seed in seeds
        ]
        stress_pressure_ratio = float(
            np.median(stress_candidate_pressure) / np.median(stress_incumbent_pressure)
        )
        stress_time_ratio = float(
            np.median(stress_candidate_time) / np.median(stress_incumbent_time)
        )
        if stress_pressure_ratio > PRESSURE_RATIO_LIMIT:
            reasons.append(
                f"stress pressure ratio is {stress_pressure_ratio:.12g} "
                f"> {PRESSURE_RATIO_LIMIT:.12g}"
            )
        if stress_time_ratio > TIME_RATIO_LIMIT:
            reasons.append(
                f"stress fit time ratio is {stress_time_ratio:.12g} "
                f"> {TIME_RATIO_LIMIT:.12g}"
            )

        candidate_rss_values = [
            by_key[(spec.name, seed, cap)].peak_rss_bytes
            for spec in specs
            for seed in seeds
        ]
        incumbent_rss_values = [
            by_key[(spec.name, seed, incumbent_cap)].peak_rss_bytes
            for spec in specs
            for seed in seeds
        ]
        if any(value is None for value in (*candidate_rss_values, *incumbent_rss_values)):
            candidate_peak_rss = None
            incumbent_peak_rss = None
            allowed_peak_rss = None
            rss_ratio = None
            reasons.append("peak RSS is unavailable for at least one paired fit")
        else:
            candidate_peak_rss = max(value for value in candidate_rss_values if value is not None)
            incumbent_peak_rss = max(value for value in incumbent_rss_values if value is not None)
            allowed_peak_rss = max(
                int(math.ceil(incumbent_peak_rss * (1.0 + RSS_RELATIVE_LIMIT))),
                incumbent_peak_rss + RSS_ABSOLUTE_LIMIT_BYTES,
            )
            rss_ratio = float(candidate_peak_rss / incumbent_peak_rss)
            if candidate_peak_rss > allowed_peak_rss:
                reasons.append(
                    f"peak RSS {candidate_peak_rss} exceeds allowed {allowed_peak_rss}"
                )

        gate_results = (
            ("complete_finite", True),
            ("median_quality", not any("median quality" in reason for reason in reasons)),
            (
                "individual_seed_quality",
                not any("individual-seed quality" in reason for reason in reasons),
            ),
            (
                "accuracy_ndcg",
                not any("accuracy loss" in reason or "NDCG loss" in reason for reason in reasons),
            ),
            ("stress_pressure", stress_pressure_ratio <= PRESSURE_RATIO_LIMIT),
            ("stress_time", stress_time_ratio <= TIME_RATIO_LIMIT),
            (
                "peak_rss",
                candidate_peak_rss is not None
                and allowed_peak_rss is not None
                and candidate_peak_rss <= allowed_peak_rss,
            ),
            ("compatibility", compatibility_passed),
        )
        if not compatibility_passed:
            reasons.extend(compatibility_reasons)
        passed = all(value for _, value in gate_results) and not reasons
        assessments.append(
            CandidateAssessment(
                cap=cap,
                passed=passed,
                gates=gate_results,
                reasons=tuple(dict.fromkeys(reasons)),
                fixture_quality_ratios=tuple(fixture_quality_ratios),
                seed_quality_ratios=tuple(seed_quality_ratios),
                fixture_accuracy_deltas=tuple(fixture_accuracy_deltas),
                fixture_ndcg_deltas=tuple(fixture_ndcg_deltas),
                stress_pressure_ratio=stress_pressure_ratio,
                stress_time_ratio=stress_time_ratio,
                rss_ratio=rss_ratio,
                candidate_peak_rss_bytes=candidate_peak_rss,
                incumbent_peak_rss_bytes=incumbent_peak_rss,
                allowed_peak_rss_bytes=allowed_peak_rss,
            )
        )

    passing_caps = [assessment.cap for assessment in assessments if assessment.passed]
    selected_cap = max(passing_caps) if passing_caps else incumbent_cap
    fallback = not passing_caps
    if fallback:
        selection_reason = "no candidate cap passed all eight gates; retained incumbent 50"
    else:
        selection_reason = f"selected largest passing candidate cap {selected_cap}"
    return DartPolicyDecision(
        candidate_caps=tuple(candidate_caps),
        incumbent_cap=incumbent_cap,
        selected_cap=selected_cap,
        fallback=fallback,
        matrix_valid=valid,
        compatibility_passed=compatibility_passed,
        selection_reason=selection_reason,
        assessments=tuple(assessments),
    )


def normalize_peak_rss_bytes(value: int, *, platform_name: str | None = None) -> int:
    """Normalize ``ru_maxrss`` to bytes on Linux and macOS."""
    if type(value) is not int or value < 0:
        raise ValueError("ru_maxrss must be a non-negative integer")
    name = platform_name or sys.platform
    return value if name == "darwin" else value * 1024


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return _sha256_bytes(array.tobytes(order="C"))


def _source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("unable to identify benchmark source commit") from error


def _import_alloygbm():
    bindings_path = str(REPO_ROOT / "bindings" / "python")
    if bindings_path not in sys.path:
        sys.path.insert(0, bindings_path)
    import alloygbm  # type: ignore[import-not-found]

    return alloygbm


def _build_model(spec: FixtureSpec, *, seed: int, arm: str, cap: int | None, n_jobs: int):
    alloygbm = _import_alloygbm()
    kwargs: dict[str, Any] = {
        "n_estimators": spec.n_estimators,
        "learning_rate": 0.06,
        "max_depth": 4,
        "lambda_l2": 1.0,
        "training_policy": "manual",
        "continuous_binning_strategy": "quantile",
        "deterministic": True,
        "seed": seed,
        "boosting_mode": "dart",
        "dart_drop_rate": spec.drop_rate,
        "dart_normalize_type": spec.normalize_type,
        "dart_sample_type": spec.sample_type,
        "tree_growth": spec.tree_growth,
        "n_jobs": n_jobs,
    }
    if spec.tree_growth == "leaf":
        kwargs["max_leaves"] = 16
    if arm != "default":
        if cap is None:
            raise ValueError(f"arm {arm!r} requires an explicit cap")
        kwargs["dart_max_drop"] = cap
    if spec.task == "regression":
        return alloygbm.GBMRegressor(**kwargs)
    if spec.task in {"binary", "multiclass"}:
        return alloygbm.GBMClassifier(**kwargs)
    if spec.task == "ranking":
        kwargs["ranking_objective"] = "rank:ndcg"
        return alloygbm.GBMRanker(**kwargs)
    raise ValueError(f"unsupported fixture task {spec.task!r}")


def _fit_record(
    spec: FixtureSpec,
    *,
    seed: int,
    arm: str,
    cap: int | None,
    n_jobs: int,
) -> DartPolicyRecord:
    fixture = make_fixture(spec, seed)
    model = _build_model(spec, seed=seed, arm=arm, cap=cap, n_jobs=n_jobs)
    started = time.perf_counter()
    if spec.task == "ranking":
        train_group = np.asarray(fixture.train_group_ids, dtype=np.int64)
        model.fit(fixture.X_train, fixture.y_train, group=train_group)
    else:
        model.fit(fixture.X_train, fixture.y_train)
    fit_seconds = float(time.perf_counter() - started)
    peak_rss_bytes = normalize_peak_rss_bytes(
        int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    )
    if spec.task == "regression":
        prediction = np.asarray(model.predict(fixture.X_test), dtype=np.float64)
        primary_value = float(np.sqrt(np.mean((fixture.y_test - prediction) ** 2)))
        secondary_value = float(np.mean(np.abs(fixture.y_test - prediction)))
    elif spec.task == "binary":
        probabilities = np.asarray(model.predict_proba(fixture.X_test), dtype=np.float64)
        prediction = probabilities
        positive = np.clip(probabilities[:, 1], 1e-15, 1.0 - 1e-15)
        y_test = np.asarray(fixture.y_test, dtype=np.float64)
        primary_value = float(
            -np.mean(y_test * np.log(positive) + (1.0 - y_test) * np.log(1.0 - positive))
        )
        secondary_value = float(np.mean(np.asarray(model.predict(fixture.X_test)) == fixture.y_test))
    elif spec.task == "multiclass":
        probabilities = np.asarray(model.predict_proba(fixture.X_test), dtype=np.float64)
        prediction = probabilities
        primary_value = float(
            -np.mean(np.log(np.clip(probabilities[np.arange(len(fixture.y_test)), fixture.y_test], 1e-15, 1.0)))
        )
        secondary_value = float(np.mean(np.argmax(probabilities, axis=1) == fixture.y_test))
    else:
        prediction = np.asarray(model.predict(fixture.X_test), dtype=np.float64)
        primary_value = _ndcg_at(fixture.y_test, prediction, fixture.test_group_ids, 10)
        secondary_value = _ndcg_at(fixture.y_test, prediction, fixture.test_group_ids, 5)
    completed_rounds = int(getattr(model, "n_estimators_"))
    requested_rounds = int(spec.n_estimators)
    if completed_rounds != requested_rounds:
        raise RuntimeError(
            f"fixture {spec.name} seed {seed} arm {arm} completed "
            f"{completed_rounds}/{requested_rounds} rounds"
        )
    actual_cap = int(getattr(model, "dart_max_drop"))
    return DartPolicyRecord(
        fixture=spec.name,
        seed=seed,
        cap=actual_cap,
        arm=arm,
        task=spec.task,
        n_rows=spec.n_rows,
        n_features=spec.n_features,
        n_estimators=spec.n_estimators,
        drop_rate=spec.drop_rate,
        sample_type=spec.sample_type,
        normalize_type=spec.normalize_type,
        tree_growth=spec.tree_growth,
        stress=spec.stress,
        trees_per_round=spec.trees_per_round,
        requested_rounds=requested_rounds,
        completed_rounds=completed_rounds,
        primary_metric=_expected_metrics(spec.task)[0],
        primary_value=primary_value,
        secondary_metric=_expected_metrics(spec.task)[1],
        secondary_value=secondary_value,
        fit_seconds=fit_seconds,
        peak_rss_bytes=peak_rss_bytes,
        configured_pressure=configured_dropout_pressure(
            spec.n_estimators, spec.drop_rate, actual_cap, spec.trees_per_round
        ),
        prediction_sha256=_sha256_array(prediction),
        artifact_sha256=_sha256_bytes(bytes(model.artifact_bytes)),
        source_commit=_source_commit(),
    )


def _ndcg_at(
    labels: np.ndarray,
    scores: np.ndarray,
    group_ids: Sequence[int],
    cutoff: int,
) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    group_ids = tuple(int(value) for value in group_ids)
    if len(labels) != len(scores) or len(labels) != len(group_ids):
        raise ValueError("NDCG inputs must have equal lengths")
    total = 0.0
    group_count = 0
    start = 0
    while start < len(group_ids):
        end = start + 1
        while end < len(group_ids) and group_ids[end] == group_ids[start]:
            end += 1
        group_labels = labels[start:end]
        group_scores = scores[start:end]
        k = min(cutoff, len(group_labels))
        order = np.argsort(-group_scores, kind="mergesort")[:k]
        ideal = np.argsort(-group_labels, kind="mergesort")[:k]
        discounts = 1.0 / np.log2(np.arange(2.0, float(k) + 2.0))
        dcg = float(np.sum((2.0 ** group_labels[order] - 1.0) * discounts))
        idcg = float(np.sum((2.0 ** group_labels[ideal] - 1.0) * discounts))
        total += 1.0 if idcg <= 0.0 else dcg / idcg
        group_count += 1
        start = end
    return 1.0 if group_count == 0 else float(total / group_count)


def _run_worker_subprocess(
    spec: FixtureSpec,
    *,
    seed: int,
    arm: str,
    cap: int | None,
    n_jobs: int = 1,
) -> DartPolicyRecord:
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "record",
        "--fixture",
        spec.name,
        "--seed",
        str(seed),
        "--arm",
        arm,
        "--n-jobs",
        str(n_jobs),
    ]
    if cap is not None:
        command.extend(("--cap", str(cap)))
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker failed for {spec.name} seed {seed} arm {arm}:\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    try:
        payload = json.loads(completed.stdout, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"worker did not emit strict JSON: {completed.stdout!r}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("worker output must be a JSON object")
    return DartPolicyRecord.from_dict(payload)


def run_matrix(
    *,
    caps: Sequence[int],
    seeds: Sequence[int],
    specs: Sequence[FixtureSpec],
    output: Path | str,
) -> None:
    caps = tuple(caps)
    seeds = tuple(seeds)
    if not caps or any(cap not in FULL_CAPS for cap in caps) or len(set(caps)) != len(caps):
        raise ValueError(f"caps must be unique values from {FULL_CAPS!r}")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty unique sequence")
    records: list[DartPolicyRecord] = []
    total = len(specs) * len(seeds) * len(caps)
    completed = 0
    for spec in specs:
        for seed in seeds:
            for cap in caps:
                print(
                    f"record {completed + 1}/{total}: {spec.name} seed={seed} cap={cap}",
                    file=sys.stderr,
                )
                records.append(
                    _run_worker_subprocess(spec, seed=seed, arm=f"cap-{cap}", cap=cap)
                )
                completed += 1
    write_matrix(output, specs, records, caps=caps, seeds=seeds)


def _compat_expected_keys(
    arms: Sequence[str], seeds: Sequence[int], specs: Sequence[FixtureSpec]
) -> set[tuple[str, int, str]]:
    return {(spec.name, seed, arm) for spec in specs for seed in seeds for arm in arms}


def _compat_checks(
    records: Sequence[DartPolicyRecord],
    *,
    arms: Sequence[str],
    seeds: Sequence[int],
    specs: Sequence[FixtureSpec],
    require_default_cap50_parity: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    by_key: dict[tuple[str, int, str], DartPolicyRecord] = {}
    for record in records:
        key = (record.fixture, record.seed, record.arm)
        if key in by_key:
            reasons.append(f"duplicate compatibility key {key!r}")
        by_key[key] = record
        if record.completed_rounds != record.requested_rounds:
            reasons.append(f"compatibility fit incomplete for {key!r}")
        if not _finite(record.primary_value) or not _finite(record.secondary_value):
            reasons.append(f"compatibility metric non-finite for {key!r}")
    expected = _compat_expected_keys(arms, seeds, specs)
    missing = expected - set(by_key)
    extra = set(by_key) - expected
    if missing:
        reasons.append(f"compatibility coverage missing {len(missing)} key(s)")
    if extra:
        reasons.append(f"compatibility coverage has {len(extra)} unexpected key(s)")

    default_cap50_equal = True
    if "default" in arms and "cap50" in arms:
        for spec in specs:
            for seed in seeds:
                default = by_key.get((spec.name, seed, "default"))
                cap50 = by_key.get((spec.name, seed, "cap50"))
                if default is None or cap50 is None:
                    continue
                if require_default_cap50_parity and default.cap != INCUMBENT_CAP:
                    reasons.append(
                        f"default arm resolved to cap {default.cap}, expected incumbent 50"
                    )
                if (default.prediction_sha256, default.artifact_sha256) != (
                    cap50.prediction_sha256,
                    cap50.artifact_sha256,
                ):
                    default_cap50_equal = False
                    if require_default_cap50_parity:
                        reasons.append(
                            f"default/cap50 parity failed for {spec.name} seed {seed}"
                        )
    return {
        "passed": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "default_cap50_equal": default_cap50_equal,
        "determinism": not any("duplicate" in reason for reason in reasons),
    }


def run_compatibility(
    *,
    arms: Sequence[str],
    seeds: Sequence[int],
    specs: Sequence[FixtureSpec],
    output: Path | str,
    selected_cap: int | None = None,
) -> None:
    allowed_arms = {"default", "cap50", "cap-selected"}
    arms = tuple(arms)
    if not arms or any(arm not in allowed_arms for arm in arms) or len(set(arms)) != len(arms):
        raise ValueError(f"arms must be unique values from {sorted(allowed_arms)!r}")
    if "cap-selected" in arms:
        if selected_cap not in CANDIDATE_CAPS:
            raise ValueError("cap-selected requires a selected candidate cap")
    records: list[DartPolicyRecord] = []
    for spec in specs:
        for seed in seeds:
            for arm in arms:
                cap = None if arm == "default" else (INCUMBENT_CAP if arm == "cap50" else selected_cap)
                print(
                    f"compat {spec.name} seed={seed} arm={arm}",
                    file=sys.stderr,
                )
                first = _run_worker_subprocess(spec, seed=seed, arm=arm, cap=cap)
                second = _run_worker_subprocess(spec, seed=seed, arm=arm, cap=cap)
                if (first.prediction_sha256, first.artifact_sha256) != (
                    second.prediction_sha256,
                    second.artifact_sha256,
                ):
                    raise RuntimeError(
                        f"determinism failed for {spec.name} seed {seed} arm {arm}"
                    )
                records.append(first)
    checks = _compat_checks(
        records,
        arms=arms,
        seeds=seeds,
        specs=specs,
        require_default_cap50_parity=arms == ("default", "cap50")
        or set(arms) == {"default", "cap50"},
    )
    if not checks["passed"]:
        raise RuntimeError(f"compatibility checks failed: {checks['reasons']}")
    write_compat(output, specs, records, arms=arms, seeds=seeds, checks=checks)


def _compat_payload_records(payload: Mapping[str, Any]) -> tuple[DartPolicyRecord, ...]:
    return tuple(DartPolicyRecord.from_dict(item) for item in payload["records"])


def _compare_compatibility(
    production_payload: Mapping[str, Any],
    candidate_payload: Mapping[str, Any] | None,
    *,
    selected_cap: int,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    production_records = _compat_payload_records(production_payload)
    production_arms = tuple(production_payload["arms"])
    production_seeds = tuple(production_payload["seeds"])
    production_by_key = {
        (record.fixture, record.seed, record.arm): record
        for record in production_records
    }
    if set(production_arms) != {"default", "cap50"}:
        reasons.append("production compatibility must contain default and cap50 arms")
    for fixture in COMPAT_FIXTURE_NAMES:
        for seed in production_seeds:
            default = production_by_key.get((fixture, seed, "default"))
            cap50 = production_by_key.get((fixture, seed, "cap50"))
            if default is None or cap50 is None:
                reasons.append(f"production compatibility missing {fixture} seed {seed}")
                continue
            if (default.prediction_sha256, default.artifact_sha256) != (
                cap50.prediction_sha256,
                cap50.artifact_sha256,
            ):
                reasons.append(f"production default/cap50 parity failed for {fixture} seed {seed}")
    if candidate_payload is None:
        return not reasons, tuple(dict.fromkeys(reasons))

    candidate_records = _compat_payload_records(candidate_payload)
    candidate_by_key = {
        (record.fixture, record.seed, record.arm): record
        for record in candidate_records
    }
    candidate_arms = set(candidate_payload["arms"])
    if candidate_arms != {"default", "cap-selected", "cap50"}:
        reasons.append("candidate compatibility must contain default, cap-selected, and cap50 arms")
    for fixture in COMPAT_FIXTURE_NAMES:
        for seed in tuple(candidate_payload["seeds"]):
            default = candidate_by_key.get((fixture, seed, "default"))
            selected = candidate_by_key.get((fixture, seed, "cap-selected"))
            cap50 = candidate_by_key.get((fixture, seed, "cap50"))
            if default is None or selected is None or cap50 is None:
                reasons.append(f"candidate compatibility missing {fixture} seed {seed}")
                continue
            if selected.cap != selected_cap:
                reasons.append(
                    f"candidate selected arm resolved to cap {selected.cap}, expected {selected_cap}"
                )
            if (default.prediction_sha256, default.artifact_sha256) != (
                selected.prediction_sha256,
                selected.artifact_sha256,
            ):
                reasons.append(f"candidate default/selected parity failed for {fixture} seed {seed}")
            production_cap50 = production_by_key.get((fixture, seed, "cap50"))
            if production_cap50 is None or (cap50.prediction_sha256, cap50.artifact_sha256) != (
                production_cap50.prediction_sha256,
                production_cap50.artifact_sha256,
            ):
                reasons.append(f"candidate/production cap50 parity failed for {fixture} seed {seed}")
    return not reasons, tuple(dict.fromkeys(reasons))


def compare_evidence(
    matrix_path: Path | str,
    *,
    production_compat_path: Path | str,
    candidate_compat_path: Path | str | None = None,
    output: Path | str | None = None,
) -> DartPolicyDecision:
    specs, records = read_matrix(matrix_path)
    production_payload = read_compat(production_compat_path)
    candidate_payload = None if candidate_compat_path is None else read_compat(candidate_compat_path)
    provisional = evaluate_candidate_caps(records, specs=specs)
    compat_passed, compat_reasons = _compare_compatibility(
        production_payload,
        candidate_payload,
        selected_cap=provisional.selected_cap,
    )
    decision = evaluate_candidate_caps(
        records,
        specs=specs,
        compatibility={"passed": compat_passed, "reasons": list(compat_reasons)},
    )
    if output is not None:
        _json_write(Path(output), decision.to_dict())
    return decision


def _find_spec(name: str, *, quick: bool) -> FixtureSpec:
    for spec in (quick_specs() if quick else full_specs()):
        if spec.name == name:
            return spec
    raise ValueError(f"unknown fixture {name!r}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the explicit-cap matrix")
    run.add_argument("--caps", nargs="+", type=int, default=list(FULL_CAPS))
    run.add_argument("--seeds", nargs="+", type=int, default=list(FULL_SEEDS))
    run.add_argument("--quick", action="store_true")
    run.add_argument("--output", type=Path, required=True)

    compat = commands.add_parser("run-compat", help="capture default/explicit compatibility")
    compat.add_argument("--arms", nargs="+", required=True)
    compat.add_argument("--seeds", nargs="+", type=int, default=list(COMPAT_SEEDS))
    compat.add_argument("--selected-cap", type=int)
    compat.add_argument("--quick", action="store_true")
    compat.add_argument("--output", type=Path, required=True)

    compare = commands.add_parser("compare", help="apply the selection rule to evidence")
    compare.add_argument("matrix", type=Path)
    compare.add_argument("--production-compat", type=Path, required=True)
    compare.add_argument("--candidate-compat", type=Path)
    compare.add_argument("--output", type=Path, required=True)

    record = commands.add_parser("record", help=argparse.SUPPRESS)
    record.add_argument("--fixture", required=True)
    record.add_argument("--seed", type=int, required=True)
    record.add_argument("--arm")
    record.add_argument("--cap", type=int)
    record.add_argument("--n-jobs", type=int, default=1)
    record.add_argument("--quick", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "record":
        spec = _find_spec(args.fixture, quick=args.quick)
        if args.arm is None:
            if args.cap is None:
                raise SystemExit("record requires --arm or --cap")
            args.arm = f"cap-{args.cap}"
        if args.arm not in {"default", "cap50", "cap-selected"} and not args.arm.startswith("cap-"):
            raise SystemExit(f"unsupported record arm {args.arm!r}")
        if args.arm == "cap50":
            args.cap = INCUMBENT_CAP
        record = _fit_record(
            spec,
            seed=args.seed,
            arm=args.arm,
            cap=args.cap,
            n_jobs=args.n_jobs,
        )
        print(json.dumps(record.to_dict(), sort_keys=True, allow_nan=False))
        return 0
    if args.command == "run":
        run_matrix(
            caps=args.caps,
            seeds=args.seeds,
            specs=quick_specs() if args.quick else full_specs(),
            output=args.output,
        )
        return 0
    if args.command == "run-compat":
        run_compatibility(
            arms=args.arms,
            seeds=args.seeds,
            specs=(
                tuple(spec for spec in (quick_specs() if args.quick else full_specs()) if spec.name in COMPAT_FIXTURE_NAMES)
            ),
            output=args.output,
            selected_cap=args.selected_cap,
        )
        return 0
    if args.command == "compare":
        decision = compare_evidence(
            args.matrix,
            production_compat_path=args.production_compat,
            candidate_compat_path=args.candidate_compat,
            output=args.output,
        )
        print(json.dumps(decision.to_dict(), sort_keys=True, indent=2, allow_nan=False))
        return 0
    raise SystemExit(f"unsupported command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
