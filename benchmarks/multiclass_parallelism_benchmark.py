#!/usr/bin/env python3
"""Deterministic correctness and scaling benchmark for multiclass tree builds."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from itertools import product
import json
import math
import os
import platform
from pathlib import Path
import resource
from statistics import median
import subprocess
import sys
import time
from typing import Sequence

import numpy as np

from alloygbm import GBMClassifier


MIN_MULTICLASS_PARALLEL_WORK = 16_384
QUICK_ROUNDS = 4
FULL_ROUNDS = 12
SHAPES = (
    ("tall-narrow", 32_768, 8),
    ("medium-wide", 4_096, 128),
    ("small-control", 512, 8),
)
CLASS_COUNTS = (3, 12)
GROWTH_MODES = ("level", "leaf")


@dataclass(frozen=True)
class Scenario:
    name: str
    shape: str
    n_rows: int
    n_features: int
    n_classes: int
    tree_growth: str
    seed: int


@dataclass(frozen=True)
class Fixture:
    X_train: np.ndarray
    y_train: np.ndarray
    X_holdout: np.ndarray
    y_holdout: np.ndarray

    def arrays(self) -> tuple[np.ndarray, ...]:
        return self.X_train, self.y_train, self.X_holdout, self.y_holdout


def _scenario_name(
    shape: str,
    n_rows: int,
    n_features: int,
    n_classes: int,
    tree_growth: str,
    seed: int,
) -> str:
    return (
        f"{shape}-{n_rows}x{n_features}-{n_classes}class-"
        f"{tree_growth}-seed{seed}"
    )


def _scenarios(seeds: Sequence[int]) -> tuple[Scenario, ...]:
    return tuple(
        Scenario(
            _scenario_name(shape, rows, features, classes, growth, seed),
            shape,
            rows,
            features,
            classes,
            growth,
            seed,
        )
        for (shape, rows, features), classes, growth, seed in product(
            SHAPES, CLASS_COUNTS, GROWTH_MODES, seeds
        )
    )


def quick_scenarios() -> tuple[Scenario, ...]:
    return _scenarios((0,))


def full_scenarios() -> tuple[Scenario, ...]:
    return _scenarios((0, 1, 2))


def _partition(
    rng: np.random.Generator,
    coefficients: np.ndarray,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_features = coefficients.shape[1]
    X = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    linear = X @ coefficients.T
    nonlinear = (
        0.4 * np.sin(X[:, [0]] * np.arange(1, coefficients.shape[0] + 1))
        + 0.2 * X[:, [min(1, n_features - 1)]] * X[:, [min(2, n_features - 1)]]
    )
    noise = rng.normal(0.0, 0.15, size=linear.shape)
    y = np.argmax(linear + nonlinear + noise, axis=1).astype(np.int32)
    if n_rows >= coefficients.shape[0]:
        y[: coefficients.shape[0]] = np.arange(coefficients.shape[0], dtype=np.int32)
    return np.ascontiguousarray(X), np.ascontiguousarray(y)


def make_fixture(scenario: Scenario) -> Fixture:
    coefficient_rng, train_rng, holdout_rng = (
        np.random.default_rng(seed)
        for seed in np.random.SeedSequence(scenario.seed).spawn(3)
    )
    coefficients = coefficient_rng.normal(
        size=(scenario.n_classes, scenario.n_features)
    ).astype(np.float32)
    coefficients /= np.maximum(
        np.linalg.norm(coefficients, axis=1, keepdims=True), np.float32(1e-6)
    )
    X_train, y_train = _partition(train_rng, coefficients, scenario.n_rows)
    holdout_rows = max(512, min(4_096, scenario.n_rows // 4))
    X_holdout, y_holdout = _partition(holdout_rng, coefficients, holdout_rows)
    return Fixture(X_train, y_train, X_holdout, y_holdout)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _multiclass_log_loss(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    selected = probabilities[np.arange(len(y_true)), y_true.astype(np.intp)]
    return float(-np.mean(np.log(np.clip(selected, 1e-15, 1.0))))


def _class_prior_log_loss(
    y_train: np.ndarray, y_holdout: np.ndarray, n_classes: int
) -> float:
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    probabilities = np.clip(counts / counts.sum(), 1e-15, 1.0)
    return float(-np.mean(np.log(probabilities[y_holdout.astype(np.intp)])))


def _peak_rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1_024


def _fit_record(
    scenario: Scenario,
    fixture: Fixture,
    *,
    n_jobs: int,
    rounds: int,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "n_estimators": rounds,
        "learning_rate": 0.12,
        "max_depth": 3,
        "min_data_in_leaf": 8,
        "training_policy": "manual",
        "continuous_binning_max_bins": 64,
        "tree_growth": scenario.tree_growth,
        "seed": scenario.seed,
        "deterministic": True,
        "n_jobs": n_jobs,
    }
    if scenario.tree_growth == "leaf":
        kwargs["max_leaves"] = 8
    estimator = GBMClassifier(**kwargs)
    rss_before = _peak_rss_bytes()
    started = time.perf_counter()
    estimator.fit(fixture.X_train, fixture.y_train)
    fit_seconds = time.perf_counter() - started
    rss_after = _peak_rss_bytes()
    probabilities = np.ascontiguousarray(
        estimator.predict_proba(fixture.X_holdout), dtype=np.float64
    )
    completed_rounds = int(estimator.n_estimators_ or 0)
    return {
        "scenario": scenario.name,
        "shape": scenario.shape,
        "n_rows": scenario.n_rows,
        "n_features": scenario.n_features,
        "n_classes": scenario.n_classes,
        "tree_growth": scenario.tree_growth,
        "seed": scenario.seed,
        "arm": "serial" if n_jobs == 1 else "parallel",
        "requested_workers": n_jobs,
        "resolved_workers": n_jobs,
        "class_parallel_eligible": (
            n_jobs > 1
            and scenario.n_classes
            * scenario.n_rows
            * max(scenario.n_features, 1)
            >= MIN_MULTICLASS_PARALLEL_WORK
        ),
        "fit_seconds": fit_seconds,
        "requested_rounds": rounds,
        "completed_rounds": completed_rounds,
        "stop_reason": (
            "completed" if completed_rounds == rounds else "stopped_before_requested"
        ),
        "artifact_sha256": _sha256(bytes(estimator.artifact_bytes)),
        "prediction_sha256": _sha256(probabilities.tobytes()),
        "predictions_finite": bool(np.isfinite(probabilities).all()),
        "multiclass_log_loss": _multiclass_log_loss(
            fixture.y_holdout, probabilities
        ),
        "class_prior_log_loss": _class_prior_log_loss(
            fixture.y_train, fixture.y_holdout, scenario.n_classes
        ),
        "peak_incremental_rss_bytes": max(0, rss_after - rss_before),
    }


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_benchmark(*, quick: bool) -> dict[str, object]:
    scenarios = quick_scenarios() if quick else full_scenarios()
    rounds = QUICK_ROUNDS if quick else FULL_ROUNDS
    parallel_workers = min(4, os.cpu_count() or 1)
    if parallel_workers < 2:
        parallel_workers = 1
    records: list[dict[str, object]] = []
    for scenario in scenarios:
        fixture = make_fixture(scenario)
        arms = (1, parallel_workers) if scenario.seed % 2 == 0 else (parallel_workers, 1)
        for n_jobs in dict.fromkeys(arms):
            records.append(
                _fit_record(
                    scenario,
                    fixture,
                    n_jobs=n_jobs,
                    rounds=rounds,
                )
            )
    return {
        "quick": quick,
        "rounds": rounds,
        "parallel_workers": parallel_workers,
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "records": records,
        "environment": {
            "source_commit": _source_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
        },
    }


def _valid_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def evaluate_gate(report: dict[str, object]) -> list[str]:
    failures: list[str] = []
    quick = report.get("quick") is True
    expected_scenarios = quick_scenarios() if quick else full_scenarios()
    expected_rounds = QUICK_ROUNDS if quick else FULL_ROUNDS
    parallel_workers = report.get("parallel_workers")
    if not isinstance(parallel_workers, int) or isinstance(parallel_workers, bool):
        failures.append("parallel_workers must be an integer")
        return failures
    expected_ids = {
        (scenario.name, arm)
        for scenario in expected_scenarios
        for arm in ("serial", "parallel")
    }
    records = report.get("records")
    if not isinstance(records, list):
        return ["records must be a list"]
    actual_ids = [
        (record.get("scenario"), record.get("arm"))
        for record in records
        if isinstance(record, dict)
    ]
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != expected_ids:
        failures.append("record identities must exactly match the canonical matrix")
    expected_specs = [asdict(scenario) for scenario in expected_scenarios]
    if report.get("scenarios") != expected_specs:
        failures.append("scenario specs must exactly match the canonical matrix")
    if report.get("rounds") != expected_rounds:
        failures.append(f"report rounds must equal {expected_rounds}")
    environment = report.get("environment")
    commit = environment.get("source_commit") if isinstance(environment, dict) else None
    if not _valid_commit(commit):
        failures.append("environment source commit must be a full lowercase SHA")

    by_scenario: dict[str, dict[str, dict[str, object]]] = {}
    for record in records:
        if not isinstance(record, dict):
            failures.append("every record must be an object")
            continue
        scenario = str(record.get("scenario"))
        arm = str(record.get("arm"))
        by_scenario.setdefault(scenario, {})[arm] = record
        context = f"{scenario}/{arm}"
        for field in ("fit_seconds", "multiclass_log_loss", "class_prior_log_loss"):
            value = record.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                failures.append(f"{context}: non-finite {field}")
        if not record.get("predictions_finite"):
            failures.append(f"{context}: predictions must be finite")
        if record.get("requested_rounds") != expected_rounds:
            failures.append(f"{context}: requested rounds must equal {expected_rounds}")
        if record.get("completed_rounds") != expected_rounds:
            failures.append(f"{context}: incomplete rounds")
        requested = record.get("requested_workers")
        resolved = record.get("resolved_workers")
        if (
            not isinstance(requested, int)
            or not isinstance(resolved, int)
            or resolved < 1
            or resolved > requested
        ):
            failures.append(f"{context}: resolved worker count exceeds worker request")
        loss = record.get("multiclass_log_loss")
        baseline = record.get("class_prior_log_loss")
        if isinstance(loss, (int, float)) and isinstance(baseline, (int, float)):
            if float(loss) >= float(baseline):
                failures.append(f"{context}: quality does not beat class-prior baseline")

    for scenario in expected_scenarios:
        pair = by_scenario.get(scenario.name, {})
        serial = pair.get("serial")
        parallel = pair.get("parallel")
        if serial is None or parallel is None:
            continue
        for field in ("artifact_sha256", "prediction_sha256"):
            if serial.get(field) != parallel.get(field):
                failures.append(f"{scenario.name}: {field.split('_')[0]} hashes differ")
        if serial.get("multiclass_log_loss") != parallel.get("multiclass_log_loss"):
            failures.append(f"{scenario.name}: quality differs across worker counts")
        if serial.get("completed_rounds") != parallel.get("completed_rounds"):
            failures.append(f"{scenario.name}: round counts differ across worker counts")
        if serial.get("stop_reason") != parallel.get("stop_reason"):
            failures.append(f"{scenario.name}: stop reasons differ across worker counts")

    eligible_high_class = any(
        isinstance(record, dict)
        and record.get("arm") == "parallel"
        and record.get("n_classes") == 12
        and record.get("class_parallel_eligible") is True
        and record.get("resolved_workers", 1) > 1
        for record in records
    )
    if parallel_workers > 1 and not eligible_high_class:
        failures.append("at least one eligible high-class parallel record is required")

    if not quick and parallel_workers > 1:
        ratios_by_classes: dict[int, list[float]] = {3: [], 12: []}
        for scenario in expected_scenarios:
            pair = by_scenario.get(scenario.name, {})
            serial = pair.get("serial")
            parallel = pair.get("parallel")
            if serial and parallel:
                ratios_by_classes[scenario.n_classes].append(
                    float(serial["fit_seconds"]) / float(parallel["fit_seconds"])
                )
        if median(ratios_by_classes[12]) <= 1.0:
            failures.append("high-class median speedup must exceed 1.0x")
        low_class_parallel_ratio = median(
            1.0 / ratio for ratio in ratios_by_classes[3]
        )
        if low_class_parallel_ratio > 1.10:
            failures.append("low-class median regression exceeds 10%")
    return failures


def render_markdown(report: dict[str, object]) -> str:
    records = report.get("records", [])
    lines = [
        "# Multiclass Parallelism Benchmark",
        "",
        "## Environment",
        "",
    ]
    environment = report.get("environment", {})
    if isinstance(environment, dict):
        for key, value in environment.items():
            lines.append(f"- {key.replace('_', ' ')}: {value}")
    lines.extend(
        [
            "",
            "## Acceptance Contract",
            "",
            "- Serial and parallel arms must have identical artifact and prediction hashes.",
            "- Both arms must complete the same requested rounds with finite probabilities.",
            "- Multiclass log loss must beat the class-prior baseline.",
            "- Explicit worker requests are upper bounds; quick-run timing is descriptive.",
            "- Full-run high-class median speedup must exceed 1.0x and low-class median regression must stay within 10%.",
            "",
            "## Records",
            "",
            "| Scenario | Arm | Workers | Eligible | Fit s | Rounds | Log loss | Prior loss | Artifact SHA-256 | Prediction SHA-256 |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for record in records:
        lines.append(
            "| {scenario} | {arm} | {resolved_workers} | {class_parallel_eligible} | "
            "{fit_seconds:.6f} | {completed_rounds} | {multiclass_log_loss:.6f} | "
            "{class_prior_log_loss:.6f} | `{artifact_sha256}` | "
            "`{prediction_sha256}` |".format(**record)
        )
    failures = evaluate_gate(report)
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- Failures: {len(failures)}",
        ]
    )
    for failure in failures:
        lines.append(f"- {failure}")
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_benchmark(quick=args.quick)
    rendered = render_markdown(report)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(rendered, encoding="utf-8")
    if not args.output_markdown:
        print(rendered, end="")
    failures = evaluate_gate(report)
    if args.gate and failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
