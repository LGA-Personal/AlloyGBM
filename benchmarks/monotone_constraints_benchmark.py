#!/usr/bin/env python3
"""Deterministic acceptance benchmark for strict monotone constraints."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import product
import platform
import sys
import time
from typing import Callable, Literal, Sequence

import numpy as np

from alloygbm import GBMClassifier, GBMRegressor


GRID_VALUES = 257
QUICK_ROUNDS = 16
FULL_ROUNDS = 64
QUICK_CONTEXTS = 8
FULL_CONTEXTS = 16
MONOTONE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Scenario:
    name: str
    n_rows: int
    n_features: int
    objective: Literal["regression", "binary"]
    direction: Literal[-1, 1]
    tree_growth: Literal["level", "leaf"]
    seed: int


@dataclass(frozen=True)
class Fixture:
    X_train: np.ndarray
    y_train: np.ndarray
    X_holdout: np.ndarray
    y_holdout: np.ndarray

    def arrays(self) -> tuple[np.ndarray, ...]:
        return self.X_train, self.y_train, self.X_holdout, self.y_holdout


@dataclass(frozen=True)
class GridCheck:
    checked_grid_pairs: int
    worst_signed_monotone_margin: float
    violation_count: int
    grid_predictions_finite: bool


def _scenario_name(
    n_rows: int,
    n_features: int,
    objective: str,
    direction: int,
    tree_growth: str,
    seed: int,
) -> str:
    sign = "increasing" if direction > 0 else "decreasing"
    return f"{n_rows}x{n_features}-{objective}-{sign}-{tree_growth}-seed{seed}"


def _make_scenarios(
    *,
    rows: Sequence[int],
    features: Sequence[int],
    objectives: Sequence[Literal["regression", "binary"]],
    directions: Sequence[Literal[-1, 1]],
    growth_modes: Sequence[Literal["level", "leaf"]],
    seeds: Sequence[int],
) -> tuple[Scenario, ...]:
    return tuple(
        Scenario(
            _scenario_name(n_rows, n_features, objective, direction, tree_growth, seed),
            n_rows,
            n_features,
            objective,
            direction,
            tree_growth,
            seed,
        )
        for n_rows, n_features, objective, direction, tree_growth, seed in product(
            rows, features, objectives, directions, growth_modes, seeds
        )
    )


def quick_scenarios() -> tuple[Scenario, ...]:
    """Return the compact CI sentinel matrix."""
    return _make_scenarios(
        rows=(128,),
        features=(2,),
        objectives=("regression", "binary"),
        directions=(-1, 1),
        growth_modes=("level", "leaf"),
        seeds=(0,),
    )


def full_scenarios() -> tuple[Scenario, ...]:
    """Return the exhaustive deterministic acceptance matrix."""
    return _make_scenarios(
        rows=(128, 4_096, 32_768),
        features=(2, 16, 128),
        objectives=("regression", "binary"),
        directions=(-1, 1),
        growth_modes=("level", "leaf"),
        seeds=(0, 1, 2),
    )


def target_components(X: np.ndarray, *, direction: int) -> dict[str, np.ndarray]:
    """Return finite target components with a monotone feature-zero derivative."""
    x0 = X[:, 0]
    x1 = X[:, 1]
    x2 = X[:, min(2, X.shape[1] - 1)]
    x3 = X[:, min(3, X.shape[1] - 1)]
    main = np.float32(direction) * np.float32(2.5) * x0
    # d / dx0 of main + interaction has the requested sign on [-1, 1].
    interaction = np.float32(direction) * np.float32(0.75) * x0 * x1
    nuisance = (
        np.float32(0.65) * np.sin(np.float32(2.0) * x1)
        + np.float32(0.35) * x2 * x2
        - np.float32(0.25) * np.cos(np.float32(3.0) * x3)
    )
    return {
        "main": np.asarray(main, dtype=np.float32),
        "interaction": np.asarray(interaction, dtype=np.float32),
        "nuisance": np.asarray(nuisance, dtype=np.float32),
    }


def _holdout_row_count(training_rows: int) -> int:
    return max(512, min(4_096, training_rows // 4))


def _make_partition(
    scenario: Scenario,
    *,
    rng: np.random.Generator,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    X = rng.uniform(-1.0, 1.0, size=(n_rows, scenario.n_features)).astype(
        np.float32
    )
    components = target_components(X, direction=scenario.direction)
    latent = components["main"] + components["interaction"] + components["nuisance"]
    if scenario.objective == "regression":
        target = latent + rng.normal(0.0, 0.15, size=n_rows).astype(np.float32)
    else:
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(latent, -12.0, 12.0)))
        target = (rng.random(n_rows) < probabilities).astype(np.float32)
        if np.unique(target).size < 2:
            target[:2] = np.asarray([0.0, 1.0], dtype=np.float32)
    return (
        np.ascontiguousarray(X),
        np.ascontiguousarray(target, dtype=np.float32),
    )


def make_fixture(scenario: Scenario) -> Fixture:
    """Generate independent deterministic float32 training and holdout data."""
    train_seed, holdout_seed = np.random.SeedSequence(scenario.seed).spawn(2)
    X_train, y_train = _make_partition(
        scenario,
        rng=np.random.default_rng(train_seed),
        n_rows=scenario.n_rows,
    )
    X_holdout, y_holdout = _make_partition(
        scenario,
        rng=np.random.default_rng(holdout_seed),
        n_rows=_holdout_row_count(scenario.n_rows),
    )
    return Fixture(
        X_train,
        y_train,
        X_holdout,
        y_holdout,
    )


def validate_monotonicity(
    estimator: object,
    X_holdout: np.ndarray,
    *,
    direction: int,
    predictor: Callable[[np.ndarray], np.ndarray] | None = None,
    context_count: int | None = None,
) -> GridCheck:
    """Sweep feature zero while holding every other feature fixed per context."""
    predict = estimator.predict if predictor is None else predictor
    contexts = X_holdout if context_count is None else X_holdout[:context_count]
    grid_values = np.linspace(-1.0, 1.0, GRID_VALUES, dtype=np.float32)
    violations = 0
    worst_margin = np.inf
    checked_pairs = 0
    grid_predictions_finite = True
    for context in contexts:
        grid = np.tile(context, (GRID_VALUES, 1))
        grid[:, 0] = grid_values
        prediction = np.asarray(predict(np.ascontiguousarray(grid)), dtype=np.float64)
        signed_margin = direction * np.diff(prediction)
        if not np.isfinite(prediction).all() or not np.isfinite(signed_margin).all():
            grid_predictions_finite = False
        violations += int(np.count_nonzero(signed_margin < -MONOTONE_TOLERANCE))
        finite_margins = signed_margin[np.isfinite(signed_margin)]
        if finite_margins.size:
            worst_margin = min(worst_margin, float(np.min(finite_margins)))
        checked_pairs += GRID_VALUES - 1
    return GridCheck(
        checked_pairs,
        float(worst_margin),
        violations,
        grid_predictions_finite,
    )


def _estimator_kwargs(scenario: Scenario, *, constrained: bool, rounds: int) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "n_estimators": rounds,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_data_in_leaf": 8,
        "min_split_gain": 0.0,
        "row_subsample": 1.0,
        "col_subsample": 1.0,
        "training_policy": "manual",
        "tree_growth": scenario.tree_growth,
        "seed": scenario.seed,
        "deterministic": True,
    }
    if scenario.tree_growth == "leaf":
        kwargs["max_leaves"] = 8
    if constrained:
        kwargs["monotone_constraints"] = [scenario.direction] + [0] * (
            scenario.n_features - 1
        )
    return kwargs


def _loss(objective: str, y_true: np.ndarray, estimator: object, X: np.ndarray) -> float:
    if objective == "regression":
        prediction = np.asarray(estimator.predict(X), dtype=np.float64)
        return float(np.sqrt(np.mean((prediction - y_true) ** 2)))
    prediction = np.asarray(estimator.predict(X), dtype=np.float64)
    return float(np.mean(prediction != y_true))


def _constant_loss(objective: str, y_train: np.ndarray, y_holdout: np.ndarray) -> float:
    if objective == "regression":
        return float(np.sqrt(np.mean((y_holdout - np.mean(y_train)) ** 2)))
    majority = float(np.mean(y_train) >= 0.5)
    return float(np.mean(y_holdout != majority))


def _fit_record(scenario: Scenario, *, rounds: int, contexts: int) -> dict[str, object]:
    fixture = make_fixture(scenario)
    estimator_type = GBMRegressor if scenario.objective == "regression" else GBMClassifier
    constrained = estimator_type(**_estimator_kwargs(scenario, constrained=True, rounds=rounds))
    unconstrained = estimator_type(**_estimator_kwargs(scenario, constrained=False, rounds=rounds))

    constrained_start = time.perf_counter()
    constrained.fit(fixture.X_train, fixture.y_train)
    constrained_seconds = time.perf_counter() - constrained_start
    unconstrained_start = time.perf_counter()
    unconstrained.fit(fixture.X_train, fixture.y_train)
    unconstrained_seconds = time.perf_counter() - unconstrained_start

    constrained_loss = _loss(
        scenario.objective, fixture.y_holdout, constrained, fixture.X_holdout
    )
    unconstrained_loss = _loss(
        scenario.objective, fixture.y_holdout, unconstrained, fixture.X_holdout
    )
    predictor: Callable[[np.ndarray], np.ndarray] | None = None
    if scenario.objective == "binary":
        predictor = lambda values: constrained.predict_proba(values)[:, 1]
    check = validate_monotonicity(
        constrained,
        fixture.X_holdout,
        direction=scenario.direction,
        predictor=predictor,
        context_count=contexts,
    )
    constrained_predictions = np.asarray(constrained.predict(fixture.X_holdout))
    unconstrained_predictions = np.asarray(unconstrained.predict(fixture.X_holdout))
    constrained_rounds = int(constrained.rounds_completed_)
    unconstrained_rounds = int(unconstrained.rounds_completed_)
    return {
        "scenario": scenario.name,
        "objective": scenario.objective,
        "direction": scenario.direction,
        "tree_growth": scenario.tree_growth,
        "seed": scenario.seed,
        "constrained_fit_seconds": float(constrained_seconds),
        "unconstrained_fit_seconds": float(unconstrained_seconds),
        "fit_time_ratio": float(constrained_seconds / max(unconstrained_seconds, np.finfo(float).tiny)),
        "constrained_loss": constrained_loss,
        "unconstrained_loss": unconstrained_loss,
        "constant_loss": _constant_loss(
            scenario.objective, fixture.y_train, fixture.y_holdout
        ),
        "checked_grid_pairs": check.checked_grid_pairs,
        "worst_signed_monotone_margin": check.worst_signed_monotone_margin,
        "violation_count": check.violation_count,
        "grid_predictions_finite": check.grid_predictions_finite,
        "constrained_finite": bool(
            np.isfinite(constrained_loss) and np.isfinite(constrained_predictions).all()
        ),
        "unconstrained_finite": bool(
            np.isfinite(unconstrained_loss) and np.isfinite(unconstrained_predictions).all()
        ),
        "constrained_completed": constrained_rounds == rounds,
        "unconstrained_completed": unconstrained_rounds == rounds,
        "constrained_completed_rounds": constrained_rounds,
        "unconstrained_completed_rounds": unconstrained_rounds,
        "requested_rounds": rounds,
    }


def _failed_record(scenario: Scenario, *, rounds: int, error: Exception) -> dict[str, object]:
    return {
        "scenario": scenario.name,
        "objective": scenario.objective,
        "direction": scenario.direction,
        "tree_growth": scenario.tree_growth,
        "seed": scenario.seed,
        "constrained_fit_seconds": float("nan"),
        "unconstrained_fit_seconds": float("nan"),
        "fit_time_ratio": float("nan"),
        "constrained_loss": float("nan"),
        "unconstrained_loss": float("nan"),
        "constant_loss": float("nan"),
        "checked_grid_pairs": 0,
        "worst_signed_monotone_margin": float("nan"),
        "violation_count": 0,
        "grid_predictions_finite": False,
        "constrained_finite": False,
        "unconstrained_finite": False,
        "constrained_completed": False,
        "unconstrained_completed": False,
        "constrained_completed_rounds": 0,
        "unconstrained_completed_rounds": 0,
        "requested_rounds": rounds,
        "error": str(error),
    }


def run_benchmark(*, quick: bool) -> dict[str, object]:
    """Fit deterministic constrained/unconstrained pairs and collect evidence."""
    scenarios = quick_scenarios() if quick else full_scenarios()
    rounds = QUICK_ROUNDS if quick else FULL_ROUNDS
    contexts = QUICK_CONTEXTS if quick else FULL_CONTEXTS
    records = []
    for scenario in scenarios:
        try:
            records.append(_fit_record(scenario, rounds=rounds, contexts=contexts))
        except Exception as error:  # Gate failures must remain reportable.
            records.append(_failed_record(scenario, rounds=rounds, error=error))
    return {
        "quick": quick,
        "rounds": rounds,
        "contexts": contexts,
        "scenarios": [scenario.name for scenario in scenarios],
        "scenario_specs": [asdict(scenario) for scenario in scenarios],
        "records": records,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool) and bool(np.isfinite(value))


def _canonical_contract(report: dict[str, object]) -> tuple[tuple[str, ...], int] | None:
    quick = report.get("quick")
    if not isinstance(quick, bool):
        return None
    scenarios = quick_scenarios() if quick else full_scenarios()
    rounds = QUICK_ROUNDS if quick else FULL_ROUNDS
    return tuple(scenario.name for scenario in scenarios), rounds


def _identity_failures(
    *,
    label: str,
    declared: object,
    expected: tuple[str, ...],
) -> tuple[list[str], dict[str, list[dict[str, object]]]]:
    if not isinstance(declared, list):
        return [f"report is missing {label}"], {}
    if not declared:
        return [f"{label} are empty"], {}

    values: list[str] = []
    records_by_identity: dict[str, list[dict[str, object]]] = {}
    for value in declared:
        if label == "record identities":
            if not isinstance(value, dict) or not isinstance(value.get("scenario"), str):
                return ["record identities include an invalid record"], {}
            identity = value["scenario"]
            records_by_identity.setdefault(identity, []).append(value)
        else:
            if not isinstance(value, str):
                return ["scenario declarations include an invalid identity"], {}
            identity = value
        values.append(identity)

    value_set = set(values)
    expected_set = set(expected)
    if len(value_set) != len(values) or value_set != expected_set:
        return [f"{label} do not exactly match canonical identities"], records_by_identity
    return [], records_by_identity


def evaluate_gate(report: dict[str, object]) -> list[str]:
    """Return acceptance failures; timing remains descriptive evidence only."""
    canonical = _canonical_contract(report)
    if canonical is None:
        return ["report quick must be a boolean"]
    expected, expected_rounds = canonical
    failures, _ = _identity_failures(
        label="scenario declarations",
        declared=report.get("scenarios"),
        expected=expected,
    )
    record_failures, by_scenario = _identity_failures(
        label="record identities",
        declared=report.get("records"),
        expected=expected,
    )
    failures.extend(record_failures)
    report_rounds = report.get("rounds")
    if (
        not isinstance(report_rounds, int)
        or isinstance(report_rounds, bool)
        or report_rounds != expected_rounds
    ):
        failures.append(f"report rounds must equal canonical value {expected_rounds}")

    for scenario in expected:
        if scenario not in by_scenario:
            failures.append(f"missing record for scenario {scenario}")
            continue
        if len(by_scenario[scenario]) != 1:
            failures.append(f"scenario {scenario} has duplicate records")
            continue
        record = by_scenario[scenario][0]
        context = f"scenario {scenario}"
        if record.get("error"):
            failures.append(f"{context}: fit error: {record['error']}")
        for field in (
            "constrained_loss",
            "unconstrained_loss",
            "constant_loss",
            "worst_signed_monotone_margin",
        ):
            if not _finite_number(record.get(field)):
                failures.append(f"{context}: non-finite {field}")
        if not record.get("constrained_finite") or not record.get("unconstrained_finite"):
            failures.append(f"{context}: non-finite fitted predictions or loss")
        if not record.get("grid_predictions_finite"):
            failures.append(f"{context}: non-finite grid predictions or differences")
        requested_rounds = record.get("requested_rounds")
        if (
            not isinstance(requested_rounds, int)
            or isinstance(requested_rounds, bool)
            or requested_rounds != expected_rounds
        ):
            failures.append(f"{context}: requested rounds must equal {expected_rounds}")
        if (
            not record.get("constrained_completed")
            or not record.get("unconstrained_completed")
            or (
                record.get("constrained_completed_rounds") != expected_rounds
                or record.get("unconstrained_completed_rounds") != expected_rounds
            )
        ):
            failures.append(f"{context}: incomplete rounds")
        if record.get("violation_count") != 0:
            failures.append(f"{context}: monotone violations detected")
        checked_pairs = record.get("checked_grid_pairs")
        if not isinstance(checked_pairs, int) or checked_pairs < GRID_VALUES - 1:
            failures.append(f"{context}: insufficient checked grid pairs")
        worst_margin = record.get("worst_signed_monotone_margin")
        if _finite_number(worst_margin) and float(worst_margin) < -MONOTONE_TOLERANCE:
            failures.append(f"{context}: worst margin exceeds tolerance")
        constrained_loss = record.get("constrained_loss")
        unconstrained_loss = record.get("unconstrained_loss")
        constant_loss = record.get("constant_loss")
        if not all(_finite_number(value) for value in (constrained_loss, unconstrained_loss, constant_loss)):
            continue
        if record.get("objective") == "regression":
            ratio = float(constrained_loss) / max(float(unconstrained_loss), np.finfo(float).tiny)
            if ratio > 1.25:
                failures.append(f"{context}: regression loss ratio {ratio:.6f} exceeds 1.25")
        elif record.get("objective") == "binary":
            degradation = float(constrained_loss) - float(unconstrained_loss)
            if degradation > 0.08:
                failures.append(f"{context}: binary error degradation {degradation:.6f} exceeds 0.08")
        else:
            failures.append(f"{context}: unknown objective")
        if float(constrained_loss) >= float(constant_loss):
            failures.append(f"{context}: constrained loss does not beat constant baseline")
    return failures


def render_markdown(report: dict[str, object]) -> str:
    """Render a self-contained benchmark report with descriptive timing."""
    records = report.get("records", [])
    lines = [
        "# Monotone Constraint Acceptance Benchmark",
        "",
        "## Environment",
        "",
    ]
    for key, value in report.get("environment", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Acceptance Contract",
            "",
            "- Strict zero monotone violations at tolerance `1e-6` with finite grid predictions and differences.",
            "- Regression constrained/unconstrained loss ratio at most `1.25`.",
            "- Binary constrained error degradation at most `0.08`.",
            "- The constrained model must beat the constant predictor.",
            "- Scenario rows are training rows; holdouts use an independent deterministic stream with 512 to 4,096 rows.",
            "- Fit timing is descriptive only and never gates acceptance.",
            "",
            "## Summary",
            "",
            f"- Scenarios: {len(report.get('scenarios', []))}",
            f"- Records: {len(records)}",
            f"- Gate failures: {len(evaluate_gate(report))}",
            "",
            "## Records",
            "",
            "| Scenario | Objective | Direction | Growth | Seed | Constrained fit s | Unconstrained fit s | Timing ratio | Constrained loss | Unconstrained loss | Constant loss | Grid pairs | Grid finite | Worst signed margin | Violations | Constrained rounds | Unconstrained rounds |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in records:
        lines.append(
            "| {scenario} | {objective} | {direction} | {tree_growth} | {seed} | "
            "{constrained_fit_seconds:.6f} | {unconstrained_fit_seconds:.6f} | "
            "{fit_time_ratio:.6f} | {constrained_loss:.6f} | {unconstrained_loss:.6f} | "
            "{constant_loss:.6f} | {checked_grid_pairs} | {grid_predictions_finite} | "
            "{worst_signed_monotone_margin:.6g} | {violation_count} | "
            "{constrained_completed_rounds} | {unconstrained_completed_rounds} |".format(**record)
        )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="run the compact CI matrix")
    parser.add_argument("--gate", action="store_true", help="return nonzero on acceptance failures")
    parser.add_argument("--output", type=str, help="write the Markdown report to this path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_benchmark(quick=args.quick)
    rendered = render_markdown(report)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    failures = evaluate_gate(report)
    if args.gate and failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
