#!/usr/bin/env python3
"""Per-mode quality gate for AlloyGBM's opt-in training modes.

Every optional mode (subsampling, GOSS, DART, MorphBoost, DRO leaves,
piecewise-linear leaves, leaf-wise growth, ...) is a separate code path
through the trainer, and each one has historically been able to regress
*silently*: the fit completes, the test suite passes, and only the model
quality is wrong. Three such regressions have actually shipped:

  * ``row_subsample`` / ``boosting_mode="goss"`` once committed trees only
    to sampled rows, so the deployed model diverged (measured test RMSE
    2.19 against a 1.15 constant-predictor baseline).
  * ``boosting_mode="dart"`` was aborted after one round by the
    training-loss gate, shipping a one-tree model.
  * ``leaf_model="linear"`` with ``lambda_l2=0`` produced unbounded leaves
    (measured training RMSE ~1e6).

Byte-equivalence tests cannot catch this class of bug — both sides of the
comparison share the defect — so this gate asserts *absolute* quality
instead, with two machine-independent invariants per mode:

  1. **Beats the constant predictor** by a healthy margin. Divergence
     (all three regressions above) violates this immediately.
  2. **Stays within a factor of the plain-boosting baseline** trained in
     the same run. This catches a mode that still "works" but has become
     much worse than simply not using it.

Both invariants are ratios computed within a single run, so they are
reproducible across CI runners and architectures where an absolute RMSE
threshold would not be. Seeds are fixed for reproducibility.

Coverage note: replaying the historical failures against this gate, the
subsampling/GOSS and DART regressions both trip invariant (1) comfortably.
The piecewise-linear blow-up does *not* reproduce on synthetic data -- it
turned out to be a numerical knife-edge tied to one specific split on one
specific real-world partition -- so the PL guards are pinned directly by
unit tests in ``crates/backend_cpu/src/pl.rs`` instead. The PL rows below
still guard against a gross PL regression; they are simply not the primary
net for that particular bug.

Usage:
  python benchmarks/mode_quality_gate.py            # full report
  python benchmarks/mode_quality_gate.py --quick    # compact CI matrix
  python benchmarks/mode_quality_gate.py --quick --gate  # nonzero on failure
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import sys
from typing import Any

import numpy as np

from alloygbm import GBMClassifier, GBMRegressor

# A mode must beat the constant predictor by at least this much: its RMSE
# must be below CONSTANT_RATIO_CEILING * std(y). Loose enough to accommodate
# genuinely lossy modes (heavy subsampling, aggressive DART dropout), tight
# enough that divergence cannot slip through.
CONSTANT_RATIO_CEILING = 0.85

# A mode's RMSE may not exceed BASELINE_RATIO_CEILING * the plain-boosting
# baseline's RMSE from the same run.
BASELINE_RATIO_CEILING = 2.5

# Classification modes are scored on accuracy instead of RMSE.
CLASSIFICATION_MIN_ACCURACY = 0.70
CLASSIFICATION_BASELINE_SLACK = 0.10


@dataclass(frozen=True)
class ModeCase:
    """One opt-in mode, expressed as kwargs layered on the baseline."""

    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    # Modes that are lossy by construction get a looser baseline bound.
    baseline_ratio_ceiling: float = BASELINE_RATIO_CEILING


def regression_modes() -> list[ModeCase]:
    return [
        ModeCase("row_subsample_0.5", {"row_subsample": 0.5}),
        ModeCase("col_subsample_0.6", {"col_subsample": 0.6}),
        ModeCase("colsample_bynode_0.6", {"colsample_bynode": 0.6}),
        ModeCase("goss", {"boosting_mode": "goss"}),
        # DART re-weights the whole ensemble every round; on a short fit it
        # is expected to trail plain boosting noticeably.
        ModeCase("dart", {"boosting_mode": "dart"}, baseline_ratio_ceiling=3.0),
        ModeCase("leaf_wise", {"tree_growth": "leaf", "max_leaves": 31}),
        ModeCase("morph", {"training_mode": "morph"}),
        ModeCase("dro_leaves", {"leaf_solver": "dro", "dro_radius": 0.05}),
        # The λ=0 case is the exact configuration that diverged; keep it.
        ModeCase("pl_leaves_unregularized", {"leaf_model": "linear"}, baseline_ratio_ceiling=3.0),
        ModeCase(
            "pl_leaves_regularized",
            {"leaf_model": "linear", "lambda_l2": 5.0},
        ),
        ModeCase(
            "pl_leaves_shortlist",
            {"leaf_model": "linear", "pl_split_candidates": 4},
            baseline_ratio_ceiling=3.0,
        ),
        ModeCase("quantile_objective", {"objective": "quantile", "quantile_alpha": 0.5}),
        ModeCase("linear_binning", {"continuous_binning_strategy": "linear"}),
    ]


def classification_modes() -> list[ModeCase]:
    return [
        ModeCase("binary_goss", {"boosting_mode": "goss"}),
        ModeCase("binary_dart", {"boosting_mode": "dart"}),
        ModeCase("binary_subsample", {"row_subsample": 0.5}),
        ModeCase("multiclass_goss", {"boosting_mode": "goss"}),
        ModeCase("multiclass_subsample", {"row_subsample": 0.5}),
    ]


def _regression_data(n_rows: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    signal = 1.3 * X[:, 0] - X[:, 2] ** 2 + 0.7 * X[:, 1] * X[:, 3]
    y = (signal + rng.normal(scale=0.3, size=n_rows)).astype(np.float32)
    split = int(n_rows * 0.75)
    return X[:split], y[:split], X[split:], y[split:]


def _classification_data(n_rows: int, n_features: int, seed: int, n_classes: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    signal = 1.3 * X[:, 0] - X[:, 2] ** 2 + 0.7 * X[:, 1] * X[:, 3]
    if n_classes == 2:
        y = (rng.random(n_rows) < 1.0 / (1.0 + np.exp(-signal))).astype(int)
    else:
        cuts = np.quantile(signal, np.linspace(0, 1, n_classes + 1)[1:-1])
        y = np.digitize(signal, cuts)
    split = int(n_rows * 0.75)
    return X[:split], y[:split], X[split:], y[split:]


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


@dataclass
class ModeResult:
    name: str
    metric: str
    value: float
    constant_reference: float
    baseline_value: float
    baseline_ratio_ceiling: float
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def _evaluate_regression(
    case: ModeCase,
    base_kwargs: dict[str, Any],
    data,
) -> ModeResult | None:
    X_train, y_train, X_test, y_test = data
    kwargs = {**base_kwargs, **case.kwargs}
    model = GBMRegressor(**kwargs).fit(X_train, y_train)
    predictions = np.asarray(model.predict(X_test))
    return ModeResult(
        name=case.name,
        metric="rmse",
        value=_rmse(y_test, predictions),
        constant_reference=float(np.std(np.asarray(y_test, dtype=np.float64))),
        baseline_value=float("nan"),
        baseline_ratio_ceiling=case.baseline_ratio_ceiling,
    )


def _evaluate_classification(
    case: ModeCase,
    base_kwargs: dict[str, Any],
    data,
) -> ModeResult:
    X_train, y_train, X_test, y_test = data
    kwargs = {**base_kwargs, **case.kwargs}
    model = GBMClassifier(**kwargs).fit(X_train, y_train)
    predictions = np.asarray(model.predict(X_test))
    accuracy = float(np.mean(predictions == np.asarray(y_test)))
    majority = float(max(np.mean(np.asarray(y_test) == label) for label in np.unique(y_test)))
    return ModeResult(
        name=case.name,
        metric="accuracy",
        value=accuracy,
        constant_reference=majority,
        baseline_value=float("nan"),
        baseline_ratio_ceiling=case.baseline_ratio_ceiling,
    )


def run(quick: bool) -> list[ModeResult]:
    n_rows = 4_000 if quick else 12_000
    n_features = 10 if quick else 16
    rounds = 40 if quick else 120
    base = {
        "n_estimators": rounds,
        "learning_rate": 0.1,
        "max_depth": 5,
        "seed": 17,
    }

    results: list[ModeResult] = []

    regression_data = _regression_data(n_rows, n_features, seed=1)
    baseline = _evaluate_regression(ModeCase("baseline"), base, regression_data)
    assert baseline is not None
    results.append(baseline)
    for case in regression_modes():
        result = _evaluate_regression(case, base, regression_data)
        if result is None:
            continue
        result.baseline_value = baseline.value
        results.append(result)

    for n_classes, cases in (
        (2, [c for c in classification_modes() if c.name.startswith("binary")]),
        (3, [c for c in classification_modes() if c.name.startswith("multiclass")]),
    ):
        data = _classification_data(n_rows, n_features, seed=2, n_classes=n_classes)
        label = "binary" if n_classes == 2 else "multiclass"
        cls_baseline = _evaluate_classification(
            ModeCase(f"{label}_baseline"), base, data
        )
        results.append(cls_baseline)
        for case in cases:
            result = _evaluate_classification(case, base, data)
            result.baseline_value = cls_baseline.value
            results.append(result)

    for result in results:
        result.failures.extend(_check(result))
    return results


def _check(result: ModeResult) -> list[str]:
    failures: list[str] = []
    if not np.isfinite(result.value):
        return [f"{result.metric} is not finite ({result.value})"]

    if result.metric == "rmse":
        ceiling = CONSTANT_RATIO_CEILING * result.constant_reference
        if result.value > ceiling:
            failures.append(
                f"rmse {result.value:.4f} exceeds "
                f"{CONSTANT_RATIO_CEILING:g}x the constant-predictor rmse "
                f"({result.constant_reference:.4f})"
            )
        if np.isfinite(result.baseline_value) and result.baseline_value > 0.0:
            limit = result.baseline_ratio_ceiling * result.baseline_value
            if result.value > limit:
                failures.append(
                    f"rmse {result.value:.4f} exceeds "
                    f"{result.baseline_ratio_ceiling:g}x the plain-boosting "
                    f"baseline ({result.baseline_value:.4f})"
                )
    else:
        if result.value < CLASSIFICATION_MIN_ACCURACY:
            failures.append(
                f"accuracy {result.value:.4f} below the "
                f"{CLASSIFICATION_MIN_ACCURACY:g} floor"
            )
        if result.value < result.constant_reference:
            failures.append(
                f"accuracy {result.value:.4f} below the majority-class rate "
                f"({result.constant_reference:.4f})"
            )
        if np.isfinite(result.baseline_value):
            floor = result.baseline_value - CLASSIFICATION_BASELINE_SLACK
            if result.value < floor:
                failures.append(
                    f"accuracy {result.value:.4f} is more than "
                    f"{CLASSIFICATION_BASELINE_SLACK:g} below the baseline "
                    f"({result.baseline_value:.4f})"
                )
    return failures


def report(results: list[ModeResult]) -> None:
    width = max(len(r.name) for r in results)
    print(f"{'mode'.ljust(width)}  metric     value   reference   baseline  status")
    for result in results:
        baseline = (
            f"{result.baseline_value:9.4f}"
            if np.isfinite(result.baseline_value)
            else "        -"
        )
        status = "ok" if result.ok else "FAIL"
        print(
            f"{result.name.ljust(width)}  {result.metric:8s} "
            f"{result.value:7.4f}  {result.constant_reference:9.4f} {baseline}  {status}"
        )
        for failure in result.failures:
            print(f"{' ' * width}    -> {failure}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="compact CI matrix")
    parser.add_argument(
        "--gate", action="store_true", help="exit nonzero when any mode fails"
    )
    args = parser.parse_args(argv)

    results = run(quick=args.quick)
    report(results)

    failed = [r for r in results if not r.ok]
    if failed:
        print(f"\n{len(failed)} mode(s) failed the quality gate.")
        if args.gate:
            return 1
    else:
        print(f"\nAll {len(results)} modes passed the quality gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
