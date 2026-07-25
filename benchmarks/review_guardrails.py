"""Deterministic experiments supporting July-review evidence guardrails."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np


DEFAULT_SEEDS = (7, 13, 29)
ALL_SECTIONS = ("quantile", "goss", "dart")
QUANTILE_ALPHAS = (0.1, 0.5, 0.9)
GOSS_RATES = ((0.1, 0.1), (0.2, 0.1), (0.2, 0.2), (0.3, 0.1))
QUICK_DART_CONFIGS = ((8, 0.1, 5), (16, 0.2, 5))
FULL_DART_CONFIGS = (
    (50, 0.05, 50),
    (100, 0.10, 50),
    (200, 0.20, 50),
    (100, 0.10, 5),
    (100, 0.10, 20),
)
DART_PROFILE_STANDARD = "standard_control"
DART_PROFILE_DEFAULT_LIKE = "default_like"
DART_PROFILE_STRESS = "stress_profile"
DART_PROFILES = {
    DART_PROFILE_STANDARD,
    DART_PROFILE_DEFAULT_LIKE,
    DART_PROFILE_STRESS,
}


@dataclass(frozen=True)
class QuantileSplitRow:
    seed: int
    alpha: float
    arm: str
    threshold: float
    gain: float
    pinball_loss: float
    baseline_loss: float
    left_count: int
    right_count: int


@dataclass(frozen=True)
class BoostingRow:
    section: str
    seed: int
    arm: str
    rmse: float
    baseline_rmse: float
    fit_seconds: float
    completed_rounds: int
    requested_rounds: int
    retained_fraction: float | None = None
    matched_control: str | None = None
    dropout_pressure: float | None = None
    dart_profile: str | None = None


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str


def make_quantile_split_data(
    *, seed: int, n_train: int, n_test: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic one-feature, heteroscedastic quantile fixture."""
    if n_train < 1 or n_test < 1:
        raise ValueError("n_train and n_test must be positive")

    rng = np.random.default_rng(seed)

    def make_partition(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feature = rng.uniform(-2.5, 2.5, size=size)
        location = 0.08 * np.sin(1.4 * feature) + 0.025 * feature**2 - 0.015 * feature
        scale = 0.25 + 0.14 * np.abs(feature)
        noise = scale * (
            0.15 * (rng.exponential(size=size) - 1.0)
            + 0.85 * rng.uniform(-1.0, 1.0, size=size)
        )
        weights = rng.lognormal(mean=0.0, sigma=0.10, size=size)
        return feature, location + noise, weights

    x_train, y_train, w_train = make_partition(n_train)
    x_test, y_test, w_test = make_partition(n_test)
    return tuple(
        np.ascontiguousarray(values, dtype=np.float64)
        for values in (x_train, y_train, w_train, x_test, y_test, w_test)
    )  # type: ignore[return-value]


def make_boosting_data(
    *, seed: int, n_train: int, n_test: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build deterministic float32 train/test data for boosting experiments."""
    if n_train < 1 or n_test < 1:
        raise ValueError("n_train and n_test must be positive")
    rng = np.random.default_rng(seed)

    def make_partition(size: int) -> tuple[np.ndarray, np.ndarray]:
        features = rng.normal(size=(size, 5)).astype(np.float32)
        target = (
            1.1 * np.sin(features[:, 0])
            + 0.65 * features[:, 1] * features[:, 2]
            - 0.3 * features[:, 3] ** 2
            + 0.2 * features[:, 4]
            + rng.normal(scale=0.18, size=size)
        )
        return np.ascontiguousarray(features), np.ascontiguousarray(target, dtype=np.float32)

    x_train, y_train = make_partition(n_train)
    x_test, y_test = make_partition(n_test)
    return x_train, y_train, x_test, y_test


def configured_dropout_pressure(*, n_estimators: int, drop_rate: float, max_drop: int) -> float:
    """Estimate configured DART dropout work, not the observed drop count."""
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")
    if not np.isfinite(drop_rate) or not 0.0 < drop_rate < 1.0:
        raise ValueError("drop_rate must be finite and in (0.0, 1.0)")
    if max_drop < 1:
        raise ValueError("max_drop must be positive")
    return float(
        sum(
            min(max_drop, max(1.0, drop_rate * existing_rounds))
            for existing_rounds in range(1, n_estimators)
        )
    )


def _rmse(y_true: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(prediction)) ** 2)))


def _median(values: Sequence[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def _finite_positive(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0.0)


def _is_matching_goss_control(row: BoostingRow, control: BoostingRow) -> bool:
    """Require a GOSS arm's uniform control to retain the same row fraction."""
    if row.retained_fraction is None or control.retained_fraction is None:
        return False
    if not np.isfinite(row.retained_fraction) or not np.isfinite(control.retained_fraction):
        return False
    return (
        control.section == "goss"
        and control.arm == f"uniform_{row.retained_fraction:.2f}"
        and control.retained_fraction == row.retained_fraction
    )


def _is_matching_dart_control(row: BoostingRow, control: BoostingRow) -> bool:
    """Require a DART arm's standard control to use the same fit horizon."""
    return (
        control.section == "dart"
        and control.dart_profile == DART_PROFILE_STANDARD
        and control.arm == f"standard_{row.requested_rounds}"
        and control.requested_rounds == row.requested_rounds
    )


def evaluate_gates(
    quantile_rows: Sequence[QuantileSplitRow],
    goss_rows: Sequence[BoostingRow],
    dart_rows: Sequence[BoostingRow],
) -> list[GateResult]:
    """Evaluate complete evidence contracts without making timing a quality gate."""
    required_quantile_arms = {"proxy", "smooth_0.05", "smooth_0.10"}
    quantile_keys = [(row.seed, row.alpha, row.arm) for row in quantile_rows]
    quantile_groups: dict[tuple[int, float], set[str]] = {}
    for row in quantile_rows:
        quantile_groups.setdefault((row.seed, row.alpha), set()).add(row.arm)
    quantile_values_valid = all(
        np.isfinite(
            [row.threshold, row.gain, row.pinball_loss, row.baseline_loss]
        ).all()
        and row.left_count > 0
        and row.right_count > 0
        for row in quantile_rows
    )
    quantile_contract = (
        bool(quantile_rows)
        and len(quantile_keys) == len(set(quantile_keys))
        and all(arms == required_quantile_arms for arms in quantile_groups.values())
        and quantile_values_valid
    )

    quantile_ratios = []
    for alpha, arm in sorted({(row.alpha, row.arm) for row in quantile_rows}):
        arm_rows = [row for row in quantile_rows if row.alpha == alpha and row.arm == arm]
        median_loss = _median([row.pinball_loss for row in arm_rows])
        median_baseline = _median([row.baseline_loss for row in arm_rows])
        quantile_ratios.append(
            median_loss / median_baseline
            if _finite_positive(median_baseline) and np.isfinite(median_loss)
            else float("inf")
        )
    quantile_quality = bool(quantile_ratios) and all(ratio <= 1.10 for ratio in quantile_ratios)

    goss_keys = [(row.seed, row.arm) for row in goss_rows]
    goss_by_seed_arm = {(row.seed, row.arm): row for row in goss_rows}
    goss_arms = [row for row in goss_rows if row.arm.startswith("goss_")]
    goss_values_valid = all(
        row.section == "goss"
        and np.isfinite([row.rmse, row.baseline_rmse, row.fit_seconds]).all()
        and row.fit_seconds > 0.0
        and row.completed_rounds > 0
        and row.requested_rounds > 0
        for row in goss_rows
    )
    goss_controls_present = all(
        row.matched_control is not None
        and (row.seed, row.matched_control) in goss_by_seed_arm
        and _is_matching_goss_control(
            row,
            goss_by_seed_arm[(row.seed, row.matched_control)],
        )
        for row in goss_arms
    )
    goss_contract = (
        bool(goss_rows)
        and len(goss_keys) == len(set(goss_keys))
        and all((seed, "standard_full") in goss_by_seed_arm for seed, _ in goss_keys)
        and bool(goss_arms)
        and goss_values_valid
        and goss_controls_present
    )
    goss_completion = bool(goss_rows) and all(
        row.completed_rounds == row.requested_rounds for row in goss_rows
    )

    goss_ratios: list[float] = []
    for arm in sorted({row.arm for row in goss_arms}):
        arm_rows = [row for row in goss_arms if row.arm == arm]
        control_rows = [
            goss_by_seed_arm[(row.seed, row.matched_control)]
            for row in arm_rows
            if row.matched_control is not None and (row.seed, row.matched_control) in goss_by_seed_arm
        ]
        goss_ratios.append(
            _median([row.rmse for row in arm_rows]) / _median([row.rmse for row in control_rows])
            if control_rows and _finite_positive(_median([row.rmse for row in control_rows]))
            else float("inf")
        )
    goss_quality = bool(goss_ratios) and all(ratio <= 1.35 for ratio in goss_ratios)
    goss_baseline = bool(goss_arms) and all(
        _median([row.rmse for row in goss_arms if row.arm == arm])
        < _median([row.baseline_rmse for row in goss_arms if row.arm == arm])
        for arm in {row.arm for row in goss_arms}
    )

    dart_keys = [(row.seed, row.arm) for row in dart_rows]
    dart_by_seed_arm = {(row.seed, row.arm): row for row in dart_rows}
    dart_arms = [
        row
        for row in dart_rows
        if row.dart_profile in (DART_PROFILE_DEFAULT_LIKE, DART_PROFILE_STRESS)
    ]
    dart_quality_rows = [
        row for row in dart_arms if row.dart_profile == DART_PROFILE_DEFAULT_LIKE
    ]
    dart_values_valid = all(
        row.section == "dart"
        and np.isfinite([row.rmse, row.baseline_rmse, row.fit_seconds]).all()
        and row.fit_seconds > 0.0
        and row.completed_rounds > 0
        and row.requested_rounds > 0
        for row in dart_rows
    )
    dart_controls_present = all(
        row.matched_control is not None
        and (row.seed, row.matched_control) in dart_by_seed_arm
        and _is_matching_dart_control(
            row,
            dart_by_seed_arm[(row.seed, row.matched_control)],
        )
        for row in dart_arms
    )
    dart_contract = (
        bool(dart_rows)
        and len(dart_keys) == len(set(dart_keys))
        and bool(dart_arms)
        and all(row.dart_profile in DART_PROFILES for row in dart_rows)
        and dart_values_valid
        and dart_controls_present
        and all(
            row.dropout_pressure is not None and _finite_positive(row.dropout_pressure)
            for row in dart_arms
        )
    )
    dart_completion = bool(dart_rows) and all(
        row.completed_rounds == row.requested_rounds for row in dart_rows
    )
    dart_ratios: list[float] = []
    for arm in sorted({row.arm for row in dart_quality_rows}):
        arm_rows = [row for row in dart_quality_rows if row.arm == arm]
        control_rows = [
            dart_by_seed_arm[(row.seed, row.matched_control)]
            for row in arm_rows
            if row.matched_control is not None and (row.seed, row.matched_control) in dart_by_seed_arm
        ]
        dart_ratios.append(
            _median([row.rmse for row in arm_rows]) / _median([row.rmse for row in control_rows])
            if control_rows and _finite_positive(_median([row.rmse for row in control_rows]))
            else float("inf")
        )
    dart_quality = all(ratio <= 1.50 for ratio in dart_ratios)

    return [
        GateResult("quantile_contract", quantile_contract, "required arms, finite values, unique rows, and children"),
        GateResult(
            "quantile_quality",
            quantile_quality,
            f"maximum loss/no-split ratio={max(quantile_ratios, default=float('inf')):.3f} (limit 1.100)",
        ),
        GateResult("goss_contract", goss_contract, "required controls, finite metrics, and unique rows"),
        GateResult("goss_completion", goss_completion, "all GOSS fits completed requested rounds"),
        GateResult(
            "goss_quality",
            goss_quality,
            f"maximum GOSS/uniform ratio={max(goss_ratios, default=float('inf')):.3f} (limit 1.350)",
        ),
        GateResult("goss_baseline", goss_baseline, "every GOSS median beats its mean-predictor baseline"),
        GateResult(
            "dart_contract",
            dart_contract,
            "explicit profiles, required controls, finite metrics, unique rows, and pressure",
        ),
        GateResult("dart_completion", dart_completion, "all DART fits completed requested rounds"),
        GateResult(
            "dart_quality",
            dart_quality,
            "maximum default-like DART/standard ratio="
            f"{max(dart_ratios, default=0.0):.3f} (limit 1.500; stress/profile arms excluded)",
        ),
    ]


def run_goss_experiment(
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_train: int = 512,
    n_test: int = 256,
    n_estimators: int = 100,
    rates: tuple[tuple[float, float], ...] = GOSS_RATES,
) -> list[BoostingRow]:
    """Compare full, uniform-subsample, and GOSS models for each rate pair."""
    from alloygbm import GBMRegressor

    def fit_row(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        *,
        seed: int,
        arm: str,
        retained_fraction: float | None = None,
        matched_control: str | None = None,
        **mode_params: float | str,
    ) -> BoostingRow:
        model = GBMRegressor(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.06,
            lambda_l2=1.0,
            seed=seed,
            deterministic=True,
            training_policy="manual",
            continuous_binning_strategy="quantile",
            **mode_params,
        )
        started = time.perf_counter()
        model.fit(x_train, y_train)
        elapsed = max(time.perf_counter() - started, np.finfo(np.float64).eps)
        baseline_prediction = np.full_like(y_test, np.mean(y_train), dtype=np.float32)
        return BoostingRow(
            section="goss",
            seed=seed,
            arm=arm,
            rmse=_rmse(y_test, model.predict(x_test)),
            baseline_rmse=_rmse(y_test, baseline_prediction),
            fit_seconds=float(elapsed),
            completed_rounds=int(model.n_estimators_),
            requested_rounds=n_estimators,
            retained_fraction=retained_fraction,
            matched_control=matched_control,
        )

    rows: list[BoostingRow] = []
    for seed in seeds:
        x_train, y_train, x_test, y_test = make_boosting_data(
            seed=seed, n_train=n_train, n_test=n_test
        )
        rows.append(
            fit_row(
                x_train,
                y_train,
                x_test,
                y_test,
                seed=seed,
                arm="standard_full",
                boosting_mode="standard",
            )
        )
        uniform_rows: dict[float, BoostingRow] = {}
        for top_rate, other_rate in rates:
            retained = top_rate + other_rate
            uniform_arm = f"uniform_{retained:.2f}"
            if retained not in uniform_rows:
                uniform_rows[retained] = fit_row(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    seed=seed,
                    arm=uniform_arm,
                    retained_fraction=retained,
                    boosting_mode="standard",
                    row_subsample=retained,
                )
                rows.append(uniform_rows[retained])
            rows.append(
                fit_row(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    seed=seed,
                    arm=f"goss_{top_rate:.2f}_{other_rate:.2f}",
                    retained_fraction=retained,
                    matched_control=uniform_arm,
                    boosting_mode="goss",
                    goss_top_rate=top_rate,
                    goss_other_rate=other_rate,
                )
            )
    return rows


def run_dart_experiment(
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_train: int = 512,
    n_test: int = 256,
    configs: tuple[tuple[int, float, int], ...] = (
        (50, 0.05, 50),
        (100, 0.10, 50),
        (200, 0.20, 50),
        (100, 0.10, 5),
        (100, 0.10, 20),
    ),
) -> list[BoostingRow]:
    """Profile DART versus standard boosting across representative settings."""
    from alloygbm import GBMRegressor

    def fit_row(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        *,
        seed: int,
        arm: str,
        horizon: int,
        matched_control: str | None = None,
        dropout_pressure: float | None = None,
        dart_profile: str = DART_PROFILE_STANDARD,
        **mode_params: float | int | str,
    ) -> BoostingRow:
        model = GBMRegressor(
            n_estimators=horizon,
            max_depth=4,
            learning_rate=0.06,
            lambda_l2=1.0,
            seed=seed,
            deterministic=True,
            training_policy="manual",
            continuous_binning_strategy="quantile",
            **mode_params,
        )
        started = time.perf_counter()
        model.fit(x_train, y_train)
        elapsed = max(time.perf_counter() - started, np.finfo(np.float64).eps)
        baseline_prediction = np.full_like(y_test, np.mean(y_train), dtype=np.float32)
        return BoostingRow(
            section="dart",
            seed=seed,
            arm=arm,
            rmse=_rmse(y_test, model.predict(x_test)),
            baseline_rmse=_rmse(y_test, baseline_prediction),
            fit_seconds=float(elapsed),
            completed_rounds=int(model.n_estimators_),
            requested_rounds=horizon,
            matched_control=matched_control,
            dropout_pressure=dropout_pressure,
            dart_profile=dart_profile,
        )

    rows: list[BoostingRow] = []
    for seed in seeds:
        x_train, y_train, x_test, y_test = make_boosting_data(
            seed=seed, n_train=n_train, n_test=n_test
        )
        standard_rows: dict[int, BoostingRow] = {}
        for horizon, drop_rate, max_drop in configs:
            standard_arm = f"standard_{horizon}"
            if horizon not in standard_rows:
                standard_rows[horizon] = fit_row(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    seed=seed,
                    arm=standard_arm,
                    horizon=horizon,
                    dart_profile=DART_PROFILE_STANDARD,
                    boosting_mode="standard",
                )
                rows.append(standard_rows[horizon])
            rows.append(
                fit_row(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    seed=seed,
                    arm=f"dart_{horizon}_{drop_rate:.2f}_{max_drop}",
                    horizon=horizon,
                    matched_control=standard_arm,
                    dropout_pressure=configured_dropout_pressure(
                        n_estimators=horizon, drop_rate=drop_rate, max_drop=max_drop
                    ),
                    dart_profile=(
                        DART_PROFILE_DEFAULT_LIKE
                        if drop_rate <= 0.10
                        else DART_PROFILE_STRESS
                    ),
                    boosting_mode="dart",
                    dart_drop_rate=drop_rate,
                    dart_max_drop=max_drop,
                )
            )
    return rows


def _validate_alpha(alpha: float) -> None:
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0.0, 1.0]")


def weighted_quantile(values: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """Return the first stable-sorted value reaching the weighted quantile."""
    _validate_alpha(alpha)
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.ndim != 1 or values.shape != weights.shape:
        raise ValueError("values and weights must be one-dimensional arrays of equal length")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("values and weights must be finite")
    if (weights < 0.0).any():
        raise ValueError("weights must be non-negative")

    keep = weights > 0.0
    if not keep.any():
        raise ValueError("weights must have positive total weight")
    values = values[keep]
    weights = weights[keep]
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative_weights = np.cumsum(weights[order])
    target = alpha * cumulative_weights[-1]
    return float(ordered_values[np.searchsorted(cumulative_weights, target, side="left")])


def pinball_loss(
    y_true: np.ndarray, prediction: np.ndarray | float, weights: np.ndarray, alpha: float
) -> float:
    """Return mean weighted pinball loss for a scalar or vector prediction."""
    _validate_alpha(alpha)
    y_true = np.asarray(y_true, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if y_true.shape != weights.shape:
        raise ValueError("y_true and weights must have matching shapes")
    if not np.isfinite(y_true).all() or not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("targets and weights must be finite, with non-negative weights")
    if not np.any(weights > 0.0):
        raise ValueError("weights must have positive total weight")
    residual = y_true - prediction
    loss = np.where(residual >= 0.0, alpha * residual, (alpha - 1.0) * residual)
    return float(np.average(loss, weights=weights))


def proxy_pinball_grad_hess(
    residual: np.ndarray, weights: np.ndarray, *, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return AlloyGBM's constant-Hessian pinball split proxy."""
    _validate_alpha(alpha)
    residual = np.asarray(residual, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if residual.shape != weights.shape:
        raise ValueError("residual and weights must have matching shapes")
    gradient = np.where(residual < 0.0, 1.0 - alpha, -alpha) * weights
    return gradient, weights.copy()


def smoothed_pinball_grad_hess(
    residual: np.ndarray, weights: np.ndarray, *, alpha: float, width: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return asymmetric Huberized-pinball gradients and Hessians."""
    _validate_alpha(alpha)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("width must be finite and positive")
    residual = np.asarray(residual, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if residual.shape != weights.shape:
        raise ValueError("residual and weights must have matching shapes")
    left_width = width * (1.0 - alpha)
    right_width = width * alpha
    gradient = np.where(
        residual <= -left_width,
        1.0 - alpha,
        np.where(
            residual >= right_width,
            -alpha,
            (1.0 - alpha) - (residual + left_width) / width,
        ),
    )
    hessian = np.where(
        (residual > -left_width) & (residual < right_width),
        1.0 / width,
        0.0,
    )
    return gradient * weights, hessian * weights


def select_quantile_split(
    feature: np.ndarray,
    gradient: np.ndarray,
    hessian: np.ndarray,
    *,
    min_rows: int = 8,
) -> tuple[float, float, int, int]:
    """Select the highest-gain valid threshold among adjacent feature values."""
    feature = np.asarray(feature, dtype=np.float64)
    gradient = np.asarray(gradient, dtype=np.float64)
    hessian = np.asarray(hessian, dtype=np.float64)
    if (
        feature.ndim != 1
        or gradient.shape != feature.shape
        or hessian.shape != feature.shape
        or min_rows < 1
    ):
        raise ValueError("feature, gradient, and hessian must be aligned one-dimensional arrays")
    if not np.isfinite(feature).all() or not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
        raise ValueError("split inputs must be finite")
    if (hessian < 0.0).any():
        raise ValueError("hessian must be non-negative")

    order = np.argsort(feature, kind="stable")
    sorted_feature = feature[order]
    sorted_gradient = gradient[order]
    sorted_hessian = hessian[order]
    left_gradient = np.cumsum(sorted_gradient)
    left_hessian = np.cumsum(sorted_hessian)
    parent_gradient = float(left_gradient[-1])
    parent_hessian = float(left_hessian[-1])

    best: tuple[float, float, int, int] | None = None
    total_rows = feature.size
    for boundary in range(min_rows, total_rows - min_rows + 1):
        if sorted_feature[boundary - 1] == sorted_feature[boundary]:
            continue
        right_rows = total_rows - boundary
        left_grad = float(left_gradient[boundary - 1])
        left_hess = float(left_hessian[boundary - 1])
        right_grad = parent_gradient - left_grad
        right_hess = parent_hessian - left_hess
        if left_hess <= 0.0 or right_hess <= 0.0:
            continue
        gain = 0.5 * (
            left_grad**2 / (left_hess + 1.0)
            + right_grad**2 / (right_hess + 1.0)
            - parent_gradient**2 / (parent_hessian + 1.0)
        )
        threshold = 0.5 * (sorted_feature[boundary - 1] + sorted_feature[boundary])
        candidate = (float(threshold), float(gain), boundary, right_rows)
        if best is None or candidate[1] > best[1]:
            best = candidate
    if best is None:
        raise ValueError("no valid split has positive Hessian children")
    return best


def run_quantile_experiment(
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    alphas: tuple[float, ...] = QUANTILE_ALPHAS,
    n_train: int = 512,
    n_test: int = 256,
) -> list[QuantileSplitRow]:
    """Compare split choices while retaining empirical-quantile leaf values."""
    rows: list[QuantileSplitRow] = []
    for seed in seeds:
        x_train, y_train, w_train, x_test, y_test, w_test = make_quantile_split_data(
            seed=seed, n_train=n_train, n_test=n_test
        )
        for alpha in alphas:
            baseline_prediction = weighted_quantile(y_train, w_train, alpha)
            baseline_loss = pinball_loss(y_test, baseline_prediction, w_test, alpha)
            residual = y_train - baseline_prediction
            mad = float(np.median(np.abs(residual - np.median(residual))))
            scale = mad if mad > 0.0 else np.finfo(np.float64).eps
            arms = [
                ("proxy", proxy_pinball_grad_hess(residual, w_train, alpha=alpha)),
                (
                    "smooth_0.05",
                    smoothed_pinball_grad_hess(residual, w_train, alpha=alpha, width=0.05 * scale),
                ),
                (
                    "smooth_0.10",
                    smoothed_pinball_grad_hess(residual, w_train, alpha=alpha, width=0.10 * scale),
                ),
            ]
            for arm, (gradient, hessian) in arms:
                threshold, gain, left_count, right_count = select_quantile_split(
                    x_train, gradient, hessian
                )
                left_train = x_train <= threshold
                left_prediction = weighted_quantile(y_train[left_train], w_train[left_train], alpha)
                right_prediction = weighted_quantile(y_train[~left_train], w_train[~left_train], alpha)
                prediction = np.where(x_test <= threshold, left_prediction, right_prediction)
                rows.append(
                    QuantileSplitRow(
                        seed=seed,
                        alpha=alpha,
                        arm=arm,
                        threshold=threshold,
                        gain=gain,
                        pinball_loss=pinball_loss(y_test, prediction, w_test, alpha),
                        baseline_loss=baseline_loss,
                        left_count=left_count,
                        right_count=right_count,
                    )
                )
    return rows


def run_benchmark(
    *,
    sections: Sequence[str] = ALL_SECTIONS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    quick: bool = False,
) -> tuple[list[QuantileSplitRow], list[BoostingRow], list[BoostingRow]]:
    """Run the requested deterministic evidence sections."""
    selected = tuple(dict.fromkeys(sections))
    unknown_sections = set(selected).difference(ALL_SECTIONS)
    if unknown_sections:
        raise ValueError(f"unknown benchmark sections: {', '.join(sorted(unknown_sections))}")
    if not selected:
        raise ValueError("at least one benchmark section is required")
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("at least one seed is required")
    if quick:
        seed_values = seed_values[:1]

    quantile_rows: list[QuantileSplitRow] = []
    goss_rows: list[BoostingRow] = []
    dart_rows: list[BoostingRow] = []
    if "quantile" in selected:
        quantile_rows = run_quantile_experiment(
            seeds=seed_values,
            n_train=160 if quick else 512,
            n_test=96 if quick else 256,
        )
    if "goss" in selected:
        goss_rows = run_goss_experiment(
            seeds=seed_values,
            n_train=256 if quick else 512,
            n_test=128 if quick else 256,
            n_estimators=8 if quick else 100,
        )
    if "dart" in selected:
        dart_rows = run_dart_experiment(
            seeds=seed_values,
            n_train=256 if quick else 512,
            n_test=128 if quick else 256,
            configs=QUICK_DART_CONFIGS if quick else FULL_DART_CONFIGS,
        )
    return quantile_rows, goss_rows, dart_rows


def _median_quantile_rows(
    rows: Sequence[QuantileSplitRow], arm: str, alpha: float
) -> tuple[float, float, float]:
    selected = [row for row in rows if row.arm == arm and row.alpha == alpha]
    return (
        _median([row.pinball_loss for row in selected]),
        _median([row.baseline_loss for row in selected]),
        _median([row.gain for row in selected]),
    )


def _median_boosting_rows(rows: Sequence[BoostingRow], arm: str) -> tuple[float, float, float, float]:
    selected = [row for row in rows if row.arm == arm]
    return (
        _median([row.rmse for row in selected]),
        _median([row.baseline_rmse for row in selected]),
        _median([row.fit_seconds for row in selected]),
        _median([row.completed_rounds for row in selected]),
    )


def render_report(
    *,
    quantile_rows: Sequence[QuantileSplitRow],
    goss_rows: Sequence[BoostingRow],
    dart_rows: Sequence[BoostingRow],
    seeds: Sequence[int],
    quick: bool,
) -> str:
    """Render medians and descriptive timings without production recommendations."""
    quantile_train, quantile_test = (160, 96) if quick else (512, 256)
    boosting_train, boosting_test = (256, 128) if quick else (512, 256)
    dart_configs = QUICK_DART_CONFIGS if quick else FULL_DART_CONFIGS
    goss_rates = ", ".join(f"({top_rate:.2f}, {other_rate:.2f})" for top_rate, other_rate in GOSS_RATES)
    dart_config_text = ", ".join(
        f"({horizon}, {drop_rate:.2f}, {max_drop})"
        for horizon, drop_rate, max_drop in dart_configs
    )
    selected_sections = [
        section
        for section, rows in (
            ("quantile", quantile_rows),
            ("goss", goss_rows),
            ("dart", dart_rows),
        )
        if rows
    ]
    lines = [
        "# Review Evidence Guardrails",
        "",
        "## Configuration",
        "",
        f"- Sections: {', '.join(selected_sections)}",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- Mode: {'quick' if quick else 'full'}",
        f"- Quantile fixture: {quantile_train} training rows, {quantile_test} held-out rows.",
        f"- Boosting fixture: {boosting_train} training rows, {boosting_test} held-out rows.",
        "- Model settings: depth 4, learning rate 0.06, lambda_l2=1.0, manual policy, deterministic quantile binning.",
        f"- GOSS rates: {goss_rates}.",
        f"- DART configs: {dart_config_text}.",
        "- Timing is descriptive only; no wall-clock threshold is a quality gate.",
    ]

    if quantile_rows:
        lines.extend(["", "## Quantile Split Selection", "", "| Alpha | Arm | Median loss | No-split loss | Median gain |", "|---:|---|---:|---:|---:|"])
        for alpha in sorted({row.alpha for row in quantile_rows}):
            for arm in ("proxy", "smooth_0.05", "smooth_0.10"):
                loss, baseline, gain = _median_quantile_rows(quantile_rows, arm, alpha)
                lines.append(f"| {alpha:.2f} | {arm} | {loss:.6f} | {baseline:.6f} | {gain:.6f} |")
        smooth_arms = ("smooth_0.05", "smooth_0.10")
        smooth_losses = {
            arm: _median([row.pinball_loss for row in quantile_rows if row.arm == arm])
            for arm in smooth_arms
        }
        best_smoothing = min(smooth_losses, key=smooth_losses.__getitem__)
        lines.extend(
            [
                "",
                f"Best smoothed-pinball median arm: `{best_smoothing}`.",
                "This identifies evidence for a later production decision; it does not recommend a production default.",
            ]
        )

    if goss_rows:
        lines.extend(["", "## GOSS Rate Sweep", "", "| Arm | Matched control | Median RMSE | Baseline RMSE | Fit seconds |", "|---|---|---:|---:|---:|"])
        for arm in sorted({row.arm for row in goss_rows}):
            rmse, baseline, fit_seconds, _ = _median_boosting_rows(goss_rows, arm)
            controls = sorted({row.matched_control for row in goss_rows if row.arm == arm and row.matched_control})
            lines.append(
                f"| {arm} | {', '.join(controls) or '-'} | {rmse:.6f} | {baseline:.6f} | {fit_seconds:.4f} |"
            )

    if dart_rows:
        lines.extend(
            [
                "",
                "## DART Dropout Profile",
                "",
                "The configured dropout pressure is an expected-work proxy, not an observed drop count.",
                "The 1.50x RMSE quality gate applies only to `default_like` rows (drop rate <= 0.10).",
                "`stress_profile` rows remain visible and must satisfy finite, control-matching, and completion contracts, but their quality is non-blocking.",
                "Standard-time ratios use unrounded median fit times; displayed fit times are rounded.",
                "",
                "| Arm | Profile | Matched standard | Median RMSE | Fit seconds | Seconds/round | Standard time ratio | Dropout pressure |",
                "|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for arm in sorted({row.arm for row in dart_rows}):
            rmse, _, fit_seconds, rounds = _median_boosting_rows(dart_rows, arm)
            matching_rows = [row for row in dart_rows if row.arm == arm]
            controls = sorted({row.matched_control for row in matching_rows if row.matched_control})
            profiles = sorted({row.dart_profile for row in matching_rows if row.dart_profile})
            control_seconds = _median(
                [
                    row.fit_seconds
                    for row in dart_rows
                    if row.arm in controls
                ]
            )
            time_ratio_text = "-" if not _finite_positive(control_seconds) else f"{fit_seconds / control_seconds:.3f}"
            pressure = _median([row.dropout_pressure for row in matching_rows if row.dropout_pressure is not None])
            pressure_text = "-" if not np.isfinite(pressure) else f"{pressure:.2f}"
            lines.append(
                f"| {arm} | {', '.join(profiles) or '-'} | {', '.join(controls) or '-'} | {rmse:.6f} | {fit_seconds:.4f} | "
                f"{fit_seconds / rounds:.6f} | {time_ratio_text} | {pressure_text} |"
            )
    return "\n".join(lines) + "\n"


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main(argv: Sequence[str] | None = None) -> int:
    """Run selected evidence sections, render Markdown, and optionally enforce gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS, help="comma-separated seeds")
    parser.add_argument("--section", action="append", choices=ALL_SECTIONS, help="section to run; repeatable")
    parser.add_argument("--quick", action="store_true", help="use the deterministic CI-sized configuration")
    parser.add_argument("--gate", action="store_true", help="exit nonzero for failed selected-section gates")
    parser.add_argument("--output", type=Path, help="write UTF-8 Markdown to this path")
    args = parser.parse_args(argv)
    sections = tuple(args.section or ALL_SECTIONS)
    effective_seeds = args.seeds[:1] if args.quick else args.seeds
    quantile_rows, goss_rows, dart_rows = run_benchmark(
        sections=sections,
        seeds=effective_seeds,
        quick=args.quick,
    )
    report = render_report(
        quantile_rows=quantile_rows,
        goss_rows=goss_rows,
        dart_rows=dart_rows,
        seeds=effective_seeds,
        quick=args.quick,
    )

    selected_gates: list[GateResult] = []
    if args.gate:
        selected_gates = [
            gate
            for gate in evaluate_gates(quantile_rows, goss_rows, dart_rows)
            if any(gate.name.startswith(section) for section in sections)
        ]
        report += "\n## Gate Summary\n\n| Gate | Result | Detail |\n|---|---|---|\n"
        for gate in selected_gates:
            report += f"| {gate.name} | {'pass' if gate.passed else 'FAIL'} | {gate.detail} |\n"

    if args.output is None:
        print(report, end="")
    else:
        args.output.write_text(report, encoding="utf-8")

    failures = [gate for gate in selected_gates if not gate.passed]
    for gate in failures:
        print(f"gate failed: {gate.name}: {gate.detail}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
