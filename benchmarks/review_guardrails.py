"""Deterministic experiments supporting July-review evidence guardrails."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


def make_quantile_split_data(
    *, seed: int, n_train: int, n_test: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic one-feature, heteroscedastic quantile fixture."""
    if n_train < 1 or n_test < 1:
        raise ValueError("n_train and n_test must be positive")

    rng = np.random.default_rng(seed)

    def make_partition(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feature = rng.uniform(-2.5, 2.5, size=size)
        location = 0.28 * np.sin(1.4 * feature) + 0.07 * feature**2 - 0.05 * feature
        scale = 0.25 + 0.14 * np.abs(feature)
        noise = scale * (rng.exponential(size=size) - 1.0)
        weights = rng.lognormal(mean=0.0, sigma=0.45, size=size)
        return feature, location + noise, weights

    x_train, y_train, w_train = make_partition(n_train)
    x_test, y_test, w_test = make_partition(n_test)
    return tuple(
        np.ascontiguousarray(values, dtype=np.float64)
        for values in (x_train, y_train, w_train, x_test, y_test, w_test)
    )  # type: ignore[return-value]


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
    seeds: tuple[int, ...] = (7, 17, 29, 43, 59),
    alphas: tuple[float, ...] = (0.1, 0.5, 0.9),
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
