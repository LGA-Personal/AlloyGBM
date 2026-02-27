"""Evaluation metric helpers for AlloyGBM Python workflows."""

from __future__ import annotations

import math
from collections.abc import Sequence


def rmse(y_true: object, y_pred: object) -> float:
    """Compute root mean squared error."""
    true_values, pred_values = _validated_pair(y_true, y_pred)
    squared_error_sum = sum(
        (true_value - pred_value) ** 2
        for true_value, pred_value in zip(true_values, pred_values)
    )
    return math.sqrt(squared_error_sum / len(true_values))


def mae(y_true: object, y_pred: object) -> float:
    """Compute mean absolute error."""
    true_values, pred_values = _validated_pair(y_true, y_pred)
    absolute_error_sum = sum(
        abs(true_value - pred_value)
        for true_value, pred_value in zip(true_values, pred_values)
    )
    return absolute_error_sum / len(true_values)


def r2_score(y_true: object, y_pred: object) -> float:
    """Compute coefficient of determination (R-squared)."""
    true_values, pred_values = _validated_pair(y_true, y_pred)
    true_mean = sum(true_values) / len(true_values)
    residual_sum_squares = sum(
        (true_value - pred_value) ** 2
        for true_value, pred_value in zip(true_values, pred_values)
    )
    total_sum_squares = sum(
        (true_value - true_mean) ** 2 for true_value in true_values
    )
    if total_sum_squares == 0.0:
        return 1.0 if residual_sum_squares == 0.0 else 0.0
    return 1.0 - (residual_sum_squares / total_sum_squares)


def pearson_correlation(y_true: object, y_pred: object) -> float:
    """Compute Pearson correlation coefficient."""
    true_values, pred_values = _validated_pair(y_true, y_pred)

    true_mean = sum(true_values) / len(true_values)
    pred_mean = sum(pred_values) / len(pred_values)

    covariance = 0.0
    true_variance = 0.0
    pred_variance = 0.0
    for true_value, pred_value in zip(true_values, pred_values):
        centered_true = true_value - true_mean
        centered_pred = pred_value - pred_mean
        covariance += centered_true * centered_pred
        true_variance += centered_true * centered_true
        pred_variance += centered_pred * centered_pred

    if true_variance == 0.0 or pred_variance == 0.0:
        return 0.0
    return covariance / math.sqrt(true_variance * pred_variance)


def rank_ic(y_true: object, y_pred: object) -> float:
    """Compute rank information coefficient (Spearman-style rank correlation)."""
    true_values, pred_values = _validated_pair(y_true, y_pred)
    true_ranks = _average_ranks(true_values)
    pred_ranks = _average_ranks(pred_values)
    return pearson_correlation(true_ranks, pred_ranks)


def hit_rate(y_true: object, y_pred: object, *, threshold: float = 0.0) -> float:
    """Compute directional hit rate using three-way sign around a threshold."""
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value):
        raise ValueError("threshold must be a finite numeric value")

    true_values, pred_values = _validated_pair(y_true, y_pred)
    hits = 0
    for true_value, pred_value in zip(true_values, pred_values):
        if _direction(true_value, threshold_value) == _direction(
            pred_value, threshold_value
        ):
            hits += 1
    return hits / len(true_values)


def icir(ic_values: object) -> float:
    """Compute information coefficient information ratio from IC observations."""
    ic_series = _coerce_numeric_sequence(ic_values, "ic_values")
    ic_mean = sum(ic_series) / len(ic_series)
    ic_variance = sum((value - ic_mean) ** 2 for value in ic_series) / len(ic_series)
    if math.isclose(ic_variance, 0.0, rel_tol=0.0, abs_tol=1e-15):
        return 0.0
    return ic_mean / math.sqrt(ic_variance)


def _validated_pair(y_true: object, y_pred: object) -> tuple[list[float], list[float]]:
    true_values = _coerce_numeric_sequence(y_true, "y_true")
    pred_values = _coerce_numeric_sequence(y_pred, "y_pred")
    if len(true_values) != len(pred_values):
        raise ValueError("y_true and y_pred must contain the same number of values")
    return true_values, pred_values


def _coerce_numeric_sequence(values: object, argument_name: str) -> list[float]:
    values_like = _coerce_sequence_like(values, argument_name)
    if len(values_like) == 0:
        raise ValueError(f"{argument_name} must contain at least one value")

    normalized: list[float] = []
    for value in values_like:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{argument_name} must contain only numeric values"
            ) from exc
        if not math.isfinite(numeric_value):
            raise ValueError(
                f"{argument_name} must contain only finite numeric values"
            )
        normalized.append(numeric_value)
    return normalized


def _coerce_sequence_like(value: object, argument_name: str) -> Sequence[object]:
    current = value
    for _ in range(4):
        if isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray, memoryview)
        ):
            return current

        next_value: object | None = None
        if hasattr(current, "to_numpy"):
            next_value = current.to_numpy()  # type: ignore[call-arg]
        elif hasattr(current, "to_list"):
            next_value = current.to_list()  # type: ignore[call-arg]
        elif hasattr(current, "tolist"):
            next_value = current.tolist()  # type: ignore[call-arg]

        if next_value is None or next_value is current:
            break
        current = next_value

    raise ValueError(
        f"{argument_name} must be a sequence or provide to_numpy/to_list/tolist conversion"
    )


def _average_ranks(values: Sequence[float]) -> list[float]:
    ranks = [0.0] * len(values)
    sorted_items = sorted(enumerate(values), key=lambda item: item[1])
    index = 0
    while index < len(sorted_items):
        tie_end = index
        while (
            tie_end + 1 < len(sorted_items)
            and sorted_items[tie_end + 1][1] == sorted_items[index][1]
        ):
            tie_end += 1
        average_rank = ((index + tie_end) / 2.0) + 1.0
        for tie_index in range(index, tie_end + 1):
            original_index = sorted_items[tie_index][0]
            ranks[original_index] = average_rank
        index = tie_end + 1
    return ranks


def _direction(value: float, threshold: float) -> int:
    if value > threshold:
        return 1
    if value < threshold:
        return -1
    return 0


__all__ = [
    "hit_rate",
    "icir",
    "mae",
    "pearson_correlation",
    "r2_score",
    "rank_ic",
    "rmse",
]
