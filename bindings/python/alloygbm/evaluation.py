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


def accuracy(y_true: object, y_pred: object) -> float:
    """Compute classification accuracy (fraction of correct predictions)."""
    true_values, pred_values = _validated_pair(y_true, y_pred)
    correct = sum(
        1 for true_value, pred_value in zip(true_values, pred_values)
        if true_value == pred_value
    )
    return correct / len(true_values)


def log_loss(y_true: object, y_prob: object) -> float:
    """Compute binary cross-entropy (log-loss).

    ``y_true`` should contain values in {0, 1} and ``y_prob`` should contain
    predicted probabilities in (0, 1).
    """
    true_values, prob_values = _validated_pair(y_true, y_prob)
    for i, y in enumerate(true_values):
        if y != 0.0 and y != 1.0:
            raise ValueError(
                f"log_loss requires y_true values in {{0, 1}}, "
                f"but found {y!r} at index {i}"
            )
    eps = 1e-15
    total = 0.0
    for y, p in zip(true_values, prob_values):
        p_clamped = max(eps, min(1.0 - eps, p))
        total += -(y * math.log(p_clamped) + (1.0 - y) * math.log(1.0 - p_clamped))
    return total / len(true_values)


def multiclass_log_loss(y_true: object, y_prob: object) -> float:
    """Compute multi-class cross-entropy (log-loss).

    Parameters
    ----------
    y_true : array-like of int
        True class labels (integers in 0..K-1).
    y_prob : array-like of shape (n_samples, K)
        Predicted class probabilities. Each row should sum to ~1.0.

    Returns
    -------
    float
        Mean negative log-likelihood: ``-mean(log(p[y_i]))``.
    """
    import numpy as np

    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    y_prob_arr = np.asarray(y_prob, dtype=float)
    if y_prob_arr.ndim != 2:
        raise ValueError(
            f"y_prob must be 2-dimensional, got shape {y_prob_arr.shape}"
        )
    n_samples, n_classes = y_prob_arr.shape
    if len(y_true_arr) != n_samples:
        raise ValueError(
            f"y_true length {len(y_true_arr)} does not match "
            f"y_prob row count {n_samples}"
        )
    if n_classes < 2:
        raise ValueError(f"y_prob must have at least 2 columns, got {n_classes}")
    # Clip probabilities for numerical stability
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob_arr, eps, 1.0 - eps)
    # Extract the predicted probability for each sample's true class
    log_probs = np.log(y_prob_clipped[np.arange(n_samples), y_true_arr])
    return float(-np.mean(log_probs))


def ndcg(
    y_true: object,
    y_pred: object,
    *,
    group: object,
    k: int | None = None,
) -> float:
    """Compute mean NDCG (Normalized Discounted Cumulative Gain) across query groups.

    Parameters
    ----------
    y_true : array-like
        True relevance labels (higher = more relevant).
    y_pred : array-like
        Predicted scores (higher = more relevant).
    group : array-like
        Query group IDs. Rows with the same group ID form a single query.
        Must be sorted so that all rows in a group are contiguous.
    k : int or None
        Truncation level. If ``None``, all documents in each group are used.

    Returns
    -------
    float
        Mean NDCG across all query groups, in [0, 1].
    """
    true_values = _coerce_numeric_sequence(y_true, "y_true")
    pred_values = _coerce_numeric_sequence(y_pred, "y_pred")
    if len(true_values) != len(pred_values):
        raise ValueError("y_true and y_pred must contain the same number of values")

    group_values = _coerce_sequence_like(group, "group")
    if len(group_values) != len(true_values):
        raise ValueError("group must contain the same number of values as y_true")

    # Find contiguous group boundaries and validate contiguity.
    seen_groups: set[object] = set()
    boundaries = [0]
    current_group = group_values[0] if group_values else None
    seen_groups.add(current_group)
    for i in range(1, len(group_values)):
        if group_values[i] != group_values[i - 1]:
            if group_values[i] in seen_groups:
                raise ValueError(
                    f"ndcg requires contiguous group IDs, but group "
                    f"{group_values[i]!r} reappears at index {i} after a "
                    f"different group. Sort your data by group before calling ndcg."
                )
            seen_groups.add(group_values[i])
            boundaries.append(i)
    boundaries.append(len(group_values))

    num_groups = len(boundaries) - 1
    if num_groups == 0:
        return 1.0

    ndcg_sum = 0.0
    for g in range(num_groups):
        start = boundaries[g]
        end = boundaries[g + 1]
        group_true = true_values[start:end]
        group_pred = pred_values[start:end]
        cutoff = len(group_true) if k is None else min(k, len(group_true))
        ndcg_sum += _ndcg_single_group(group_true, group_pred, cutoff)

    return ndcg_sum / num_groups


def _ndcg_single_group(labels: list[float], scores: list[float], k: int) -> float:
    """NDCG for a single query group."""
    # DCG: sort by scores descending, compute discounted gains.
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    dcg = 0.0
    for rank, idx in enumerate(order[:k]):
        gain = (2.0 ** labels[idx]) - 1.0
        discount = 1.0 / math.log2(rank + 2.0)
        dcg += gain * discount

    # Ideal DCG: sort by labels descending.
    sorted_labels = sorted(labels, reverse=True)
    idcg = 0.0
    for rank, label in enumerate(sorted_labels[:k]):
        gain = (2.0 ** label) - 1.0
        discount = 1.0 / math.log2(rank + 2.0)
        idcg += gain * discount

    if idcg <= 0.0:
        return 1.0  # degenerate: all labels identical or zero
    return dcg / idcg


__all__ = [
    "accuracy",
    "hit_rate",
    "icir",
    "log_loss",
    "mae",
    "ndcg",
    "pearson_correlation",
    "r2_score",
    "rank_ic",
    "rmse",
]
