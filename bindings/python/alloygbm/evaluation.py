"""Evaluation metric helpers for AlloyGBM Python workflows."""

from __future__ import annotations

import math

import numpy as np


def rmse(y_true: object, y_pred: object) -> float:
    """Compute root mean squared error."""
    true_values, pred_values = _validated_pair(y_true, y_pred)
    diff = true_values - pred_values
    return float(np.sqrt(np.mean(diff * diff)))


def mae(y_true: object, y_pred: object) -> float:
    """Compute mean absolute error."""
    true_values, pred_values = _validated_pair(y_true, y_pred)
    return float(np.mean(np.abs(true_values - pred_values)))


def r2_score(y_true: object, y_pred: object) -> float:
    """Compute coefficient of determination (R-squared)."""
    true_values, pred_values = _validated_pair(y_true, y_pred)
    residuals = true_values - pred_values
    centered_true = true_values - float(np.mean(true_values))
    residual_sum_squares = float(np.sum(residuals * residuals))
    total_sum_squares = float(np.sum(centered_true * centered_true))
    if total_sum_squares == 0.0:
        return 1.0 if residual_sum_squares == 0.0 else 0.0
    return 1.0 - (residual_sum_squares / total_sum_squares)


def pearson_correlation(y_true: object, y_pred: object) -> float:
    """Compute Pearson correlation coefficient."""
    true_values, pred_values = _validated_pair(y_true, y_pred)

    centered_true = true_values - float(np.mean(true_values))
    centered_pred = pred_values - float(np.mean(pred_values))
    covariance = float(np.dot(centered_true, centered_pred))
    true_variance = float(np.dot(centered_true, centered_true))
    pred_variance = float(np.dot(centered_pred, centered_pred))

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
    true_direction = np.sign(true_values - threshold_value)
    pred_direction = np.sign(pred_values - threshold_value)
    return float(np.mean(true_direction == pred_direction))


def icir(ic_values: object) -> float:
    """Compute information coefficient information ratio from IC observations."""
    ic_series = _coerce_numeric_array(ic_values, "ic_values")
    ic_mean = float(np.mean(ic_series))
    centered = ic_series - ic_mean
    ic_variance = float(np.mean(centered * centered))
    if math.isclose(ic_variance, 0.0, rel_tol=0.0, abs_tol=1e-15):
        return 0.0
    return ic_mean / math.sqrt(ic_variance)


def _validated_pair(y_true: object, y_pred: object) -> tuple[np.ndarray, np.ndarray]:
    true_values = _coerce_numeric_array(y_true, "y_true")
    pred_values = _coerce_numeric_array(y_pred, "y_pred")
    if len(true_values) != len(pred_values):
        raise ValueError("y_true and y_pred must contain the same number of values")
    return true_values, pred_values


def _coerce_numeric_array(values: object, argument_name: str) -> np.ndarray:
    arr = _coerce_array_like(values, argument_name, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError(
            f"{argument_name} must be a sequence or provide to_numpy/to_list/tolist conversion"
        )
    if arr.ndim != 1:
        raise ValueError(f"{argument_name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{argument_name} must contain at least one value")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{argument_name} must contain only finite numeric values")
    return arr


def _coerce_group_array(values: object, argument_name: str) -> np.ndarray:
    arr = _coerce_array_like(values, argument_name, dtype=None)
    if arr.ndim == 0:
        raise ValueError(
            f"{argument_name} must be a sequence or provide to_numpy/to_list/tolist conversion"
        )
    if arr.ndim != 1:
        raise ValueError(f"{argument_name} must be one-dimensional")
    return arr


def _coerce_array_like(
    value: object, argument_name: str, *, dtype: object | None
) -> np.ndarray:
    current = value
    last_error: Exception | None = None
    for _ in range(4):
        if isinstance(current, (str, bytes, bytearray, memoryview)):
            break
        try:
            arr = np.asarray(current, dtype=dtype)
        except (TypeError, ValueError) as exc:
            last_error = exc
        else:
            if arr.ndim != 0 or not _has_conversion_adapter(current):
                return arr

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

    if dtype is None:
        raise ValueError(
            f"{argument_name} must be a sequence or provide to_numpy/to_list/tolist conversion"
        ) from last_error
    raise ValueError(f"{argument_name} must contain only numeric values") from last_error


def _has_conversion_adapter(value: object) -> bool:
    return not isinstance(value, np.ndarray) and (
        hasattr(value, "to_numpy")
        or hasattr(value, "to_list")
        or hasattr(value, "tolist")
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    tie_starts = np.concatenate(
        ([0], np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1)
    )
    tie_ends = np.concatenate((tie_starts[1:], [len(values)]))
    ranks = np.empty(len(values), dtype=np.float64)
    for start, end in zip(tie_starts, tie_ends):
        average_rank = ((float(start) + float(end - 1)) / 2.0) + 1.0
        ranks[order[start:end]] = average_rank
    return ranks


def _first_indexed_value(values: np.ndarray, indices: np.ndarray) -> float:
    return float(values[int(indices[0])])


def accuracy(y_true: object, y_pred: object) -> float:
    """Compute classification accuracy (fraction of correct predictions)."""
    true_values, pred_values = _validated_pair(y_true, y_pred)
    return float(np.mean(true_values == pred_values))


def log_loss(y_true: object, y_prob: object) -> float:
    """Compute binary cross-entropy (log-loss).

    ``y_true`` should contain values in {0, 1} and ``y_prob`` should contain
    predicted probabilities in (0, 1).
    """
    true_values, prob_values = _validated_pair(y_true, y_prob)
    invalid = np.flatnonzero((true_values != 0.0) & (true_values != 1.0))
    if invalid.size > 0:
        i = int(invalid[0])
        y = float(true_values[i])
        raise ValueError(
            f"log_loss requires y_true values in {{0, 1}}, "
            f"but found {y!r} at index {i}"
        )
    eps = 1e-15
    p_clamped = np.clip(prob_values, eps, 1.0 - eps)
    losses = -(
        true_values * np.log(p_clamped)
        + (1.0 - true_values) * np.log(1.0 - p_clamped)
    )
    return float(np.mean(losses))


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
    true_values = _coerce_numeric_array(y_true, "y_true")
    pred_values = _coerce_numeric_array(y_pred, "y_pred")
    if len(true_values) != len(pred_values):
        raise ValueError("y_true and y_pred must contain the same number of values")

    group_values = _coerce_group_array(group, "group")
    if len(group_values) != len(true_values):
        raise ValueError("group must contain the same number of values as y_true")

    # Find contiguous group boundaries and validate contiguity.
    seen_groups: set[object] = set()
    boundaries = [0]
    current_group = group_values[0] if len(group_values) > 0 else None
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


def _ndcg_single_group(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """NDCG for a single query group."""
    order = np.argsort(-scores, kind="mergesort")[:k]
    ideal_order = np.argsort(-labels, kind="mergesort")[:k]
    discounts = 1.0 / np.log2(np.arange(2.0, float(k) + 2.0))
    dcg = float(np.sum(((2.0 ** labels[order]) - 1.0) * discounts[: len(order)]))
    idcg = float(
        np.sum(((2.0 ** labels[ideal_order]) - 1.0) * discounts[: len(ideal_order)])
    )

    if idcg <= 0.0:
        return 1.0  # degenerate: all labels identical or zero
    return dcg / idcg


def poisson_deviance(y_true: object, y_pred: object) -> float:
    """Mean Poisson deviance for log-link regression.

    For ``y >= 0`` and ``mu > 0``:
        ``D = 2 · mean_i [ y_i · log(y_i / mu_i) − (y_i − mu_i) ]``

    With the convention ``y log(y/mu) → 0`` when ``y → 0``.

    Inputs are coerced via the same path as other metrics, so numpy arrays,
    pandas Series, and other sequence-likes are accepted; non-finite values
    are rejected before the GLM domain check.
    """
    true_values, pred_values = _validated_pair(y_true, y_pred)
    invalid_pred = np.flatnonzero(pred_values <= 0.0)
    if invalid_pred.size > 0:
        raise ValueError(
            f"y_pred must be > 0, got {_first_indexed_value(pred_values, invalid_pred)}"
        )
    invalid_true = np.flatnonzero(true_values < 0.0)
    if invalid_true.size > 0:
        raise ValueError(
            f"y_true must be >= 0, got {_first_indexed_value(true_values, invalid_true)}"
        )
    terms = -(true_values - pred_values)
    positive_mask = true_values > 0.0
    terms[positive_mask] += true_values[positive_mask] * np.log(
        true_values[positive_mask] / pred_values[positive_mask]
    )
    return float(np.mean(2.0 * terms))


def gamma_deviance(y_true: object, y_pred: object) -> float:
    """Mean Gamma deviance for log-link regression.

    For ``y > 0`` and ``mu > 0``:
        ``D = 2 · mean_i [ -log(y_i / mu_i) + (y_i - mu_i) / mu_i ]``

    Inputs are coerced via the same path as other metrics, so numpy arrays,
    pandas Series, and other sequence-likes are accepted; non-finite values
    are rejected before the GLM domain check.
    """
    true_values, pred_values = _validated_pair(y_true, y_pred)
    invalid_pred = np.flatnonzero(pred_values <= 0.0)
    if invalid_pred.size > 0:
        raise ValueError(
            f"y_pred must be > 0, got {_first_indexed_value(pred_values, invalid_pred)}"
        )
    invalid_true = np.flatnonzero(true_values <= 0.0)
    if invalid_true.size > 0:
        raise ValueError(
            f"y_true must be > 0, got {_first_indexed_value(true_values, invalid_true)}"
        )
    terms = 2.0 * (
        -np.log(true_values / pred_values) + (true_values - pred_values) / pred_values
    )
    return float(np.mean(terms))


def tweedie_deviance(
    y_true: object, y_pred: object, *, variance_power: float
) -> float:
    """Mean Tweedie deviance for ``variance_power p ∈ (1, 2)``.

    For ``y >= 0``, ``mu > 0``, ``1 < p < 2``::

        D = 2 · mean_i [ y^(2-p) / ((1-p)(2-p))
                        − y · mu^(1-p) / (1-p)
                        + mu^(2-p) / (2-p) ]

    The ``y^(2-p)`` term is 0 when ``y == 0``.

    Inputs are coerced via the same path as other metrics, so numpy arrays,
    pandas Series, and other sequence-likes are accepted; non-finite values
    are rejected before the GLM domain check.
    """
    if not (1.0 < variance_power < 2.0):
        raise ValueError(
            f"tweedie_deviance requires 1 < variance_power < 2 (got {variance_power})"
        )
    true_values, pred_values = _validated_pair(y_true, y_pred)
    p = variance_power
    invalid_pred = np.flatnonzero(pred_values <= 0.0)
    if invalid_pred.size > 0:
        raise ValueError(
            f"y_pred must be > 0, got {_first_indexed_value(pred_values, invalid_pred)}"
        )
    invalid_true = np.flatnonzero(true_values < 0.0)
    if invalid_true.size > 0:
        raise ValueError(
            f"y_true must be >= 0, got {_first_indexed_value(true_values, invalid_true)}"
        )
    term1 = np.zeros_like(true_values)
    positive_mask = true_values > 0.0
    term1[positive_mask] = (true_values[positive_mask] ** (2.0 - p)) / (
        (1.0 - p) * (2.0 - p)
    )
    term2 = true_values * (pred_values ** (1.0 - p)) / (1.0 - p)
    term3 = (pred_values ** (2.0 - p)) / (2.0 - p)
    return float(np.mean(2.0 * (term1 - term2 + term3)))


__all__ = [
    "accuracy",
    "gamma_deviance",
    "hit_rate",
    "icir",
    "log_loss",
    "mae",
    "ndcg",
    "pearson_correlation",
    "poisson_deviance",
    "r2_score",
    "rank_ic",
    "rmse",
    "tweedie_deviance",
]
