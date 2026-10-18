"""Automatic round count and learning rate (``n_estimators="auto"``).

The single most effective default in modern GBDTs is "many small steps,
stopped by validation". With ``n_estimators="auto"`` (the default) and
``training_policy="auto"``, a fit:

1. holds out a seeded :data:`AUTO_VALIDATION_FRACTION` of the training rows
   (stratified by class for classifiers, by whole query groups for rankers);
2. trains on the rest at :data:`AUTO_LEARNING_RATE` for at most
   :data:`AUTO_ROUND_CAP` rounds, stopping after :data:`AUTO_PATIENCE`
   rounds without a relative validation improvement of
   :data:`AUTO_RELATIVE_TOLERANCE`;
3. refits on every row for the chosen round count, scaled up by
   ``1 / (1 - fraction)`` because the refit sees more rows per round.

When the caller passes ``time_index``, step 1 holds out the latest periods
instead (:func:`time_holdout_split`): neighbouring periods look alike, so a
random holdout of time-ordered rows overrates late rounds and picks too many.

When the caller passes ``eval_set``, that set drives the early stopping
directly and there is no refit. Configurations whose round semantics do not
survive a refit (warm start, DART, MorphBoost, learning-rate schedules,
custom objectives, time-aware categorical or neutralized fits, tiny datasets) and
``training_policy="manual"`` fall back to the historical fixed defaults of
:data:`LEGACY_N_ESTIMATORS` rounds at :data:`LEGACY_LEARNING_RATE`.

The constants were chosen on the default-quality suite; see
``docs/benchmarks/auto_rounds_v1.md``.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

AUTO = "auto"

LEGACY_N_ESTIMATORS = 100
LEGACY_LEARNING_RATE = 0.1

AUTO_LEARNING_RATE = 0.05
AUTO_ROUND_CAP = 2000
AUTO_PATIENCE = 50
AUTO_RELATIVE_TOLERANCE = 1e-3
AUTO_VALIDATION_FRACTION = 0.1
# Below this many training rows a 10% holdout is too small to choose a round
# count from, so the fit keeps the fixed legacy defaults.
AUTO_MIN_ROWS = 50

MODE_EXPLICIT = "explicit"
MODE_EVAL_SET = "eval_set"
MODE_HOLDOUT = "holdout"
MODE_LEGACY = "legacy"


def validate_n_estimators(value: object) -> int | str:
    if isinstance(value, str):
        if value != AUTO:
            raise ValueError("n_estimators must be a positive integer or 'auto'")
        return AUTO
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("n_estimators must be a positive integer or 'auto'")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError("n_estimators must be greater than 0")
    return normalized


def validate_learning_rate(value: object) -> float | str:
    if isinstance(value, str):
        if value != AUTO:
            raise ValueError("learning_rate must be in (0.0, 1.0] or 'auto'")
        return AUTO
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("learning_rate must be in (0.0, 1.0] or 'auto'")
    normalized = float(value)
    if not (0.0 < normalized <= 1.0):
        raise ValueError("learning_rate must be in (0.0, 1.0]")
    return normalized


def best_round_with_relative_tolerance(curve: list[float], patience: int, tolerance: float) -> int:
    """Round count to keep under a relative-improvement rule.

    ``curve[i]`` is the validation loss after ``i + 1`` rounds. A round counts
    as an improvement only when it beats the running best by ``tolerance`` of
    that best, so a long, nearly flat tail does not keep pushing the chosen
    round count out (the engine's own early stopping uses an absolute
    threshold, which would depend on the target's units).
    """
    best_value = curve[0]
    best_index = 0
    for index, value in enumerate(curve):
        if value < best_value - tolerance * abs(best_value):
            best_value = value
            best_index = index
        if index - best_index >= patience:
            break
    return best_index + 1


def holdout_split(
    row_count: int,
    fraction: float,
    seed: int,
    *,
    class_labels: np.ndarray | None = None,
    group_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Seeded train/validation row indices, both in ascending order.

    Classifiers split every class in proportion, rankers split whole query
    groups, and everything else splits rows uniformly. Returns ``None`` when
    no usable split exists (for example a class with a single row, or a
    ranker with a single group).
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFF_FFFF, 0xA070]))
    if group_ids is not None:
        unique_groups, inverse = np.unique(np.asarray(group_ids), return_inverse=True)
        group_count = len(unique_groups)
        if group_count < 2:
            return None
        holdout_groups = max(1, int(round(fraction * group_count)))
        if holdout_groups >= group_count:
            return None
        chosen = np.zeros(group_count, dtype=bool)
        chosen[rng.permutation(group_count)[:holdout_groups]] = True
        is_validation = chosen[inverse]
    elif class_labels is not None:
        labels = np.asarray(class_labels)
        is_validation = np.zeros(row_count, dtype=bool)
        for label in np.unique(labels):
            members = np.flatnonzero(labels == label)
            if len(members) < 2:
                return None
            take = min(len(members) - 1, max(1, int(round(fraction * len(members)))))
            is_validation[rng.permutation(members)[:take]] = True
    else:
        take = max(1, int(round(fraction * row_count)))
        if take >= row_count:
            return None
        is_validation = np.zeros(row_count, dtype=bool)
        is_validation[rng.permutation(row_count)[:take]] = True
    return np.flatnonzero(~is_validation), np.flatnonzero(is_validation)


def time_holdout_split(
    time_index: np.ndarray,
    fraction: float,
    *,
    group_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Hold out the latest time periods instead of random rows.

    Takes whole periods, newest first, until at least ``fraction`` of the
    rows are held out, and always leaves at least one period to train on.
    Returns ``None`` with fewer than two periods, or when a query group
    would straddle the cut.
    """
    times = np.asarray(time_index)
    periods, counts = np.unique(times, return_counts=True)
    if len(periods) < 2:
        return None
    newest_first = np.cumsum(counts[::-1])
    held_periods = int(np.searchsorted(newest_first, fraction * len(times))) + 1
    held_periods = min(held_periods, len(periods) - 1)
    is_validation = times >= periods[-held_periods]
    if group_ids is not None:
        groups = np.asarray(group_ids)
        if np.intersect1d(groups[is_validation], groups[~is_validation]).size:
            return None
    return np.flatnonzero(~is_validation), np.flatnonzero(is_validation)


def take_rows(values: object, indices: np.ndarray) -> object:
    """Row subset of an array-like, keeping DataFrames as DataFrames."""
    if values is None:
        return None
    if hasattr(values, "iloc"):
        return values.iloc[indices]
    if isinstance(values, np.ndarray):
        return values[indices]
    as_list = list(values)
    return [as_list[int(i)] for i in indices]


def refit_rounds(best_round: int, fraction: float) -> int:
    return max(1, int(round(best_round / (1.0 - fraction))))


def is_positive_int(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and int(value) > 0
