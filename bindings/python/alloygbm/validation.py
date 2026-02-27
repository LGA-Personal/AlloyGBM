"""Leakage-aware validation split helpers for AlloyGBM workflows."""

from __future__ import annotations

from collections.abc import Sequence


def purged_time_series_splits(
    time_index: object,
    *,
    n_splits: int = 5,
    purge_gap: int = 0,
    embargo: int = 0,
) -> list[tuple[list[int], list[int]]]:
    """Build contiguous time-based folds with purge/embargo exclusions."""
    times = _coerce_sequence(time_index, "time_index")
    split_count = _validate_split_count(n_splits)
    purge_periods = _validate_non_negative_int(purge_gap, "purge_gap")
    embargo_periods = _validate_non_negative_int(embargo, "embargo")

    unique_periods = _sorted_unique_periods(times)
    if split_count > len(unique_periods):
        raise ValueError(
            "n_splits must not exceed the number of unique time periods in time_index"
        )

    period_to_position = {period: position for position, period in enumerate(unique_periods)}
    fold_bounds = _contiguous_fold_bounds(len(unique_periods), split_count)

    splits: list[tuple[list[int], list[int]]] = []
    for start_position, end_position_exclusive in fold_bounds:
        test_positions = set(range(start_position, end_position_exclusive))
        blocked_start = max(0, start_position - purge_periods)
        blocked_end = min(
            len(unique_periods) - 1,
            (end_position_exclusive - 1) + embargo_periods,
        )
        blocked_positions = set(range(blocked_start, blocked_end + 1))

        train_indices: list[int] = []
        test_indices: list[int] = []
        for row_index, period in enumerate(times):
            position = period_to_position[period]
            if position in test_positions:
                test_indices.append(row_index)
            elif position not in blocked_positions:
                train_indices.append(row_index)

        if not train_indices:
            raise ValueError(
                "purge_gap/embargo configuration leaves no training rows for at least one fold"
            )
        splits.append((train_indices, test_indices))

    return splits


def purged_panel_splits(
    time_index: object,
    group_index: object,
    *,
    n_splits: int = 5,
    purge_gap: int = 0,
    embargo: int = 0,
) -> list[tuple[list[int], list[int]]]:
    """Build panel splits using time buckets across all groups."""
    times = _coerce_sequence(time_index, "time_index")
    groups = _coerce_sequence(group_index, "group_index")
    if len(times) != len(groups):
        raise ValueError(
            "time_index and group_index must contain the same number of rows"
        )

    # Panel behavior is defined as time-bucketed splitting across all groups.
    return purged_time_series_splits(
        times,
        n_splits=n_splits,
        purge_gap=purge_gap,
        embargo=embargo,
    )


def _coerce_sequence(values: object, argument_name: str) -> list[object]:
    values_like = _coerce_sequence_like(values, argument_name)
    if len(values_like) == 0:
        raise ValueError(f"{argument_name} must contain at least one value")
    return list(values_like)


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


def _sorted_unique_periods(periods: Sequence[object]) -> list[object]:
    try:
        return sorted(set(periods))
    except TypeError as exc:
        raise ValueError("time_index must contain orderable values") from exc


def _validate_split_count(n_splits: int) -> int:
    if not isinstance(n_splits, int):
        raise ValueError("n_splits must be an integer >= 2")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    return n_splits


def _validate_non_negative_int(value: int, argument_name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{argument_name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{argument_name} must be >= 0")
    return value


def _contiguous_fold_bounds(total_periods: int, n_splits: int) -> list[tuple[int, int]]:
    base_size, remainder = divmod(total_periods, n_splits)
    bounds: list[tuple[int, int]] = []
    start = 0
    for fold_index in range(n_splits):
        fold_size = base_size + (1 if fold_index < remainder else 0)
        end = start + fold_size
        bounds.append((start, end))
        start = end
    return bounds


__all__ = ["purged_panel_splits", "purged_time_series_splits"]
