"""Quantization and binning mixin for GBMRegressor."""

from __future__ import annotations

import bisect
import math
import struct
from collections.abc import Sequence

from . import _base
from ._base import (
    _PRE_BINNED_INTEGER_TOLERANCE,
    _max_data_bin_for_max_bins,
    _nan_bin_for_max_bins,
)


class _QuantizationMixin:
    """Mixin carrying quantization/binning methods for GBMRegressor.

    All 33 methods are moved verbatim from GBMRegressor in _core.py.
    ``GBMRegressor`` references inside static method bodies resolve at
    call-time from this module's globals: after defining the class, _core.py
    injects ``_quantization.GBMRegressor = GBMRegressor`` (a top-level
    ``from ._core import GBMRegressor`` would create a circular import).
    """

    @staticmethod
    def predict_from_artifact(
        artifact_bytes: bytes | bytearray | memoryview, X: object
    ) -> list[float]:
        """Run predictor-backed inference from serialized model artifact bytes."""
        if not isinstance(artifact_bytes, (bytes, bytearray, memoryview)):
            raise TypeError("artifact_bytes must be bytes-like")
        dense_payload = GBMRegressor._native_matrix_flat_payload(X)
        if dense_payload is not None:
            flat_values, row_count, feature_count = dense_payload
            predictor_predict_batch_dense = _base._load_native_predictor_predict_batch_dense()
            return list(
                predictor_predict_batch_dense(
                    bytes(artifact_bytes),
                    flat_values,
                    row_count=row_count,
                    feature_count=feature_count,
                )
            )
        rows = GBMRegressor._validate_rows(X)
        predictor_predict_batch = _base._load_native_predictor_predict_batch()
        return list(predictor_predict_batch(bytes(artifact_bytes), rows))

    @staticmethod
    def _validate_rows(
        X: object,
        *,
        categorical_feature_index: int | None = None,
        categorical_feature_indices: list[int] | None = None,
    ) -> list[list[float]]:
        rows_like = GBMRegressor._coerce_sequence_like(X, "X")
        if not isinstance(rows_like, Sequence) or isinstance(rows_like, (str, bytes)):
            raise TypeError("X must be a sequence of feature rows")
        if len(rows_like) == 0:
            raise ValueError("X must contain at least one row")

        # Build a set of all categorical column indices for fast lookup
        cat_indices_set: set[int] = set()
        if categorical_feature_indices is not None:
            cat_indices_set.update(categorical_feature_indices)
        elif categorical_feature_index is not None:
            cat_indices_set.add(categorical_feature_index)

        normalized: list[list[float]] = []
        expected_width: int | None = None
        for row in rows_like:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                raise TypeError("each X row must be a sequence of numeric values")
            if len(row) == 0:
                raise ValueError("each X row must contain at least one feature value")
            row_values: list[float] = []
            for feature_index, value in enumerate(row):
                if feature_index in cat_indices_set:
                    try:
                        row_values.append(float(value))
                    except (TypeError, ValueError):
                        row_values.append(0.0)
                else:
                    row_values.append(float(value))
            if expected_width is None:
                expected_width = len(row_values)
            elif len(row_values) != expected_width:
                raise ValueError("all X rows must have the same feature count")
            normalized.append(row_values)

        return normalized

    @staticmethod
    def _adapt_native_array_candidate(value: object) -> object | None:
        current = value
        for _ in range(2):
            try:
                view = memoryview(current)
            except TypeError:
                view = None
            if view is not None and getattr(view, "ndim", 0) == 2:
                return current
            if hasattr(current, "to_numpy"):
                next_value = current.to_numpy()  # type: ignore[call-arg]
                if next_value is current:
                    break
                current = next_value
                continue
            break
        return None

    @staticmethod
    def _native_matrix_shape(value: object) -> tuple[int, int]:
        try:
            view = memoryview(value)
        except TypeError as exc:
            raise TypeError("X is not a native dense matrix candidate") from exc
        shape = getattr(view, "shape", None)
        if view.ndim != 2 or shape is None or len(shape) != 2:
            raise TypeError("X is not a 2D native dense matrix candidate")
        row_count = int(shape[0])
        feature_count = int(shape[1])
        if row_count <= 0 or feature_count <= 0:
            raise ValueError("X must contain at least one row and one feature")
        return row_count, feature_count

    @staticmethod
    def _buffer_format_is_integer(value: object) -> bool:
        try:
            view = memoryview(value)
        except TypeError:
            return False
        format_code = getattr(view, "format", "") or ""
        normalized = str(format_code).lower()
        return normalized not in {"f", "d", "e"}

    @staticmethod
    def _native_matrix_fast_path_candidate(
        X: object, *, require_integer: bool = False
    ) -> object | None:
        candidate = GBMRegressor._adapt_native_array_candidate(X)
        if candidate is None:
            return None
        GBMRegressor._native_matrix_shape(candidate)
        if require_integer and not GBMRegressor._buffer_format_is_integer(candidate):
            return None
        return candidate

    @staticmethod
    def _native_matrix_flat_payload(
        X: object, *, require_integer: bool = False
    ) -> tuple[list[float], int, int] | None:
        candidate = GBMRegressor._native_matrix_fast_path_candidate(
            X, require_integer=require_integer
        )
        if candidate is None:
            return None
        row_count, feature_count = GBMRegressor._native_matrix_shape(candidate)
        return (
            GBMRegressor._flatten_native_matrix_candidate(candidate),
            row_count,
            feature_count,
        )

    @staticmethod
    def _native_matrix_bytes_payload(
        X: object,
    ) -> tuple[bytes, int, int] | None:
        """Return raw f32 bytes of the matrix for zero-copy transfer to Rust."""
        try:
            import numpy as np
            candidate = GBMRegressor._native_matrix_fast_path_candidate(X)
            if candidate is None:
                return None
            row_count, feature_count = GBMRegressor._native_matrix_shape(candidate)
            arr = np.ascontiguousarray(candidate, dtype=np.float32)
            return (arr.tobytes(), row_count, feature_count)
        except ImportError:
            return None

    @staticmethod
    def _flatten_native_matrix_candidate(candidate: object) -> list[float]:
        # Fast path: numpy arrays can use .astype(float32).ravel().tolist()
        # which is 10-100× faster than struct.iter_unpack for large arrays
        try:
            import numpy as np
            if isinstance(candidate, np.ndarray):
                flat = np.ascontiguousarray(candidate, dtype=np.float32).ravel()
                return flat.tolist()
        except ImportError:
            pass

        view = memoryview(candidate)
        format_code = getattr(view, "format", "") or ""
        normalized = str(format_code).strip()
        type_code = next(
            (character for character in reversed(normalized) if character.isalpha()),
            "",
        )
        if type_code == "":
            raise TypeError("native dense matrix format is not supported")
        if type_code == "?":
            unpack_code = "?"
        else:
            unpack_code = type_code
        if unpack_code not in {"b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "f", "d"}:
            raise TypeError(
                f"native dense matrix format '{normalized}' is not supported"
            )
        raw_bytes = view.tobytes()
        return [float(value[0]) for value in struct.iter_unpack("@" + unpack_code, raw_bytes)]

    @staticmethod
    def _column_values_from_flat_payload(
        flat_values: Sequence[float], row_count: int, feature_count: int, feature_index: int
    ) -> list[float]:
        return [
            float(flat_values[row_index * feature_count + feature_index])
            for row_index in range(row_count)
        ]

    @staticmethod
    def _derive_dense_feature_bounds(
        flat_values: Sequence[float], row_count: int, feature_count: int
    ) -> tuple[list[float], list[float]]:
        try:
            import numpy as np
            arr = np.asarray(flat_values, dtype=np.float32).reshape(row_count, feature_count)
            return np.nanmin(arr, axis=0).tolist(), np.nanmax(arr, axis=0).tolist()
        except (ImportError, ValueError):
            pass
        mins = [float("inf")] * feature_count
        maxs = [float("-inf")] * feature_count
        for row_index in range(row_count):
            row_base = row_index * feature_count
            for feature_index in range(feature_count):
                value = float(flat_values[row_base + feature_index])
                if math.isnan(value):
                    continue
                if value < mins[feature_index]:
                    mins[feature_index] = value
                if value > maxs[feature_index]:
                    maxs[feature_index] = value
        return mins, maxs

    @staticmethod
    def _try_derive_dense_sorted_feature_values_numpy(
        flat_values: Sequence[float], row_count: int, feature_count: int
    ) -> list[list[float]] | None:
        try:
            import numpy as np
            arr = np.asarray(flat_values, dtype=np.float32).reshape(row_count, feature_count)
        except (ImportError, TypeError, ValueError):
            return None

        sorted_values: list[list[float]] = []
        for feature_index in range(feature_count):
            values = arr[:, feature_index]
            values = values[~np.isnan(values)]
            sorted_values.append([float(value) for value in np.sort(values).tolist()])
        return sorted_values

    @staticmethod
    def _derive_dense_sorted_feature_values(
        flat_values: Sequence[float], row_count: int, feature_count: int
    ) -> list[list[float]]:
        numpy_sorted_values = GBMRegressor._try_derive_dense_sorted_feature_values_numpy(
            flat_values, row_count, feature_count
        )
        if numpy_sorted_values is not None:
            return numpy_sorted_values

        sorted_values: list[list[float]] = []
        for feature_index in range(feature_count):
            values = GBMRegressor._column_values_from_flat_payload(
                flat_values, row_count, feature_count, feature_index
            )
            values = [v for v in values if not math.isnan(v)]
            values.sort()
            sorted_values.append(values)
        return sorted_values

    @staticmethod
    def _derive_dense_feature_quantile_cuts(
        flat_values: Sequence[float], row_count: int, feature_count: int, max_bins: int
    ) -> list[list[float]]:
        sorted_feature_values = GBMRegressor._try_derive_dense_sorted_feature_values_numpy(
            flat_values, row_count, feature_count
        )
        if sorted_feature_values is not None:
            return GBMRegressor._feature_quantile_cuts_from_sorted_values(
                sorted_feature_values, max_bins
            )

        feature_cuts: list[list[float]] = []
        for feature_index in range(feature_count):
            values = GBMRegressor._column_values_from_flat_payload(
                flat_values, row_count, feature_count, feature_index
            )
            values = [v for v in values if not math.isnan(v)]
            values.sort()
            feature_cuts.append(
                GBMRegressor._single_feature_quantile_cuts_from_sorted_values(
                    values, max_bins
                )
            )
        return feature_cuts

    @staticmethod
    def _feature_quantile_cuts_from_sorted_values(
        sorted_feature_values: Sequence[Sequence[float]], max_bins: int
    ) -> list[list[float]]:
        return [
            GBMRegressor._single_feature_quantile_cuts_from_sorted_values(
                values, max_bins
            )
            for values in sorted_feature_values
        ]

    @staticmethod
    def _single_feature_quantile_cuts_from_sorted_values(
        values: Sequence[float], max_bins: int
    ) -> list[float]:
        if len(values) <= 1:
            return []

        bin_count = min(max_bins, len(values))
        cuts: list[float] = []
        for quantile_index in range(1, bin_count):
            rank = (quantile_index * len(values)) // bin_count
            if rank >= len(values):
                rank = len(values) - 1
            cut_value = values[rank]
            if cuts and cut_value <= cuts[-1]:
                continue
            cuts.append(cut_value)
        return cuts

    @staticmethod
    def _quantize_dense_values_linear(
        flat_values: Sequence[float],
        row_count: int,
        feature_count: int,
        feature_mins: Sequence[float],
        feature_maxs: Sequence[float],
        max_bins: int = 256,
    ) -> list[float]:
        max_bin = _max_data_bin_for_max_bins(max_bins)
        nan_bin = _nan_bin_for_max_bins(max_bins)
        try:
            import numpy as np
            arr = np.asarray(flat_values, dtype=np.float32).reshape(row_count, feature_count)
            mins = np.asarray(feature_mins, dtype=np.float32)
            maxs = np.asarray(feature_maxs, dtype=np.float32)
            span = maxs - mins
            span_ok = span > _PRE_BINNED_INTEGER_TOLERANCE
            nan_mask = np.isnan(arr)
            # Vectorized quantization
            result = np.zeros_like(arr)
            for fi in range(feature_count):
                if not span_ok[fi]:
                    result[:, fi] = 0.0
                else:
                    scaled = ((arr[:, fi] - mins[fi]) / span[fi]) * max_bin
                    rounded = np.floor(scaled + 0.5)
                    result[:, fi] = np.clip(rounded, 0, max_bin)
            # Clamp min/max boundaries
            result = np.where(arr <= mins, 0.0, result)
            result = np.where(arr >= maxs, float(max_bin), result)
            result = np.where(nan_mask, float(nan_bin), result)
            return result.ravel().tolist()
        except (ImportError, ValueError):
            pass
        quantized: list[float] = []
        for row_index in range(row_count):
            row_base = row_index * feature_count
            for feature_index in range(feature_count):
                value = float(flat_values[row_base + feature_index])
                if math.isnan(value):
                    quantized.append(float(nan_bin))
                    continue
                min_value = feature_mins[feature_index]
                max_value = feature_maxs[feature_index]
                if value <= min_value:
                    clamped = 0
                elif value >= max_value:
                    clamped = max_bin
                else:
                    s = max_value - min_value
                    if s <= _PRE_BINNED_INTEGER_TOLERANCE:
                        clamped = 0
                    else:
                        scaled = ((value - min_value) / s) * max_bin
                        rounded = GBMRegressor._round_half_away_from_zero(scaled)
                        clamped = min(max_bin, max(0, rounded))
                quantized.append(float(clamped))
        return quantized

    @staticmethod
    def _quantize_dense_values_linear_with_selective_rank(
        flat_values: Sequence[float],
        row_count: int,
        feature_count: int,
        feature_mins: Sequence[float],
        feature_maxs: Sequence[float],
        rank_flags: Sequence[bool],
        feature_sorted_values: Sequence[Sequence[float]],
        max_bins: int = 256,
    ) -> list[float]:
        quantized: list[float] = []
        max_bin = _max_data_bin_for_max_bins(max_bins)
        nan_bin = _nan_bin_for_max_bins(max_bins)
        for row_index in range(row_count):
            row_base = row_index * feature_count
            for feature_index in range(feature_count):
                value = float(flat_values[row_base + feature_index])
                if math.isnan(value):
                    quantized.append(float(nan_bin))
                    continue
                if rank_flags[feature_index]:
                    sorted_values = feature_sorted_values[feature_index]
                    if len(sorted_values) <= 1:
                        clamped = 0
                    else:
                        rank = bisect.bisect_right(sorted_values, value) - 1
                        if rank < 0:
                            rank = 0
                        elif rank >= len(sorted_values):
                            rank = len(sorted_values) - 1
                        scaled = (rank * max_bin) / (len(sorted_values) - 1)
                        rounded = GBMRegressor._round_half_away_from_zero(scaled)
                        clamped = min(max_bin, max(0, rounded))
                else:
                    min_value = feature_mins[feature_index]
                    max_value = feature_maxs[feature_index]
                    if value <= min_value:
                        clamped = 0
                    elif value >= max_value:
                        clamped = max_bin
                    else:
                        span = max_value - min_value
                        if span <= _PRE_BINNED_INTEGER_TOLERANCE:
                            clamped = 0
                        else:
                            scaled = ((value - min_value) / span) * max_bin
                            rounded = GBMRegressor._round_half_away_from_zero(scaled)
                            clamped = min(max_bin, max(0, rounded))
                quantized.append(float(clamped))
        return quantized

    @staticmethod
    def _quantize_dense_values_rank(
        flat_values: Sequence[float],
        row_count: int,
        feature_count: int,
        feature_sorted_values: Sequence[Sequence[float]],
        max_bins: int = 256,
    ) -> list[float]:
        quantized: list[float] = []
        max_bin = _max_data_bin_for_max_bins(max_bins)
        nan_bin = _nan_bin_for_max_bins(max_bins)
        for row_index in range(row_count):
            row_base = row_index * feature_count
            for feature_index in range(feature_count):
                value = float(flat_values[row_base + feature_index])
                if math.isnan(value):
                    quantized.append(float(nan_bin))
                    continue
                sorted_values = feature_sorted_values[feature_index]
                if len(sorted_values) <= 1:
                    clamped = 0
                else:
                    rank = bisect.bisect_right(sorted_values, value) - 1
                    if rank < 0:
                        rank = 0
                    elif rank >= len(sorted_values):
                        rank = len(sorted_values) - 1
                    scaled = (rank * max_bin) / (len(sorted_values) - 1)
                    rounded = GBMRegressor._round_half_away_from_zero(scaled)
                    clamped = min(max_bin, max(0, rounded))
                quantized.append(float(clamped))
        return quantized

    @staticmethod
    def _quantize_dense_values_quantile(
        flat_values: Sequence[float],
        row_count: int,
        feature_count: int,
        feature_quantile_cuts: Sequence[Sequence[float]],
        max_bins: int = 256,
    ) -> list[float]:
        quantized: list[float] = []
        max_bin = _max_data_bin_for_max_bins(max_bins)
        nan_bin = _nan_bin_for_max_bins(max_bins)
        for row_index in range(row_count):
            row_base = row_index * feature_count
            for feature_index in range(feature_count):
                value = float(flat_values[row_base + feature_index])
                if math.isnan(value):
                    quantized.append(float(nan_bin))
                    continue
                cuts = feature_quantile_cuts[feature_index]
                bucket = bisect.bisect_right(cuts, value)
                clamped = min(max_bin, max(0, bucket))
                quantized.append(float(clamped))
        return quantized

    @staticmethod
    def _check_pre_binned_integers(flat_values: Sequence[float]) -> bool:
        """Check if flat values are pre-binned non-negative integers. Uses numpy fast path."""
        try:
            import numpy as np
            arr = np.asarray(flat_values, dtype=np.float32)
            if np.any(np.isnan(arr)):
                return False
            if np.any(arr < 0.0):
                return False
            rounded = np.where(arr >= 0.0, np.floor(arr + 0.5), np.ceil(arr - 0.5))
            return bool(np.all(np.abs(arr - rounded) <= _PRE_BINNED_INTEGER_TOLERANCE))
        except ImportError:
            pass
        for value in flat_values:
            if math.isnan(value):
                return False
            if value < 0.0:
                return False
            rounded = GBMRegressor._round_half_away_from_zero(value)
            if abs(value - float(rounded)) > _PRE_BINNED_INTEGER_TOLERANCE:
                return False
        return True

    @staticmethod
    def _round_half_away_from_zero(value: float) -> int:
        if value >= 0.0:
            return int(math.floor(value + 0.5))
        return int(math.ceil(value - 0.5))

    @staticmethod
    def _rows_are_pre_binned(rows: Sequence[Sequence[float]]) -> bool:
        for row in rows:
            for value in row:
                if math.isnan(value):
                    return False
                if value < 0.0:
                    return False
                rounded = float(GBMRegressor._round_half_away_from_zero(value))
                if abs(value - rounded) > _PRE_BINNED_INTEGER_TOLERANCE:
                    return False
        return True

    @staticmethod
    def _derive_continuous_feature_bounds(
        rows: Sequence[Sequence[float]],
    ) -> tuple[list[float], list[float]]:
        feature_count = len(rows[0])
        mins = [float("inf")] * feature_count
        maxs = [float("-inf")] * feature_count
        for row in rows:
            for feature_index, value in enumerate(row):
                if value < mins[feature_index]:
                    mins[feature_index] = value
                if value > maxs[feature_index]:
                    maxs[feature_index] = value
        return mins, maxs

    @staticmethod
    def _derive_continuous_feature_sorted_values(
        rows: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        feature_count = len(rows[0])
        columns: list[list[float]] = [[] for _ in range(feature_count)]
        for row in rows:
            for feature_index, value in enumerate(row):
                if not math.isnan(value):
                    columns[feature_index].append(value)
        for feature_index in range(feature_count):
            columns[feature_index].sort()
        return columns

    @staticmethod
    def _derive_continuous_feature_tail_rank_plan(
        rows: Sequence[Sequence[float]],
        core_span_ratio_threshold: float,
    ) -> tuple[list[bool], list[list[float]]]:
        columns = GBMRegressor._derive_continuous_feature_sorted_values(rows)
        flags: list[bool] = []
        for values in columns:
            value_count = len(values)
            if value_count < 5:
                flags.append(False)
                continue
            full_span = values[-1] - values[0]
            if full_span <= _PRE_BINNED_INTEGER_TOLERANCE:
                flags.append(False)
                continue
            trim_count = max(1, int(math.floor(value_count * 0.1)))
            if trim_count * 2 >= value_count:
                flags.append(False)
                continue
            core_low = values[trim_count]
            core_high = values[value_count - 1 - trim_count]
            core_span = core_high - core_low
            ratio = core_span / full_span
            flags.append(
                math.isfinite(ratio) and ratio <= core_span_ratio_threshold
            )
        return flags, columns

    @staticmethod
    def _quantize_rows_linear(
        rows: Sequence[Sequence[float]],
        feature_mins: Sequence[float],
        feature_maxs: Sequence[float],
        max_bins: int = 256,
    ) -> list[list[float]]:
        quantized: list[list[float]] = []
        max_bin = _max_data_bin_for_max_bins(max_bins)
        nan_bin = _nan_bin_for_max_bins(max_bins)
        for row in rows:
            quantized_row: list[float] = []
            for feature_index, value in enumerate(row):
                if math.isnan(value):
                    quantized_row.append(float(nan_bin))
                    continue
                min_value = feature_mins[feature_index]
                max_value = feature_maxs[feature_index]
                if value <= min_value:
                    clamped = 0
                elif value >= max_value:
                    clamped = max_bin
                else:
                    span = max_value - min_value
                    if span <= _PRE_BINNED_INTEGER_TOLERANCE:
                        clamped = 0
                    else:
                        scaled = ((value - min_value) / span) * max_bin
                        rounded = GBMRegressor._round_half_away_from_zero(scaled)
                        clamped = min(max_bin, max(0, rounded))
                quantized_row.append(float(clamped))
            quantized.append(quantized_row)
        return quantized

    @staticmethod
    def _quantize_rows_linear_with_selective_rank(
        rows: Sequence[Sequence[float]],
        feature_mins: Sequence[float],
        feature_maxs: Sequence[float],
        rank_flags: Sequence[bool],
        feature_sorted_values: Sequence[Sequence[float]],
        max_bins: int = 256,
    ) -> list[list[float]]:
        quantized: list[list[float]] = []
        max_bin = _max_data_bin_for_max_bins(max_bins)
        nan_bin = _nan_bin_for_max_bins(max_bins)
        for row in rows:
            quantized_row: list[float] = []
            for feature_index, value in enumerate(row):
                if math.isnan(value):
                    quantized_row.append(float(nan_bin))
                    continue
                if rank_flags[feature_index]:
                    sorted_values = feature_sorted_values[feature_index]
                    if len(sorted_values) <= 1:
                        clamped = 0
                    else:
                        rank = bisect.bisect_right(sorted_values, value) - 1
                        if rank < 0:
                            rank = 0
                        elif rank >= len(sorted_values):
                            rank = len(sorted_values) - 1
                        scaled = (rank * max_bin) / (len(sorted_values) - 1)
                        rounded = GBMRegressor._round_half_away_from_zero(scaled)
                        clamped = min(max_bin, max(0, rounded))
                else:
                    min_value = feature_mins[feature_index]
                    max_value = feature_maxs[feature_index]
                    if value <= min_value:
                        clamped = 0
                    elif value >= max_value:
                        clamped = max_bin
                    else:
                        span = max_value - min_value
                        if span <= _PRE_BINNED_INTEGER_TOLERANCE:
                            clamped = 0
                        else:
                            scaled = ((value - min_value) / span) * max_bin
                            rounded = GBMRegressor._round_half_away_from_zero(scaled)
                            clamped = min(max_bin, max(0, rounded))
                quantized_row.append(float(clamped))
            quantized.append(quantized_row)
        return quantized

    @staticmethod
    def _quantize_rows_rank(
        rows: Sequence[Sequence[float]],
        feature_sorted_values: Sequence[Sequence[float]],
        max_bins: int = 256,
    ) -> list[list[float]]:
        quantized: list[list[float]] = []
        max_bin = _max_data_bin_for_max_bins(max_bins)
        nan_bin = _nan_bin_for_max_bins(max_bins)
        for row in rows:
            quantized_row: list[float] = []
            for feature_index, value in enumerate(row):
                if math.isnan(value):
                    quantized_row.append(float(nan_bin))
                    continue
                sorted_values = feature_sorted_values[feature_index]
                if len(sorted_values) <= 1:
                    clamped = 0
                else:
                    rank = bisect.bisect_right(sorted_values, value) - 1
                    if rank < 0:
                        rank = 0
                    elif rank >= len(sorted_values):
                        rank = len(sorted_values) - 1
                    scaled = (rank * max_bin) / (len(sorted_values) - 1)
                    rounded = GBMRegressor._round_half_away_from_zero(scaled)
                    clamped = min(max_bin, max(0, rounded))
                quantized_row.append(float(clamped))
            quantized.append(quantized_row)
        return quantized

    @staticmethod
    def _derive_continuous_feature_quantile_cuts(
        rows: Sequence[Sequence[float]],
        max_bins: int,
    ) -> list[list[float]]:
        feature_count = len(rows[0])
        columns: list[list[float]] = [[] for _ in range(feature_count)]
        for row in rows:
            for feature_index, value in enumerate(row):
                if not math.isnan(value):
                    columns[feature_index].append(value)

        feature_cuts: list[list[float]] = []
        for feature_index in range(feature_count):
            values = columns[feature_index]
            values.sort()
            if len(values) <= 1:
                feature_cuts.append([])
                continue

            bin_count = min(max_bins, len(values))
            cuts: list[float] = []
            for quantile_index in range(1, bin_count):
                rank = (quantile_index * len(values)) // bin_count
                if rank >= len(values):
                    rank = len(values) - 1
                cut_value = values[rank]
                if cuts and cut_value <= cuts[-1]:
                    continue
                cuts.append(cut_value)
            feature_cuts.append(cuts)
        return feature_cuts

    @staticmethod
    def _quantize_rows_quantile(
        rows: Sequence[Sequence[float]],
        feature_quantile_cuts: Sequence[Sequence[float]],
        max_bins: int = 256,
    ) -> list[list[float]]:
        quantized: list[list[float]] = []
        max_bin = _max_data_bin_for_max_bins(max_bins)
        nan_bin = _nan_bin_for_max_bins(max_bins)
        for row in rows:
            quantized_row: list[float] = []
            for feature_index, value in enumerate(row):
                if math.isnan(value):
                    quantized_row.append(float(nan_bin))
                    continue
                cuts = feature_quantile_cuts[feature_index]
                bucket = bisect.bisect_right(cuts, value)
                clamped = min(max_bin, max(0, bucket))
                quantized_row.append(float(clamped))
            quantized.append(quantized_row)
        return quantized

    def _quantize_rows_for_prediction(
        self, rows: Sequence[Sequence[float]]
    ) -> list[list[float]]:
        if self.continuous_binning_strategy == "linear":
            mins, maxs = self._require_continuous_feature_bounds()
            rank_flags = self._continuous_feature_linear_rank_flags
            if rank_flags is not None and any(rank_flags):
                sorted_values = self._require_continuous_feature_sorted_values()
                return self._quantize_rows_linear_with_selective_rank(
                    rows, mins, maxs, rank_flags, sorted_values,
                    max_bins=self.continuous_binning_max_bins,
                )
            return self._quantize_rows_linear(
                rows, mins, maxs,
                max_bins=self.continuous_binning_max_bins,
            )
        if self.continuous_binning_strategy == "rank":
            sorted_values = self._require_continuous_feature_sorted_values()
            return self._quantize_rows_rank(
                rows, sorted_values,
                max_bins=self.continuous_binning_max_bins,
            )
        quantile_cuts = self._require_continuous_feature_quantile_cuts()
        return self._quantize_rows_quantile(
            rows, quantile_cuts,
            max_bins=self.continuous_binning_max_bins,
        )

    def _require_continuous_feature_bounds(self) -> tuple[list[float], list[float]]:
        if self._continuous_feature_mins is None or self._continuous_feature_maxs is None:
            raise RuntimeError(
                "continuous-feature quantization bounds are missing; refit the model"
            )
        return self._continuous_feature_mins, self._continuous_feature_maxs

    def _require_continuous_feature_sorted_values(self) -> list[list[float]]:
        if self._continuous_feature_sorted_values is None:
            raise RuntimeError(
                "continuous-feature quantization bounds are missing; refit the model"
            )
        return self._continuous_feature_sorted_values

    def _require_continuous_feature_quantile_cuts(self) -> list[list[float]]:
        if self._continuous_feature_quantile_cuts is None:
            raise RuntimeError(
                "continuous-feature quantization cuts are missing; refit the model"
            )
        return self._continuous_feature_quantile_cuts

    def _apply_continuous_binning_metadata(self, metadata: object) -> None:
        self._uses_continuous_binning = bool(
            getattr(metadata, "uses_continuous_binning", False)
        )
        self._continuous_feature_mins = getattr(metadata, "feature_mins", None)
        self._continuous_feature_maxs = getattr(metadata, "feature_maxs", None)
        self._continuous_feature_sorted_values = getattr(
            metadata, "feature_sorted_values", None
        )
        self._continuous_feature_quantile_cuts = getattr(
            metadata, "feature_quantile_cuts", None
        )
        self.feature_quantile_cut_methods_ = getattr(
            metadata, "feature_quantile_cut_methods", None
        )
        self._continuous_feature_linear_rank_flags = getattr(
            metadata, "feature_linear_rank_flags", None
        )


# GBMRegressor is injected into this module's namespace by _core.py after the
# class is defined (see the bottom of _core.py).  Static methods in
# _QuantizationMixin reference GBMRegressor by name; those names are resolved at
# call-time against this module's globals, which by then contain the real class.
