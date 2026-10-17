"""Split-conformal prediction intervals for fitted AlloyGBM regressors.

Two calibrators share one interface: fit a model on training data, call
``calibrate(X_cal, y_cal, alpha)`` on rows the model never saw, then
``predict_interval(X)`` returns ``(lower, upper)`` arrays that cover the true
target with probability about ``1 - alpha``.

- :class:`SplitConformal` is the classic split-conformal baseline: one global
  quantile of the absolute residual ``|y - prediction|``, so every interval
  has the same width.
- :class:`LoBoost` (Santos et al., *LoBoost*, arXiv 2602.22432) groups
  calibration rows by the leaves they share along the boosting path, so the
  interval width adapts to the region of feature space a row falls in while
  keeping finite-sample calibration within each group.

Neither touches training or the model artifact; both pickle with the model
they wrap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["LoBoost", "SplitConformal"]


def _conformal_quantile(sorted_scores: np.ndarray, alpha: float) -> float:
    """Level ``ceil((n + 1)(1 - alpha)) / n`` empirical quantile; ``inf`` when ``n`` is too small."""
    n = sorted_scores.shape[0]
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if n == 0 or rank > n:
        return math.inf
    return float(sorted_scores[rank - 1])


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return alpha


def _calibration_targets(model: object, X: object, y: object) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "predict_proba"):
        raise TypeError("conformal intervals need a regression model, not a classifier")
    targets = np.asarray(y, dtype=np.float64).reshape(-1)
    predictions = np.asarray(model.predict(X), dtype=np.float64).reshape(-1)
    if targets.shape[0] != predictions.shape[0]:
        raise ValueError(
            f"y has {targets.shape[0]} rows but X has {predictions.shape[0]}"
        )
    if targets.shape[0] == 0:
        raise ValueError("calibration data must contain at least one row")
    if not np.isfinite(targets).all():
        raise ValueError("calibration targets must be finite (no NaN or inf)")
    if not np.isfinite(predictions).all():
        raise ValueError("model predictions on the calibration rows must be finite")
    return targets, predictions


class SplitConformal:
    """Split-conformal intervals with one global residual quantile.

    Parameters
    ----------
    model:
        A fitted AlloyGBM regressor (or any object with ``predict``).
    """

    def __init__(self, model: object) -> None:
        self.model = model
        self.alpha_: float | None = None
        self.scores_: np.ndarray | None = None

    def calibrate(self, X: object, y: object, alpha: float = 0.1) -> "SplitConformal":
        """Store the sorted absolute residuals of the calibration rows."""
        self.alpha_ = _validate_alpha(alpha)
        targets, predictions = _calibration_targets(self.model, X, y)
        self.scores_ = np.sort(np.abs(targets - predictions))
        return self

    def predict_interval(
        self, X: object, alpha: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)``; ``alpha`` defaults to the calibrated level."""
        if self.scores_ is None or self.alpha_ is None:
            raise RuntimeError("call calibrate() before predict_interval()")
        alpha = self.alpha_ if alpha is None else _validate_alpha(alpha)
        predictions = np.asarray(self.model.predict(X), dtype=np.float64).reshape(-1)
        half_width = _conformal_quantile(self.scores_, alpha)
        return predictions - half_width, predictions + half_width


@dataclass
class _Node:
    """A node of the leaf-prefix refinement tree built during calibration."""

    scores: np.ndarray
    split_tree: int = -1
    children: dict[int, int] = field(default_factory=dict)
    region: int = -1


class LoBoost:
    """Local conformal intervals from shared boosting-path leaves (LoBoost).

    Calibration rows are grouped by the leaves they reach, one tree at a time:
    a group with at least ``N_min = max(n_part, ceil(p_min * n_cal))`` rows is
    split by its leaves in the next tree that separates them, and a group below
    ``N_min`` stops. Undersized groups are then merged, deepest first, into the
    group whose leaf path is closest under a Hamming distance weighted by each
    tree's output range. Each final group gets its own conformal quantile of
    ``|y - prediction|``.

    A new row follows the same refinement tree using its leaves. When it
    reaches a leaf no calibration row reached, it gets the quantile of the
    deepest group it did match (the paper falls back to the global quantile;
    the matched ancestor is the more local of the two and is still a group of
    at least ``N_min`` calibration rows).

    Parameters
    ----------
    model:
        A fitted single-output AlloyGBM regressor (``GBMRegressor`` or
        ``GBMRanker``); it must support ``apply``.
    n_part:
        Absolute floor on group size (the paper's ``N_part``, default 50).
    p_min:
        Proportional floor on group size, as a fraction of the calibration
        rows (the paper's ``p_min``, default 0.005).
    """

    def __init__(self, model: object, *, n_part: int = 50, p_min: float = 0.005) -> None:
        if int(n_part) < 1:
            raise ValueError(f"n_part must be >= 1, got {n_part}")
        if not 0.0 <= float(p_min) < 1.0:
            raise ValueError(f"p_min must be in [0, 1), got {p_min}")
        self.model = model
        self.n_part = int(n_part)
        self.p_min = float(p_min)
        self.alpha_: float | None = None
        self.min_group_size_: int | None = None
        self.n_regions_: int | None = None
        self.depth_: int | None = None
        self._nodes: list[_Node] = []
        self._region_scores: list[np.ndarray] = []
        self._global_scores: np.ndarray | None = None

    # -- Calibration ---------------------------------------------------------

    def calibrate(self, X: object, y: object, alpha: float = 0.1) -> "LoBoost":
        """Build the leaf-prefix partition on calibration rows and store residuals."""
        self.alpha_ = _validate_alpha(alpha)
        targets, predictions = _calibration_targets(self.model, X, y)
        scores = np.abs(targets - predictions)
        leaves, outputs = self.model._leaf_assignments(X)
        leaves = np.asarray(leaves, dtype=np.int64)
        n_cal, n_trees = leaves.shape
        min_size = max(self.n_part, math.ceil(self.p_min * n_cal))
        self.min_group_size_ = min_size
        self._global_scores = np.sort(scores)

        # Refinement: grow the prefix tree one boosting tree at a time. Only
        # covariates decide the partition; responses enter through `scores`.
        nodes: list[_Node] = [_Node(scores=self._global_scores)]
        node_rows: list[np.ndarray] = [np.arange(n_cal)]
        node_depth: list[int] = [0]
        active = [0] if n_cal >= min_size else []
        depth = 0
        for tree in range(n_trees):
            if not active:
                break
            next_active: list[int] = []
            for node_index in active:
                rows = node_rows[node_index]
                column = leaves[rows, tree]
                distinct = np.unique(column)
                if distinct.shape[0] == 1:
                    next_active.append(node_index)
                    continue
                nodes[node_index].split_tree = tree
                for leaf in distinct:
                    child_rows = rows[column == leaf]
                    child_index = len(nodes)
                    nodes[node_index].children[int(leaf)] = child_index
                    nodes.append(_Node(scores=np.sort(scores[child_rows])))
                    node_rows.append(child_rows)
                    node_depth.append(tree + 1)
                    if child_rows.shape[0] >= min_size:
                        next_active.append(child_index)
            active = next_active
            depth = tree + 1
        self.depth_ = depth

        terminal = [i for i, node in enumerate(nodes) if not node.children]
        region_rows = [node_rows[i] for i in terminal]
        region_depth = np.array([node_depth[i] for i in terminal], dtype=np.int64)
        region_size = np.array([rows.shape[0] for rows in region_rows], dtype=np.int64)
        merged_into = self._merge_undersized(
            leaves[:, :depth], outputs[:, :depth], region_rows, region_depth, region_size,
            min_size,
        )

        # Compact surviving regions and attach their residual quantile inputs.
        survivors = sorted(set(merged_into))
        compact = {raw: index for index, raw in enumerate(survivors)}
        members: list[list[np.ndarray]] = [[] for _ in survivors]
        for raw_index, target in enumerate(merged_into):
            members[compact[target]].append(region_rows[raw_index])
        self._region_scores = [np.sort(scores[np.concatenate(rows)]) for rows in members]
        for raw_index, node_index in enumerate(terminal):
            nodes[node_index].region = compact[merged_into[raw_index]]
        self._nodes = nodes
        self.n_regions_ = len(survivors)
        return self

    @staticmethod
    def _merge_undersized(
        leaves: np.ndarray,
        outputs: np.ndarray,
        region_rows: list[np.ndarray],
        region_depth: np.ndarray,
        region_size: np.ndarray,
        min_size: int,
    ) -> list[int]:
        """Merge undersized regions into their nearest neighbour; return each raw region's target."""
        region_count = len(region_rows)
        merged_into = list(range(region_count))
        if region_count <= 1 or not (region_size < min_size).any():
            return merged_into
        depth = leaves.shape[1]

        # Tree weights: each tree's output range across its leaves (mean leaf
        # output on calibration rows, so linear leaves get one value per leaf).
        weights = np.zeros(depth, dtype=np.float64)
        for tree in range(depth):
            leaf_column = leaves[:, tree]
            counts = np.bincount(leaf_column)
            sums = np.bincount(leaf_column, weights=outputs[:, tree])
            seen = counts > 0
            means = sums[seen] / counts[seen]
            weights[tree] = float(means.max() - means.min())

        # Representative path: the shared prefix, then the per-tree mode of
        # the members' leaves for trees after the region stopped refining.
        leaf_span = int(leaves.max()) + 1 if leaves.size else 1
        offsets = np.arange(depth, dtype=np.int64) * leaf_span
        paths = np.empty((region_count, depth), dtype=np.int64)
        for index, rows in enumerate(region_rows):
            block = leaves[rows]
            if block.shape[0] == 1:
                paths[index] = block[0]
                continue
            counts = np.bincount(
                (block + offsets).ravel(), minlength=depth * leaf_span
            ).reshape(depth, leaf_span)
            paths[index] = counts.argmax(axis=1)

        alive = np.ones(region_count, dtype=bool)
        size = region_size.astype(np.int64).copy()
        while alive.sum() > 1:
            undersized = np.flatnonzero(alive & (size < min_size))
            if undersized.shape[0] == 0:
                break
            # Deepest first, ties broken in favour of the smaller region.
            order = np.lexsort((size[undersized], -region_depth[undersized]))
            source = int(undersized[order[0]])
            candidates = np.flatnonzero(alive)
            candidates = candidates[candidates != source]
            distance = (paths[candidates] != paths[source]) @ weights
            target = int(candidates[int(np.argmin(distance))])
            size[target] += size[source]
            alive[source] = False
            merged_into[source] = target
        # Resolve chains (a region merged into one that merged later).
        for index in range(region_count):
            target = merged_into[index]
            while merged_into[target] != target:
                target = merged_into[target]
            merged_into[index] = target
        return merged_into

    # -- Prediction ----------------------------------------------------------

    def _half_widths(self, X: object, alpha: float) -> np.ndarray:
        leaves, _ = self.model._leaf_assignments(X)
        leaves = np.asarray(leaves, dtype=np.int64)
        row_count = leaves.shape[0]
        region_q = np.array(
            [_conformal_quantile(s, alpha) for s in self._region_scores], dtype=np.float64
        )
        half_width = np.empty(row_count, dtype=np.float64)
        stack: list[tuple[int, np.ndarray]] = [(0, np.arange(row_count))]
        while stack:
            node_index, rows = stack.pop()
            node = self._nodes[node_index]
            if not node.children:
                if node.region >= 0:
                    half_width[rows] = region_q[node.region]
                else:
                    half_width[rows] = _conformal_quantile(node.scores, alpha)
                continue
            column = leaves[rows, node.split_tree]
            matched = np.zeros(rows.shape[0], dtype=bool)
            for leaf, child_index in node.children.items():
                mask = column == leaf
                if mask.any():
                    matched |= mask
                    stack.append((child_index, rows[mask]))
            if not matched.all():
                half_width[rows[~matched]] = _conformal_quantile(node.scores, alpha)
        return half_width

    def predict_interval(
        self, X: object, alpha: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)``; ``alpha`` defaults to the calibrated level."""
        if not self._nodes or self.alpha_ is None:
            raise RuntimeError("call calibrate() before predict_interval()")
        alpha = self.alpha_ if alpha is None else _validate_alpha(alpha)
        predictions = np.asarray(self.model.predict(X), dtype=np.float64).reshape(-1)
        half_width = self._half_widths(X, alpha)
        return predictions - half_width, predictions + half_width
