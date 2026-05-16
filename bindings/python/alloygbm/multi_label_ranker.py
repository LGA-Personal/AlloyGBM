"""Multi-label learning-to-rank estimator for AlloyGBM.

`MultiLabelGBMRanker` exposes a unified multi-output ranking API: a single
estimator with one ``fit`` / ``predict`` interface that trains and scores
across multiple ranking labels at once.  ``y`` is shaped ``(n_rows,
n_labels)`` and ``predict`` returns scores with the same column layout.

Internally v0.7.1 trains one independent `GBMRanker` per label.  This makes
the implementation a thin wrapper that reuses every existing feature on
`GBMRanker` (warm-start, factor neutralization, MorphBoost, PL leaves, DRO
leaves, interaction constraints, custom eval metrics).  The labels share
``group`` and ``factor_exposures`` so the per-label fits remain comparable.

Numerically this is equivalent to training each label separately; joint
shared-tree multi-label training is a remaining v0.7.x follow-up.
Users get the ergonomic API today and a clean upgrade path tomorrow.
"""

from __future__ import annotations

import copy
import struct
from typing import Any

import numpy as np

from .ranker import GBMRanker

_MULTI_LABEL_RANKER_MAGIC = b"MLRK"
_MULTI_LABEL_RANKER_VERSION = 1


class MultiLabelGBMRanker:
    """Gradient Boosted Decision Tree learning-to-rank estimator with
    multiple ranking labels per item.

    Parameters
    ----------
    ranking_labels : list[str] | None, optional
        Optional names for each ranking label.  When provided, the model
        records them on ``self.ranking_labels_`` and exposes them via
        ``get_params``.  When omitted, labels are positional (``"label_0"``,
        ``"label_1"``, ...).  The number of labels is inferred from the
        shape of ``y`` at fit time.

    ranking_objective : str | list[str], default ``"rank:ndcg"``
        Either a single objective applied to every label, or a list of
        per-label objectives with length ``n_labels``.  Supported values
        match :class:`GBMRanker`.

    **kwargs
        Forwarded to :class:`GBMRanker` for every per-label fit.
    """

    def __init__(
        self,
        *,
        ranking_labels: list[str] | None = None,
        ranking_objective: str | list[str] = "rank:ndcg",
        **kwargs: Any,
    ) -> None:
        self.ranking_labels = (
            [str(label) for label in ranking_labels]
            if ranking_labels is not None
            else None
        )
        # Per-label objective normalization happens at fit time once we know
        # n_labels — we just stash whatever the user supplied here.
        self.ranking_objective = ranking_objective
        # Stash kwargs so we can clone the same configuration into every
        # per-label `GBMRanker`.  `_per_label_kwargs` is the canonical store.
        self._per_label_kwargs: dict[str, Any] = dict(kwargs)
        self._is_fitted = False
        self._sub_rankers: list[GBMRanker] = []
        self.ranking_labels_: list[str] | None = None
        self.n_labels_: int | None = None
        self.rounds_completed_: list[int] | None = None

    # ── Configuration ──────────────────────────────────────────────────

    def _resolve_objectives(self, n_labels: int) -> list[str]:
        if isinstance(self.ranking_objective, str):
            return [self.ranking_objective] * n_labels
        objs = list(self.ranking_objective)
        if len(objs) != n_labels:
            raise ValueError(
                f"ranking_objective list length {len(objs)} does not match "
                f"y's label count {n_labels}"
            )
        return [str(obj) for obj in objs]

    def _resolve_label_names(self, n_labels: int) -> list[str]:
        if self.ranking_labels is None:
            return [f"label_{i}" for i in range(n_labels)]
        if len(self.ranking_labels) != n_labels:
            raise ValueError(
                f"ranking_labels length {len(self.ranking_labels)} does not "
                f"match y's label count {n_labels}"
            )
        return list(self.ranking_labels)

    def get_params(self, deep: bool = True) -> dict:
        """Return estimator parameters in sklearn-compatible shape."""
        del deep
        params = dict(self._per_label_kwargs)
        params["ranking_objective"] = (
            list(self.ranking_objective)
            if isinstance(self.ranking_objective, list)
            else self.ranking_objective
        )
        params["ranking_labels"] = (
            list(self.ranking_labels) if self.ranking_labels is not None else None
        )
        return params

    def set_params(self, **params: object) -> "MultiLabelGBMRanker":
        if "ranking_objective" in params:
            self.ranking_objective = params.pop("ranking_objective")  # type: ignore[assignment]
        if "ranking_labels" in params:
            v = params.pop("ranking_labels")
            self.ranking_labels = (
                [str(label) for label in v] if v is not None else None  # type: ignore[union-attr]
            )
        self._per_label_kwargs.update(params)
        return self

    # ── Training ───────────────────────────────────────────────────────

    def fit(
        self,
        X: object,
        y: object,
        *,
        group: object | None = None,
        factor_exposures: object | None = None,
        **fit_kwargs: Any,
    ) -> "MultiLabelGBMRanker":
        """Train one ranker per label using shared ``group`` and (optionally)
        shared ``factor_exposures``.

        Parameters
        ----------
        X : array-like, shape ``(n_rows, n_features)``
            Feature matrix.
        y : array-like, shape ``(n_rows, n_labels)``
            Per-label relevance/target columns.  ``n_labels`` is inferred
            from the array shape.  A 1-D ``y`` is rejected — use
            :class:`GBMRanker` for single-label ranking.
        group : array-like, optional
            Group sizes shared across all labels (the per-label rankers all
            see the same query-document grouping).
        factor_exposures : array-like, optional
            Shared factor exposures forwarded to every per-label fit.
        **fit_kwargs
            Forwarded as-is to every per-label ``GBMRanker.fit`` call.
        """
        y_arr = np.asarray(y)
        if y_arr.ndim != 2:
            raise ValueError(
                "y must be 2-D with shape (n_rows, n_labels); for single-label "
                "ranking use GBMRanker"
            )
        n_rows, n_labels = y_arr.shape
        if n_labels == 0:
            raise ValueError("y must have at least one label column")

        objectives = self._resolve_objectives(n_labels)
        names = self._resolve_label_names(n_labels)

        # `eval_set=(X_val, y_val)` arrives with a 2-D ``y_val`` shaped
        # ``(m, n_labels)`` so callers can use validation-dependent features
        # like ``early_stopping_rounds`` / ``eval_metric``.  Each per-label
        # ``GBMRanker`` only sees its slice of the target, so we must slice
        # the validation target column-wise too — otherwise ``GBMRegressor``
        # tries to cast each row vector to ``float`` and raises during
        # ``_validate_targets``.  Sample weights / group / time index on the
        # validation set are shape-preserving across labels so they pass
        # through unchanged.
        eval_set = fit_kwargs.pop("eval_set", None)

        self._sub_rankers = []
        self.rounds_completed_ = []
        for label_idx in range(n_labels):
            ranker = GBMRanker(
                ranking_objective=objectives[label_idx],
                **copy.deepcopy(self._per_label_kwargs),
            )
            label_targets = np.asarray(y_arr[:, label_idx], dtype=np.float32)
            label_fit_kwargs = dict(fit_kwargs)
            if eval_set is not None:
                X_val, y_val = eval_set
                y_val_arr = np.asarray(y_val)
                if y_val_arr.ndim == 2:
                    if y_val_arr.shape[1] != n_labels:
                        raise ValueError(
                            f"eval_set y has {y_val_arr.shape[1]} label columns "
                            f"but training y has {n_labels}; columns must match"
                        )
                    label_y_val = np.asarray(
                        y_val_arr[:, label_idx], dtype=np.float32
                    )
                else:
                    # Permit a 1-D ``y_val`` when ``n_labels == 1`` — same
                    # column for every sub-ranker is the only sensible
                    # interpretation otherwise.
                    if n_labels != 1:
                        raise ValueError(
                            "eval_set y must be 2-D with shape "
                            "(m, n_labels) when training has multiple labels"
                        )
                    label_y_val = np.asarray(y_val_arr, dtype=np.float32)
                label_fit_kwargs["eval_set"] = (X_val, label_y_val)
            ranker.fit(
                X,
                label_targets,
                group=group,
                factor_exposures=factor_exposures,
                **label_fit_kwargs,
            )
            self._sub_rankers.append(ranker)
            self.rounds_completed_.append(int(ranker.rounds_completed_ or 0))

        self.ranking_labels_ = names
        self.n_labels_ = n_labels
        self._is_fitted = True
        return self

    # ── Prediction ─────────────────────────────────────────────────────

    def predict(self, X: object) -> np.ndarray:
        """Predict ranking scores for each label.

        Returns
        -------
        np.ndarray of shape ``(n_rows, n_labels)``
            Column ``i`` holds the scores produced by the ranker fit for
            label ``i`` (matching ``self.ranking_labels_``).
        """
        if not self._is_fitted:
            raise RuntimeError("MultiLabelGBMRanker must be fit before predict")
        cols = [np.asarray(ranker.predict(X), dtype=np.float64) for ranker in self._sub_rankers]
        return np.stack(cols, axis=1)

    # ── Persistence ────────────────────────────────────────────────────

    def save_model(self, path: str) -> None:
        """Serialise every per-label ranker into a single bundle.

        Uses :mod:`pickle` for the per-ranker payloads so the bundle picks
        up all of :class:`GBMRanker`'s ``__getstate__`` / ``__setstate__``
        plumbing — meaning the same set of features that survive a
        ``pickle.dumps(ranker)`` survive the multi-label save/load.  The
        outer container is a small framed format with a magic word, version,
        label count, and label names so the file is self-describing.
        """
        if not self._is_fitted:
            raise RuntimeError("MultiLabelGBMRanker must be fit before save_model")
        import pickle

        with open(path, "wb") as f:
            f.write(_MULTI_LABEL_RANKER_MAGIC)
            f.write(struct.pack("<II", _MULTI_LABEL_RANKER_VERSION, len(self._sub_rankers)))
            names = self.ranking_labels_ or [f"label_{i}" for i in range(len(self._sub_rankers))]
            for name in names:
                encoded = name.encode("utf-8")
                f.write(struct.pack("<I", len(encoded)))
                f.write(encoded)
            for ranker in self._sub_rankers:
                blob = pickle.dumps(ranker, protocol=pickle.HIGHEST_PROTOCOL)
                f.write(struct.pack("<Q", len(blob)))
                f.write(blob)

    @classmethod
    def load_model(cls, path: str) -> "MultiLabelGBMRanker":
        import pickle

        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != _MULTI_LABEL_RANKER_MAGIC:
                raise ValueError(
                    "file is not a MultiLabelGBMRanker bundle "
                    f"(magic {magic!r} != {_MULTI_LABEL_RANKER_MAGIC!r})"
                )
            version, n_labels = struct.unpack("<II", f.read(8))
            if version != _MULTI_LABEL_RANKER_VERSION:
                raise ValueError(
                    f"unsupported MultiLabelGBMRanker bundle version {version}"
                )
            names: list[str] = []
            for _ in range(n_labels):
                (name_len,) = struct.unpack("<I", f.read(4))
                names.append(f.read(name_len).decode("utf-8"))
            rankers: list[GBMRanker] = []
            for _ in range(n_labels):
                (blob_len,) = struct.unpack("<Q", f.read(8))
                blob = f.read(blob_len)
                rankers.append(pickle.loads(blob))
        # Reconstruct wrapper-level configuration from the sub-rankers so
        # `get_params` after load matches the original.  At fit time every
        # sub-ranker was configured with the same ``_per_label_kwargs``
        # modulo ``ranking_objective``, so lifting params from sub_rankers[0]
        # and collapsing the per-label objective is enough.
        first_params = rankers[0].get_params()
        per_label_objectives = [r.ranking_objective for r in rankers]
        ranking_objective: str | list[str]
        if len(set(per_label_objectives)) == 1:
            ranking_objective = per_label_objectives[0]
        else:
            ranking_objective = list(per_label_objectives)
        # Drop the per-label objective from the shared kwargs — it's
        # represented separately on `ranking_objective` and must not leak
        # back into `_per_label_kwargs` (where it would override at fit time).
        first_params.pop("ranking_objective", None)
        inst = cls(
            ranking_labels=names,
            ranking_objective=ranking_objective,
            **first_params,
        )
        inst._sub_rankers = rankers
        inst.ranking_labels_ = names
        inst.n_labels_ = n_labels
        inst._is_fitted = True
        inst.rounds_completed_ = [int(r.rounds_completed_ or 0) for r in rankers]
        return inst

    # ── Introspection ──────────────────────────────────────────────────

    @property
    def sub_rankers_(self) -> list[GBMRanker]:
        """Return the underlying per-label :class:`GBMRanker` instances."""
        return list(self._sub_rankers)

    def __repr__(self) -> str:
        return (
            f"MultiLabelGBMRanker(n_labels={self.n_labels_}, "
            f"ranking_labels={self.ranking_labels_}, "
            f"ranking_objective={self.ranking_objective!r}, "
            f"fitted={self._is_fitted})"
        )
