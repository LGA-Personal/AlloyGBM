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
# v2 (v0.10.1+) bundles include a `mode` byte after the version word so
# joint-mode and independent-mode bundles can coexist. v1 bundles always
# implied independent mode and load through a back-compat branch.
_MULTI_LABEL_RANKER_VERSION = 2


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
        multi_label_mode: str = "independent",
        **kwargs: Any,
    ) -> None:
        if multi_label_mode not in ("independent", "joint"):
            raise ValueError(
                f"multi_label_mode must be 'independent' or 'joint', got "
                f"{multi_label_mode!r}"
            )
        self.multi_label_mode = multi_label_mode
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
        # Joint-mode state (populated only when multi_label_mode == "joint")
        self._joint_handle = None  # JointPredictorHandle
        self._joint_artifact_bytes: bytes | None = None
        self._joint_baselines: list[float] | None = None
        self._joint_feature_count: int | None = None
        self.ranking_labels_: list[str] | None = None
        self.n_labels_: int | None = None
        self.rounds_completed_: list[int] | int | None = None

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
        params["multi_label_mode"] = self.multi_label_mode
        return params

    def set_params(self, **params: object) -> "MultiLabelGBMRanker":
        if "multi_label_mode" in params:
            tm = params.pop("multi_label_mode")
            if tm not in ("independent", "joint"):
                raise ValueError(
                    f"multi_label_mode must be 'independent' or 'joint', got {tm!r}"
                )
            self.multi_label_mode = tm  # type: ignore[assignment]
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

        if self.multi_label_mode == "joint":
            self._fit_joint(
                X,
                y_arr,
                group=group,
                factor_exposures=factor_exposures,
                objectives=objectives,
                names=names,
                n_labels=n_labels,
                fit_kwargs=fit_kwargs,
                eval_set=eval_set,
            )
            return self

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

    # ── Joint shared-tree training (v0.10.1) ───────────────────────────

    # Per-label kwargs that the v0.10.1 joint trainer actually consumes
    # (`engine::joint::fit_joint_multi_output` reads
    # `learning_rate`, `seed`, `max_depth`, `min_data_in_leaf`,
    # `lambda_l2`; `max_bin` is consumed by `prepare_training_matrices`
    # one layer up; `n_estimators` is the round count).  Anything
    # outside this set is rejected by `_fit_joint` so callers never
    # silently lose a configured knob — joint-path feature parity
    # (row/col subsample, min_split_gain, MorphBoost, neutralization,
    # interaction constraints, etc.) is tracked as v0.10.x follow-ups
    # in docs/limitations.md.
    _JOINT_SUPPORTED_KWARGS = frozenset({
        "n_estimators",
        "learning_rate",
        "seed",
        "max_depth",
        "min_data_in_leaf",
        "lambda_l2",
        "max_bin",
    })

    @staticmethod
    def _normalize_group_for_joint(group: object, n_rows: int) -> list[int]:
        """Normalize ``group`` to a length-``n_rows`` per-row group ID
        list as the joint trainer (and the underlying ranking
        objectives) require.

        Accepts two LightGBM-compatible input shapes:

        * **per-row IDs** — ``len(group) == n_rows``.  Returned as-is
          (after dtype coercion).
        * **group sizes** — ``len(group) < n_rows`` AND
          ``sum(group) == n_rows``.  Expanded via ``np.repeat`` so
          group ``i`` becomes ``size[i]`` consecutive copies of ``i``.

        Anything else raises ``ValueError`` with a clear message.
        """
        g_arr = np.asarray(group)
        if g_arr.ndim != 1:
            raise ValueError("group must be 1-D")
        g_arr = g_arr.astype(np.int64, copy=False)
        if len(g_arr) == n_rows:
            return g_arr.astype(np.uint32).tolist()
        if len(g_arr) < n_rows and int(g_arr.sum()) == n_rows:
            ids = np.repeat(np.arange(len(g_arr), dtype=np.uint32), g_arr.tolist())
            return ids.tolist()
        raise ValueError(
            f"multi_label_mode='joint' could not interpret group: length "
            f"{len(g_arr)} matches neither per-row IDs (n_rows={n_rows}) "
            f"nor group sizes summing to n_rows (sum={int(g_arr.sum())}). "
            f"Pass either a length-{n_rows} array of per-row group IDs, or "
            f"a shorter array of contiguous group sizes."
        )

    def _fit_joint(
        self,
        X: object,
        y_arr: np.ndarray,
        *,
        group: object | None,
        factor_exposures: object | None,
        objectives: list[str],
        names: list[str],
        n_labels: int,
        fit_kwargs: dict[str, Any],
        eval_set: object | None,
    ) -> None:
        """Train one shared tree ensemble for all K labels via the joint
        multi-output trainer.

        Routes through ``alloygbm._alloygbm.train_joint_multi_label_ranker``
        which calls ``engine::joint::fit_joint_multi_output``.  Per-label
        features that the joint trainer does NOT yet support are rejected
        here with a clear pointer back to ``multi_label_mode='independent'``.

        PR review (C2): rows are sorted by group ID before fitting,
        mirroring ``GBMRanker._sort_by_group``, because ranking
        objectives derive query boundaries from `compute_group_boundaries`
        which requires equal group IDs to be adjacent.

        PR review (C3, C7): every kwarg in ``_per_label_kwargs`` must
        be in ``_JOINT_SUPPORTED_KWARGS`` or this method raises.
        Defaults (when a kwarg is missing) match ``GBMRanker`` /
        ``GBMRegressor``'s public Python defaults rather than the
        engine's internal ``TrainParams::default()`` values, so
        ``MultiLabelGBMRanker(multi_label_mode='joint', n_estimators=20)``
        trains the same number of rounds as
        ``MultiLabelGBMRanker(multi_label_mode='independent', n_estimators=20)``.
        """
        if factor_exposures is not None:
            raise NotImplementedError(
                "multi_label_mode='joint' does not support factor_exposures "
                "in v0.10.1 (joint-path feature parity is tracked in "
                "docs/limitations.md). Use multi_label_mode='independent'."
            )
        if eval_set is not None:
            raise NotImplementedError(
                "multi_label_mode='joint' does not support eval_set / "
                "early_stopping_rounds in v0.10.1"
            )
        if fit_kwargs:
            raise NotImplementedError(
                f"multi_label_mode='joint' rejects fit kwargs "
                f"{sorted(fit_kwargs.keys())} in v0.10.1; pass them via "
                f"__init__ instead, or use multi_label_mode='independent'."
            )

        # Strict allow-list: every per-label kwarg must be in the set
        # that `train_joint_multi_label_ranker` actually forwards into
        # `TrainParams`.  Silently dropping a knob is a reproducibility
        # bug (e.g. setting `row_subsample=0.5` and then training on
        # the full dataset would be a debugging nightmare).
        unsupported = set(self._per_label_kwargs.keys()) - self._JOINT_SUPPORTED_KWARGS
        if unsupported:
            raise NotImplementedError(
                f"multi_label_mode='joint' rejects per-label kwargs "
                f"{sorted(unsupported)} in v0.10.1; either pass only "
                f"{sorted(self._JOINT_SUPPORTED_KWARGS)} or use "
                f"multi_label_mode='independent' (joint-path feature "
                f"parity is tracked in docs/limitations.md)."
            )

        from . import _alloygbm as _native

        x_arr = np.ascontiguousarray(np.asarray(X), dtype=np.float32)
        row_count = int(x_arr.shape[0])
        feature_count = int(x_arr.shape[1])

        # PR review (C2): joint mode must reorder rows so per-query
        # group IDs are contiguous before handing the data to the
        # engine's ranking objectives.  Done here (not in the wrapper
        # of the predictor) so prediction order is preserved.
        group_arr: list[int] | None = None
        if group is not None:
            per_row_ids = self._normalize_group_for_joint(group, row_count)
            ids_np = np.asarray(per_row_ids, dtype=np.uint32)
            sort_idx = np.argsort(ids_np, kind="stable")
            x_arr = x_arr[sort_idx]
            y_arr = y_arr[sort_idx]
            group_arr = ids_np[sort_idx].tolist()

        x_flat = x_arr.reshape(-1).tolist()

        targets_per_output: list[list[float]] = [
            np.ascontiguousarray(y_arr[:, k], dtype=np.float32).tolist()
            for k in range(n_labels)
        ]

        # JointObjective::parse in Rust accepts: "squared_error" | "queryrmse"
        # | "rank:pairwise" | "rank:ndcg" | "rank:xendcg". GBMRanker's
        # ranking_objective strings already use that style, so pass through.
        per_output_objective_names = list(objectives)

        kw = self._per_label_kwargs
        artifact, baselines, _fc, rounds_completed = _native.train_joint_multi_label_ranker(
            x_flat,
            row_count,
            feature_count,
            targets_per_output,
            n_labels,
            per_output_objective_names,
            group_arr,
            # Defaults below match GBMRegressor / GBMRanker's public
            # Python defaults (NOT the engine's TrainParams::default()).
            int(kw.get("n_estimators", 6)),
            float(kw.get("learning_rate", 0.1)),
            int(kw.get("seed", 0)),
            int(kw.get("max_depth", 6)),
            int(kw.get("min_data_in_leaf", 1)),
            float(kw.get("lambda_l2", 0.0)),
            int(kw.get("max_bin", 256)),
        )

        self._joint_artifact_bytes = bytes(artifact)
        self._joint_baselines = list(baselines)
        self._joint_feature_count = feature_count
        try:
            self._joint_handle = _native.JointPredictorHandle(
                self._joint_artifact_bytes,
                self._joint_baselines,
                feature_count,
            )
        except ValueError as e:
            # The joint trainer returns an artifact without
            # MultiOutputLeafValues when every round was rejected
            # (e.g. no valid split candidate under the current
            # min_data_in_leaf / max_depth / min_split_gain settings).
            # Surface a user-actionable message rather than the raw
            # PyO3 ValueError.
            if "MultiOutputLeafValues" in str(e):
                raise RuntimeError(
                    "multi_label_mode='joint' produced an empty ensemble: "
                    "no valid split candidate was found in any of "
                    f"n_estimators={kw.get('n_estimators', 6)} rounds. "
                    "Try lowering `min_data_in_leaf` (currently "
                    f"{int(kw.get('min_data_in_leaf', 1))}), increasing "
                    "the training row count, or using "
                    "multi_label_mode='independent'."
                ) from e
            raise
        self.ranking_labels_ = names
        self.n_labels_ = n_labels
        self.rounds_completed_ = int(rounds_completed)
        self._is_fitted = True

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
        if self.multi_label_mode == "joint":
            assert self._joint_handle is not None
            assert self.n_labels_ is not None
            x_arr = np.ascontiguousarray(np.asarray(X), dtype=np.float32)
            n_rows = int(x_arr.shape[0])
            flat = self._joint_handle.predict_dense(x_arr.reshape(-1).tolist())
            return np.asarray(flat, dtype=np.float64).reshape(n_rows, self.n_labels_)
        cols = [np.asarray(ranker.predict(X), dtype=np.float64) for ranker in self._sub_rankers]
        return np.stack(cols, axis=1)

    # ── Pickle ─────────────────────────────────────────────────────────

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # JointPredictorHandle is a PyO3 class with no Python pickle support.
        # Drop it; reconstruct from `_joint_artifact_bytes` + `_joint_baselines`
        # + `_joint_feature_count` on `__setstate__`.
        state["_joint_handle"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if (
            getattr(self, "multi_label_mode", "independent") == "joint"
            and self._joint_artifact_bytes is not None
        ):
            from . import _alloygbm as _native

            self._joint_handle = _native.JointPredictorHandle(
                self._joint_artifact_bytes,
                self._joint_baselines,
                self._joint_feature_count,
            )

    # ── Persistence ────────────────────────────────────────────────────

    def save_model(self, path: str) -> None:
        """Serialise a multi-label ranker into a self-describing bundle.

        Format v2 (v0.10.1+):
            magic[4]      = b"MLRK"
            version[u32]  = 2
            mode[u32]     = 0 (independent) | 1 (joint)
            n_labels[u32]
            n_labels × (name_len[u32], name[name_len bytes])
            mode == 0:
                n_labels × (blob_len[u64], pickle(sub_ranker))
            mode == 1:
                feature_count[u32]
                n_baselines[u32]
                n_baselines × baseline[f32]
                artifact_len[u64]
                artifact[artifact_len bytes]
        """
        if not self._is_fitted:
            raise RuntimeError("MultiLabelGBMRanker must be fit before save_model")
        import pickle

        mode_int = 1 if self.multi_label_mode == "joint" else 0
        n_labels = int(self.n_labels_ or 0)
        names = self.ranking_labels_ or [f"label_{i}" for i in range(n_labels)]

        with open(path, "wb") as f:
            f.write(_MULTI_LABEL_RANKER_MAGIC)
            f.write(
                struct.pack("<III", _MULTI_LABEL_RANKER_VERSION, mode_int, n_labels)
            )
            for name in names:
                encoded = name.encode("utf-8")
                f.write(struct.pack("<I", len(encoded)))
                f.write(encoded)
            if mode_int == 0:
                for ranker in self._sub_rankers:
                    blob = pickle.dumps(ranker, protocol=pickle.HIGHEST_PROTOCOL)
                    f.write(struct.pack("<Q", len(blob)))
                    f.write(blob)
            else:
                assert self._joint_artifact_bytes is not None
                assert self._joint_baselines is not None
                assert self._joint_feature_count is not None
                f.write(struct.pack("<I", int(self._joint_feature_count)))
                f.write(struct.pack("<I", len(self._joint_baselines)))
                for b in self._joint_baselines:
                    f.write(struct.pack("<f", float(b)))
                f.write(struct.pack("<Q", len(self._joint_artifact_bytes)))
                f.write(self._joint_artifact_bytes)

    @classmethod
    def load_model(cls, path: str) -> "MultiLabelGBMRanker":
        """Load a bundle written by :meth:`save_model`.

        Accepts both v1 (pre-v0.10.1, always independent mode) and v2
        (v0.10.1+, explicit mode byte) on-disk layouts.
        """
        import pickle

        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != _MULTI_LABEL_RANKER_MAGIC:
                raise ValueError(
                    "file is not a MultiLabelGBMRanker bundle "
                    f"(magic {magic!r} != {_MULTI_LABEL_RANKER_MAGIC!r})"
                )
            # v1 header: version[u32], n_labels[u32]
            # v2 header: version[u32], mode[u32], n_labels[u32]
            (version,) = struct.unpack("<I", f.read(4))
            if version == 1:
                mode_int = 0
                (n_labels,) = struct.unpack("<I", f.read(4))
            elif version == 2:
                mode_int, n_labels = struct.unpack("<II", f.read(8))
            else:
                raise ValueError(
                    f"unsupported MultiLabelGBMRanker bundle version {version}"
                )

            names: list[str] = []
            for _ in range(n_labels):
                (name_len,) = struct.unpack("<I", f.read(4))
                names.append(f.read(name_len).decode("utf-8"))

            if mode_int == 0:
                rankers: list[GBMRanker] = []
                for _ in range(n_labels):
                    (blob_len,) = struct.unpack("<Q", f.read(8))
                    blob = f.read(blob_len)
                    rankers.append(pickle.loads(blob))
                first_params = rankers[0].get_params()
                per_label_objectives = [r.ranking_objective for r in rankers]
                ranking_objective: str | list[str]
                if len(set(per_label_objectives)) == 1:
                    ranking_objective = per_label_objectives[0]
                else:
                    ranking_objective = list(per_label_objectives)
                first_params.pop("ranking_objective", None)
                inst = cls(
                    ranking_labels=names,
                    ranking_objective=ranking_objective,
                    multi_label_mode="independent",
                    **first_params,
                )
                inst._sub_rankers = rankers
                inst.ranking_labels_ = names
                inst.n_labels_ = n_labels
                inst._is_fitted = True
                inst.rounds_completed_ = [
                    int(r.rounds_completed_ or 0) for r in rankers
                ]
                return inst

            # Joint mode (mode_int == 1)
            (feature_count,) = struct.unpack("<I", f.read(4))
            (n_baselines,) = struct.unpack("<I", f.read(4))
            baselines = list(struct.unpack(f"<{n_baselines}f", f.read(4 * n_baselines)))
            (artifact_len,) = struct.unpack("<Q", f.read(8))
            artifact = f.read(artifact_len)

            from . import _alloygbm as _native

            inst = cls(
                ranking_labels=names,
                ranking_objective="rank:ndcg",  # placeholder; joint predict ignores it
                multi_label_mode="joint",
            )
            inst._joint_artifact_bytes = bytes(artifact)
            inst._joint_baselines = baselines
            inst._joint_feature_count = int(feature_count)
            inst._joint_handle = _native.JointPredictorHandle(
                inst._joint_artifact_bytes,
                inst._joint_baselines,
                inst._joint_feature_count,
            )
            inst.ranking_labels_ = names
            inst.n_labels_ = n_labels
            inst._is_fitted = True
            return inst

    # ── Introspection ──────────────────────────────────────────────────

    @property
    def sub_rankers_(self) -> list[GBMRanker]:
        """Return the underlying per-label :class:`GBMRanker` instances."""
        return list(self._sub_rankers)

    def __repr__(self) -> str:
        return (
            f"MultiLabelGBMRanker(n_labels={self.n_labels_}, "
            f"multi_label_mode={self.multi_label_mode!r}, "
            f"ranking_labels={self.ranking_labels_}, "
            f"ranking_objective={self.ranking_objective!r}, "
            f"fitted={self._is_fitted})"
        )
