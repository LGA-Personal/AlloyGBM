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
import json
import struct
from typing import Any

import numpy as np

from .ranker import GBMRanker
from ._regressor._quantization import _QuantizationMixin
from ._regressor._shap import _ShapMixin

_MULTI_LABEL_RANKER_MAGIC = b"MLRK"
# v2 (v0.10.1+) bundles include a `mode` byte after the version word so
# joint-mode and independent-mode bundles can coexist. v1 bundles always
# implied independent mode and load through a back-compat branch.
_MULTI_LABEL_RANKER_VERSION = 3


def _build_joint_morph_config(kw: dict[str, Any]) -> dict[str, Any] | None:
    """Build the ``morph_config`` dict for the joint PyO3 bridge.

    Returns ``None`` when ``training_mode != 'morph'`` so the bridge skips
    MorphBoost wiring entirely. When ``training_mode == 'morph'``, calls
    ``alloygbm._morph.build_morph_config_dict`` with the per-label kwargs
    using ``GBMRegressor`` / ``GBMRanker`` defaults for any missing fields.

    PR #37 review (C1, C4): validate ``training_mode`` strictly against the
    same set ``GBMRegressor`` / ``GBMRanker`` accept (``auto``, ``manual``,
    ``morph``) so typos like ``"morhp"`` fail fast instead of silently
    running standard training.
    """
    training_mode = str(kw.get("training_mode", "auto"))
    if training_mode not in ("auto", "manual", "morph"):
        raise ValueError(
            f"training_mode must be 'auto', 'manual', or 'morph', "
            f"got {training_mode!r}"
        )
    if training_mode != "morph":
        return None
    from ._morph import build_morph_config_dict

    morph_kwargs: dict[str, Any] = {}
    for k in (
        "morph_rate",
        "evolution_pressure",
        "morph_warmup_iters",
        "info_score_weight",
        "depth_penalty_base",
        "balance_penalty",
        "lr_schedule",
        "lr_warmup_frac",
    ):
        if k in kw:
            morph_kwargs[k] = kw[k]
    return build_morph_config_dict(**morph_kwargs)


class MultiLabelGBMRanker(_QuantizationMixin, _ShapMixin):
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
        self._uses_continuous_binning = False
        self._artifact_bytes: bytes | None = None
        self._continuous_feature_mins = None
        self._continuous_feature_maxs = None
        self._continuous_feature_linear_rank_flags = None
        self._continuous_feature_sorted_values = None
        self._continuous_feature_quantile_cuts = None
        self.ranking_labels_: list[str] | None = None
        self.n_labels_: int | None = None
        self.rounds_completed_: list[int] | int | None = None

    @property
    def continuous_binning_strategy(self) -> str:
        return self._per_label_kwargs.get("continuous_binning_strategy", "linear")

    @property
    def continuous_binning_max_bins(self) -> int:
        return self._per_label_kwargs.get("continuous_binning_max_bins", 256)

    @property
    def _n_features_in(self) -> int:
        return self._joint_feature_count or 0

    # ── Configuration ──────────────────────────────────────────────────

    def _resolve_objectives(self, n_labels: int) -> list[str]:
        if isinstance(self.ranking_objective, str):
            objs = [self.ranking_objective] * n_labels
        else:
            objs = [str(obj) for obj in self.ranking_objective]
            if len(objs) != n_labels:
                raise ValueError(
                    f"ranking_objective list length {len(objs)} does not match "
                    f"y's label count {n_labels}"
                )

        return objs

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
        # v0.10.2 Phase 1: small/medium joint-trainer features.
        "min_split_gain",
        "row_subsample",
        "col_subsample",
        "interaction_constraints",
        # v0.10.2 Phase 2: leaf-wise growth.
        "tree_growth",
        "max_leaves",
        # v0.10.3: native-categorical splits (Python wiring finally honest).
        # The Rust-level joint trainer
        # (`fit_joint_multi_output_with_categorical` +
        # `find_best_multi_output_categorical_split`) was already in
        # place in v0.10.2; v0.10.3 adds the re-binning step in the
        # PyO3 bridge so `bin_index == category_id` is preserved for
        # the requested columns before the trainer sees them.
        "categorical_feature_indices",
        "max_cat_threshold",
        # v0.10.3: joint boosting modes — GOSS, DART.
        "boosting_mode",
        "goss_top_rate",
        "goss_other_rate",
        "dart_drop_rate",
        "dart_max_drop",
        "dart_normalize_type",
        "dart_sample_type",
        # v0.10.4: MorphBoost on joint trainer. `training_mode="morph"`
        # activates the MorphBoost split-gain blend + LR schedule + leaf
        # shrinkage + depth penalty. Other values ("auto", "manual") are
        # treated as no-op for joint (joint trainer doesn't have an
        # auto-tuned policy in v0.10.x; "auto" means "use the user's
        # explicit params unmodified", matching "manual").
        "training_mode",
        "morph_rate",
        "evolution_pressure",
        "morph_warmup_iters",
        "info_score_weight",
        "depth_penalty_base",
        "balance_penalty",
        "lr_schedule",
        "lr_warmup_frac",
        # v0.10.5: joint DRO leaves (Wasserstein-radius leaf shrinkage).
        # Mirrors GBMRegressor / GBMRanker's leaf_solver kwargs.
        "leaf_solver",
        "dro_radius",
        "dro_metric",
        # v0.10.6: joint factor neutralization (all three modes).
        # The shared `factor_exposures` kwarg is consumed in `fit()`; only the
        # configuration kwargs flow through `_per_label_kwargs`.
        "neutralization",
        "factor_neutralization_lambda",
        "factor_penalty",
        "tweedie_variance_power",
        "quantile_alpha",
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
        # v0.10.6: joint factor neutralization is now supported. Cross-validate
        # the exposures-vs-config invariant up front; the PyO3 bridge enforces
        # the same contract as a backstop. Empty / non-active configs cannot
        # accept exposures (signals user confusion); active configs require
        # exposures (otherwise the projector / split-penalty path has no input).
        neutralization_kind = str(self._per_label_kwargs.get("neutralization", "none"))
        if neutralization_kind != "none" and factor_exposures is None:
            raise ValueError(
                "factor_exposures are required when neutralization is active "
                f"(neutralization={neutralization_kind!r})"
            )
        if factor_exposures is not None and neutralization_kind == "none":
            raise ValueError(
                "factor_exposures were provided but neutralization='none'"
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

        # v0.10.3: joint warm-start. `init_model` is a previously fit
        # MultiLabelGBMRanker (joint mode); we crack open its artifact
        # + baselines + rounds_completed and feed them to the engine
        # entry point. These are "managed" kwargs — not part of
        # `_JOINT_SUPPORTED_KWARGS` because they map to a separate
        # warm-start argument on the bridge.
        init_model = self._per_label_kwargs.get("init_model")
        warm_start_flag = bool(self._per_label_kwargs.get("warm_start", False))
        if init_model is not None and not warm_start_flag:
            raise ValueError("init_model requires warm_start=True")
        if warm_start_flag and init_model is None:
            raise ValueError(
                "warm_start=True requires init_model=<fitted MultiLabelGBMRanker>"
            )
        init_artifact: bytes | None = None
        init_baselines: list[float] | None = None
        init_rounds: int | None = None
        if init_model is not None:
            if not isinstance(init_model, MultiLabelGBMRanker):
                raise TypeError(
                    "joint warm-start expects init_model to be a "
                    f"MultiLabelGBMRanker (got {type(init_model).__name__})"
                )
            if init_model.multi_label_mode != "joint":
                raise ValueError(
                    "joint warm-start requires init_model.multi_label_mode == 'joint'"
                )
            if not init_model._is_fitted:
                raise ValueError("init_model must be fitted")
            if init_model._joint_artifact_bytes is None:
                raise ValueError("init_model is missing joint artifact bytes")
            init_artifact = init_model._joint_artifact_bytes
            init_baselines = list(init_model._joint_baselines or [])
            init_rounds = int(init_model.rounds_completed_ or 0)

        # Strict allow-list: every per-label kwarg must be in the set
        # that `train_joint_multi_label_ranker` actually forwards into
        # `TrainParams`.  Silently dropping a knob is a reproducibility
        # bug (e.g. setting `row_subsample=0.5` and then training on
        # the full dataset would be a debugging nightmare).
        #
        # `init_model` and `warm_start` are managed separately above
        # so they're allowed in `_per_label_kwargs` without appearing
        # in the bridge-forwarded set.
        _MANAGED_KWARGS = {"init_model", "warm_start"}
        unsupported = (
            set(self._per_label_kwargs.keys())
            - self._JOINT_SUPPORTED_KWARGS
            - _MANAGED_KWARGS
        )
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

        # PR #36 review (C4): validate init_model schema before sending
        # the artifact to Rust. Without these checks, a mismatched prior
        # fit can panic the Rust side (e.g. out-of-bounds feature index
        # in `walk_tree_into_predictions`) instead of raising a clean
        # Python ValueError. C2 added defense-in-depth on the Rust side;
        # this is the corresponding defense-in-depth on the Python side.
        if init_model is not None:
            prior_features = int(init_model._joint_feature_count or 0)
            if prior_features != feature_count:
                raise ValueError(
                    f"joint warm-start: init_model was trained on "
                    f"{prior_features} features, but X has {feature_count}. "
                    f"Schemas must match exactly."
                )
            prior_n_labels = int(init_model.n_labels_ or 0)
            if prior_n_labels != n_labels:
                raise ValueError(
                    f"joint warm-start: init_model has {prior_n_labels} "
                    f"labels, but y has {n_labels}. Schemas must match."
                )
            # DART <-> non-DART mode mismatch: a DART prior resumed as
            # standard is replayed at weight 1.0 (the per-tree weight
            # is discarded for non-DART training), which silently
            # changes the residual stream the new rounds see. A
            # non-DART prior resumed as DART is also surprising — the
            # new fit starts with `dart_state.tree_weights` reconstructed
            # from `tree_weight=1.0` for every prior tree, which is
            # numerically correct but means dropouts in early new rounds
            # only ever drop prior trees with equal weight (no real
            # dropout asymmetry until new DART rounds run). Both are
            # confusing footguns; reject explicitly.
            prior_bm = init_model._per_label_kwargs.get("boosting_mode", "standard")
            curr_bm = self._per_label_kwargs.get("boosting_mode", "standard")
            prior_is_dart = prior_bm == "dart"
            curr_is_dart = curr_bm == "dart"
            if prior_is_dart != curr_is_dart:
                raise ValueError(
                    f"joint warm-start: boosting_mode mismatch — init_model used "
                    f"{prior_bm!r}, current fit uses {curr_bm!r}. DART <-> non-DART "
                    f"transitions across warm-resume are rejected because the per-tree "
                    f"`tree_weight` semantics differ. Use the same `boosting_mode` "
                    f"for both fits, or fit fresh without `warm_start=True`."
                )

        # PR review (C2): joint mode must reorder rows so per-query
        # group IDs are contiguous before handing the data to the
        # engine's ranking objectives.  Done here (not in the wrapper
        # of the predictor) so prediction order is preserved.
        group_arr: list[int] | None = None
        sort_idx: np.ndarray | None = None
        if group is not None:
            per_row_ids = self._normalize_group_for_joint(group, row_count)
            ids_np = np.asarray(per_row_ids, dtype=np.uint32)
            sort_idx = np.argsort(ids_np, kind="stable")
            x_arr = x_arr[sort_idx]
            y_arr = y_arr[sort_idx]
            group_arr = ids_np[sort_idx].tolist()

        # v0.10.6: factor_exposures must follow the same row order as X / y.
        # Validated against `row_count` here (post-coercion); the PyO3 bridge
        # re-checks the (values_len, row_count, factor_count) triple against
        # the binned matrix.
        fe_values: list[float] | None = None
        fe_row_count: int | None = None
        fe_factor_count: int | None = None
        if factor_exposures is not None:
            fe_arr = np.ascontiguousarray(factor_exposures, dtype=np.float32)
            if fe_arr.ndim != 2:
                raise ValueError("factor_exposures must be a 2D array")
            if fe_arr.shape[0] != row_count:
                raise ValueError(
                    f"factor_exposures row count {fe_arr.shape[0]} does not "
                    f"match X row count {row_count}"
                )
            if sort_idx is not None:
                fe_arr = fe_arr[sort_idx]
            fe_values = fe_arr.reshape(-1).tolist()
            fe_row_count = int(fe_arr.shape[0])
            fe_factor_count = int(fe_arr.shape[1])

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
        # interaction_constraints: accept list[list[int]] | list[list[np.int*]] |
        # np.array-of-objects; coerce to a strict list[list[int]] for the
        # PyO3 bridge (it wants Vec<Vec<u32>>).
        ic_raw = kw.get("interaction_constraints", [])
        if ic_raw is None:
            ic_list: list[list[int]] = []
        else:
            ic_list = [[int(x) for x in group] for group in ic_raw]
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
            # v0.10.2 Phase 1 kwargs (keyword args so the order in the PyO3
            # signature can keep evolving across point releases).
            min_split_gain=float(kw.get("min_split_gain", 0.0)),
            row_subsample=float(kw.get("row_subsample", 1.0)),
            col_subsample=float(kw.get("col_subsample", 1.0)),
            interaction_constraints=ic_list,
            # v0.10.2 Phase 2 kwargs.
            tree_growth=str(kw.get("tree_growth", "level")),
            max_leaves=(
                int(kw["max_leaves"]) if kw.get("max_leaves") is not None else None
            ),
            # v0.10.3: native-categorical kwargs are now passed through.
            # The PyO3 bridge re-bins requested columns to
            # `bin_index == category_id` before calling the Rust trainer,
            # which is the invariant the joint native-cat path requires.
            categorical_feature_indices=[
                int(fi) for fi in (kw.get("categorical_feature_indices") or [])
            ],
            max_cat_threshold=int(kw.get("max_cat_threshold", 0)),
            # v0.10.3: joint boosting_mode (GOSS / DART).
            boosting_mode=str(kw.get("boosting_mode", "standard")),
            goss_top_rate=(
                float(kw["goss_top_rate"]) if "goss_top_rate" in kw else None
            ),
            goss_other_rate=(
                float(kw["goss_other_rate"]) if "goss_other_rate" in kw else None
            ),
            dart_drop_rate=(
                float(kw["dart_drop_rate"]) if "dart_drop_rate" in kw else None
            ),
            dart_max_drop=(
                int(kw["dart_max_drop"]) if "dart_max_drop" in kw else None
            ),
            dart_normalize_type=(
                str(kw["dart_normalize_type"]) if "dart_normalize_type" in kw else None
            ),
            dart_sample_type=(
                str(kw["dart_sample_type"]) if "dart_sample_type" in kw else None
            ),
            # v0.10.3: joint warm-start.
            init_artifact_bytes=init_artifact,
            init_baselines=init_baselines,
            init_rounds_completed=init_rounds,
            # v0.10.4: MorphBoost on joint trainer. `morph_config` dict is
            # built from per-label kwargs when training_mode='morph';
            # `None` means non-morph training (the bridge ignores all
            # other morph_* kwargs in that case).
            morph_config=_build_joint_morph_config(kw),
            # v0.10.5: joint DRO leaves. Defaults match the PyO3 bridge
            # defaults (which themselves match single-output train_python).
            leaf_solver=str(kw.get("leaf_solver", "standard")),
            dro_radius=float(kw.get("dro_radius", 0.05)),
            dro_metric=str(kw.get("dro_metric", "wasserstein")),
            # v0.10.6: joint factor neutralization.
            factor_exposure_values=fe_values,
            factor_exposure_row_count=fe_row_count,
            factor_exposure_factor_count=fe_factor_count,
            neutralization=str(kw.get("neutralization", "none")),
            factor_neutralization_lambda=float(
                kw.get("factor_neutralization_lambda", 1e-6)
            ),
            factor_penalty=float(kw.get("factor_penalty", 0.0)),
            tweedie_variance_power=(
                float(kw["tweedie_variance_power"]) if "tweedie_variance_power" in kw else None
            ),
            quantile_alpha=(
                float(kw["quantile_alpha"]) if "quantile_alpha" in kw else None
            ),
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
        self._artifact_bytes = self._joint_artifact_bytes
        self._uses_continuous_binning = not self._rows_are_pre_binned(x_arr)
        if self._uses_continuous_binning:
            strategy = self.continuous_binning_strategy
            if strategy == "linear":
                self._continuous_feature_mins = np.nanmin(x_arr, axis=0)
                self._continuous_feature_maxs = np.nanmax(x_arr, axis=0)
                self._continuous_feature_linear_rank_flags = None
                self._continuous_feature_sorted_values = None
            elif strategy == "quantile":
                self._continuous_feature_quantile_cuts = self._derive_continuous_feature_quantile_cuts(
                    x_arr, self.continuous_binning_max_bins
                )

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
            if self._joint_handle is None:
                raise RuntimeError("Joint predictor handle is not initialized; model must be fitted first.")
            if self.n_labels_ is None:
                raise RuntimeError("n_labels_ is not initialized; model must be fitted first.")
            x_arr = np.ascontiguousarray(np.asarray(X), dtype=np.float32)
            n_rows = int(x_arr.shape[0])
            if self._uses_continuous_binning:
                rows = self._quantize_rows_for_prediction(x_arr.tolist())
                cat_indices = self._per_label_kwargs.get("categorical_feature_indices")
                if cat_indices:
                    cat_indices = [int(i) for i in cat_indices]
                    raw_rows = x_arr.tolist()
                    for r_idx in range(len(rows)):
                        for c_idx in cat_indices:
                            if c_idx < len(rows[r_idx]):
                                rows[r_idx][c_idx] = raw_rows[r_idx][c_idx]
                flat = [v for row in rows for v in row]
            else:
                flat = x_arr.reshape(-1).tolist()
            raw = self._joint_handle.predict_dense(flat)
            preds = np.asarray(raw, dtype=np.float64).reshape(n_rows, self.n_labels_)
            objs = self._resolve_objectives(self.n_labels_)
            for col_idx, obj in enumerate(objs):
                if obj in ("poisson", "gamma", "tweedie"):
                    preds[:, col_idx] = np.exp(np.clip(preds[:, col_idx], -50.0, 50.0))
                elif obj == "binary_crossentropy":
                    x = preds[:, col_idx]
                    preds[:, col_idx] = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
            return preds
        cols = [np.asarray(ranker.predict(X), dtype=np.float64) for ranker in self._sub_rankers]
        return np.stack(cols, axis=1)

    # ── SHAP ───────────────────────────────────────────────────────────

    def shap_values(
        self, X: object, *, include_expected_value: bool = False
    ) -> list[np.ndarray] | tuple[list[float], list[np.ndarray]]:
        """Compute SHAP values for each label.

        Returns
        -------
        list[np.ndarray] or tuple[list[float], list[np.ndarray]]
            If include_expected_value is False, returns a list of ``n_labels`` matrices,
            where each matrix has shape ``(n_samples, n_features)``.
            If include_expected_value is True, returns a tuple of (expected_values, shap_values),
            where expected_values is a list of length ``n_labels``.
        """
        if not self._is_fitted:
            raise RuntimeError("MultiLabelGBMRanker must be fit before computing SHAP values")
        
        if self.multi_label_mode == "joint":
            if self._joint_artifact_bytes is None:
                raise RuntimeError("Joint model is fit but artifact bytes are missing.")
            if include_expected_value:
                expected_vals, vals = _ShapMixin.shap_values(self, X, include_expected_value=True)
                return expected_vals, [np.array(v, dtype=np.float64) for v in vals]
            else:
                vals = _ShapMixin.shap_values(self, X, include_expected_value=False)
                return [np.array(v, dtype=np.float64) for v in vals]
            
        results = [
            ranker.shap_values(X, include_expected_value=include_expected_value)
            for ranker in self._sub_rankers
        ]
        if include_expected_value:
            expected_vals = [res[0] for res in results]
            shap_vals = [np.array(res[1], dtype=np.float64) for res in results]
            return expected_vals, shap_vals
        else:
            return [np.array(res, dtype=np.float64) for res in results]

    def shap_interaction_values(
        self, X: object, *, include_expected_value: bool = False
    ) -> list[np.ndarray] | tuple[list[float], list[np.ndarray]]:
        """Compute SHAP interaction values for each label.

        Returns
        -------
        list[np.ndarray] or tuple[list[float], list[np.ndarray]]
            A list of ``n_labels`` 3D tensors, where each tensor has shape
            ``(n_samples, n_features, n_features)``.
        """
        if not self._is_fitted:
            raise RuntimeError("MultiLabelGBMRanker must be fit before computing SHAP interaction values")
            
        if self.multi_label_mode == "joint":
            if self._joint_artifact_bytes is None:
                raise RuntimeError("Joint model is fit but artifact bytes are missing.")
            if include_expected_value:
                expected_vals, vals = _ShapMixin.shap_interaction_values(self, X, include_expected_value=True)
                return expected_vals, [np.array(v, dtype=np.float64) for v in vals]
            else:
                vals = _ShapMixin.shap_interaction_values(self, X, include_expected_value=False)
                return [np.array(v, dtype=np.float64) for v in vals]
            
        results = [
            ranker.shap_interaction_values(X, include_expected_value=include_expected_value)
            for ranker in self._sub_rankers
        ]
        if include_expected_value:
            expected_vals = [res[0] for res in results]
            shap_vals = [np.array(res[1], dtype=np.float64) for res in results]
            return expected_vals, shap_vals
        else:
            return [np.array(res, dtype=np.float64) for res in results]

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
                if self._joint_artifact_bytes is None:
                    raise ValueError("Cannot save joint model: artifact bytes are missing.")
                if self._joint_baselines is None:
                    raise ValueError("Cannot save joint model: baselines are missing.")
                if self._joint_feature_count is None:
                    raise ValueError("Cannot save joint model: feature count is missing.")
                f.write(struct.pack("<I", int(self._joint_feature_count)))
                f.write(struct.pack("<I", len(self._joint_baselines)))
                for b in self._joint_baselines:
                    f.write(struct.pack("<f", float(b)))
                f.write(struct.pack("<Q", len(self._joint_artifact_bytes)))
                f.write(self._joint_artifact_bytes)
                
                # Append continuous binning metadata as JSON block for joint mode in v3

                def to_list(val):
                    if isinstance(val, np.ndarray):
                        return val.tolist()
                    if isinstance(val, (list, tuple)):
                        return [to_list(v) for v in val]
                    if isinstance(val, dict):
                        return {k: to_list(v) for k, v in val.items()}
                    if hasattr(val, "item") and callable(val.item):
                        return val.item()
                    if callable(val):
                        return None
                    return val

                metadata = {
                    "uses_continuous_binning": getattr(self, "_uses_continuous_binning", False),
                    "continuous_binning_strategy": getattr(self, "continuous_binning_strategy", "linear"),
                    "continuous_binning_max_bins": getattr(self, "continuous_binning_max_bins", 256),
                    "continuous_feature_mins": getattr(self, "_continuous_feature_mins", None),
                    "continuous_feature_maxs": getattr(self, "_continuous_feature_maxs", None),
                    "continuous_feature_sorted_values": getattr(self, "_continuous_feature_sorted_values", None),
                    "continuous_feature_quantile_cuts": getattr(self, "_continuous_feature_quantile_cuts", None),
                    "continuous_feature_linear_rank_flags": getattr(self, "_continuous_feature_linear_rank_flags", None),
                    "per_label_kwargs": self._per_label_kwargs,
                }
                metadata_json = json.dumps(to_list(metadata)).encode("utf-8")
                f.write(struct.pack("<Q", len(metadata_json)))
                f.write(metadata_json)

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
            elif version == 2 or version == 3:
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

            if version == 3:
                (metadata_len,) = struct.unpack("<Q", f.read(8))
                metadata_json = f.read(metadata_len)
                metadata = json.loads(metadata_json.decode("utf-8"))
                inst._uses_continuous_binning = metadata.get("uses_continuous_binning", False)
                inst._continuous_feature_mins = metadata.get("continuous_feature_mins")
                inst._continuous_feature_maxs = metadata.get("continuous_feature_maxs")
                inst._continuous_feature_sorted_values = metadata.get("continuous_feature_sorted_values")
                inst._continuous_feature_quantile_cuts = metadata.get("continuous_feature_quantile_cuts")
                inst._continuous_feature_linear_rank_flags = metadata.get("continuous_feature_linear_rank_flags")
                inst._per_label_kwargs = metadata.get("per_label_kwargs", {})
            else:
                inst._uses_continuous_binning = False
                inst._per_label_kwargs = {}

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
