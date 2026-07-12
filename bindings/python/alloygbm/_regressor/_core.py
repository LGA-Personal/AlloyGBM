"""Python-facing estimator baseline scaffold for AlloyGBM."""

from __future__ import annotations

import math
import time

import numpy as np

from ._validation import _ValidationMixin
from ._quantization import _QuantizationMixin
from ._shap import _ShapMixin
from ._persistence import _PersistenceMixin
from . import _base
from ._base import (
    _GBMRegressorBase,
    _MAX_CONTINUOUS_QUANTIZED_BIN,
    _max_data_bin_for_max_bins,
    _MIN_CONTINUOUS_QUANTIZED_BINS,
    _VALID_CONTINUOUS_BINNING_STRATEGIES,
    _VALID_BOOSTING_MODES,
    _linear_tail_rank_enabled_from_env,
    _linear_tail_core_span_ratio_threshold_from_env,
    _diagnostics_to_dicts,
    _validate_quantile_alpha,
)


class GBMRegressor(_ValidationMixin, _QuantizationMixin, _ShapMixin, _PersistenceMixin, _GBMRegressorBase):
    """Gradient Boosted Decision Tree regressor with sklearn-compatible API."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        n_estimators: int = 6,
        row_subsample: float = 1.0,
        col_subsample: float = 1.0,
        early_stopping_rounds: int | None = None,
        min_validation_improvement: float = 0.0,
        min_data_in_leaf: int = 1,
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0,
        min_child_hessian: float = 0.0,
        min_split_gain: float = 0.0,
        seed: int = 0,
        deterministic: bool = True,
        continuous_binning_strategy: str = "quantile",
        continuous_binning_max_bins: int = 256,
        categorical_feature_index: int | None = None,
        categorical_feature_indices: list[int] | None = None,
        training_policy: str = "auto",
        store_node_stats: bool = False,
        categorical_smoothing: float = 20.0,
        categorical_min_samples_leaf: int = 1,
        categorical_time_aware: bool = False,
        monotone_constraints: list[int] | dict[int, int] | None = None,
        feature_weights: list[float] | dict[int, float] | None = None,
        interaction_constraints: list[list[int]] | None = None,
        max_leaves: int | None = None,
        tree_growth: str = "level",
        warm_start: bool = False,
        objective: "str | None | object" = None,
        max_cat_threshold: int = 0,
        training_mode: str = "auto",
        morph_rate: float = 0.1,
        evolution_pressure: float = 0.2,
        morph_warmup_iters: int = 5,
        info_score_weight: float = 0.3,
        depth_penalty_base: float = 0.9,
        balance_penalty: bool = True,
        lr_schedule: str = "constant",
        lr_warmup_frac: float = 0.1,
        leaf_model: str = "constant",
        leaf_solver: str = "standard",
        dro_radius: float = 0.05,
        dro_metric: str = "wasserstein",
        neutralization: str = "none",
        factor_neutralization_lambda: float = 1e-6,
        factor_penalty: float = 0.0,
        factor_exposure_transform: str = "none",
        boosting_mode: str = "standard",
        goss_top_rate: float = 0.2,
        goss_other_rate: float = 0.1,
        dart_drop_rate: float = 0.1,
        dart_max_drop: int = 50,
        dart_normalize_type: str = "tree",
        dart_sample_type: str = "uniform",
        tweedie_variance_power: float = 1.5,
        poisson_max_delta_step: float = 0.7,
        quantile_alpha: float = 0.5,
    ) -> None:
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError("learning_rate must be in (0.0, 1.0]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than 0")
        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")
        if not (0.0 < row_subsample <= 1.0):
            raise ValueError("row_subsample must be in (0.0, 1.0]")
        if not (0.0 < col_subsample <= 1.0):
            raise ValueError("col_subsample must be in (0.0, 1.0]")
        if early_stopping_rounds is not None and int(early_stopping_rounds) <= 0:
            raise ValueError("early_stopping_rounds must be greater than 0 when set")
        if min_validation_improvement < 0.0:
            raise ValueError("min_validation_improvement must be >= 0")
        if int(min_data_in_leaf) <= 0:
            raise ValueError("min_data_in_leaf must be greater than 0")
        if lambda_l1 < 0.0:
            raise ValueError("lambda_l1 must be >= 0")
        if lambda_l2 < 0.0:
            raise ValueError("lambda_l2 must be >= 0")
        if min_child_hessian < 0.0:
            raise ValueError("min_child_hessian must be >= 0")
        if not math.isfinite(min_split_gain) or min_split_gain < 0.0:
            raise ValueError("min_split_gain must be a finite value >= 0")
        if categorical_feature_index is not None and int(categorical_feature_index) < 0:
            raise ValueError("categorical_feature_index must be >= 0 when set")
        if categorical_feature_indices is not None:
            if not isinstance(categorical_feature_indices, (list, tuple)):
                raise TypeError("categorical_feature_indices must be a list of ints")
            for idx in categorical_feature_indices:
                if int(idx) < 0:
                    raise ValueError(
                        "all values in categorical_feature_indices must be >= 0"
                    )
            if len(set(int(i) for i in categorical_feature_indices)) != len(
                categorical_feature_indices
            ):
                raise ValueError(
                    "categorical_feature_indices must not contain duplicates"
                )
        if categorical_feature_index is not None and categorical_feature_indices is not None:
            raise ValueError(
                "categorical_feature_index and categorical_feature_indices are mutually exclusive; use categorical_feature_indices for multiple columns"
            )
        if categorical_smoothing < 0.0:
            raise ValueError("categorical_smoothing must be >= 0")
        if int(categorical_min_samples_leaf) <= 0:
            raise ValueError("categorical_min_samples_leaf must be greater than 0")
        if training_policy not in {"auto", "manual"}:
            raise ValueError("training_policy must be 'auto' or 'manual'")
        if continuous_binning_strategy not in _VALID_CONTINUOUS_BINNING_STRATEGIES:
            raise ValueError(
                "continuous_binning_strategy must be one of: "
                + ", ".join(sorted(_VALID_CONTINUOUS_BINNING_STRATEGIES))
            )
        max_bins = int(continuous_binning_max_bins)
        if not (
            _MIN_CONTINUOUS_QUANTIZED_BINS
            <= max_bins
            <= (_MAX_CONTINUOUS_QUANTIZED_BIN + 1)
        ):
            raise ValueError(
                "continuous_binning_max_bins must be in "
                f"[{_MIN_CONTINUOUS_QUANTIZED_BINS}, {_MAX_CONTINUOUS_QUANTIZED_BIN + 1}]"
            )
        if monotone_constraints is not None:
            if isinstance(monotone_constraints, dict):
                for v in monotone_constraints.values():
                    if int(v) not in (-1, 0, 1):
                        raise ValueError(
                            "monotone_constraints values must be -1, 0, or +1"
                        )
            else:
                for v in monotone_constraints:
                    if int(v) not in (-1, 0, 1):
                        raise ValueError(
                            "monotone_constraints values must be -1, 0, or +1"
                        )
        if feature_weights is not None:
            vals = (
                feature_weights.values()
                if isinstance(feature_weights, dict)
                else feature_weights
            )
            for w in vals:
                if not math.isfinite(float(w)) or float(w) < 0.0:
                    raise ValueError(
                        "feature_weights values must be finite and >= 0"
                    )
        if interaction_constraints is not None:
            if not isinstance(interaction_constraints, (list, tuple)):
                raise TypeError(
                    "interaction_constraints must be a sequence of feature-index groups"
                )
            if len(interaction_constraints) > 64:
                raise ValueError(
                    "interaction_constraints supports at most 64 groups "
                    f"(got {len(interaction_constraints)})"
                )
            for gi, group in enumerate(interaction_constraints):
                if not isinstance(group, (list, tuple)) or len(group) == 0:
                    raise ValueError(
                        f"interaction_constraints group {gi} must be a non-empty "
                        "sequence of feature indices"
                    )
                seen: set[int] = set()
                for f in group:
                    fi = int(f)
                    if fi < 0:
                        raise ValueError(
                            f"interaction_constraints group {gi} contains negative "
                            f"feature index {fi}"
                        )
                    if fi in seen:
                        raise ValueError(
                            f"interaction_constraints group {gi} contains duplicate "
                            f"feature index {fi}"
                        )
                    seen.add(fi)
        if max_leaves is not None:
            if int(max_leaves) < 2:
                raise ValueError(
                    "max_leaves must be >= 2 when set"
                )
        _VALID_TREE_GROWTH = {"level", "leaf"}
        if tree_growth not in _VALID_TREE_GROWTH:
            raise ValueError(
                "tree_growth must be one of: " + ", ".join(sorted(_VALID_TREE_GROWTH))
            )
        if tree_growth == "leaf" and max_leaves is None:
            raise ValueError(
                "max_leaves must be set when tree_growth='leaf'"
            )
        if objective is not None and not callable(objective) and not isinstance(objective, str):
            raise TypeError(
                "objective must be a string, a callable, or None"
            )
        if isinstance(objective, str) and objective == "tweedie":
            if not isinstance(tweedie_variance_power, (int, float)) or not (
                1.0 < float(tweedie_variance_power) < 2.0
            ):
                raise ValueError(
                    "tweedie_variance_power must satisfy 1 < p < 2 when objective='tweedie' "
                    f"(got {tweedie_variance_power!r})"
                )
        if (
            not math.isfinite(float(poisson_max_delta_step))
            or float(poisson_max_delta_step) < 0.0
        ):
            raise ValueError("poisson_max_delta_step must be finite and >= 0")
        _validate_quantile_alpha(quantile_alpha)
        if int(max_cat_threshold) < 0:
            raise ValueError("max_cat_threshold must be >= 0")
        if training_mode not in ("auto", "manual", "morph"):
            raise ValueError(
                f"training_mode must be 'auto', 'manual', or 'morph', got {training_mode!r}"
            )
        if not (0.0 <= float(morph_rate) <= 1.0):
            raise ValueError("morph_rate must be in [0.0, 1.0]")
        if not (0.0 <= float(evolution_pressure) <= 1.0):
            raise ValueError("evolution_pressure must be in [0.0, 1.0]")
        if int(morph_warmup_iters) < 0:
            raise ValueError("morph_warmup_iters must be >= 0")
        if lr_schedule not in ("constant", "warmup_cosine"):
            raise ValueError(
                f"lr_schedule must be 'constant' or 'warmup_cosine', got {lr_schedule!r}"
            )
        if not (0.0 <= float(lr_warmup_frac) <= 1.0):
            raise ValueError("lr_warmup_frac must be in [0.0, 1.0]")
        # lr_warmup_frac is only meaningful when lr_schedule == "warmup_cosine".
        # We treat 0.1 as the inert default and reject any other value with a
        # non-warmup-cosine schedule (matches the Rust-side validation contract
        # and prevents silent drops in the bridge).
        if lr_schedule != "warmup_cosine" and float(lr_warmup_frac) != 0.1:
            raise ValueError(
                f"lr_warmup_frac={lr_warmup_frac} is only valid with "
                f"lr_schedule='warmup_cosine'; got lr_schedule='{lr_schedule}'"
            )
        if not (0.0 <= float(info_score_weight) <= 1.0):
            raise ValueError("info_score_weight must be in [0.0, 1.0]")
        # depth_penalty_base must be strictly > 0 (matches the Rust core
        # validation: morph_config.depth_penalty_base must be in (0, 1]).
        if not (0.0 < float(depth_penalty_base) <= 1.0):
            raise ValueError("depth_penalty_base must be in (0.0, 1.0]")
        if str(leaf_model) not in ("constant", "linear"):
            raise ValueError(
                f"leaf_model must be 'constant' or 'linear', got {leaf_model!r}"
            )
        if str(leaf_solver) not in ("standard", "dro"):
            raise ValueError(
                f"leaf_solver must be 'standard' or 'dro', got {leaf_solver!r}"
            )
        if not math.isfinite(float(dro_radius)) or float(dro_radius) < 0.0:
            raise ValueError("dro_radius must be finite and >= 0")
        if str(dro_metric) != "wasserstein":
            raise ValueError(
                f"dro_metric must be 'wasserstein', got {dro_metric!r}"
            )
        if str(leaf_solver) == "dro" and str(leaf_model) != "constant":
            raise ValueError(
                "leaf_solver='dro' requires leaf_model='constant'"
            )
        if str(neutralization) not in (
            "none",
            "pre_target",
            "per_round_gradient",
            "split_penalty",
        ):
            raise ValueError(
                "neutralization must be 'none', 'pre_target', "
                "'per_round_gradient', or 'split_penalty'"
            )
        if (
            not math.isfinite(float(factor_neutralization_lambda))
            or float(factor_neutralization_lambda) < 0.0
        ):
            raise ValueError("factor_neutralization_lambda must be finite and >= 0")
        if not math.isfinite(float(factor_penalty)) or float(factor_penalty) < 0.0:
            raise ValueError("factor_penalty must be finite and >= 0")
        if str(factor_exposure_transform) not in ("none", "center", "standardize"):
            raise ValueError(
                "factor_exposure_transform must be 'none', 'center', or 'standardize'"
            )
        if str(neutralization) != "split_penalty" and float(factor_penalty) != 0.0:
            raise ValueError(
                "factor_penalty is only valid with neutralization='split_penalty'"
            )
        if str(neutralization) == "split_penalty" and str(leaf_model) == "linear":
            raise ValueError(
                "neutralization='split_penalty' requires leaf_model='constant'"
            )
        if str(boosting_mode) not in _VALID_BOOSTING_MODES:
            raise ValueError(
                "boosting_mode must be one of: "
                + ", ".join(sorted(_VALID_BOOSTING_MODES))
                + f", got {boosting_mode!r}"
            )
        if str(boosting_mode) == "goss":
            if not (0.0 < float(goss_top_rate) < 1.0):
                raise ValueError(
                    "boosting_mode='goss' requires goss_top_rate in (0, 1)"
                )
            if not (0.0 < float(goss_other_rate) < 1.0):
                raise ValueError(
                    "boosting_mode='goss' requires goss_other_rate in (0, 1)"
                )
            if float(goss_top_rate) + float(goss_other_rate) > 1.0:
                raise ValueError(
                    "boosting_mode='goss' requires goss_top_rate + goss_other_rate <= 1.0"
                )
        if str(boosting_mode) == "dart":
            # v0.9.0: DART is fully wired through the single-output trainer
            # (regression / binary classification / single-label ranking).
            # Multiclass DART is rejected at fit time; warm-start + DART
            # is rejected at fit time.
            if not (0.0 < float(dart_drop_rate) < 1.0):
                raise ValueError(
                    "boosting_mode='dart' requires dart_drop_rate in (0, 1), "
                    f"got {dart_drop_rate}"
                )
            if int(dart_max_drop) < 1:
                raise ValueError(
                    "boosting_mode='dart' requires dart_max_drop >= 1, "
                    f"got {dart_max_drop}"
                )
            if str(dart_normalize_type) not in {"tree", "forest"}:
                raise ValueError(
                    "boosting_mode='dart' requires "
                    "dart_normalize_type in {'tree', 'forest'}, "
                    f"got {dart_normalize_type!r}"
                )
            if str(dart_sample_type) not in {"uniform", "weighted"}:
                raise ValueError(
                    "boosting_mode='dart' requires "
                    "dart_sample_type in {'uniform', 'weighted'}, "
                    f"got {dart_sample_type!r}"
                )

        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.n_estimators = int(n_estimators)
        self.row_subsample = float(row_subsample)
        self.col_subsample = float(col_subsample)
        self.early_stopping_rounds = (
            int(early_stopping_rounds)
            if early_stopping_rounds is not None
            else None
        )
        self.min_validation_improvement = float(min_validation_improvement)
        self.min_data_in_leaf = int(min_data_in_leaf)
        self.lambda_l1 = float(lambda_l1)
        self.lambda_l2 = float(lambda_l2)
        self.min_child_hessian = float(min_child_hessian)
        self.min_split_gain = float(min_split_gain)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.continuous_binning_strategy = str(continuous_binning_strategy)
        self.continuous_binning_max_bins = max_bins
        self.categorical_feature_index = (
            int(categorical_feature_index)
            if categorical_feature_index is not None
            else None
        )
        self.categorical_feature_indices = (
            [int(i) for i in categorical_feature_indices]
            if categorical_feature_indices is not None
            else None
        )
        self.training_policy = str(training_policy)
        self.store_node_stats = bool(store_node_stats)
        self.categorical_smoothing = float(categorical_smoothing)
        self.categorical_min_samples_leaf = int(categorical_min_samples_leaf)
        self.categorical_time_aware = bool(categorical_time_aware)
        self.monotone_constraints = (
            monotone_constraints if monotone_constraints is not None else None
        )
        self.feature_weights = (
            feature_weights if feature_weights is not None else None
        )
        self.interaction_constraints = (
            [list(map(int, g)) for g in interaction_constraints]
            if interaction_constraints is not None
            else None
        )
        self.max_leaves = int(max_leaves) if max_leaves is not None else None
        self.tree_growth = str(tree_growth)
        self.warm_start = bool(warm_start)
        self.objective = objective
        self.max_cat_threshold = int(max_cat_threshold)
        self.training_mode = str(training_mode)
        self.morph_rate = float(morph_rate)
        self.evolution_pressure = float(evolution_pressure)
        self.morph_warmup_iters = int(morph_warmup_iters)
        self.info_score_weight = float(info_score_weight)
        self.depth_penalty_base = float(depth_penalty_base)
        self.balance_penalty = bool(balance_penalty)
        self.lr_schedule = str(lr_schedule)
        self.lr_warmup_frac = float(lr_warmup_frac)
        self.leaf_model = str(leaf_model)
        self.leaf_solver = str(leaf_solver)
        self.dro_radius = float(dro_radius)
        self.dro_metric = str(dro_metric)
        self.neutralization = str(neutralization)
        self.factor_neutralization_lambda = float(factor_neutralization_lambda)
        self.factor_penalty = float(factor_penalty)
        self.factor_exposure_transform = str(factor_exposure_transform)
        # v0.8.0+: per-round boosting strategy.  Default "standard" is
        # byte-identical to v0.7.5 behaviour.  "goss" enables
        # gradient-based one-side sampling.  v0.9.0+: "dart" enables
        # Dropouts-meet-MART for single-output objectives.
        self.boosting_mode = str(boosting_mode)
        self.goss_top_rate = float(goss_top_rate)
        self.goss_other_rate = float(goss_other_rate)
        self.dart_drop_rate = float(dart_drop_rate)
        self.dart_max_drop = int(dart_max_drop)
        self.dart_normalize_type = str(dart_normalize_type)
        self.dart_sample_type = str(dart_sample_type)
        self.tweedie_variance_power = float(tweedie_variance_power)
        self.poisson_max_delta_step = float(poisson_max_delta_step)
        self.quantile_alpha = float(quantile_alpha)
        self._fit_neutralization: str | None = None
        self._fit_factor_neutralization_lambda: float | None = None
        self._fit_factor_penalty: float | None = None
        self._is_fitted = False
        self._artifact_bytes: bytes | None = None
        self._native_predictor_handle: object | None = None
        self._float_thresholds_converted: bool = False
        self._n_features_in = 0
        self._uses_continuous_binning = False
        self._continuous_feature_mins: list[float] | None = None
        self._continuous_feature_maxs: list[float] | None = None
        self._continuous_feature_sorted_values: list[list[float]] | None = None
        self._continuous_feature_quantile_cuts: list[list[float]] | None = None
        self._continuous_feature_linear_rank_flags: list[bool] | None = None
        self.feature_names_in_: list[str] | None = None
        self._native_cat_mappings_: dict[int, dict[str, int]] | None = None
        self.best_iteration_: int | None = None
        self.best_score_: float | None = None
        self.n_estimators_: int | None = None
        self.evals_result_: dict[str, dict[str, list[float]]] | None = None
        self.fit_timing_: dict[str, float] | None = None
        self.diagnostics_per_round_: list[dict] | None = None
        self.factor_exposure_diagnostics_: dict | None = None

    def __repr__(self) -> str:
        return (
            "GBMRegressor("
            f"learning_rate={self.learning_rate}, "
            f"max_depth={self.max_depth}, "
            f"n_estimators={self.n_estimators}, "
            f"row_subsample={self.row_subsample}, "
            f"col_subsample={self.col_subsample}, "
            f"early_stopping_rounds={self.early_stopping_rounds}, "
            f"min_validation_improvement={self.min_validation_improvement}, "
            f"min_data_in_leaf={self.min_data_in_leaf}, "
            f"lambda_l1={self.lambda_l1}, "
            f"lambda_l2={self.lambda_l2}, "
            f"min_child_hessian={self.min_child_hessian}, "
            f"min_split_gain={self.min_split_gain}, "
            f"seed={self.seed}, "
            f"deterministic={self.deterministic}, "
            f"continuous_binning_strategy='{self.continuous_binning_strategy}', "
            f"continuous_binning_max_bins={self.continuous_binning_max_bins}, "
            f"categorical_feature_index={self.categorical_feature_index}, "
            f"categorical_feature_indices={self.categorical_feature_indices}, "
            f"training_policy='{self.training_policy}', "
            f"store_node_stats={self.store_node_stats}, "
            f"categorical_smoothing={self.categorical_smoothing}, "
            f"categorical_min_samples_leaf={self.categorical_min_samples_leaf}, "
            f"categorical_time_aware={self.categorical_time_aware}, "
            f"monotone_constraints={self.monotone_constraints}, "
            f"feature_weights={self.feature_weights}, "
            f"interaction_constraints={self.interaction_constraints}, "
            f"max_leaves={self.max_leaves}, "
            f"tree_growth='{self.tree_growth}', "
            f"warm_start={self.warm_start}, "
            f"objective={self.objective!r}, "
            f"max_cat_threshold={self.max_cat_threshold}, "
            f"training_mode='{self.training_mode}', "
            f"morph_rate={self.morph_rate}, "
            f"evolution_pressure={self.evolution_pressure}, "
            f"morph_warmup_iters={self.morph_warmup_iters}, "
            f"info_score_weight={self.info_score_weight}, "
            f"depth_penalty_base={self.depth_penalty_base}, "
            f"balance_penalty={self.balance_penalty}, "
            f"lr_schedule='{self.lr_schedule}', "
            f"lr_warmup_frac={self.lr_warmup_frac}, "
            f"leaf_model='{self.leaf_model}', "
            f"leaf_solver='{self.leaf_solver}', "
            f"dro_radius={self.dro_radius}, "
            f"dro_metric='{self.dro_metric}', "
            f"neutralization='{self.neutralization}', "
            f"factor_neutralization_lambda={self.factor_neutralization_lambda}, "
            f"factor_penalty={self.factor_penalty}, "
            f"factor_exposure_transform='{self.factor_exposure_transform}', "
            f"boosting_mode='{self.boosting_mode}', "
            f"goss_top_rate={self.goss_top_rate}, "
            f"goss_other_rate={self.goss_other_rate}, "
            f"dart_drop_rate={self.dart_drop_rate}, "
            f"dart_max_drop={self.dart_max_drop}, "
            f"dart_normalize_type='{self.dart_normalize_type}', "
            f"dart_sample_type='{self.dart_sample_type}', "
            f"poisson_max_delta_step={self.poisson_max_delta_step}, "
            f"quantile_alpha={self.quantile_alpha}"
            ")"
        )

    def get_params(self, deep: bool = True) -> dict:
        """Return estimator parameters in sklearn-compatible shape."""
        del deep  # Not used until nested estimators exist.
        return {
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "row_subsample": self.row_subsample,
            "col_subsample": self.col_subsample,
            "early_stopping_rounds": self.early_stopping_rounds,
            "min_validation_improvement": self.min_validation_improvement,
            "min_data_in_leaf": self.min_data_in_leaf,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "min_child_hessian": self.min_child_hessian,
            "min_split_gain": self.min_split_gain,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "continuous_binning_strategy": self.continuous_binning_strategy,
            "continuous_binning_max_bins": self.continuous_binning_max_bins,
            "categorical_feature_index": self.categorical_feature_index,
            "categorical_feature_indices": self.categorical_feature_indices,
            "training_policy": self.training_policy,
            "store_node_stats": self.store_node_stats,
            "categorical_smoothing": self.categorical_smoothing,
            "categorical_min_samples_leaf": self.categorical_min_samples_leaf,
            "categorical_time_aware": self.categorical_time_aware,
            "monotone_constraints": self.monotone_constraints,
            "feature_weights": self.feature_weights,
            "interaction_constraints": self.interaction_constraints,
            "max_leaves": self.max_leaves,
            "tree_growth": self.tree_growth,
            "warm_start": self.warm_start,
            "objective": self.objective,
            "max_cat_threshold": self.max_cat_threshold,
            "training_mode": self.training_mode,
            "morph_rate": self.morph_rate,
            "evolution_pressure": self.evolution_pressure,
            "morph_warmup_iters": self.morph_warmup_iters,
            "info_score_weight": self.info_score_weight,
            "depth_penalty_base": self.depth_penalty_base,
            "balance_penalty": self.balance_penalty,
            "lr_schedule": self.lr_schedule,
            "lr_warmup_frac": self.lr_warmup_frac,
            "leaf_model": self.leaf_model,
            "leaf_solver": self.leaf_solver,
            "dro_radius": self.dro_radius,
            "dro_metric": self.dro_metric,
            "neutralization": self.neutralization,
            "factor_neutralization_lambda": self.factor_neutralization_lambda,
            "factor_penalty": self.factor_penalty,
            "factor_exposure_transform": self.factor_exposure_transform,
            "boosting_mode": self.boosting_mode,
            "goss_top_rate": self.goss_top_rate,
            "goss_other_rate": self.goss_other_rate,
            "dart_drop_rate": self.dart_drop_rate,
            "dart_max_drop": self.dart_max_drop,
            "dart_normalize_type": self.dart_normalize_type,
            "dart_sample_type": self.dart_sample_type,
            "tweedie_variance_power": self.tweedie_variance_power,
            "poisson_max_delta_step": self.poisson_max_delta_step,
            "quantile_alpha": self.quantile_alpha,
        }

    def set_params(self, **params: object) -> "GBMRegressor":
        """Set estimator parameters with constructor-equivalent validation."""
        allowed = {
            "learning_rate",
            "max_depth",
            "n_estimators",
            "row_subsample",
            "col_subsample",
            "early_stopping_rounds",
            "min_validation_improvement",
            "min_data_in_leaf",
            "lambda_l1",
            "lambda_l2",
            "min_child_hessian",
            "min_split_gain",
            "seed",
            "deterministic",
            "continuous_binning_strategy",
            "continuous_binning_max_bins",
            "categorical_feature_index",
            "categorical_feature_indices",
            "training_policy",
            "store_node_stats",
            "categorical_smoothing",
            "categorical_min_samples_leaf",
            "categorical_time_aware",
            "monotone_constraints",
            "feature_weights",
            "interaction_constraints",
            "max_leaves",
            "tree_growth",
            "warm_start",
            "objective",
            "max_cat_threshold",
            "training_mode",
            "morph_rate",
            "evolution_pressure",
            "morph_warmup_iters",
            "info_score_weight",
            "depth_penalty_base",
            "balance_penalty",
            "lr_schedule",
            "lr_warmup_frac",
            "leaf_model",
            "leaf_solver",
            "dro_radius",
            "dro_metric",
            "neutralization",
            "factor_neutralization_lambda",
            "factor_penalty",
            "factor_exposure_transform",
            "boosting_mode",
            "goss_top_rate",
            "goss_other_rate",
            "dart_drop_rate",
            "dart_max_drop",
            "dart_normalize_type",
            "dart_sample_type",
            "tweedie_variance_power",
            "poisson_max_delta_step",
            "quantile_alpha",
        }
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise ValueError(f"Unknown parameter(s): {', '.join(unknown)}")

        if (
            "neutralization" in params
            or "factor_neutralization_lambda" in params
            or "factor_penalty" in params
            or "factor_exposure_transform" in params
            or "leaf_model" in params
        ):
            candidate_neutralization = str(
                params.get("neutralization", self.neutralization)
            )
            if candidate_neutralization not in (
                "none",
                "pre_target",
                "per_round_gradient",
                "split_penalty",
            ):
                raise ValueError(
                    "neutralization must be 'none', 'pre_target', "
                    "'per_round_gradient', or 'split_penalty'"
                )
            candidate_factor_neutralization_lambda = float(
                params.get(
                    "factor_neutralization_lambda",
                    self.factor_neutralization_lambda,
                )
            )
            if (
                not math.isfinite(candidate_factor_neutralization_lambda)
                or candidate_factor_neutralization_lambda < 0.0
            ):
                raise ValueError("factor_neutralization_lambda must be finite and >= 0")
            candidate_factor_penalty = float(
                params.get("factor_penalty", self.factor_penalty)
            )
            if (
                not math.isfinite(candidate_factor_penalty)
                or candidate_factor_penalty < 0.0
            ):
                raise ValueError("factor_penalty must be finite and >= 0")
            candidate_factor_exposure_transform = str(
                params.get("factor_exposure_transform", self.factor_exposure_transform)
            )
            if candidate_factor_exposure_transform not in (
                "none",
                "center",
                "standardize",
            ):
                raise ValueError(
                    "factor_exposure_transform must be 'none', 'center', or 'standardize'"
                )
            candidate_leaf_model = str(params.get("leaf_model", self.leaf_model))
            if (
                candidate_neutralization != "split_penalty"
                and candidate_factor_penalty != 0.0
            ):
                raise ValueError(
                    "factor_penalty is only valid with neutralization='split_penalty'"
                )
            if (
                candidate_neutralization == "split_penalty"
                and candidate_leaf_model == "linear"
            ):
                raise ValueError(
                    "neutralization='split_penalty' requires leaf_model='constant'"
                )

        if "learning_rate" in params:
            learning_rate = float(params["learning_rate"])
            if not (0.0 < learning_rate <= 1.0):
                raise ValueError("learning_rate must be in (0.0, 1.0]")
            self.learning_rate = learning_rate

        if "max_depth" in params:
            max_depth = int(params["max_depth"])
            if max_depth <= 0:
                raise ValueError("max_depth must be greater than 0")
            self.max_depth = max_depth

        if "n_estimators" in params:
            n_estimators = int(params["n_estimators"])
            if n_estimators <= 0:
                raise ValueError("n_estimators must be greater than 0")
            self.n_estimators = n_estimators

        if "row_subsample" in params:
            row_subsample = float(params["row_subsample"])
            if not (0.0 < row_subsample <= 1.0):
                raise ValueError("row_subsample must be in (0.0, 1.0]")
            self.row_subsample = row_subsample

        if "col_subsample" in params:
            col_subsample = float(params["col_subsample"])
            if not (0.0 < col_subsample <= 1.0):
                raise ValueError("col_subsample must be in (0.0, 1.0]")
            self.col_subsample = col_subsample

        if "early_stopping_rounds" in params:
            if params["early_stopping_rounds"] is None:
                self.early_stopping_rounds = None
            else:
                early_stopping_rounds = int(params["early_stopping_rounds"])
                if early_stopping_rounds <= 0:
                    raise ValueError(
                        "early_stopping_rounds must be greater than 0 when set"
                    )
                self.early_stopping_rounds = early_stopping_rounds

        if "min_validation_improvement" in params:
            min_validation_improvement = float(params["min_validation_improvement"])
            if min_validation_improvement < 0.0:
                raise ValueError("min_validation_improvement must be >= 0")
            self.min_validation_improvement = min_validation_improvement

        if "min_data_in_leaf" in params:
            min_data_in_leaf = int(params["min_data_in_leaf"])
            if min_data_in_leaf <= 0:
                raise ValueError("min_data_in_leaf must be greater than 0")
            self.min_data_in_leaf = min_data_in_leaf

        if "lambda_l1" in params:
            lambda_l1 = float(params["lambda_l1"])
            if lambda_l1 < 0.0:
                raise ValueError("lambda_l1 must be >= 0")
            self.lambda_l1 = lambda_l1

        if "lambda_l2" in params:
            lambda_l2 = float(params["lambda_l2"])
            if lambda_l2 < 0.0:
                raise ValueError("lambda_l2 must be >= 0")
            self.lambda_l2 = lambda_l2

        if "min_child_hessian" in params:
            min_child_hessian = float(params["min_child_hessian"])
            if min_child_hessian < 0.0:
                raise ValueError("min_child_hessian must be >= 0")
            self.min_child_hessian = min_child_hessian

        if "min_split_gain" in params:
            min_split_gain = float(params["min_split_gain"])
            if not math.isfinite(min_split_gain) or min_split_gain < 0.0:
                raise ValueError("min_split_gain must be a finite value >= 0")
            self.min_split_gain = min_split_gain

        if "seed" in params:
            self.seed = int(params["seed"])

        if "deterministic" in params:
            self.deterministic = bool(params["deterministic"])

        if "continuous_binning_strategy" in params:
            strategy = str(params["continuous_binning_strategy"])
            if strategy not in _VALID_CONTINUOUS_BINNING_STRATEGIES:
                raise ValueError(
                    "continuous_binning_strategy must be one of: "
                    + ", ".join(sorted(_VALID_CONTINUOUS_BINNING_STRATEGIES))
                )
            if strategy != self.continuous_binning_strategy and self._is_fitted:
                self._reset_fitted_state()
            self.continuous_binning_strategy = strategy

        if "continuous_binning_max_bins" in params:
            max_bins = int(params["continuous_binning_max_bins"])
            if not (
                _MIN_CONTINUOUS_QUANTIZED_BINS
                <= max_bins
                <= (_MAX_CONTINUOUS_QUANTIZED_BIN + 1)
            ):
                raise ValueError(
                    "continuous_binning_max_bins must be in "
                    f"[{_MIN_CONTINUOUS_QUANTIZED_BINS}, {_MAX_CONTINUOUS_QUANTIZED_BIN + 1}]"
                )
            if max_bins != self.continuous_binning_max_bins and self._is_fitted:
                self._reset_fitted_state()
            self.continuous_binning_max_bins = max_bins

        if "categorical_feature_index" in params:
            if params["categorical_feature_index"] is None:
                self.categorical_feature_index = None
            else:
                categorical_feature_index = int(params["categorical_feature_index"])
                if categorical_feature_index < 0:
                    raise ValueError("categorical_feature_index must be >= 0 when set")
                self.categorical_feature_index = categorical_feature_index

        if "categorical_feature_indices" in params:
            if params["categorical_feature_indices"] is None:
                self.categorical_feature_indices = None
            else:
                indices = params["categorical_feature_indices"]
                if not isinstance(indices, (list, tuple)):
                    raise TypeError("categorical_feature_indices must be a list of ints")
                int_indices = [int(i) for i in indices]
                for idx in int_indices:
                    if idx < 0:
                        raise ValueError(
                            "all values in categorical_feature_indices must be >= 0"
                        )
                if len(set(int_indices)) != len(int_indices):
                    raise ValueError(
                        "categorical_feature_indices must not contain duplicates"
                    )
                self.categorical_feature_indices = int_indices

        # Mutual exclusion check after both may have been set
        if self.categorical_feature_index is not None and self.categorical_feature_indices is not None:
            raise ValueError(
                "categorical_feature_index and categorical_feature_indices are mutually exclusive; use categorical_feature_indices for multiple columns"
            )

        if "training_policy" in params:
            training_policy = str(params["training_policy"])
            if training_policy not in {"auto", "manual"}:
                raise ValueError("training_policy must be 'auto' or 'manual'")
            self.training_policy = training_policy

        if "store_node_stats" in params:
            self.store_node_stats = bool(params["store_node_stats"])

        if "categorical_smoothing" in params:
            categorical_smoothing = float(params["categorical_smoothing"])
            if categorical_smoothing < 0.0:
                raise ValueError("categorical_smoothing must be >= 0")
            self.categorical_smoothing = categorical_smoothing

        if "categorical_min_samples_leaf" in params:
            categorical_min_samples_leaf = int(params["categorical_min_samples_leaf"])
            if categorical_min_samples_leaf <= 0:
                raise ValueError("categorical_min_samples_leaf must be greater than 0")
            self.categorical_min_samples_leaf = categorical_min_samples_leaf

        if "categorical_time_aware" in params:
            self.categorical_time_aware = bool(params["categorical_time_aware"])

        if "monotone_constraints" in params:
            mc = params["monotone_constraints"]
            if mc is not None:
                vals = mc.values() if isinstance(mc, dict) else mc
                for v in vals:
                    if int(v) not in (-1, 0, 1):
                        raise ValueError(
                            "monotone_constraints values must be -1, 0, or +1"
                        )
            self.monotone_constraints = mc

        if "feature_weights" in params:
            fw = params["feature_weights"]
            if fw is not None:
                vals = fw.values() if isinstance(fw, dict) else fw
                for w in vals:
                    if not math.isfinite(float(w)) or float(w) < 0.0:
                        raise ValueError(
                            "feature_weights values must be finite and >= 0"
                        )
            self.feature_weights = fw

        if "interaction_constraints" in params:
            ic = params["interaction_constraints"]
            if ic is None:
                self.interaction_constraints = None
            else:
                if not isinstance(ic, (list, tuple)):
                    raise TypeError(
                        "interaction_constraints must be a sequence of feature-index groups"
                    )
                if len(ic) > 64:
                    raise ValueError(
                        "interaction_constraints supports at most 64 groups "
                        f"(got {len(ic)})"
                    )
                normalized: list[list[int]] = []
                for gi, group in enumerate(ic):
                    if not isinstance(group, (list, tuple)) or len(group) == 0:
                        raise ValueError(
                            f"interaction_constraints group {gi} must be a non-empty "
                            "sequence of feature indices"
                        )
                    seen: set[int] = set()
                    canonical: list[int] = []
                    for f in group:
                        fi = int(f)
                        if fi < 0:
                            raise ValueError(
                                f"interaction_constraints group {gi} contains negative "
                                f"feature index {fi}"
                            )
                        if fi in seen:
                            raise ValueError(
                                f"interaction_constraints group {gi} contains duplicate "
                                f"feature index {fi}"
                            )
                        seen.add(fi)
                        canonical.append(fi)
                    normalized.append(canonical)
                self.interaction_constraints = normalized

        if "max_leaves" in params:
            if params["max_leaves"] is None:
                self.max_leaves = None
            else:
                ml = int(params["max_leaves"])
                if ml < 2:
                    raise ValueError("max_leaves must be >= 2 when set")
                self.max_leaves = ml

        if "tree_growth" in params:
            tg = str(params["tree_growth"])
            if tg not in {"level", "leaf"}:
                raise ValueError("tree_growth must be 'level' or 'leaf'")
            self.tree_growth = tg

        if "warm_start" in params:
            self.warm_start = bool(params["warm_start"])

        if "objective" in params:
            obj = params["objective"]
            if obj is not None and not callable(obj) and not isinstance(obj, str):
                raise TypeError("objective must be a string, a callable, or None")
            self.objective = obj

        if "max_cat_threshold" in params:
            mct = int(params["max_cat_threshold"])
            if mct < 0:
                raise ValueError("max_cat_threshold must be >= 0")
            self.max_cat_threshold = mct

        if "training_mode" in params:
            tm = str(params["training_mode"])
            if tm not in ("auto", "manual", "morph"):
                raise ValueError(
                    f"training_mode must be 'auto', 'manual', or 'morph', got {tm!r}"
                )
            self.training_mode = tm

        if "morph_rate" in params:
            mr = float(params["morph_rate"])
            if not (0.0 <= mr <= 1.0):
                raise ValueError("morph_rate must be in [0.0, 1.0]")
            self.morph_rate = mr

        if "evolution_pressure" in params:
            ep = float(params["evolution_pressure"])
            if not (0.0 <= ep <= 1.0):
                raise ValueError("evolution_pressure must be in [0.0, 1.0]")
            self.evolution_pressure = ep

        if "morph_warmup_iters" in params:
            mwi = int(params["morph_warmup_iters"])
            if mwi < 0:
                raise ValueError("morph_warmup_iters must be >= 0")
            self.morph_warmup_iters = mwi

        if "info_score_weight" in params:
            isw = float(params["info_score_weight"])
            if not (0.0 <= isw <= 1.0):
                raise ValueError("info_score_weight must be in [0.0, 1.0]")
            self.info_score_weight = isw

        if "depth_penalty_base" in params:
            dpb = float(params["depth_penalty_base"])
            if not (0.0 < dpb <= 1.0):
                raise ValueError("depth_penalty_base must be in (0.0, 1.0]")
            self.depth_penalty_base = dpb

        if "balance_penalty" in params:
            self.balance_penalty = bool(params["balance_penalty"])

        if "lr_schedule" in params:
            lrs = str(params["lr_schedule"])
            if lrs not in ("constant", "warmup_cosine"):
                raise ValueError(
                    f"lr_schedule must be 'constant' or 'warmup_cosine', got {lrs!r}"
                )
            self.lr_schedule = lrs

        if "lr_warmup_frac" in params:
            lwf = float(params["lr_warmup_frac"])
            if not (0.0 <= lwf <= 1.0):
                raise ValueError("lr_warmup_frac must be in [0.0, 1.0]")
            self.lr_warmup_frac = lwf

        if "leaf_model" in params:
            lm = str(params["leaf_model"])
            if lm not in ("constant", "linear"):
                raise ValueError(
                    f"leaf_model must be 'constant' or 'linear', got {lm!r}"
                )
            self.leaf_model = lm

        if "leaf_solver" in params:
            ls = str(params["leaf_solver"])
            if ls not in ("standard", "dro"):
                raise ValueError(
                    f"leaf_solver must be 'standard' or 'dro', got {ls!r}"
                )
            self.leaf_solver = ls

        if "dro_radius" in params:
            dr = float(params["dro_radius"])
            if not math.isfinite(dr) or dr < 0.0:
                raise ValueError("dro_radius must be finite and >= 0")
            self.dro_radius = dr

        if "dro_metric" in params:
            dm = str(params["dro_metric"])
            if dm != "wasserstein":
                raise ValueError(f"dro_metric must be 'wasserstein', got {dm!r}")
            self.dro_metric = dm

        if "neutralization" in params:
            nz = str(params["neutralization"])
            if nz not in ("none", "pre_target", "per_round_gradient", "split_penalty"):
                raise ValueError(
                    "neutralization must be 'none', 'pre_target', "
                    "'per_round_gradient', or 'split_penalty'"
                )
            self.neutralization = nz

        if "factor_neutralization_lambda" in params:
            fnl = float(params["factor_neutralization_lambda"])
            if not math.isfinite(fnl) or fnl < 0.0:
                raise ValueError("factor_neutralization_lambda must be finite and >= 0")
            self.factor_neutralization_lambda = fnl

        if "factor_penalty" in params:
            fp = float(params["factor_penalty"])
            if not math.isfinite(fp) or fp < 0.0:
                raise ValueError("factor_penalty must be finite and >= 0")
            self.factor_penalty = fp

        if "factor_exposure_transform" in params:
            fet = str(params["factor_exposure_transform"])
            if fet not in ("none", "center", "standardize"):
                raise ValueError(
                    "factor_exposure_transform must be 'none', 'center', or 'standardize'"
                )
            self.factor_exposure_transform = fet

        if "boosting_mode" in params:
            bm = str(params["boosting_mode"])
            if bm not in _VALID_BOOSTING_MODES:
                raise ValueError(
                    "boosting_mode must be one of: "
                    + ", ".join(sorted(_VALID_BOOSTING_MODES))
                )
            self.boosting_mode = bm
        if "goss_top_rate" in params:
            r = float(params["goss_top_rate"])
            if not (0.0 < r < 1.0):
                raise ValueError("goss_top_rate must be in (0, 1)")
            self.goss_top_rate = r
        if "goss_other_rate" in params:
            r = float(params["goss_other_rate"])
            if not (0.0 < r < 1.0):
                raise ValueError("goss_other_rate must be in (0, 1)")
            self.goss_other_rate = r
        if "dart_drop_rate" in params:
            r = float(params["dart_drop_rate"])
            if not (0.0 < r < 1.0):
                raise ValueError("dart_drop_rate must be in (0, 1)")
            self.dart_drop_rate = r
        if "dart_max_drop" in params:
            d = int(params["dart_max_drop"])
            if d < 1:
                raise ValueError("dart_max_drop must be >= 1")
            self.dart_max_drop = d
        if "dart_normalize_type" in params:
            n = str(params["dart_normalize_type"])
            if n not in {"tree", "forest"}:
                raise ValueError(
                    "dart_normalize_type must be 'tree' or 'forest', "
                    f"got {n!r}"
                )
            self.dart_normalize_type = n
        if "dart_sample_type" in params:
            s = str(params["dart_sample_type"])
            if s not in {"uniform", "weighted"}:
                raise ValueError(
                    "dart_sample_type must be 'uniform' or 'weighted', "
                    f"got {s!r}"
                )
            self.dart_sample_type = s
        if "tweedie_variance_power" in params:
            v = float(params["tweedie_variance_power"])
            self.tweedie_variance_power = v
        if "poisson_max_delta_step" in params:
            v = float(params["poisson_max_delta_step"])
            if not math.isfinite(v) or v < 0.0:
                raise ValueError("poisson_max_delta_step must be finite and >= 0")
            self.poisson_max_delta_step = v
        if "quantile_alpha" in params:
            qa = float(params["quantile_alpha"])
            _validate_quantile_alpha(qa)
            self.quantile_alpha = qa
        # Cross-field validation for goss top+other rates.
        if self.boosting_mode == "goss" and (
            self.goss_top_rate + self.goss_other_rate > 1.0
        ):
            raise ValueError(
                "boosting_mode='goss' requires goss_top_rate + goss_other_rate <= 1.0"
            )
        # PR #41 review C3: cross-field validation for Tweedie variance
        # power.  The constructor rejects out-of-range powers when
        # `objective='tweedie'`; set_params must do the same regardless of
        # whether the objective, the power, or both change in this call —
        # otherwise sklearn-style updates leave the estimator in a state
        # the constructor would have refused, and the failure only
        # surfaces from the native bridge much later.
        post_objective = self._objective_name()
        if post_objective == "tweedie":
            v = self.tweedie_variance_power
            if not isinstance(v, (int, float)) or not (1.0 < float(v) < 2.0):
                raise ValueError(
                    "tweedie_variance_power must satisfy 1 < p < 2 when "
                    f"objective='tweedie' (got {v!r})"
                )
        if post_objective == "quantile":
            _validate_quantile_alpha(self.quantile_alpha)

        # Cross-field validation: leaf growth requires max_leaves
        if self.tree_growth == "leaf" and self.max_leaves is None:
            raise ValueError("max_leaves must be set when tree_growth='leaf'")

        # Cross-field validation: lr_warmup_frac is only meaningful with
        # lr_schedule="warmup_cosine". Reject any non-default value with a
        # constant schedule (matches __init__ and the Rust-side contract).
        if self.lr_schedule != "warmup_cosine" and self.lr_warmup_frac != 0.1:
            raise ValueError(
                f"lr_warmup_frac={self.lr_warmup_frac} is only valid with "
                f"lr_schedule='warmup_cosine'; got lr_schedule='{self.lr_schedule}'"
            )

        if self.leaf_solver == "dro" and self.leaf_model != "constant":
            raise ValueError(
                "leaf_solver='dro' requires leaf_model='constant'"
            )
        if self.neutralization != "split_penalty" and self.factor_penalty != 0.0:
            raise ValueError(
                "factor_penalty is only valid with neutralization='split_penalty'"
            )
        if self.neutralization == "split_penalty" and self.leaf_model == "linear":
            raise ValueError(
                "neutralization='split_penalty' requires leaf_model='constant'"
            )

        return self

    def fit(
        self,
        X: object,
        y: object,
        *,
        sample_weight: object | None = None,
        eval_set: tuple[object, object] | None = None,
        eval_sample_weight: object | None = None,
        group: object | None = None,
        eval_group: object | None = None,
        eval_time_index: object | None = None,
        categorical_feature_values: object | None = None,
        categorical_feature_values_list: object | None = None,
        time_index: object | None = None,
        init_model: "GBMRegressor | None" = None,
        eval_metric: object | None = None,
        factor_exposures: object | None = None,
    ) -> "GBMRegressor":
        """Fit native-backed regression model artifact state.

        Parameters
        ----------
        init_model : GBMRegressor or None, optional
            A previously fitted model to continue training from (warm-start).
            When provided, training resumes from this model's trees and
            ``n_estimators`` additional rounds are trained. If ``warm_start=True``
            is set on the estimator, the model's own previous state is used
            instead. ``init_model`` takes priority over ``warm_start``.
        eval_metric : callable or None, optional
            Custom evaluation metric callable. The function signature must be
            ``(y_true: np.ndarray, y_pred: np.ndarray) -> (name, value, higher_is_better)``
            where *name* is a string, *value* is a float, and *higher_is_better*
            is a bool.  When provided together with ``eval_set``, the metric is
            evaluated after each boosting round and can drive early stopping
            (instead of the built-in loss) when ``early_stopping_rounds`` is set.
        """
        fit_start = time.perf_counter()
        self._fit_start_time = fit_start
        targets = self._validate_targets(y)
        obj = self._objective_name()
        if obj == "quantile":
            _validate_quantile_alpha(self.quantile_alpha)
        if self.early_stopping_rounds is not None and eval_set is None:
            raise ValueError("early_stopping_rounds requires eval_set to be provided")
        if eval_time_index is not None and eval_set is None:
            raise ValueError("eval_time_index requires eval_set to be provided")
        if eval_metric is not None and not callable(eval_metric):
            raise TypeError("eval_metric must be a callable or None")
        if eval_metric is not None and eval_set is None:
            raise ValueError("eval_metric requires eval_set to be provided")
        if self.neutralization == "pre_target" and eval_set is not None:
            raise ValueError(
                "neutralization='pre_target' does not support eval_set in this "
                "release because validation factor_exposures are not accepted"
            )
        if categorical_feature_values is not None and categorical_feature_values_list is not None:
            raise ValueError(
                "categorical_feature_values and categorical_feature_values_list are "
                "mutually exclusive"
            )

        # ── Resolve warm-start artifact bytes ──────────────────────────
        init_artifact_bytes: bytes | None = None
        if init_model is not None:
            if not hasattr(init_model, "_artifact_bytes") or init_model._artifact_bytes is None:
                raise ValueError("init_model must be a fitted GBMRegressor with artifact bytes")
            init_neutralization, init_lambda, init_penalty = (
                init_model._fitted_neutralization_contract()
            )
            # Settings-mismatch fires first so dropping/changing the
            # neutralization mode produces a clear "does not match" error
            # instead of a misleading "factor_exposures required" one (the
            # latter would otherwise win when the user passes None on a
            # mode-switch to "none").
            self._raise_if_neutralization_settings_mismatch(
                init_neutralization,
                init_lambda,
                init_penalty,
                origin="init_model",
            )
            self._raise_if_neutralized_warm_start_contract(
                init_neutralization, factor_exposures
            )
            if hasattr(init_model, "_objective_name"):
                init_objective = init_model._objective_name()
                current_objective = self._objective_name()
                if init_objective != current_objective:
                    raise ValueError(
                        f"init_model objective '{init_objective}' does not match "
                        f"current objective '{current_objective}'"
                    )
            init_artifact_bytes = init_model._artifact_bytes
        elif self.warm_start and self._is_fitted and self._artifact_bytes is not None:
            fit_neutralization, fit_lambda, fit_penalty = (
                self._fitted_neutralization_contract()
            )
            self._raise_if_neutralization_settings_mismatch(
                fit_neutralization,
                fit_lambda,
                fit_penalty,
                origin="warm_start",
            )
            self._raise_if_neutralized_warm_start_contract(
                fit_neutralization, factor_exposures
            )
            init_artifact_bytes = self._artifact_bytes

        # ── Normalize categorical configuration to plural form ──────────
        # effective_categorical_indices: list of column indices (or None if no categoricals)
        # categorical_values_list: list of per-column string values (or None)
        effective_categorical_indices: list[int] | None = None
        categorical_values_list: list[list[str]] | None = None

        if self.categorical_feature_indices is not None:
            effective_categorical_indices = list(self.categorical_feature_indices)
        elif self.categorical_feature_index is not None:
            effective_categorical_indices = [self.categorical_feature_index]

        if categorical_feature_values_list is not None:
            categorical_values_list = self._validate_categorical_values_list(
                categorical_feature_values_list, len(targets)
            )
        elif categorical_feature_values is not None:
            categorical_values_list = [
                self._validate_categorical_values(
                    categorical_feature_values, len(targets)
                )
            ]
        elif effective_categorical_indices is None:
            # Auto-infer from DataFrame dtypes
            inferred = self._infer_explicit_categorical_features(X)
            if inferred is not None:
                effective_categorical_indices, categorical_values_list = inferred

        has_categorical = (
            effective_categorical_indices is not None
            and len(effective_categorical_indices) > 0
        )

        # Backward-compat aliases used by some downstream code paths
        effective_categorical_feature_index: int | None = (
            effective_categorical_indices[0]
            if has_categorical and len(effective_categorical_indices) == 1
            else None
        )
        categorical_values: list[str] | None = (
            categorical_values_list[0]
            if categorical_values_list is not None and len(categorical_values_list) == 1
            else None
        )

        # Prefer bytes payload (zero-copy numpy→Rust) over list[float] payload
        dense_training_bytes_payload = (
            self._native_matrix_bytes_payload(X)
            if not has_categorical
            else None
        )
        dense_training_payload = (
            self._native_matrix_flat_payload(X)
            if not has_categorical and dense_training_bytes_payload is None
            else None
        )
        training_rows: list[list[float]] | None = None
        if dense_training_bytes_payload is not None:
            _, row_count, feature_count = dense_training_bytes_payload
        elif dense_training_payload is not None:
            _, row_count, feature_count = dense_training_payload
        else:
            training_rows = self._validate_rows(
                X, categorical_feature_indices=effective_categorical_indices
            )
            row_count = len(training_rows)
            feature_count = len(training_rows[0])
        if row_count != len(targets):
            raise ValueError("X and y must contain the same number of rows")
        (
            factor_exposure_values,
            factor_exposure_row_count,
            factor_exposure_factor_count,
            transformed_factor_exposures,
        ) = self._prepare_factor_exposures(factor_exposures, row_count)

        if init_model is not None and hasattr(init_model, "_n_features_in"):
            if (
                init_model._n_features_in is not None
                and init_model._n_features_in != feature_count
            ):
                raise ValueError(
                    f"init_model was fitted with {init_model._n_features_in} features, "
                    f"but X has {feature_count} features"
                )

        # Validate sample_weight if provided.
        validated_sample_weights: list[float] | None = None
        if sample_weight is not None:
            validated_sample_weights = self._validate_sample_weight(sample_weight, row_count)
            obj = self._objective_name()
            if obj in ("rank_pairwise", "rank_ndcg", "rank_xendcg", "yetirank"):
                import warnings

                warnings.warn(
                    f"sample_weight is ignored by ranking objective '{obj}'",
                    UserWarning,
                    stacklevel=2,
                )

        # Target-domain validation for GLM objectives — applied to training
        # targets here, and again to validation targets after `eval_set` is
        # unpacked below (early-stopping and validation-loss reporting also
        # need to respect the GLM domain — see PR #41 review C2).
        self._validate_glm_target_domain(y, role="training")

        # Validate group if provided.
        validated_group_id: list[int] | None = None
        if group is not None:
            validated_group_id = self._validate_group(group, row_count)

        if eval_sample_weight is not None and eval_set is None:
            raise ValueError("eval_sample_weight requires eval_set to be provided")
        if eval_group is not None and eval_set is None:
            raise ValueError("eval_group requires eval_set to be provided")

        # Capture feature names from DataFrame columns if available.
        columns = getattr(X, "columns", None)
        if columns is not None:
            names = [str(c) for c in columns]
            if len(names) == feature_count:
                self.feature_names_in_ = names
            else:
                self.feature_names_in_ = None
        else:
            self.feature_names_in_ = None

        if has_categorical:
            assert effective_categorical_indices is not None
            for idx in effective_categorical_indices:
                if idx >= feature_count:
                    raise ValueError(
                        f"categorical feature index {idx} must be within fitted feature bounds (feature_count={feature_count})"
                    )

        if not has_categorical and categorical_values_list is not None:
            raise ValueError(
                "categorical_feature_values requires categorical_feature_index or categorical_feature_indices to be set"
            )
        if has_categorical and categorical_values_list is None:
            raise ValueError(
                "categorical_feature_values must be provided when categorical feature indices are set"
            )
        if (
            has_categorical
            and categorical_values_list is not None
            and len(categorical_values_list) != len(effective_categorical_indices)  # type: ignore[arg-type]
        ):
            raise ValueError(
                f"categorical_feature_values_list length ({len(categorical_values_list)}) "
                f"does not match categorical_feature_indices length ({len(effective_categorical_indices)})"  # type: ignore[arg-type]
            )

        validated_time_index = (
            self._validate_time_index(time_index, row_count) if time_index is not None else None
        )

        validation_X: object | None = None
        validation_targets: list[float] | None = None
        validation_dense_payload: tuple[list[float], int, int] | None = None
        validation_dense_bytes_payload: tuple[bytes, int, int] | None = None
        validation_rows: list[list[float]] | None = None
        validation_categorical_values: list[str] | None = None
        validation_categorical_values_list: list[list[str]] | None = None
        validated_eval_time_index: list[int] | None = None
        validated_eval_sample_weights: list[float] | None = None
        validated_eval_group_id: list[int] | None = None
        if eval_set is not None:
            validation_X, validation_y = eval_set
            validation_targets = self._validate_targets(validation_y)
            # PR #41 review C2: GLM domain check must also gate validation
            # targets — without this, early-stopping reads losses computed
            # on out-of-domain y and reports nonsensical numbers (Gamma loss
            # on y=0 hits `log(0)`, Poisson/Tweedie hit negative-y paths).
            self._validate_glm_target_domain(validation_targets, role="validation")
            validation_dense_bytes_payload = (
                self._native_matrix_bytes_payload(validation_X)
                if not has_categorical
                else None
            )
            validation_dense_payload = (
                self._native_matrix_flat_payload(validation_X)
                if not has_categorical and validation_dense_bytes_payload is None
                else None
            )
            if validation_dense_bytes_payload is not None:
                _, validation_row_count, validation_feature_count = validation_dense_bytes_payload
            elif validation_dense_payload is not None:
                _, validation_row_count, validation_feature_count = validation_dense_payload
            else:
                validation_rows = self._validate_rows(
                    validation_X,
                    categorical_feature_indices=effective_categorical_indices,
                )
                validation_row_count = len(validation_rows)
                validation_feature_count = len(validation_rows[0])
            if validation_row_count != len(validation_targets):
                raise ValueError("eval_set X and y must contain the same number of rows")
            if validation_feature_count != feature_count:
                raise ValueError(
                    "eval_set feature count must match training feature count"
                )
            if eval_time_index is not None:
                validated_eval_time_index = self._validate_time_index(
                    eval_time_index, validation_row_count
                )
            if has_categorical:
                validation_categorical_values_list = (
                    self._extract_categorical_values_for_indices(
                        validation_X,
                        effective_categorical_indices,  # type: ignore[arg-type]
                        validation_row_count,
                    )
                )
                # Keep backward-compat alias for singular case
                if len(effective_categorical_indices) == 1:  # type: ignore[arg-type]
                    validation_categorical_values = validation_categorical_values_list[0]
            if eval_sample_weight is not None:
                validated_eval_sample_weights = self._validate_sample_weight(
                    eval_sample_weight, validation_row_count
                )
            if eval_group is not None:
                validated_eval_group_id = self._validate_group(
                    eval_group, validation_row_count
                )

        if (
            has_categorical
            and self.categorical_time_aware
            and validated_time_index is None
        ):
            raise ValueError(
                "time_index must be provided when categorical_time_aware=True and categorical features are set"
            )
        if (
            has_categorical
            and self.categorical_time_aware
            and eval_set is not None
            and validated_eval_time_index is None
        ):
            raise ValueError(
                "eval_time_index must be provided when categorical_time_aware=True and eval_set is used"
            )

        import numpy as np
        targets_bytes = np.asarray(targets, dtype=np.float32).tobytes() if dense_training_bytes_payload is not None else None
        validation_targets_bytes = (
            np.asarray(validation_targets, dtype=np.float32).tobytes()
            if validation_targets is not None and dense_training_bytes_payload is not None
            else None
        )

        input_adaptation_seconds = time.perf_counter() - fit_start
        ranking_sigma_kwargs = (
            {
                "ranking_sigma": self.ranking_sigma,
                "lambdarank_truncation_level": self.lambdarank_truncation_level,
            }
            if hasattr(self, "ranking_sigma")
            else {}
        )

        # Resolve custom objective / metric callables for the native bridge.
        _custom_objective_fn = self.objective if callable(self.objective) else None
        _custom_loss_fn = None  # reserved for future extension
        _custom_metric_fn = eval_metric if callable(eval_metric) else None

        # Resolve morph-mode training profile.
        _effective_max_depth = self.max_depth
        self._morph_config_: dict | None = None
        if self.training_mode == "morph":
            from alloygbm._morph import build_morph_config_dict, compute_morph_fingerprint
            fp = compute_morph_fingerprint(X, y, fast_mode=True)
            if self.max_depth is None:
                _effective_max_depth = fp["suggested_max_depth"]
            self._morph_config_ = build_morph_config_dict(
                morph_rate=self.morph_rate,
                evolution_pressure=self.evolution_pressure,
                morph_warmup_iters=self.morph_warmup_iters,
                info_score_weight=self.info_score_weight,
                depth_penalty_base=self.depth_penalty_base,
                balance_penalty=self.balance_penalty,
                lr_schedule=self.lr_schedule,
                lr_warmup_frac=self.lr_warmup_frac,
            )
        elif self.training_mode in ("auto", "manual") and self.lr_schedule != "constant":
            # User opted into an LR schedule without enabling morph mode. The
            # public API documents lr_schedule as independent of training_mode,
            # so we build a neutral morph_config that activates only the LR
            # schedule (and the schedule-aware early-stop logic). Setting
            # morph_rate=0.0, info_score_weight=0.0, depth_penalty_base=1.0,
            # balance_penalty=False, and morph_warmup_iters=0 keeps the gain
            # criterion and leaf-value formula bit-identical to the standard
            # path; only the per-iteration LR is overridden by the schedule.
            from alloygbm._morph import build_morph_config_dict
            self._morph_config_ = build_morph_config_dict(
                morph_rate=0.0,
                evolution_pressure=0.0,
                morph_warmup_iters=0,
                info_score_weight=0.0,
                depth_penalty_base=1.0,
                balance_penalty=False,
                lr_schedule=self.lr_schedule,
                lr_warmup_frac=self.lr_warmup_frac,
            )
        elif self.training_mode not in ("auto", "manual"):
            raise ValueError(
                f"training_mode must be 'auto', 'manual', or 'morph', got {self.training_mode!r}"
            )

        # Try bytes path first (avoids Python list→Vec<f32> conversion overhead)
        if dense_training_bytes_payload is not None:
            try:
                from alloygbm._alloygbm import train_regression_artifact_dense_with_summary_bytes
                native_result = train_regression_artifact_dense_with_summary_bytes(
                    values_bytes=dense_training_bytes_payload[0],
                    row_count=dense_training_bytes_payload[1],
                    feature_count=dense_training_bytes_payload[2],
                    targets_bytes=targets_bytes,
                    learning_rate=self.learning_rate,
                    max_depth=_effective_max_depth,
                    row_subsample=self.row_subsample,
                    col_subsample=self.col_subsample,
                    min_validation_improvement=self.min_validation_improvement,
                    seed=self.seed,
                    deterministic=self.deterministic,
                    rounds=self.n_estimators,
                    early_stopping_rounds=self.early_stopping_rounds,
                    min_data_in_leaf=self.min_data_in_leaf,
                    lambda_l1=self.lambda_l1,
                    lambda_l2=self.lambda_l2,
                    min_child_hessian=self.min_child_hessian,
                    sample_weights=validated_sample_weights,
                    group_id=validated_group_id,
                    min_split_gain=self.min_split_gain,
                    validation_values_bytes=(
                        validation_dense_bytes_payload[0]
                        if validation_dense_bytes_payload is not None
                        else None
                    ),
                    validation_row_count=(
                        validation_dense_bytes_payload[1]
                        if validation_dense_bytes_payload is not None
                        else None
                    ),
                    validation_targets_bytes=validation_targets_bytes,
                    validation_sample_weights=validated_eval_sample_weights,
                    validation_group_id=validated_eval_group_id,
                    validation_time_index=validated_eval_time_index,
                    categorical_feature_index=effective_categorical_feature_index,
                    categorical_feature_values=categorical_values,
                    validation_categorical_feature_values=validation_categorical_values,
                    training_policy=self.training_policy,
                    store_node_stats=self.store_node_stats,
                    categorical_smoothing=self.categorical_smoothing,
                    categorical_min_samples_leaf=self.categorical_min_samples_leaf,
                    categorical_time_aware=self.categorical_time_aware,
                    time_index=validated_time_index,
                    continuous_binning_strategy=self.continuous_binning_strategy,
                    continuous_binning_max_bins=self.continuous_binning_max_bins,
                    objective=self._objective_name(),
                    monotone_constraints=self._resolve_monotone_constraints(feature_count),
                    feature_weights=self._resolve_feature_weights(feature_count),
                    interaction_constraints=self._resolve_interaction_constraints(feature_count),
                    max_leaves=self.max_leaves,
                    tree_growth=self.tree_growth,
                    categorical_feature_indices=effective_categorical_indices if has_categorical else None,
                    categorical_feature_values_list=categorical_values_list if has_categorical else None,
                    validation_categorical_feature_values_list=validation_categorical_values_list if has_categorical else None,
                    init_artifact_bytes=init_artifact_bytes,
                    num_classes=getattr(self, '_num_classes_for_training', None),
                    custom_objective_fn=_custom_objective_fn,
                    custom_loss_fn=_custom_loss_fn,
                    custom_metric_fn=_custom_metric_fn,
                    max_cat_threshold=self.max_cat_threshold,
                    morph_config=self._morph_config_,
                    leaf_model=self.leaf_model,
                    leaf_solver=self.leaf_solver,
                    dro_radius=self.dro_radius,
                    dro_metric=self.dro_metric,
                    neutralization=self.neutralization,
                    factor_neutralization_lambda=self.factor_neutralization_lambda,
                    factor_penalty=self.factor_penalty,
                    factor_exposure_values=factor_exposure_values,
                    factor_exposure_row_count=(
                        factor_exposure_row_count
                        if factor_exposure_values is not None
                        else None
                    ),
                    factor_exposure_factor_count=(
                        factor_exposure_factor_count
                        if factor_exposure_values is not None
                        else None
                    ),
                    boosting_mode=self.boosting_mode,
                    goss_top_rate=(
                        self.goss_top_rate if self.boosting_mode == "goss" else None
                    ),
                    goss_other_rate=(
                        self.goss_other_rate if self.boosting_mode == "goss" else None
                    ),
                    dart_drop_rate=(
                        self.dart_drop_rate if self.boosting_mode == "dart" else None
                    ),
                    dart_max_drop=(
                        self.dart_max_drop if self.boosting_mode == "dart" else None
                    ),
                    dart_normalize_type=(
                        self.dart_normalize_type if self.boosting_mode == "dart" else None
                    ),
                    dart_sample_type=(
                        self.dart_sample_type if self.boosting_mode == "dart" else None
                    ),
                    tweedie_variance_power=(
                        self.tweedie_variance_power
                        if self._objective_name() == "tweedie"
                        else None
                    ),
                    poisson_max_delta_step=(
                        self.poisson_max_delta_step
                        if self._objective_name() == "poisson"
                        else None
                    ),
                    quantile_alpha=(
                        self.quantile_alpha
                        if self._objective_name() == "quantile"
                        else None
                    ),
                    **ranking_sigma_kwargs,
                )
                return self._finalize_training_result(
                    native_result,
                    input_adaptation_seconds,
                    feature_count=feature_count,
                    fit_X=X,
                    transformed_factor_exposures=transformed_factor_exposures,
                )
            except (ImportError, AttributeError):
                pass  # Fall through to list-based path
            except Exception:
                pass  # Bytes path not available or failed; fall through

        # Compute dense_training_payload lazily if only bytes payload was prepared
        if dense_training_payload is None and dense_training_bytes_payload is not None:
            dense_training_payload = self._native_matrix_flat_payload(X)

        try:
            if dense_training_payload is not None:
                train_with_summary = _base._load_native_train_regression_artifact_dense_with_summary()
            else:
                train_with_summary = _base._load_native_train_regression_artifact_with_summary()
        except RuntimeError:
            return self._fit_with_legacy_native_bridge(
                X=X,
                targets=targets,
                dense_training_payload=dense_training_payload,
                training_rows=training_rows if dense_training_payload is None else None,
                feature_count=feature_count,
                categorical_feature_index=effective_categorical_feature_index,
                categorical_values=categorical_values,
                time_index=validated_time_index,
                eval_set=eval_set,
                input_adaptation_seconds=input_adaptation_seconds,
                factor_exposure_values=factor_exposure_values,
                factor_exposure_row_count=factor_exposure_row_count,
                factor_exposure_factor_count=factor_exposure_factor_count,
                transformed_factor_exposures=transformed_factor_exposures,
            )

        if dense_training_payload is not None:
            native_result = train_with_summary(
                values=dense_training_payload[0],
                row_count=dense_training_payload[1],
                feature_count=dense_training_payload[2],
                targets=targets,
                learning_rate=self.learning_rate,
                max_depth=_effective_max_depth,
                row_subsample=self.row_subsample,
                col_subsample=self.col_subsample,
                min_validation_improvement=self.min_validation_improvement,
                seed=self.seed,
                deterministic=self.deterministic,
                rounds=self.n_estimators,
                early_stopping_rounds=self.early_stopping_rounds,
                min_data_in_leaf=self.min_data_in_leaf,
                lambda_l1=self.lambda_l1,
                lambda_l2=self.lambda_l2,
                min_child_hessian=self.min_child_hessian,
                sample_weights=validated_sample_weights,
                group_id=validated_group_id,
                min_split_gain=self.min_split_gain,
                validation_values=(
                    validation_dense_payload[0]
                    if validation_dense_payload is not None
                    else None
                ),
                validation_row_count=(
                    validation_dense_payload[1]
                    if validation_dense_payload is not None
                    else None
                ),
                validation_targets=validation_targets,
                validation_sample_weights=validated_eval_sample_weights,
                validation_group_id=validated_eval_group_id,
                validation_time_index=validated_eval_time_index,
                categorical_feature_index=effective_categorical_feature_index,
                categorical_feature_values=categorical_values,
                validation_categorical_feature_values=validation_categorical_values,
                training_policy=self.training_policy,
                store_node_stats=self.store_node_stats,
                categorical_smoothing=self.categorical_smoothing,
                categorical_min_samples_leaf=self.categorical_min_samples_leaf,
                categorical_time_aware=self.categorical_time_aware,
                time_index=validated_time_index,
                continuous_binning_strategy=self.continuous_binning_strategy,
                continuous_binning_max_bins=self.continuous_binning_max_bins,
                objective=self._objective_name(),
                monotone_constraints=self._resolve_monotone_constraints(feature_count),
                feature_weights=self._resolve_feature_weights(feature_count),
                interaction_constraints=self._resolve_interaction_constraints(feature_count),
                max_leaves=self.max_leaves,
                tree_growth=self.tree_growth,
                categorical_feature_indices=effective_categorical_indices if has_categorical else None,
                categorical_feature_values_list=categorical_values_list if has_categorical else None,
                validation_categorical_feature_values_list=validation_categorical_values_list if has_categorical else None,
                init_artifact_bytes=init_artifact_bytes,
                num_classes=getattr(self, '_num_classes_for_training', None),
                custom_objective_fn=_custom_objective_fn,
                custom_loss_fn=_custom_loss_fn,
                custom_metric_fn=_custom_metric_fn,
                max_cat_threshold=self.max_cat_threshold,
                morph_config=self._morph_config_,
                leaf_model=self.leaf_model,
                leaf_solver=self.leaf_solver,
                dro_radius=self.dro_radius,
                dro_metric=self.dro_metric,
                neutralization=self.neutralization,
                factor_neutralization_lambda=self.factor_neutralization_lambda,
                factor_penalty=self.factor_penalty,
                factor_exposure_values=factor_exposure_values,
                factor_exposure_row_count=(
                    factor_exposure_row_count if factor_exposure_values is not None else None
                ),
                factor_exposure_factor_count=(
                    factor_exposure_factor_count
                    if factor_exposure_values is not None
                    else None
                ),
                boosting_mode=self.boosting_mode,
                goss_top_rate=(
                    self.goss_top_rate if self.boosting_mode == "goss" else None
                ),
                goss_other_rate=(
                    self.goss_other_rate if self.boosting_mode == "goss" else None
                ),
                dart_drop_rate=(
                    self.dart_drop_rate if self.boosting_mode == "dart" else None
                ),
                dart_max_drop=(
                    self.dart_max_drop if self.boosting_mode == "dart" else None
                ),
                dart_normalize_type=(
                    self.dart_normalize_type if self.boosting_mode == "dart" else None
                ),
                dart_sample_type=(
                    self.dart_sample_type if self.boosting_mode == "dart" else None
                ),
                tweedie_variance_power=(
                    self.tweedie_variance_power
                    if self._objective_name() == "tweedie"
                    else None
                ),
                poisson_max_delta_step=(
                    self.poisson_max_delta_step
                    if self._objective_name() == "poisson"
                    else None
                ),
                quantile_alpha=(
                    self.quantile_alpha
                    if self._objective_name() == "quantile"
                    else None
                ),
                **ranking_sigma_kwargs,
            )
        else:
            assert training_rows is not None
            native_result = train_with_summary(
                rows=training_rows,
                targets=targets,
                learning_rate=self.learning_rate,
                max_depth=_effective_max_depth,
                row_subsample=self.row_subsample,
                col_subsample=self.col_subsample,
                min_validation_improvement=self.min_validation_improvement,
                seed=self.seed,
                deterministic=self.deterministic,
                rounds=self.n_estimators,
                early_stopping_rounds=self.early_stopping_rounds,
                min_data_in_leaf=self.min_data_in_leaf,
                lambda_l1=self.lambda_l1,
                lambda_l2=self.lambda_l2,
                min_child_hessian=self.min_child_hessian,
                sample_weights=validated_sample_weights,
                group_id=validated_group_id,
                min_split_gain=self.min_split_gain,
                validation_rows=validation_rows,
                validation_targets=validation_targets,
                validation_sample_weights=validated_eval_sample_weights,
                validation_group_id=validated_eval_group_id,
                validation_time_index=validated_eval_time_index,
                categorical_feature_index=effective_categorical_feature_index,
                categorical_feature_values=categorical_values,
                validation_categorical_feature_values=validation_categorical_values,
                training_policy=self.training_policy,
                store_node_stats=self.store_node_stats,
                categorical_smoothing=self.categorical_smoothing,
                categorical_min_samples_leaf=self.categorical_min_samples_leaf,
                categorical_time_aware=self.categorical_time_aware,
                time_index=validated_time_index,
                continuous_binning_strategy=self.continuous_binning_strategy,
                continuous_binning_max_bins=self.continuous_binning_max_bins,
                objective=self._objective_name(),
                monotone_constraints=self._resolve_monotone_constraints(feature_count),
                feature_weights=self._resolve_feature_weights(feature_count),
                interaction_constraints=self._resolve_interaction_constraints(feature_count),
                max_leaves=self.max_leaves,
                tree_growth=self.tree_growth,
                categorical_feature_indices=effective_categorical_indices if has_categorical else None,
                categorical_feature_values_list=categorical_values_list if has_categorical else None,
                validation_categorical_feature_values_list=validation_categorical_values_list if has_categorical else None,
                init_artifact_bytes=init_artifact_bytes,
                num_classes=getattr(self, '_num_classes_for_training', None),
                custom_objective_fn=_custom_objective_fn,
                custom_loss_fn=_custom_loss_fn,
                custom_metric_fn=_custom_metric_fn,
                max_cat_threshold=self.max_cat_threshold,
                morph_config=self._morph_config_,
                leaf_model=self.leaf_model,
                leaf_solver=self.leaf_solver,
                dro_radius=self.dro_radius,
                dro_metric=self.dro_metric,
                neutralization=self.neutralization,
                factor_neutralization_lambda=self.factor_neutralization_lambda,
                factor_penalty=self.factor_penalty,
                factor_exposure_values=factor_exposure_values,
                factor_exposure_row_count=(
                    factor_exposure_row_count if factor_exposure_values is not None else None
                ),
                factor_exposure_factor_count=(
                    factor_exposure_factor_count
                    if factor_exposure_values is not None
                    else None
                ),
                boosting_mode=self.boosting_mode,
                goss_top_rate=(
                    self.goss_top_rate if self.boosting_mode == "goss" else None
                ),
                goss_other_rate=(
                    self.goss_other_rate if self.boosting_mode == "goss" else None
                ),
                dart_drop_rate=(
                    self.dart_drop_rate if self.boosting_mode == "dart" else None
                ),
                dart_max_drop=(
                    self.dart_max_drop if self.boosting_mode == "dart" else None
                ),
                dart_normalize_type=(
                    self.dart_normalize_type if self.boosting_mode == "dart" else None
                ),
                dart_sample_type=(
                    self.dart_sample_type if self.boosting_mode == "dart" else None
                ),
                tweedie_variance_power=(
                    self.tweedie_variance_power
                    if self._objective_name() == "tweedie"
                    else None
                ),
                poisson_max_delta_step=(
                    self.poisson_max_delta_step
                    if self._objective_name() == "poisson"
                    else None
                ),
                quantile_alpha=(
                    self.quantile_alpha
                    if self._objective_name() == "quantile"
                    else None
                ),
                **ranking_sigma_kwargs,
            )

        self._apply_continuous_binning_metadata(native_result.continuous_binning_metadata)
        self._n_features_in = feature_count
        self._artifact_bytes = bytes(native_result.artifact_bytes)
        self._native_predictor_handle = self._build_native_predictor_handle(
            self._artifact_bytes
        )
        self._convert_predictor_thresholds_to_float()
        raw_mappings = getattr(native_result, "native_cat_mappings", None)
        if raw_mappings:
            self._native_cat_mappings_ = {
                int(k): {str(ck): int(cv) for ck, cv in v.items()}
                for k, v in raw_mappings.items()
            }
        else:
            self._native_cat_mappings_ = None
        summary = native_result.summary
        self.best_iteration_ = summary.best_validation_round
        self.best_score_ = (
            float(summary.best_validation_loss)
            if summary.best_validation_loss is not None
            else None
        )
        self.n_estimators_ = int(summary.rounds_completed)
        self.rounds_completed_ = int(summary.rounds_completed)
        self.stop_reason_ = (
            str(summary.stop_reason) if summary.stop_reason is not None else None
        )
        self.diagnostics_per_round_ = _diagnostics_to_dicts(
            getattr(summary, "diagnostics_per_round", None)
        )
        self.evals_result_ = self._build_evals_result(summary)
        total_fit_seconds = time.perf_counter() - fit_start
        self.fit_timing_ = {
            "input_adaptation_seconds": float(input_adaptation_seconds),
            "native_bridge_prepare_seconds": float(summary.bridge_prepare_seconds),
            "native_train_seconds": float(summary.native_train_seconds),
            "total_fit_seconds": float(total_fit_seconds),
        }
        self._record_fit_neutralization_contract()
        self._is_fitted = True
        self._record_post_fit_factor_exposure_diagnostics(
            X, transformed_factor_exposures
        )
        return self

    def _finalize_training_result(
        self,
        native_result: object,
        input_adaptation_seconds: float,
        feature_count: int | None = None,
        fit_X: object | None = None,
        transformed_factor_exposures: object | None = None,
    ) -> "GBMRegressor":
        self._apply_continuous_binning_metadata(native_result.continuous_binning_metadata)
        if feature_count is not None:
            self._n_features_in = feature_count
        self._artifact_bytes = bytes(native_result.artifact_bytes)
        self._native_predictor_handle = self._build_native_predictor_handle(
            self._artifact_bytes
        )
        self._convert_predictor_thresholds_to_float()
        raw_mappings = getattr(native_result, "native_cat_mappings", None)
        if raw_mappings:
            self._native_cat_mappings_ = {
                int(k): {str(ck): int(cv) for ck, cv in v.items()}
                for k, v in raw_mappings.items()
            }
        else:
            self._native_cat_mappings_ = None
        summary = native_result.summary
        self.best_iteration_ = summary.best_validation_round
        self.best_score_ = (
            float(summary.best_validation_loss)
            if summary.best_validation_loss is not None
            else None
        )
        self.n_estimators_ = int(summary.rounds_completed)
        self.rounds_completed_ = int(summary.rounds_completed)
        self.stop_reason_ = (
            str(summary.stop_reason) if summary.stop_reason is not None else None
        )
        self.diagnostics_per_round_ = _diagnostics_to_dicts(
            getattr(summary, "diagnostics_per_round", None)
        )
        self.evals_result_ = self._build_evals_result(summary)
        total_fit_seconds = time.perf_counter() - self._fit_start_time
        self.fit_timing_ = {
            "input_adaptation_seconds": float(input_adaptation_seconds),
            "native_bridge_prepare_seconds": float(summary.bridge_prepare_seconds),
            "native_train_seconds": float(summary.native_train_seconds),
            "total_fit_seconds": float(total_fit_seconds),
        }
        self._record_fit_neutralization_contract()
        self._is_fitted = True
        if fit_X is not None:
            self._record_post_fit_factor_exposure_diagnostics(
                fit_X, transformed_factor_exposures
            )
        return self

    def _fit_with_legacy_native_bridge(
        self,
        *,
        X: object,
        targets: list[float],
        dense_training_payload: tuple[list[float], int, int] | None,
        training_rows: list[list[float]] | None,
        feature_count: int,
        categorical_feature_index: int | None,
        categorical_values: list[str] | None,
        time_index: list[int] | None,
        eval_set: tuple[object, object] | None,
        input_adaptation_seconds: float,
        factor_exposure_values: list[float] | None,
        factor_exposure_row_count: int,
        factor_exposure_factor_count: int,
        transformed_factor_exposures: object | None,
    ) -> "GBMRegressor":
        if eval_set is not None:
            raise RuntimeError(
                "eval_set requires a native alloygbm build with training summary support"
            )
        if self.training_mode == "morph":
            raise RuntimeError(
                "training_mode='morph' requires a native alloygbm build with training summary support"
            )
        if (
            self.min_data_in_leaf != 1
            or self.lambda_l1 != 0.0
            or self.lambda_l2 != 0.0
            or self.min_child_hessian != 0.0
        ):
            raise RuntimeError(
                "explicit regularization and min_data_in_leaf require a native alloygbm build with training summary support"
            )

        fit_start = time.perf_counter()
        ranking_sigma_kwargs = (
            {
                "ranking_sigma": self.ranking_sigma,
                "lambdarank_truncation_level": self.lambdarank_truncation_level,
            }
            if hasattr(self, "ranking_sigma")
            else {}
        )
        native_training_rows: object | None = None
        active_dense_training_payload = dense_training_payload
        rows = training_rows
        if active_dense_training_payload is None:
            assert rows is not None
            self._uses_continuous_binning = not self._rows_are_pre_binned(rows)
            if self._uses_continuous_binning:
                if self.continuous_binning_strategy == "linear":
                    mins, maxs = self._derive_continuous_feature_bounds(rows)
                    self._continuous_feature_mins = mins
                    self._continuous_feature_maxs = maxs
                    if _linear_tail_rank_enabled_from_env():
                        core_span_ratio_threshold = (
                            _linear_tail_core_span_ratio_threshold_from_env()
                        )
                        rank_flags, sorted_values = (
                            self._derive_continuous_feature_tail_rank_plan(
                                rows, core_span_ratio_threshold
                            )
                        )
                        self._continuous_feature_linear_rank_flags = rank_flags
                        if any(rank_flags):
                            self._continuous_feature_sorted_values = sorted_values
                            native_training_rows = (
                                self._quantize_rows_linear_with_selective_rank(
                                    rows, mins, maxs, rank_flags, sorted_values,
                                    max_bins=self.continuous_binning_max_bins,
                                )
                            )
                        else:
                            self._continuous_feature_sorted_values = None
                            native_training_rows = self._quantize_rows_linear(
                                rows, mins, maxs,
                                max_bins=self.continuous_binning_max_bins,
                            )
                    else:
                        self._continuous_feature_sorted_values = None
                        self._continuous_feature_linear_rank_flags = None
                        native_training_rows = self._quantize_rows_linear(
                            rows, mins, maxs,
                            max_bins=self.continuous_binning_max_bins,
                        )
                    self._continuous_feature_quantile_cuts = None
                elif self.continuous_binning_strategy == "rank":
                    sorted_values = self._derive_continuous_feature_sorted_values(rows)
                    self._continuous_feature_sorted_values = sorted_values
                    self._continuous_feature_mins = None
                    self._continuous_feature_maxs = None
                    self._continuous_feature_quantile_cuts = None
                    self._continuous_feature_linear_rank_flags = None
                    native_training_rows = self._quantize_rows_rank(
                        rows, sorted_values,
                        max_bins=self.continuous_binning_max_bins,
                    )
                else:
                    quantile_cuts = self._derive_continuous_feature_quantile_cuts(
                        rows, self.continuous_binning_max_bins
                    )
                    self._continuous_feature_quantile_cuts = quantile_cuts
                    self._continuous_feature_sorted_values = None
                    self._continuous_feature_mins = None
                    self._continuous_feature_maxs = None
                    self._continuous_feature_linear_rank_flags = None
                    native_training_rows = self._quantize_rows_quantile(
                        rows, quantile_cuts,
                        max_bins=self.continuous_binning_max_bins,
                    )
            else:
                self._continuous_feature_mins = None
                self._continuous_feature_maxs = None
                self._continuous_feature_sorted_values = None
                self._continuous_feature_quantile_cuts = None
                self._continuous_feature_linear_rank_flags = None
                native_training_rows = rows
        else:
            flat_values, row_count, dense_feature_count = active_dense_training_payload
            if self._check_pre_binned_integers(flat_values):
                self._uses_continuous_binning = False
                self._continuous_feature_mins = None
                self._continuous_feature_maxs = None
                self._continuous_feature_sorted_values = None
                self._continuous_feature_quantile_cuts = None
                self._continuous_feature_linear_rank_flags = None
            else:
                self._uses_continuous_binning = True
                rows = self._validate_rows(X)
                if self.continuous_binning_strategy == "linear":
                    mins, maxs = self._derive_dense_feature_bounds(
                        flat_values, row_count, dense_feature_count
                    )
                    self._continuous_feature_mins = mins
                    self._continuous_feature_maxs = maxs
                    if _linear_tail_rank_enabled_from_env():
                        core_span_ratio_threshold = (
                            _linear_tail_core_span_ratio_threshold_from_env()
                        )
                        rank_flags, sorted_values = (
                            self._derive_continuous_feature_tail_rank_plan(
                                rows, core_span_ratio_threshold
                            )
                        )
                        self._continuous_feature_linear_rank_flags = rank_flags
                        if any(rank_flags):
                            self._continuous_feature_sorted_values = sorted_values
                            active_dense_training_payload = (
                                self._quantize_dense_values_linear_with_selective_rank(
                                    flat_values,
                                    row_count,
                                    dense_feature_count,
                                    mins,
                                    maxs,
                                    rank_flags,
                                    sorted_values,
                                    max_bins=self.continuous_binning_max_bins,
                                ),
                                row_count,
                                dense_feature_count,
                            )
                        else:
                            self._continuous_feature_sorted_values = None
                            active_dense_training_payload = (
                                self._quantize_dense_values_linear(
                                    flat_values, row_count, dense_feature_count, mins, maxs,
                                    max_bins=self.continuous_binning_max_bins,
                                ),
                                row_count,
                                dense_feature_count,
                            )
                    else:
                        self._continuous_feature_sorted_values = None
                        self._continuous_feature_linear_rank_flags = None
                        active_dense_training_payload = (
                            self._quantize_dense_values_linear(
                                flat_values, row_count, dense_feature_count, mins, maxs,
                                max_bins=self.continuous_binning_max_bins,
                            ),
                            row_count,
                            dense_feature_count,
                        )
                    self._continuous_feature_quantile_cuts = None
                elif self.continuous_binning_strategy == "rank":
                    sorted_values = self._derive_dense_sorted_feature_values(
                        flat_values, row_count, dense_feature_count
                    )
                    self._continuous_feature_sorted_values = sorted_values
                    self._continuous_feature_mins = None
                    self._continuous_feature_maxs = None
                    self._continuous_feature_quantile_cuts = None
                    self._continuous_feature_linear_rank_flags = None
                    active_dense_training_payload = (
                        self._quantize_dense_values_rank(
                            flat_values, row_count, dense_feature_count, sorted_values,
                            max_bins=self.continuous_binning_max_bins,
                        ),
                        row_count,
                        dense_feature_count,
                    )
                else:
                    quantile_cuts = self._derive_dense_feature_quantile_cuts(
                        flat_values,
                        row_count,
                        dense_feature_count,
                        self.continuous_binning_max_bins,
                    )
                    self._continuous_feature_quantile_cuts = quantile_cuts
                    self._continuous_feature_sorted_values = None
                    self._continuous_feature_mins = None
                    self._continuous_feature_maxs = None
                    self._continuous_feature_linear_rank_flags = None
                    active_dense_training_payload = (
                        self._quantize_dense_values_quantile(
                            flat_values, row_count, dense_feature_count, quantile_cuts,
                            max_bins=self.continuous_binning_max_bins,
                        ),
                        row_count,
                        dense_feature_count,
                    )

        if active_dense_training_payload is not None:
            train_regression_artifact_dense = _base._load_native_train_regression_artifact_dense()
            artifact_bytes = train_regression_artifact_dense(
                values=active_dense_training_payload[0],
                row_count=active_dense_training_payload[1],
                feature_count=active_dense_training_payload[2],
                targets=targets,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                row_subsample=self.row_subsample,
                col_subsample=self.col_subsample,
                min_validation_improvement=self.min_validation_improvement,
                seed=self.seed,
                deterministic=self.deterministic,
                rounds=self.n_estimators,
                early_stopping_rounds=self.early_stopping_rounds,
                categorical_feature_index=categorical_feature_index,
                categorical_feature_values=categorical_values,
                training_policy=self.training_policy,
                store_node_stats=self.store_node_stats,
                categorical_smoothing=self.categorical_smoothing,
                categorical_min_samples_leaf=self.categorical_min_samples_leaf,
                categorical_time_aware=self.categorical_time_aware,
                time_index=time_index,
                continuous_binning_strategy=self.continuous_binning_strategy,
                continuous_binning_max_bins=self.continuous_binning_max_bins,
                objective=self._objective_name(),
                leaf_model=self.leaf_model,
                leaf_solver=self.leaf_solver,
                dro_radius=self.dro_radius,
                dro_metric=self.dro_metric,
                neutralization=self.neutralization,
                factor_neutralization_lambda=self.factor_neutralization_lambda,
                factor_penalty=self.factor_penalty,
                factor_exposure_values=factor_exposure_values,
                factor_exposure_row_count=(
                    factor_exposure_row_count if factor_exposure_values is not None else None
                ),
                factor_exposure_factor_count=(
                    factor_exposure_factor_count
                    if factor_exposure_values is not None
                    else None
                ),
                boosting_mode=self.boosting_mode,
                goss_top_rate=(
                    self.goss_top_rate if self.boosting_mode == "goss" else None
                ),
                goss_other_rate=(
                    self.goss_other_rate if self.boosting_mode == "goss" else None
                ),
                dart_drop_rate=(
                    self.dart_drop_rate if self.boosting_mode == "dart" else None
                ),
                dart_max_drop=(
                    self.dart_max_drop if self.boosting_mode == "dart" else None
                ),
                dart_normalize_type=(
                    self.dart_normalize_type if self.boosting_mode == "dart" else None
                ),
                dart_sample_type=(
                    self.dart_sample_type if self.boosting_mode == "dart" else None
                ),
                tweedie_variance_power=(
                    self.tweedie_variance_power
                    if self._objective_name() == "tweedie"
                    else None
                ),
                poisson_max_delta_step=(
                    self.poisson_max_delta_step
                    if self._objective_name() == "poisson"
                    else None
                ),
                quantile_alpha=(
                    self.quantile_alpha
                    if self._objective_name() == "quantile"
                    else None
                ),
                **ranking_sigma_kwargs,
            )
        else:
            train_regression_artifact = _base._load_native_train_regression_artifact()
            artifact_bytes = train_regression_artifact(
                rows=native_training_rows,
                targets=targets,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                row_subsample=self.row_subsample,
                col_subsample=self.col_subsample,
                min_validation_improvement=self.min_validation_improvement,
                seed=self.seed,
                deterministic=self.deterministic,
                rounds=self.n_estimators,
                early_stopping_rounds=self.early_stopping_rounds,
                categorical_feature_index=categorical_feature_index,
                categorical_feature_values=categorical_values,
                training_policy=self.training_policy,
                store_node_stats=self.store_node_stats,
                categorical_smoothing=self.categorical_smoothing,
                categorical_min_samples_leaf=self.categorical_min_samples_leaf,
                categorical_time_aware=self.categorical_time_aware,
                time_index=time_index,
                continuous_binning_strategy=self.continuous_binning_strategy,
                continuous_binning_max_bins=self.continuous_binning_max_bins,
                objective=self._objective_name(),
                leaf_model=self.leaf_model,
                leaf_solver=self.leaf_solver,
                dro_radius=self.dro_radius,
                dro_metric=self.dro_metric,
                neutralization=self.neutralization,
                factor_neutralization_lambda=self.factor_neutralization_lambda,
                factor_penalty=self.factor_penalty,
                factor_exposure_values=factor_exposure_values,
                factor_exposure_row_count=(
                    factor_exposure_row_count if factor_exposure_values is not None else None
                ),
                factor_exposure_factor_count=(
                    factor_exposure_factor_count
                    if factor_exposure_values is not None
                    else None
                ),
                boosting_mode=self.boosting_mode,
                goss_top_rate=(
                    self.goss_top_rate if self.boosting_mode == "goss" else None
                ),
                goss_other_rate=(
                    self.goss_other_rate if self.boosting_mode == "goss" else None
                ),
                dart_drop_rate=(
                    self.dart_drop_rate if self.boosting_mode == "dart" else None
                ),
                dart_max_drop=(
                    self.dart_max_drop if self.boosting_mode == "dart" else None
                ),
                dart_normalize_type=(
                    self.dart_normalize_type if self.boosting_mode == "dart" else None
                ),
                dart_sample_type=(
                    self.dart_sample_type if self.boosting_mode == "dart" else None
                ),
                tweedie_variance_power=(
                    self.tweedie_variance_power
                    if self._objective_name() == "tweedie"
                    else None
                ),
                poisson_max_delta_step=(
                    self.poisson_max_delta_step
                    if self._objective_name() == "poisson"
                    else None
                ),
                quantile_alpha=(
                    self.quantile_alpha
                    if self._objective_name() == "quantile"
                    else None
                ),
                **ranking_sigma_kwargs,
            )

        self._n_features_in = feature_count
        self._artifact_bytes = bytes(artifact_bytes)
        self._native_predictor_handle = self._build_native_predictor_handle(
            self._artifact_bytes
        )
        self._convert_predictor_thresholds_to_float()
        self.best_iteration_ = None
        self.best_score_ = None
        self.n_estimators_ = self.n_estimators
        loss_metric = self._loss_metric_name()
        self.evals_result_ = {"train": {"rmse": [], loss_metric: []}}
        total_fit_seconds = time.perf_counter() - fit_start
        self.fit_timing_ = {
            "input_adaptation_seconds": float(input_adaptation_seconds),
            "native_bridge_prepare_seconds": 0.0,
            "native_train_seconds": float(total_fit_seconds),
            "total_fit_seconds": float(total_fit_seconds),
        }
        self._record_fit_neutralization_contract()
        self._is_fitted = True
        self._record_post_fit_factor_exposure_diagnostics(
            X, transformed_factor_exposures
        )
        return self

    @staticmethod
    def _prediction_array(values: object) -> np.ndarray:
        if isinstance(values, np.ndarray):
            return values.reshape(-1)
        return np.asarray(values, dtype=np.float32).reshape(-1)

    def predict(self, X: object) -> np.ndarray:
        """Predict using the fitted native artifact."""
        if not self._is_fitted:
            raise RuntimeError("GBMRegressor must be fit before predict")
        if self._artifact_bytes is None:
            raise RuntimeError("GBMRegressor native artifact is not available")
        # Lazily reconstruct the native predictor after pickle roundtrip.
        if self._native_predictor_handle is None and getattr(
            self, "_predictor_needs_rebuild", False
        ):
            self._predictor_needs_rebuild = False
            self._native_predictor_handle = self._build_native_predictor_handle(
                self._artifact_bytes
            )
            self._convert_predictor_thresholds_to_float()
        # Convert native categorical columns to integer IDs before prediction.
        X = self._apply_native_cat_mappings_for_predict(X)
        rows: object
        # Fast path: float thresholds + zero-copy numpy — no data copying
        if self._float_thresholds_converted:
            try:
                import numpy as _np

                candidate = self._native_matrix_fast_path_candidate(X)
                if candidate is not None:
                    arr = _np.ascontiguousarray(candidate, dtype=_np.float32)
                    if arr.shape[1] != self._n_features_in:
                        raise ValueError(
                            f"X feature count {arr.shape[1]} does not match fitted "
                            f"feature count {self._n_features_in}"
                        )
                    predict_numpy = getattr(
                        self._native_predictor_handle, "predict_numpy", None
                    )
                    if callable(predict_numpy):
                        return self._prediction_array(predict_numpy(arr))
            except ImportError:
                pass

        if self._uses_continuous_binning:
            dense_payload = self._native_matrix_flat_payload(X)
            if dense_payload is not None:
                flat_values, row_count, feature_count = dense_payload
                if feature_count != self._n_features_in:
                    raise ValueError(
                        f"X feature count {feature_count} does not match fitted feature count "
                        f"{self._n_features_in}"
                    )
                # Float threshold fallback (when bytes path unavailable)
                if self._float_thresholds_converted:
                    predict_dense = getattr(
                        self._native_predictor_handle, "predict_dense", None
                    )
                    if callable(predict_dense):
                        return self._prediction_array(
                            predict_dense(flat_values, row_count, feature_count)
                        )

                # Use Rust-side fused quantize+predict when native handle is available
                if self.continuous_binning_strategy == "linear":
                    mins, maxs = self._require_continuous_feature_bounds()
                    rank_flags = self._continuous_feature_linear_rank_flags
                    if rank_flags is not None and any(rank_flags):
                        sorted_values = self._require_continuous_feature_sorted_values()
                        predict_fn = getattr(
                            self._native_predictor_handle,
                            "predict_dense_quantized_linear_rank",
                            None,
                        )
                        if callable(predict_fn):
                            return self._prediction_array(
                                predict_fn(
                                    flat_values,
                                    row_count,
                                    feature_count,
                                    list(mins),
                                    list(maxs),
                                    list(rank_flags),
                                    [list(sv) for sv in sorted_values],
                                    _max_data_bin_for_max_bins(
                                        self.continuous_binning_max_bins
                                    ),
                                )
                            )
                        # Fallback to Python quantization
                        dense_payload = (
                            self._quantize_dense_values_linear_with_selective_rank(
                                flat_values,
                                row_count,
                                feature_count,
                                mins,
                                maxs,
                                rank_flags,
                                sorted_values,
                                max_bins=self.continuous_binning_max_bins,
                            ),
                            row_count,
                            feature_count,
                        )
                    else:
                        predict_fn = getattr(
                            self._native_predictor_handle,
                            "predict_dense_quantized_linear",
                            None,
                        )
                        if callable(predict_fn):
                            return self._prediction_array(
                                predict_fn(
                                    flat_values,
                                    row_count,
                                    feature_count,
                                    list(mins),
                                    list(maxs),
                                    _max_data_bin_for_max_bins(
                                        self.continuous_binning_max_bins
                                    ),
                                )
                            )
                        # Fallback to Python quantization
                        dense_payload = (
                            self._quantize_dense_values_linear(
                                flat_values,
                                row_count,
                                feature_count,
                                mins,
                                maxs,
                                max_bins=self.continuous_binning_max_bins,
                            ),
                            row_count,
                            feature_count,
                        )
                elif self.continuous_binning_strategy == "rank":
                    sorted_values = self._require_continuous_feature_sorted_values()
                    dense_payload = (
                        self._quantize_dense_values_rank(
                            flat_values,
                            row_count,
                            feature_count,
                            sorted_values,
                            max_bins=self.continuous_binning_max_bins,
                        ),
                        row_count,
                        feature_count,
                    )
                else:
                    quantile_cuts = self._require_continuous_feature_quantile_cuts()
                    dense_payload = (
                        self._quantize_dense_values_quantile(
                            flat_values,
                            row_count,
                            feature_count,
                            quantile_cuts,
                            max_bins=self.continuous_binning_max_bins,
                        ),
                        row_count,
                        feature_count,
                    )
                flat_values, row_count, feature_count = dense_payload
                predict_dense = getattr(self._native_predictor_handle, "predict_dense", None)
                if callable(predict_dense):
                    return self._prediction_array(
                        predict_dense(flat_values, row_count, feature_count)
                    )
                predictor_predict_batch_dense = _base._load_native_predictor_predict_batch_dense()
                return self._prediction_array(
                    predictor_predict_batch_dense(
                        self._artifact_bytes, flat_values, row_count, feature_count
                    )
                )
            if self._float_thresholds_converted:
                # Float thresholds: send raw (unquantized) rows directly
                validated_rows = self._validate_rows(X)
                if len(validated_rows[0]) != self._n_features_in:
                    raise ValueError(
                        f"X feature count {len(validated_rows[0])} does not match fitted "
                        f"feature count {self._n_features_in}"
                    )
                rows = validated_rows
            else:
                quantized_rows = self._quantize_rows_for_prediction(self._validate_rows(X))
                if len(quantized_rows[0]) != self._n_features_in:
                    raise ValueError(
                        f"X feature count {len(quantized_rows[0])} does not match fitted "
                        f"feature count {self._n_features_in}"
                    )
                rows = quantized_rows
        else:
            dense_payload = self._native_matrix_flat_payload(X)
            if dense_payload is not None:
                _, _, feature_count = dense_payload
                if feature_count != self._n_features_in:
                    raise ValueError(
                        f"X feature count {feature_count} does not match fitted feature count "
                        f"{self._n_features_in}"
                    )
                rows = dense_payload
            else:
                validated_rows = self._validate_rows(X)
                if len(validated_rows[0]) != self._n_features_in:
                    raise ValueError(
                        f"X feature count {len(validated_rows[0])} does not match fitted feature count "
                        f"{self._n_features_in}"
                    )
                rows = validated_rows
        if self._native_predictor_handle is not None:
            if isinstance(rows, tuple):
                predict_dense = getattr(self._native_predictor_handle, "predict_dense", None)
                if callable(predict_dense):
                    try:
                        flat_values, row_count, feature_count = rows
                        return self._prediction_array(
                            predict_dense(flat_values, row_count, feature_count)
                        )
                    except RuntimeError:
                        self._native_predictor_handle = None
            predict_batch = getattr(self._native_predictor_handle, "predict_batch", None)
            if callable(predict_batch) and not isinstance(rows, tuple):
                try:
                    return self._prediction_array(predict_batch(rows))
                except RuntimeError:
                    self._native_predictor_handle = None
        if isinstance(rows, tuple):
            flat_values, row_count, feature_count = rows
            predictor_predict_batch_canonical_dense = (
                _base._load_native_predictor_predict_batch_canonical_dense()
            )
            return self._prediction_array(
                predictor_predict_batch_canonical_dense(
                    self._artifact_bytes,
                    flat_values,
                    row_count=row_count,
                    feature_count=feature_count,
                )
            )
        predictor_predict_batch_canonical = _base._load_native_predictor_predict_batch_canonical()
        return self._prediction_array(
            predictor_predict_batch_canonical(self._artifact_bytes, rows)
        )

    def score(self, X: object, y: object, sample_weight: object = None) -> float:
        """Return R² score for the given test data and labels."""
        from alloygbm.evaluation import r2_score

        predictions = self.predict(X)
        targets = self._validate_targets(y)
        return float(r2_score(targets, predictions))

    def __sklearn_tags__(self):
        if not hasattr(super(), "__sklearn_tags__"):
            return {
                "non_deterministic": not self.deterministic,
                "requires_y": True,
                "allow_nan": True,
                "X_types": ["2darray"],
            }
        tags = super().__sklearn_tags__()
        # sklearn >= 1.6 returns a Tags dataclass
        if hasattr(tags, "non_deterministic"):
            tags.non_deterministic = not self.deterministic
        if hasattr(tags, "input_tags") and hasattr(tags.input_tags, "allow_nan"):
            tags.input_tags.allow_nan = True
        return tags

    def _more_tags(self):
        return {"allow_nan": True, "requires_y": True}


# Inject GBMRegressor into _validation's namespace so its static-method bodies
# (which reference GBMRegressor by name) can resolve the class.  This must run
# after the class statement above completes; by this point _validation is already
# imported (it was imported at the top of this file), so updating its globals is
# safe and avoids any circular-import.
import alloygbm._regressor._validation as _validation_module
_validation_module.GBMRegressor = GBMRegressor
del _validation_module

# Inject GBMRegressor into _quantization's namespace so its static-method bodies
# (which reference GBMRegressor by name) can resolve the class.
import alloygbm._regressor._quantization as _quantization_module
_quantization_module.GBMRegressor = GBMRegressor
del _quantization_module

# Advertise the public import path. `alloygbm.regressor` is the stable
# compatibility surface (the back-compat shim that re-exports this class);
# `alloygbm._regressor._core` is the private implementation module. Without
# this assignment, `GBMRegressor.__module__` would expose the private path,
# repr would print `<class 'alloygbm._regressor._core.GBMRegressor'>`, and
# newly-created pickles would store the private path — tying the pickle
# format to the internal package layout. The shim re-export means
# `alloygbm.regressor.GBMRegressor` always resolves to this class object, so
# pickles using the public path load correctly.
GBMRegressor.__module__ = "alloygbm.regressor"
