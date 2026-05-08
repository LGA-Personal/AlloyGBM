#!/usr/bin/env python3
"""Run cross-model benchmark comparisons for AlloyGBM and peer GBM libraries."""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    log_loss as sklearn_log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

AVAILABLE_SCENARIOS = [
    # Regression
    "california_housing",
    "bike_sharing",
    "dense_numeric",
    "panel_time_series",
    "histogram_stress",
    "dow_jones_financial",
    "abalone_regression",
    "synthetic_categorical",
    # Binary classification
    "breast_cancer",
    "adult_income",
    "synthetic_classification",
    # Multiclass classification
    "wine_multiclass",
    "digits_multiclass",
    "synthetic_multiclass",
    # Ranking
    "synthetic_ranking",
    "california_ranking",
    # "news_ranking",  # Uncomment once prepare.py is implemented
]


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    learning_rate: float
    max_depth: int
    rounds: int


DEFAULT_PROFILES = [
    BenchmarkProfile("shallow_high_lr", learning_rate=0.20, max_depth=4, rounds=200),
    BenchmarkProfile("mid_balanced", learning_rate=0.05, max_depth=6, rounds=1200),
    BenchmarkProfile("deep_low_lr", learning_rate=0.01, max_depth=8, rounds=5000),
]

ULTRA_PROFILE = BenchmarkProfile(
    "ultra_low_lr", learning_rate=0.005, max_depth=8, rounds=10000
)

REQUIRED_ALLOY_INIT_PARAMS = (
    "learning_rate",
    "max_depth",
    "n_estimators",
    "row_subsample",
    "col_subsample",
)
VALID_ALLOY_CONTINUOUS_BINNING_STRATEGIES = ("linear", "rank", "quantile")


@dataclass
class BenchmarkRecord:
    scenario: str
    task_type: str
    profile_name: str
    profile_index: int
    run_index: int
    seed: int
    learning_rate: float
    max_depth: int
    rounds: int
    model: str
    train_rows: int
    test_rows: int
    n_features: int
    input_adaptation_seconds: float
    native_bridge_prepare_seconds: float
    native_train_seconds: float
    fit_seconds: float
    predict_seconds: float
    rmse: float
    mae: float
    r2: float
    accuracy: float
    log_loss_val: float
    auc: float
    ndcg_5: float
    ndcg_10: float
    ndcg_full: float
    status: str
    error: str


def _verify_alloygbm_runtime_contract(
    gbm_regressor_cls: type, native_module: object
) -> tuple[str, ...]:
    signature = inspect.signature(gbm_regressor_cls.__init__)
    parameter_names = tuple(signature.parameters.keys())
    missing = sorted(
        required
        for required in REQUIRED_ALLOY_INIT_PARAMS
        if required not in signature.parameters
    )
    if missing:
        raise RuntimeError(
            "loaded alloygbm.GBMRegressor is not benchmark-compatible; "
            f"missing required __init__ parameters: {', '.join(missing)}"
        )
    if not hasattr(native_module, "train_regression_artifact"):
        raise RuntimeError(
            "loaded alloygbm native extension is not benchmark-compatible; "
            "missing train_regression_artifact"
        )
    return parameter_names


def _load_alloygbm_runtime() -> tuple[type, dict[str, object]]:
    try:
        import alloygbm
        import alloygbm._alloygbm as native_module
        from alloygbm import GBMRegressor
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "failed to import benchmark runtime package 'alloygbm'; "
            "install a current local alloygbm build before running benchmarks"
        ) from exc

    init_parameters = _verify_alloygbm_runtime_contract(GBMRegressor, native_module)
    runtime = {
        "module_path": str(Path(alloygbm.__file__).resolve()),
        "native_module_path": str(Path(native_module.__file__).resolve()),
        "init_parameters": list(init_parameters),
    }
    return GBMRegressor, runtime


def _load_optional_catboost_regressor() -> tuple[type | None, dict[str, object]]:
    try:
        import catboost
        from catboost import CatBoostRegressor
    except Exception as exc:  # noqa: BLE001
        return None, {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return CatBoostRegressor, {
        "available": True,
        "version": getattr(catboost, "__version__", "unknown"),
    }


def _load_alloygbm_classifier_runtime() -> tuple[type, dict[str, object]]:
    try:
        from alloygbm import GBMClassifier
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "failed to import GBMClassifier from alloygbm"
        ) from exc
    return GBMClassifier, {"available": True}


def _load_alloygbm_ranker_runtime() -> tuple[type, dict[str, object]]:
    try:
        from alloygbm import GBMRanker
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "failed to import GBMRanker from alloygbm"
        ) from exc
    return GBMRanker, {"available": True}


def _load_optional_catboost_classifier() -> tuple[type | None, dict[str, object]]:
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier, {"available": True}
    except Exception:  # noqa: BLE001
        return None, {"available": False}


def _group_ids_to_sizes(group_ids: np.ndarray) -> np.ndarray:
    """Convert sorted per-row group IDs to group size array for LightGBM."""
    changes = np.where(np.diff(group_ids) != 0)[0] + 1
    boundaries = np.concatenate([[0], changes, [len(group_ids)]])
    return np.diff(boundaries)


class _LGBMRankerAdapter:
    """Wraps LGBMRanker to accept per-row group IDs like AlloyGBM."""

    def __init__(self, **kwargs: object) -> None:
        from lightgbm import LGBMRanker
        self._model = LGBMRanker(**kwargs)

    def fit(self, X: object, y: object, group: object = None) -> "_LGBMRankerAdapter":
        group_sizes = _group_ids_to_sizes(np.asarray(group))
        self._model.fit(X, y, group=group_sizes)
        return self

    def predict(self, X: object) -> object:
        return self._model.predict(X)


class _XGBRankerAdapter:
    """Wraps XGBRanker to accept per-row group IDs like AlloyGBM."""

    def __init__(self, **kwargs: object) -> None:
        from xgboost import XGBRanker
        self._model = XGBRanker(**kwargs)

    def fit(self, X: object, y: object, group: object = None) -> "_XGBRankerAdapter":
        self._model.fit(X, y, qid=group)
        return self

    def predict(self, X: object) -> object:
        return self._model.predict(X)


class _CatBoostRankerAdapter:
    """Wraps CatBoost with YetiRank for ranking."""

    def __init__(self, **kwargs: object) -> None:
        from catboost import CatBoost
        self._model = CatBoost({**kwargs, "loss_function": "YetiRank"})

    def fit(self, X: object, y: object, group: object = None) -> "_CatBoostRankerAdapter":
        from catboost import Pool
        train_pool = Pool(data=X, label=y, group_id=group)
        self._model.fit(train_pool)
        return self

    def predict(self, X: object) -> object:
        return self._model.predict(X)


def _prepare_dataset(
    repo_root: Path,
    scenario: str,
    force_prepare: bool,
    prepared_path: Path,
    manifest_kind: str,
) -> None:
    if prepared_path.exists() and not force_prepare:
        return

    scenario_script = repo_root / "benchmarks" / scenario / "prepare.py"
    command = [sys.executable, "-B", str(scenario_script)]
    if force_prepare and manifest_kind == "uci_download":
        command.append("--force-download")
    subprocess.run(command, cwd=repo_root, check=True)


def _load_manifest(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        return yaml.safe_load(manifest_file)


def _load_dataset(
    repo_root: Path, scenario: str, force_prepare: bool
) -> tuple[pd.DataFrame, str, str, str | None]:
    manifest_path = repo_root / "benchmarks" / scenario / "manifest.yaml"
    manifest = _load_manifest(manifest_path)
    target_column = str(manifest["prepared"]["target"])
    prepared_file = str(manifest["prepared"]["filename"])
    manifest_kind = str(manifest.get("kind", ""))
    task_type = str(manifest.get("task_type", "regression"))
    group_column = manifest["prepared"].get("group_column")
    if group_column is not None:
        group_column = str(group_column)
    prepared_path = repo_root / "benchmarks" / "data" / scenario / "prepared" / prepared_file

    _prepare_dataset(repo_root, scenario, force_prepare, prepared_path, manifest_kind)
    if not prepared_path.exists():
        raise FileNotFoundError(f"prepared dataset missing: {prepared_path}")

    frame = pd.read_csv(prepared_path)
    if target_column not in frame.columns:
        raise ValueError(f"target column '{target_column}' missing in {prepared_path}")
    return frame, target_column, task_type, group_column


def _split_by_timestamp(frame: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Split on unique timestamps so no timestamp appears in both train and test.
    ordered = frame.sort_values(["timestamp", "group_id"], kind="mergesort")
    unique_timestamps = sorted({str(value) for value in ordered["timestamp"]})
    if len(unique_timestamps) < 2:
        raise ValueError("need at least 2 unique timestamps for split")

    train_timestamp_count = max(1, int(len(unique_timestamps) * (1.0 - test_size)))
    if train_timestamp_count >= len(unique_timestamps):
        train_timestamp_count = len(unique_timestamps) - 1
    train_timestamps = set(unique_timestamps[:train_timestamp_count])

    train = ordered[ordered["timestamp"].astype(str).isin(train_timestamps)]
    test = ordered[~ordered["timestamp"].astype(str).isin(train_timestamps)]
    if train.empty or test.empty:
        raise ValueError("timestamp split produced empty train/test partition")
    return train, test


def _split_dataset(
    scenario: str,
    frame: pd.DataFrame,
    target_column: str,
    seed: int,
    test_size: float,
    task_type: str = "regression",
    group_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    # Leakage check: skip group/timestamp/target columns.
    skip_columns = {target_column, "group_id", "timestamp"}
    if group_column is not None:
        skip_columns.add(group_column)

    target_equivalent_features: list[str] = []
    target_series = pd.to_numeric(frame[target_column], errors="coerce")
    for column in frame.columns:
        if column in skip_columns:
            continue
        feature_series = pd.to_numeric(frame[column], errors="coerce")
        comparable = feature_series.notna() & target_series.notna()
        if comparable.any() and (feature_series[comparable] == target_series[comparable]).all():
            target_equivalent_features.append(column)

    if target_equivalent_features:
        raise ValueError(
            f"{scenario}: target-equivalent features detected: "
            + ", ".join(sorted(target_equivalent_features))
        )

    # Ranking: group-aware split keeping entire queries together.
    if task_type == "ranking" and group_column is not None:
        group_arr = frame[group_column].to_numpy()
        unique_groups = np.unique(group_arr)
        rng = np.random.RandomState(seed)
        rng.shuffle(unique_groups)
        split_point = max(1, int(len(unique_groups) * (1.0 - test_size)))
        train_groups = set(unique_groups[:split_point].tolist())

        train_mask = frame[group_column].isin(train_groups)
        test_mask = ~train_mask

        # Sort by group to ensure contiguous group IDs for ranking adapters.
        train_frame = frame.loc[train_mask].sort_values(group_column)
        test_frame = frame.loc[test_mask].sort_values(group_column)

        drop_cols = [target_column, group_column]
        x_train = train_frame.drop(columns=drop_cols, errors="ignore")
        x_test = test_frame.drop(columns=drop_cols, errors="ignore")
        y_train = train_frame[target_column]
        y_test = test_frame[target_column]
        g_train = train_frame[group_column].to_numpy(dtype=np.uint32)
        g_test = test_frame[group_column].to_numpy(dtype=np.uint32)

        x_train = x_train.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        x_test = x_test.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        y_train = pd.to_numeric(y_train, errors="coerce")
        y_test = pd.to_numeric(y_test, errors="coerce")

        if len(x_train) == 0 or len(x_test) == 0:
            raise ValueError(f"{scenario}: no rows left after ranking split")

        return (
            x_train.to_numpy(dtype=float),
            x_test.to_numpy(dtype=float),
            y_train.to_numpy(dtype=float),
            y_test.to_numpy(dtype=float),
            g_train,
            g_test,
        )

    # Classification (binary and multiclass): stratified split.
    if task_type in ("classification", "multiclass_classification"):
        feature_frame = frame.drop(
            columns=[target_column, "group_id", "timestamp"], errors="ignore"
        ).copy()
        feature_frame = feature_frame.apply(
            pd.to_numeric, errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        target_vals = frame[target_column]

        x_train, x_test, y_train, y_test = train_test_split(
            feature_frame,
            target_vals,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=target_vals,
        )

        y_train = pd.to_numeric(y_train, errors="coerce")
        y_test = pd.to_numeric(y_test, errors="coerce")

        train_mask = x_train.notna().all(axis=1) & y_train.notna()
        test_mask = x_test.notna().all(axis=1) & y_test.notna()
        x_train = x_train[train_mask]
        y_train = y_train[train_mask]
        x_test = x_test[test_mask]
        y_test = y_test[test_mask]

        if len(x_train) == 0 or len(x_test) == 0:
            raise ValueError(f"{scenario}: no rows left after numeric/finite filtering")

        return (
            x_train.to_numpy(dtype=float),
            x_test.to_numpy(dtype=float),
            y_train.to_numpy(dtype=float),
            y_test.to_numpy(dtype=float),
            None,
            None,
        )

    # Regression: existing logic (timestamp-based or random split).
    feature_frame = frame.drop(columns=[target_column]).copy()
    if "group_id" in feature_frame.columns:
        feature_frame = feature_frame.drop(columns=["group_id"])

    if "timestamp" in feature_frame.columns:
        try:
            train, test = _split_by_timestamp(frame, test_size)
        except ValueError as exc:
            raise ValueError(f"{scenario}: {exc}") from exc

        x_train = train.drop(columns=[target_column, "group_id", "timestamp"], errors="ignore")
        x_test = test.drop(columns=[target_column, "group_id", "timestamp"], errors="ignore")
        y_train = train[target_column]
        y_test = test[target_column]
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            feature_frame,
            frame[target_column],
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )

    x_train = x_train.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x_test = x_test.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y_train = pd.to_numeric(y_train, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y_test = pd.to_numeric(y_test, errors="coerce").replace([np.inf, -np.inf], np.nan)

    train_mask = x_train.notna().all(axis=1) & y_train.notna()
    test_mask = x_test.notna().all(axis=1) & y_test.notna()
    x_train = x_train[train_mask]
    y_train = y_train[train_mask]
    x_test = x_test[test_mask]
    y_test = y_test[test_mask]

    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError(f"{scenario}: no rows left after numeric/finite filtering")

    return (
        x_train.to_numpy(dtype=float),
        x_test.to_numpy(dtype=float),
        y_train.to_numpy(dtype=float),
        y_test.to_numpy(dtype=float),
        None,
        None,
    )


def _run_model(
    model_name: str,
    factory: Callable[[], object],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    scenario: str,
    profile: BenchmarkProfile,
    profile_index: int,
    run_index: int,
    seed: int,
    task_type: str = "regression",
    group_train: np.ndarray | None = None,
    group_test: np.ndarray | None = None,
) -> BenchmarkRecord:
    nan = float("nan")
    try:
        model = factory()

        fit_start = time.perf_counter()
        if task_type == "ranking":
            model.fit(x_train, y_train, group=group_train)
        else:
            model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - fit_start

        fit_timing = getattr(model, "fit_timing_", None)
        input_adaptation_seconds = float(
            fit_timing.get("input_adaptation_seconds", nan)
        ) if isinstance(fit_timing, dict) else nan
        native_bridge_prepare_seconds = float(
            fit_timing.get("native_bridge_prepare_seconds", nan)
        ) if isinstance(fit_timing, dict) else nan
        native_train_seconds = float(
            fit_timing.get("native_train_seconds", nan)
        ) if isinstance(fit_timing, dict) else nan

        predict_start = time.perf_counter()
        class_predictions = None
        multiclass_proba: np.ndarray | None = None
        if task_type in ("classification", "multiclass_classification") and hasattr(model, "predict_proba"):
            proba = np.array(model.predict_proba(x_test), dtype=float)
            if task_type == "multiclass_classification" and proba.ndim == 2:
                multiclass_proba = proba
                col_indices = np.argmax(proba, axis=1)
                classes = getattr(model, "classes_", None)
                class_predictions = np.asarray(classes)[col_indices] if classes is not None else col_indices
                predictions = class_predictions.astype(float)
            else:
                prob_positive = proba[:, 1] if proba.ndim == 2 else proba
                class_predictions = (prob_positive >= 0.5).astype(int)
                predictions = prob_positive
        else:
            predictions = np.array(model.predict(x_test), dtype=float)
        predict_seconds = time.perf_counter() - predict_start

        rmse_val = mae_val = r2_val = nan
        accuracy_val = log_loss_metric = auc_val = nan
        ndcg_5_val = ndcg_10_val = ndcg_full_val = nan

        if task_type == "regression":
            rmse_val = float(np.sqrt(mean_squared_error(y_test, predictions)))
            mae_val = float(mean_absolute_error(y_test, predictions))
            r2_val = float(r2_score(y_test, predictions))
        elif task_type == "classification":
            if class_predictions is None:
                class_predictions = (predictions >= 0.5).astype(int)
            accuracy_val = float(accuracy_score(y_test, class_predictions))
            log_loss_metric = float(
                sklearn_log_loss(y_test, np.clip(predictions, 1e-15, 1 - 1e-15))
            )
            try:
                auc_val = float(roc_auc_score(y_test, predictions))
            except ValueError:
                auc_val = nan
        elif task_type == "multiclass_classification":
            if class_predictions is None:
                class_predictions = predictions.astype(int)
            accuracy_val = float(accuracy_score(y_test, class_predictions))
            if multiclass_proba is not None:
                log_loss_metric = float(sklearn_log_loss(y_test, multiclass_proba))
            # auc_val stays nan for multiclass
        elif task_type == "ranking":
            from alloygbm.evaluation import ndcg as alloy_ndcg
            ndcg_5_val = float(
                alloy_ndcg(y_test.tolist(), predictions.tolist(), group=group_test.tolist(), k=5)
            )
            ndcg_10_val = float(
                alloy_ndcg(y_test.tolist(), predictions.tolist(), group=group_test.tolist(), k=10)
            )
            ndcg_full_val = float(
                alloy_ndcg(y_test.tolist(), predictions.tolist(), group=group_test.tolist())
            )

        return BenchmarkRecord(
            scenario=scenario,
            task_type=task_type,
            profile_name=profile.name,
            profile_index=profile_index,
            run_index=run_index,
            seed=seed,
            learning_rate=profile.learning_rate,
            max_depth=profile.max_depth,
            rounds=profile.rounds,
            model=model_name,
            train_rows=int(len(x_train)),
            test_rows=int(len(x_test)),
            n_features=int(x_train.shape[1]),
            input_adaptation_seconds=input_adaptation_seconds,
            native_bridge_prepare_seconds=native_bridge_prepare_seconds,
            native_train_seconds=native_train_seconds,
            fit_seconds=float(fit_seconds),
            predict_seconds=float(predict_seconds),
            rmse=rmse_val,
            mae=mae_val,
            r2=r2_val,
            accuracy=accuracy_val,
            log_loss_val=log_loss_metric,
            auc=auc_val,
            ndcg_5=ndcg_5_val,
            ndcg_10=ndcg_10_val,
            ndcg_full=ndcg_full_val,
            status="PASS",
            error="",
        )
    except Exception as exc:  # noqa: BLE001
        return BenchmarkRecord(
            scenario=scenario,
            task_type=task_type,
            profile_name=profile.name,
            profile_index=profile_index,
            run_index=run_index,
            seed=seed,
            learning_rate=profile.learning_rate,
            max_depth=profile.max_depth,
            rounds=profile.rounds,
            model=model_name,
            train_rows=0,
            test_rows=0,
            n_features=0,
            input_adaptation_seconds=nan,
            native_bridge_prepare_seconds=nan,
            native_train_seconds=nan,
            fit_seconds=0.0,
            predict_seconds=0.0,
            rmse=nan,
            mae=nan,
            r2=nan,
            accuracy=nan,
            log_loss_val=nan,
            auc=nan,
            ndcg_5=nan,
            ndcg_10=nan,
            ndcg_full=nan,
            status="FAIL",
            error=f"{type(exc).__name__}: {exc}",
        )


def _make_alloygbm_morph(task_type, **kwargs):
    from alloygbm import GBMClassifier, GBMRanker, GBMRegressor
    cls = {"regression": GBMRegressor, "binary": GBMClassifier,
           "multiclass": GBMClassifier, "ranking": GBMRanker}[task_type]
    return cls(training_mode="morph", **kwargs)


def _make_alloygbm_morph_cosine(task_type, **kwargs):
    from alloygbm import GBMClassifier, GBMRanker, GBMRegressor
    cls = {"regression": GBMRegressor, "binary": GBMClassifier,
           "multiclass": GBMClassifier, "ranking": GBMRanker}[task_type]
    return cls(training_mode="morph", lr_schedule="warmup_cosine", lr_warmup_frac=0.1, **kwargs)


def _make_alloygbm_linear(task_type, **kwargs):
    from alloygbm import GBMClassifier, GBMRanker, GBMRegressor
    cls = {"regression": GBMRegressor, "binary": GBMClassifier,
           "multiclass": GBMClassifier, "ranking": GBMRanker}[task_type]
    # Linear leaves need weight regularisation to avoid divergence at high round counts.
    # Default lambda_l2=0.01 (from pl_trees_benchmark sweep); overridable via --linear-lambda-l2.
    kwargs.setdefault("lambda_l2", 0.01)
    return cls(leaf_model="linear", **kwargs)


def _make_alloygbm_morph_linear(task_type, **kwargs):
    from alloygbm import GBMClassifier, GBMRanker, GBMRegressor
    cls = {"regression": GBMRegressor, "binary": GBMClassifier,
           "multiclass": GBMClassifier, "ranking": GBMRanker}[task_type]
    kwargs.setdefault("lambda_l2", 0.01)
    return cls(training_mode="morph", leaf_model="linear", **kwargs)


def _model_factories(
    gbm_regressor_cls: type,
    catboost_regressor_cls: type | None,
    seed: int,
    learning_rate: float,
    max_depth: int,
    rounds: int,
    alloy_continuous_binning_strategy: str,
    alloy_continuous_binning_max_bins: int,
    linear_lambda_l2: float = 0.01,
) -> dict:
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    alloy_signature = inspect.signature(gbm_regressor_cls.__init__)
    alloy_params: dict[str, object] = {}
    if "learning_rate" in alloy_signature.parameters:
        alloy_params["learning_rate"] = learning_rate
    if "max_depth" in alloy_signature.parameters:
        alloy_params["max_depth"] = max_depth
    if "n_estimators" in alloy_signature.parameters:
        alloy_params["n_estimators"] = rounds
    if "rounds" in alloy_signature.parameters:
        alloy_params["rounds"] = rounds
    if "row_subsample" in alloy_signature.parameters:
        alloy_params["row_subsample"] = 0.8
    if "col_subsample" in alloy_signature.parameters:
        alloy_params["col_subsample"] = 0.8
    if "seed" in alloy_signature.parameters:
        alloy_params["seed"] = seed
    if "deterministic" in alloy_signature.parameters:
        alloy_params["deterministic"] = True
    if "continuous_binning_strategy" in alloy_signature.parameters:
        alloy_params["continuous_binning_strategy"] = alloy_continuous_binning_strategy
    if "continuous_binning_max_bins" in alloy_signature.parameters:
        alloy_params["continuous_binning_max_bins"] = alloy_continuous_binning_max_bins

    factories = {
        "alloygbm": lambda: gbm_regressor_cls(**alloy_params),
        "alloygbm_linear": lambda: _make_alloygbm_linear("regression", lambda_l2=linear_lambda_l2, **alloy_params),
        "alloygbm_morph": lambda: _make_alloygbm_morph("regression", **alloy_params),
        "alloygbm_morph_cosine": lambda: _make_alloygbm_morph_cosine("regression", **alloy_params),
        "alloygbm_morph_linear": lambda: _make_alloygbm_morph_linear("regression", lambda_l2=linear_lambda_l2, **alloy_params),
        "lightgbm": lambda: LGBMRegressor(
            objective="regression",
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=rounds,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        ),
        "xgboost": lambda: XGBRegressor(
            objective="reg:squarederror",
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=rounds,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        ),
    }
    if catboost_regressor_cls is not None:
        factories["catboost"] = lambda: catboost_regressor_cls(
            loss_function="RMSE",
            learning_rate=learning_rate,
            depth=max_depth,
            iterations=rounds,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )
    return factories


def _build_alloy_params(
    cls: type,
    seed: int,
    learning_rate: float,
    max_depth: int,
    rounds: int,
    alloy_continuous_binning_strategy: str,
    alloy_continuous_binning_max_bins: int,
) -> dict[str, object]:
    sig = inspect.signature(cls.__init__)
    params: dict[str, object] = {}
    if "learning_rate" in sig.parameters:
        params["learning_rate"] = learning_rate
    if "max_depth" in sig.parameters:
        params["max_depth"] = max_depth
    if "n_estimators" in sig.parameters:
        params["n_estimators"] = rounds
    if "rounds" in sig.parameters:
        params["rounds"] = rounds
    if "row_subsample" in sig.parameters:
        params["row_subsample"] = 0.8
    if "col_subsample" in sig.parameters:
        params["col_subsample"] = 0.8
    if "seed" in sig.parameters:
        params["seed"] = seed
    if "deterministic" in sig.parameters:
        params["deterministic"] = True
    if "continuous_binning_strategy" in sig.parameters:
        params["continuous_binning_strategy"] = alloy_continuous_binning_strategy
    if "continuous_binning_max_bins" in sig.parameters:
        params["continuous_binning_max_bins"] = alloy_continuous_binning_max_bins
    return params


def _classifier_factories(
    gbm_classifier_cls: type,
    catboost_classifier_cls: type | None,
    seed: int,
    learning_rate: float,
    max_depth: int,
    rounds: int,
    alloy_continuous_binning_strategy: str,
    alloy_continuous_binning_max_bins: int,
    linear_lambda_l2: float = 0.01,
) -> dict:
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    alloy_params = _build_alloy_params(
        gbm_classifier_cls, seed, learning_rate, max_depth, rounds,
        alloy_continuous_binning_strategy, alloy_continuous_binning_max_bins,
    )
    factories: dict[str, Callable[[], object]] = {
        "alloygbm": lambda: gbm_classifier_cls(**alloy_params),
        "alloygbm_linear": lambda: _make_alloygbm_linear("binary", lambda_l2=linear_lambda_l2, **alloy_params),
        "alloygbm_morph": lambda: _make_alloygbm_morph("binary", **alloy_params),
        "alloygbm_morph_cosine": lambda: _make_alloygbm_morph_cosine("binary", **alloy_params),
        "alloygbm_morph_linear": lambda: _make_alloygbm_morph_linear("binary", lambda_l2=linear_lambda_l2, **alloy_params),
        "lightgbm": lambda: LGBMClassifier(
            objective="binary",
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=rounds,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        ),
        "xgboost": lambda: XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=rounds,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        ),
    }
    if catboost_classifier_cls is not None:
        factories["catboost"] = lambda: catboost_classifier_cls(
            loss_function="Logloss",
            learning_rate=learning_rate,
            depth=max_depth,
            iterations=rounds,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )
    return factories


def _multiclass_classifier_factories(
    gbm_classifier_cls: type,
    catboost_classifier_cls: type | None,
    n_classes: int,
    seed: int,
    learning_rate: float,
    max_depth: int,
    rounds: int,
    alloy_continuous_binning_strategy: str,
    alloy_continuous_binning_max_bins: int,
    linear_lambda_l2: float = 0.01,
) -> dict:
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    alloy_params = _build_alloy_params(
        gbm_classifier_cls, seed, learning_rate, max_depth, rounds,
        alloy_continuous_binning_strategy, alloy_continuous_binning_max_bins,
    )
    factories: dict[str, Callable[[], object]] = {
        "alloygbm": lambda: gbm_classifier_cls(**alloy_params),
        "alloygbm_linear": lambda: _make_alloygbm_linear("multiclass", lambda_l2=linear_lambda_l2, **alloy_params),
        "alloygbm_morph": lambda: _make_alloygbm_morph("multiclass", **alloy_params),
        "alloygbm_morph_cosine": lambda: _make_alloygbm_morph_cosine("multiclass", **alloy_params),
        "alloygbm_morph_linear": lambda: _make_alloygbm_morph_linear("multiclass", lambda_l2=linear_lambda_l2, **alloy_params),
        "lightgbm": lambda: LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=rounds,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        ),
        "xgboost": lambda: XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=rounds,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        ),
    }
    if catboost_classifier_cls is not None:
        factories["catboost"] = lambda: catboost_classifier_cls(
            loss_function="MultiClass",
            learning_rate=learning_rate,
            depth=max_depth,
            iterations=rounds,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )
    return factories


def _ranker_factories(
    gbm_ranker_cls: type,
    catboost_available: bool,
    seed: int,
    learning_rate: float,
    max_depth: int,
    rounds: int,
    alloy_continuous_binning_strategy: str,
    alloy_continuous_binning_max_bins: int,
    linear_lambda_l2: float = 0.01,
) -> dict:
    alloy_params = _build_alloy_params(
        gbm_ranker_cls, seed, learning_rate, max_depth, rounds,
        alloy_continuous_binning_strategy, alloy_continuous_binning_max_bins,
    )
    factories: dict[str, Callable[[], object]] = {
        "alloygbm": lambda: gbm_ranker_cls(**alloy_params),
        "alloygbm_linear": lambda: _make_alloygbm_linear("ranking", lambda_l2=linear_lambda_l2, **alloy_params),
        "alloygbm_morph": lambda: _make_alloygbm_morph("ranking", **alloy_params),
        "alloygbm_morph_cosine": lambda: _make_alloygbm_morph_cosine("ranking", **alloy_params),
        "alloygbm_morph_linear": lambda: _make_alloygbm_morph_linear("ranking", lambda_l2=linear_lambda_l2, **alloy_params),
        "lightgbm": lambda: _LGBMRankerAdapter(
            objective="lambdarank",
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=rounds,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        ),
        "xgboost": lambda: _XGBRankerAdapter(
            objective="rank:ndcg",
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=rounds,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        ),
    }
    if catboost_available:
        factories["catboost"] = lambda: _CatBoostRankerAdapter(
            learning_rate=learning_rate,
            depth=max_depth,
            iterations=rounds,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )
    return factories


def _parse_seed_list(seed_text: str) -> list[int]:
    raw_parts = [part.strip() for part in seed_text.split(",") if part.strip()]
    if not raw_parts:
        raise ValueError("profile seed list cannot be empty")

    seeds: list[int] = []
    seen: set[int] = set()
    for part in raw_parts:
        value = int(part)
        if value in seen:
            continue
        seeds.append(value)
        seen.add(value)
    return seeds


def _parse_profile_spec(spec: str) -> BenchmarkProfile:
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) != 4:
        raise ValueError(
            "profile must use 'name:learning_rate:max_depth:rounds' format"
        )

    name, learning_rate_raw, max_depth_raw, rounds_raw = parts
    learning_rate = float(learning_rate_raw)
    max_depth = int(max_depth_raw)
    rounds = int(rounds_raw)

    if not name:
        raise ValueError("profile name cannot be empty")
    if learning_rate <= 0.0:
        raise ValueError("profile learning_rate must be > 0")
    if max_depth <= 0:
        raise ValueError("profile max_depth must be > 0")
    if rounds <= 0:
        raise ValueError("profile rounds must be > 0")

    return BenchmarkProfile(
        name=name,
        learning_rate=learning_rate,
        max_depth=max_depth,
        rounds=rounds,
    )


def _resolve_profiles(
    args: argparse.Namespace,
) -> tuple[list[BenchmarkProfile], list[int], bool]:
    if args.profile and args.profile_grid != "none":
        raise ValueError("use either --profile-grid or --profile, not both")

    if args.profile:
        profiles = [_parse_profile_spec(spec) for spec in args.profile]
        seeds = _parse_seed_list(args.profile_seeds)
        return profiles, seeds, True

    if args.profile_grid == "default":
        profiles = list(DEFAULT_PROFILES)
        seeds = _parse_seed_list(args.profile_seeds)
        return profiles, seeds, True

    if args.profile_grid == "default_ultra":
        profiles = list(DEFAULT_PROFILES) + [ULTRA_PROFILE]
        seeds = _parse_seed_list(args.profile_seeds)
        return profiles, seeds, True

    profile = BenchmarkProfile(
        name="single",
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        rounds=args.rounds,
    )
    return [profile], [args.seed], False


def _summarize_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    passed = frame[frame["status"] == "PASS"].copy()
    if passed.empty:
        return pd.DataFrame()

    summary = (
        passed.groupby(
            [
                "scenario",
                "task_type",
                "profile_name",
                "model",
                "learning_rate",
                "max_depth",
                "rounds",
            ],
            as_index=False,
        )
        .agg(
            runs=("fit_seconds", "count"),
            input_adaptation_seconds_median=("input_adaptation_seconds", "median"),
            native_bridge_prepare_seconds_median=(
                "native_bridge_prepare_seconds",
                "median",
            ),
            native_train_seconds_median=("native_train_seconds", "median"),
            fit_seconds_median=("fit_seconds", "median"),
            predict_seconds_median=("predict_seconds", "median"),
            rmse_median=("rmse", "median"),
            mae_median=("mae", "median"),
            r2_median=("r2", "median"),
            accuracy_median=("accuracy", "median"),
            log_loss_val_median=("log_loss_val", "median"),
            auc_median=("auc", "median"),
            ndcg_5_median=("ndcg_5", "median"),
            ndcg_10_median=("ndcg_10", "median"),
            ndcg_full_median=("ndcg_full", "median"),
        )
        .sort_values(["scenario", "profile_name", "model"])
        .reset_index(drop=True)
    )
    return summary


def _best_rows_by_scenario(
    summary: pd.DataFrame, metric: str, ascending: bool
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    grouped = summary.groupby("scenario", as_index=False)
    index = grouped[metric].idxmin() if ascending else grouped[metric].idxmax()
    return summary.loc[index[metric]].sort_values("scenario")


def _render_results_markdown(
    run_id: str,
    params: dict,
    frame: pd.DataFrame,
    summary: pd.DataFrame,
) -> str:
    lines = [
        f"# Model Comparison ({run_id})",
        "",
        "## Params",
        (
            "- alloy_continuous_binning_strategy: "
            f"`{params['alloy_continuous_binning_strategy']}`"
        ),
        (
            "- alloy_continuous_binning_max_bins: "
            f"`{params['alloy_continuous_binning_max_bins']}`"
        ),
        f"- profile_mode: `{params['profile_mode']}`",
        f"- scenarios: `{', '.join(params['scenarios'])}`",
        f"- test_size: `{params['test_size']}`",
    ]

    if params["profile_mode"] == "single":
        lines.extend(
            [
                f"- seed: `{params['seed']}`",
                f"- learning_rate: `{params['learning_rate']}`",
                f"- max_depth: `{params['max_depth']}`",
                f"- rounds: `{params['rounds']}`",
            ]
        )
    else:
        lines.extend(
            [
                f"- profile_grid: `{params['profile_grid']}`",
                f"- profile_seeds: `{params['profile_seeds']}`",
                "",
                "### Profiles",
            ]
        )
        for profile in params["profiles"]:
            lines.append(
                "- "
                f"`{profile['name']}`: "
                f"lr={profile['learning_rate']}, "
                f"max_depth={profile['max_depth']}, "
                f"rounds={profile['rounds']}"
            )

    lines.extend(["", "## Raw Results", ""])

    if frame.empty:
        lines.append("No benchmark records were produced.")
        return "\n".join(lines) + "\n"

    # Render raw results by task type.
    timing_cols = "input_adaptation_seconds | native_bridge_prepare_seconds | native_train_seconds | fit_seconds | predict_seconds"
    timing_align = "---: | ---: | ---: | ---: | ---:"

    for task_type, task_label, metric_header, metric_align, metric_fmt in [
        ("regression", "Regression", "rmse | mae | r2", "---: | ---: | ---:",
         lambda r: f"{r['rmse']:.6f} | {r['mae']:.6f} | {r['r2']:.6f}"),
        ("classification", "Classification", "accuracy | log_loss | auc", "---: | ---: | ---:",
         lambda r: f"{r['accuracy']:.6f} | {r['log_loss_val']:.6f} | {r['auc']:.6f}"),
        ("multiclass_classification", "Multiclass Classification", "accuracy | log_loss", "---: | ---:",
         lambda r: f"{r['accuracy']:.6f} | {r['log_loss_val']:.6f}"),
        ("ranking", "Ranking", "ndcg@5 | ndcg@10 | ndcg", "---: | ---: | ---:",
         lambda r: f"{r['ndcg_5']:.6f} | {r['ndcg_10']:.6f} | {r['ndcg_full']:.6f}"),
    ]:
        task_frame = frame[frame["task_type"] == task_type] if "task_type" in frame.columns else frame
        if task_frame.empty:
            continue
        lines.extend([f"### {task_label}", ""])
        lines.extend([
            f"| scenario | profile | model | seed | status | {timing_cols} | {metric_header} |",
            f"| --- | --- | --- | ---: | --- | {timing_align} | {metric_align} |",
        ])
        for _, row in task_frame.iterrows():
            lines.append(
                f"| {row['scenario']} | {row['profile_name']} | {row['model']} | {int(row['seed'])} | "
                f"{row['status']} | {row['input_adaptation_seconds']:.6f} | "
                f"{row['native_bridge_prepare_seconds']:.6f} | {row['native_train_seconds']:.6f} | "
                f"{row['fit_seconds']:.6f} | {row['predict_seconds']:.6f} | "
                f"{metric_fmt(row)} |"
            )
        lines.append("")

    lines.extend(["## Profile Median Summary", ""])
    if summary.empty:
        lines.append("No PASS records available for profile summary.")
        return "\n".join(lines) + "\n"

    for task_type, task_label, metric_header, metric_align, metric_fmt in [
        ("regression", "Regression", "rmse_median | mae_median | r2_median", "---: | ---: | ---:",
         lambda r: f"{r['rmse_median']:.6f} | {r['mae_median']:.6f} | {r['r2_median']:.6f}"),
        ("classification", "Classification", "accuracy_median | log_loss_median | auc_median", "---: | ---: | ---:",
         lambda r: f"{r['accuracy_median']:.6f} | {r['log_loss_val_median']:.6f} | {r['auc_median']:.6f}"),
        ("multiclass_classification", "Multiclass Classification", "accuracy_median | log_loss_median", "---: | ---:",
         lambda r: f"{r['accuracy_median']:.6f} | {r['log_loss_val_median']:.6f}"),
        ("ranking", "Ranking", "ndcg@5_median | ndcg@10_median | ndcg_median", "---: | ---: | ---:",
         lambda r: f"{r['ndcg_5_median']:.6f} | {r['ndcg_10_median']:.6f} | {r['ndcg_full_median']:.6f}"),
    ]:
        task_summary = summary[summary["task_type"] == task_type] if "task_type" in summary.columns else summary
        if task_summary.empty:
            continue
        lines.extend([f"### {task_label}", ""])
        lines.extend([
            f"| scenario | profile | model | runs | lr | depth | rounds | fit_seconds_median | predict_seconds_median | {metric_header} |",
            f"| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | {metric_align} |",
        ])
        for _, row in task_summary.iterrows():
            lines.append(
                f"| {row['scenario']} | {row['profile_name']} | {row['model']} | {int(row['runs'])} | "
                f"{row['learning_rate']:.6f} | {int(row['max_depth'])} | {int(row['rounds'])} | "
                f"{row['fit_seconds_median']:.6f} | {row['predict_seconds_median']:.6f} | "
                f"{metric_fmt(row)} |"
            )
        lines.append("")

    # Best-metric sections by task type.
    reg_summary = summary[summary["task_type"] == "regression"] if "task_type" in summary.columns else summary
    clf_summary = summary[summary["task_type"] == "classification"] if "task_type" in summary.columns else pd.DataFrame()
    multiclf_summary = summary[summary["task_type"] == "multiclass_classification"] if "task_type" in summary.columns else pd.DataFrame()
    rank_summary = summary[summary["task_type"] == "ranking"] if "task_type" in summary.columns else pd.DataFrame()

    best_rmse = _best_rows_by_scenario(reg_summary, metric="rmse_median", ascending=True)
    if not best_rmse.empty:
        lines.extend(["## Best RMSE By Scenario (Regression)", ""])
        lines.extend(["| scenario | profile | model | rmse_median |", "| --- | --- | --- | ---: |"])
        for _, row in best_rmse.iterrows():
            lines.append(f"| {row['scenario']} | {row['profile_name']} | {row['model']} | {row['rmse_median']:.6f} |")
        lines.append("")

    best_acc = _best_rows_by_scenario(clf_summary, metric="accuracy_median", ascending=False)
    if not best_acc.empty:
        lines.extend(["## Best Accuracy By Scenario (Classification)", ""])
        lines.extend(["| scenario | profile | model | accuracy_median |", "| --- | --- | --- | ---: |"])
        for _, row in best_acc.iterrows():
            lines.append(f"| {row['scenario']} | {row['profile_name']} | {row['model']} | {row['accuracy_median']:.6f} |")
        lines.append("")

    best_accuracy_multiclass = _best_rows_by_scenario(multiclf_summary, metric="accuracy_median", ascending=False)
    if not best_accuracy_multiclass.empty:
        lines.extend(["#### Best Accuracy by Scenario (Multiclass Classification)", ""])
        lines.extend([
            "| scenario | model | profile | accuracy_median | log_loss_median |",
            "| --- | --- | --- | ---: | ---: |",
        ])
        for _, row in best_accuracy_multiclass.iterrows():
            lines.append(
                f"| {row['scenario']} | {row['model']} | {row['profile_name']} "
                f"| {row['accuracy_median']:.6f} | {row['log_loss_val_median']:.6f} |"
            )
        lines.append("")

    best_ndcg = _best_rows_by_scenario(rank_summary, metric="ndcg_full_median", ascending=False)
    if not best_ndcg.empty:
        lines.extend(["## Best NDCG By Scenario (Ranking)", ""])
        lines.extend(["| scenario | profile | model | ndcg_median |", "| --- | --- | --- | ---: |"])
        for _, row in best_ndcg.iterrows():
            lines.append(f"| {row['scenario']} | {row['profile_name']} | {row['model']} | {row['ndcg_full_median']:.6f} |")
        lines.append("")

    fastest_fit = _best_rows_by_scenario(summary, metric="fit_seconds_median", ascending=True)
    lines.extend(["## Fastest Fit By Scenario", ""])
    if fastest_fit.empty:
        lines.append("No fastest-fit rows available.")
    else:
        lines.extend(["| scenario | profile | model | fit_seconds_median |", "| --- | --- | --- | ---: |"])
        for _, row in fastest_fit.iterrows():
            lines.append(f"| {row['scenario']} | {row['profile_name']} | {row['model']} | {row['fit_seconds_median']:.6f} |")

    return "\n".join(lines) + "\n"


def _write_outputs(
    output_dir: Path,
    run_id: str,
    records: list[BenchmarkRecord],
    params: dict,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame([asdict(record) for record in records])
    if not frame.empty:
        frame = frame.sort_values(
            ["scenario", "profile_index", "run_index", "model"]
        ).reset_index(drop=True)

    csv_timestamped = output_dir / f"model_comparison_{run_id}.csv"
    csv_latest = output_dir / "model_comparison_latest.csv"
    frame.to_csv(csv_timestamped, index=False)
    frame.to_csv(csv_latest, index=False)

    json_timestamped = output_dir / f"model_comparison_{run_id}.json"
    json_latest = output_dir / "model_comparison_latest.json"
    payload = {
        "run_id": run_id,
        "params": params,
        "records": frame.to_dict(orient="records"),
    }
    json_timestamped.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    json_latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = _summarize_profiles(frame)

    summary_csv_timestamped = output_dir / f"model_comparison_profile_summary_{run_id}.csv"
    summary_csv_latest = output_dir / "model_comparison_profile_summary_latest.csv"
    summary.to_csv(summary_csv_timestamped, index=False)
    summary.to_csv(summary_csv_latest, index=False)

    summary_json_timestamped = output_dir / f"model_comparison_profile_summary_{run_id}.json"
    summary_json_latest = output_dir / "model_comparison_profile_summary_latest.json"
    summary_payload = {
        "run_id": run_id,
        "params": params,
        "summary": summary.to_dict(orient="records"),
    }
    summary_json_timestamped.write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )
    summary_json_latest.write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )

    markdown_text = _render_results_markdown(run_id, params, frame, summary)
    markdown_timestamped = output_dir / f"model_comparison_{run_id}.md"
    markdown_latest = output_dir / "model_comparison_latest.md"
    markdown_timestamped.write_text(markdown_text, encoding="utf-8")
    markdown_latest.write_text(markdown_text, encoding="utf-8")

    return {
        "csv": csv_timestamped,
        "json": json_timestamped,
        "markdown": markdown_timestamped,
        "summary_csv": summary_csv_timestamped,
        "summary_json": summary_json_timestamped,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=AVAILABLE_SCENARIOS,
        choices=AVAILABLE_SCENARIOS,
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=120)
    parser.add_argument(
        "--alloy-continuous-binning-strategy",
        choices=VALID_ALLOY_CONTINUOUS_BINNING_STRATEGIES,
        default="linear",
        help=(
            "continuous-feature binning strategy passed to alloygbm runtime when "
            "supported by the installed GBMRegressor"
        ),
    )
    parser.add_argument(
        "--alloy-continuous-binning-max-bins",
        type=int,
        default=256,
        help="max quantized bins for alloy continuous binning modes that use cut points",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument(
        "--profile-grid",
        choices=["none", "default", "default_ultra"],
        default="none",
        help="named profile matrix; use none for single-profile compatibility mode",
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help="custom profile spec: name:learning_rate:max_depth:rounds (repeatable)",
    )
    parser.add_argument(
        "--profile-seeds",
        default="7,17,29",
        help="comma-separated seeds for profile-grid/custom-profile modes",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help=(
            "filter to only these model names (e.g. alloygbm alloygbm_linear lightgbm). "
            "Default: run all models. Valid names depend on task type but include: "
            "alloygbm, alloygbm_linear, alloygbm_morph, alloygbm_morph_cosine, "
            "alloygbm_morph_linear, lightgbm, xgboost, catboost"
        ),
    )
    parser.add_argument(
        "--linear-lambda-l2",
        type=float,
        default=0.01,
        metavar="LAMBDA",
        help=(
            "L2 regularisation applied to linear leaf weights for alloygbm_linear and "
            "alloygbm_morph_linear variants. Default 0.01 (from pl_trees_benchmark sweep). "
            "Constant-leaf variants always use lambda_l2=0.0."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks") / "results",
    )
    args = parser.parse_args(argv)

    if not (2 <= args.alloy_continuous_binning_max_bins <= 256):
        print(
            "invalid --alloy-continuous-binning-max-bins: expected integer in [2, 256]",
            file=sys.stderr,
        )
        return 2

    try:
        profiles, seeds, matrix_mode = _resolve_profiles(args)
    except (TypeError, ValueError) as exc:
        print(f"invalid profile configuration: {exc}", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    try:
        gbm_regressor_cls, alloy_runtime = _load_alloygbm_runtime()
    except RuntimeError as exc:
        print(f"alloygbm runtime check failed: {exc}", file=sys.stderr)
        return 2
    gbm_classifier_cls, _ = _load_alloygbm_classifier_runtime()
    gbm_ranker_cls, _ = _load_alloygbm_ranker_runtime()
    catboost_regressor_cls, catboost_runtime = _load_optional_catboost_regressor()
    catboost_classifier_cls, _ = _load_optional_catboost_classifier()
    catboost_ranker_available = catboost_runtime.get("available", False)
    print(
        "alloygbm runtime: "
        f"module={alloy_runtime['module_path']} "
        f"native={alloy_runtime['native_module_path']}"
    )
    if catboost_runtime.get("available"):
        print(f"catboost runtime: version={catboost_runtime['version']}")
    else:
        print(f"catboost runtime: unavailable ({catboost_runtime.get('error', 'unknown')})")

    datasets: dict[str, tuple[pd.DataFrame, str, str, str | None]] = {}
    dataset_errors: dict[str, str] = {}
    for scenario in args.scenarios:
        try:
            frame, target_column, task_type, group_column = _load_dataset(
                repo_root, scenario, args.force_prepare
            )
            datasets[scenario] = (frame, target_column, task_type, group_column)
        except Exception as exc:  # noqa: BLE001
            dataset_errors[scenario] = f"{type(exc).__name__}: {exc}"

    nan = float("nan")

    def _make_fail_record(
        scenario: str, task_type: str, profile: BenchmarkProfile,
        profile_index: int, run_index: int, seed: int, model_name: str, error: str,
    ) -> BenchmarkRecord:
        return BenchmarkRecord(
            scenario=scenario, task_type=task_type, profile_name=profile.name,
            profile_index=profile_index, run_index=run_index, seed=seed,
            learning_rate=profile.learning_rate, max_depth=profile.max_depth,
            rounds=profile.rounds, model=model_name,
            train_rows=0, test_rows=0, n_features=0,
            input_adaptation_seconds=nan, native_bridge_prepare_seconds=nan,
            native_train_seconds=nan, fit_seconds=0.0, predict_seconds=0.0,
            rmse=nan, mae=nan, r2=nan,
            accuracy=nan, log_loss_val=nan, auc=nan,
            ndcg_5=nan, ndcg_10=nan, ndcg_full=nan,
            status="FAIL", error=error,
        )

    records: list[BenchmarkRecord] = []

    for profile_index, profile in enumerate(profiles, start=1):
        for run_index, seed in enumerate(seeds, start=1):
            common_factory_args = dict(
                seed=seed,
                learning_rate=profile.learning_rate,
                max_depth=profile.max_depth,
                rounds=profile.rounds,
                alloy_continuous_binning_strategy=args.alloy_continuous_binning_strategy,
                alloy_continuous_binning_max_bins=args.alloy_continuous_binning_max_bins,
                linear_lambda_l2=args.linear_lambda_l2,
            )

            for scenario in args.scenarios:
                # Determine task type (default to regression for error cases).
                if scenario in datasets:
                    _, _, task_type, group_column = datasets[scenario]
                else:
                    task_type = "regression"
                    group_column = None

                # Select factories for this task type.
                # For multiclass_classification, factories are deferred until
                # after _split_dataset because n_classes requires y_train.
                if task_type == "classification":
                    factories = _classifier_factories(
                        gbm_classifier_cls=gbm_classifier_cls,
                        catboost_classifier_cls=catboost_classifier_cls,
                        **common_factory_args,
                    )
                    if args.models:
                        factories = {k: v for k, v in factories.items() if k in args.models}
                elif task_type == "multiclass_classification":
                    # Use a placeholder factory dict for error-record model names;
                    # real factories are built after _split_dataset below.
                    factories = _multiclass_classifier_factories(
                        gbm_classifier_cls=gbm_classifier_cls,
                        catboost_classifier_cls=catboost_classifier_cls,
                        n_classes=2,  # placeholder; overwritten after split
                        **common_factory_args,
                    )
                    if args.models:
                        factories = {k: v for k, v in factories.items() if k in args.models}
                elif task_type == "ranking":
                    factories = _ranker_factories(
                        gbm_ranker_cls=gbm_ranker_cls,
                        catboost_available=catboost_ranker_available,
                        **common_factory_args,
                    )
                    if args.models:
                        factories = {k: v for k, v in factories.items() if k in args.models}
                else:
                    factories = _model_factories(
                        gbm_regressor_cls=gbm_regressor_cls,
                        catboost_regressor_cls=catboost_regressor_cls,
                        **common_factory_args,
                    )
                    if args.models:
                        factories = {k: v for k, v in factories.items() if k in args.models}

                if scenario in dataset_errors:
                    error = dataset_errors[scenario]
                    for model_name in factories:
                        records.append(_make_fail_record(
                            scenario, task_type, profile, profile_index,
                            run_index, seed, model_name, error,
                        ))
                    continue

                frame, target_column, task_type, group_column = datasets[scenario]
                try:
                    x_train, x_test, y_train, y_test, g_train, g_test = _split_dataset(
                        scenario, frame, target_column, seed, args.test_size,
                        task_type=task_type, group_column=group_column,
                    )
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                    for model_name in factories:
                        records.append(_make_fail_record(
                            scenario, task_type, profile, profile_index,
                            run_index, seed, model_name, error,
                        ))
                    continue

                # Rebuild multiclass factories now that y_train is available.
                if task_type == "multiclass_classification":
                    n_classes = int(len(np.unique(y_train)))
                    factories = _multiclass_classifier_factories(
                        gbm_classifier_cls=gbm_classifier_cls,
                        catboost_classifier_cls=catboost_classifier_cls,
                        n_classes=n_classes,
                        **common_factory_args,
                    )
                    if args.models:
                        factories = {k: v for k, v in factories.items() if k in args.models}

                for model_name, factory in factories.items():
                    record = _run_model(
                        model_name=model_name,
                        factory=factory,
                        x_train=x_train,
                        y_train=y_train,
                        x_test=x_test,
                        y_test=y_test,
                        scenario=scenario,
                        profile=profile,
                        profile_index=profile_index,
                        run_index=run_index,
                        seed=seed,
                        task_type=task_type,
                        group_train=g_train,
                        group_test=g_test,
                    )
                    records.append(record)
                    timing_str = (
                        f"fit={record.fit_seconds:.4f}s "
                        f"(adapt={record.input_adaptation_seconds:.4f}s "
                        f"bridge={record.native_bridge_prepare_seconds:.4f}s "
                        f"native={record.native_train_seconds:.4f}s) "
                        f"pred={record.predict_seconds:.4f}s"
                    )
                    if task_type in ("classification", "multiclass_classification"):
                        metric_str = f"acc={record.accuracy:.4f} logloss={record.log_loss_val:.6f} auc={record.auc:.4f}"
                    elif task_type == "ranking":
                        metric_str = f"ndcg@5={record.ndcg_5:.4f} ndcg@10={record.ndcg_10:.4f} ndcg={record.ndcg_full:.4f}"
                    else:
                        metric_str = f"rmse={record.rmse:.6f} mae={record.mae:.6f} r2={record.r2:.6f}"
                    print(
                        f"[{scenario}][{profile.name}][seed={seed}] {model_name}: "
                        f"{record.status} {timing_str} {metric_str}"
                    )
                    if record.status != "PASS":
                        print(f"  error: {record.error}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    params = {
        "profile_mode": "matrix" if matrix_mode else "single",
        "profile_grid": args.profile_grid,
        "profiles": [asdict(profile) for profile in profiles],
        "profile_seeds": seeds,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "rounds": args.rounds,
        "alloy_continuous_binning_strategy": args.alloy_continuous_binning_strategy,
        "alloy_continuous_binning_max_bins": args.alloy_continuous_binning_max_bins,
        "test_size": args.test_size,
        "scenarios": args.scenarios,
        "models_filter": args.models,
        "alloygbm_runtime": alloy_runtime,
        "catboost_runtime": catboost_runtime,
    }
    paths = _write_outputs(args.output_dir, run_id, records, params)
    print(f"wrote comparison csv: {paths['csv']}")
    print(f"wrote comparison json: {paths['json']}")
    print(f"wrote comparison markdown: {paths['markdown']}")
    print(f"wrote profile summary csv: {paths['summary_csv']}")
    print(f"wrote profile summary json: {paths['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
