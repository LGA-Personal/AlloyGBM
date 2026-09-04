"""Lazy comparator adapters and the common measured result contract."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import time
from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from .datasets import DatasetCase


@dataclass(frozen=True)
class AdapterResult:
    predictions: np.ndarray
    preprocessing_seconds: float
    fit_seconds: float
    predict_seconds: float
    peak_rss_bytes: int
    library: str
    library_version: str
    effective_params: Mapping[str, object]
    input_representation: str
    rounds_completed: int


class Adapter(Protocol):
    def fit_predict(self, case: DatasetCase, seed: int, threads: int) -> AdapterResult: ...


def _import_optional(module: str, library: str):
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise RuntimeError(f"{library} optional dependency unavailable: {exc}") from exc


def _seconds(start_ns: int, end_ns: int) -> float:
    return max(end_ns - start_ns, 1) / 1_000_000_000.0


def _rss_bytes() -> int:
    import resource
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes, Linux reports KiB.
    if os.uname().sysname.lower() == "darwin":
        return max(value, 1)
    return max(value * 1024, 1)


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _completed_rounds(model: object, default: int) -> int:
    for name in ("rounds_completed_", "rounds_completed", "n_estimators_", "tree_count_"):
        value = getattr(model, name, None)
        if isinstance(value, (int, np.integer)) and int(value) > 0:
            return int(value)
    return default


def _base_params(case: DatasetCase, seed: int, threads: int) -> dict[str, object]:
    return {
        "learning_rate": 0.05, "row_subsample": 0.8, "col_subsample": 0.8,
        "rounds": case.rounds, "max_depth": case.depth, "seed": seed,
        "threads": threads,
        "categorical_handling": "native" if case.categorical_feature_indices else "none",
        "sparse_fallback": "none", "multi_output_strategy": "single_target",
    }


def _prepare_input(case: DatasetCase, library: str):
    import scipy.sparse as sp
    started = time.perf_counter_ns()
    representation = case.input_representation
    X_train, X_test = case.X_train, case.X_test
    params: dict[str, object] = {}
    params["sparse_fallback"] = "none"
    fit_kwargs: dict[str, object] = {}
    if representation == "csr" and library == "alloygbm":
        X_train, X_test = X_train.toarray(), X_test.toarray()
        representation = "dense_fallback"
        params["sparse_fallback"] = "dense"
    elif representation == "csr" and not sp.isspmatrix_csr(X_train):
        X_train, X_test = X_train.tocsr(), X_test.tocsr()
    if case.categorical_feature_indices and library in {"lightgbm", "xgboost"}:
        import pandas as pd
        X_train = pd.DataFrame(np.asarray(X_train).copy())
        X_test = pd.DataFrame(np.asarray(X_test).copy())
        for index in case.categorical_feature_indices:
            levels = np.unique(np.asarray(X_train.iloc[:, index]))
            # XGBoost rejects floating-point category labels.  Encode levels
            # as integer IDs derived only from training values; unseen test
            # values become missing categorical values.
            level_to_id = {value: position for position, value in enumerate(levels.tolist())}
            train_ids = np.asarray([level_to_id[value] for value in np.asarray(X_train.iloc[:, index])], dtype=np.int32)
            test_ids = np.asarray([level_to_id.get(value, -1) for value in np.asarray(X_test.iloc[:, index])], dtype=np.int32)
            category_ids = np.arange(len(levels), dtype=np.int32)
            X_train[index] = pd.Series(pd.Categorical(train_ids, categories=category_ids), index=X_train.index)
            X_test[index] = pd.Series(pd.Categorical(test_ids, categories=category_ids), index=X_test.index)
    elif case.categorical_feature_indices and library == "catboost":
        import pandas as pd
        X_train = pd.DataFrame(np.asarray(X_train).copy())
        X_test = pd.DataFrame(np.asarray(X_test).copy())
        for index in case.categorical_feature_indices:
            X_train[index] = pd.Series([str(value) for value in X_train.iloc[:, index]], dtype=object)
            X_test[index] = pd.Series([str(value) for value in X_test.iloc[:, index]], dtype=object)
    if case.categorical_feature_indices and library == "catboost":
        params["categorical_indices"] = list(case.categorical_feature_indices)
    if case.categorical_feature_indices and library == "alloygbm":
        fit_kwargs["categorical_feature_values_list"] = [
            [str(value) for value in np.asarray(X_train)[:, index]]
            for index in case.categorical_feature_indices
        ]
    return X_train, X_test, representation, params, _seconds(started, time.perf_counter_ns()), fit_kwargs


def _fit_alloy(case: DatasetCase, seed: int, threads: int) -> AdapterResult:
    alloy = _import_optional("alloygbm", "alloygbm")
    X_train, X_test, representation, prep_params, prep_seconds, fit_kwargs = _prepare_input(case, "alloygbm")
    params = _base_params(case, seed, threads) | prep_params
    params.update({"continuous_binning_max_bins": 256, "deterministic": True})
    if case.categorical_feature_indices:
        params["categorical_indices"] = list(case.categorical_feature_indices)
        # Enable Fisher/native categorical partitions.  With the 256-bin
        # budget, bin 255 is reserved for missing values, so Alloy's native
        # API must cap candidate categories at 255 for the 256-cardinality
        # fixture; the high-cardinality column is explicitly reported as the
        # library's target-encoding fallback, never mislabeled as native.
        params["max_cat_threshold"] = 255
        params["categorical_handling"] = "mixed_native_target_encoding"
        params["categorical_fallback"] = "target_encoding"
    if case.task == "regression":
        params.update({"objective": "squared_error", "topology": "levelwise"})
        model = alloy.GBMRegressor(n_estimators=case.rounds, max_depth=case.depth, learning_rate=0.05, row_subsample=0.8, col_subsample=0.8, continuous_binning_max_bins=256, seed=seed, deterministic=True, n_jobs=threads, categorical_feature_indices=list(case.categorical_feature_indices) or None, max_cat_threshold=255 if case.categorical_feature_indices else 0)
    elif case.task == "binary_classification":
        params["objective"] = "binary_crossentropy"
        model = alloy.GBMClassifier(n_estimators=case.rounds, max_depth=case.depth, learning_rate=0.05, row_subsample=0.8, col_subsample=0.8, continuous_binning_max_bins=256, seed=seed, deterministic=True, n_jobs=threads, categorical_feature_indices=list(case.categorical_feature_indices) or None, max_cat_threshold=255 if case.categorical_feature_indices else 0)
    elif case.task == "ranking":
        params["objective"] = "rank:ndcg"
        model = alloy.GBMRanker(ranking_objective="rank:ndcg", n_estimators=case.rounds, max_depth=case.depth, learning_rate=0.05, row_subsample=0.8, col_subsample=0.8, continuous_binning_max_bins=256, seed=seed, deterministic=True, n_jobs=threads, categorical_feature_indices=list(case.categorical_feature_indices) or None, max_cat_threshold=255 if case.categorical_feature_indices else 0)
    elif case.task == "multi_output_regression":
        # The currently installed joint bridge rejects ``deterministic`` as a
        # per-label kwarg; record the limitation rather than claiming it was applied.
        model = alloy.MultiLabelGBMRanker(multi_label_mode="joint", ranking_objective="squared_error", n_estimators=case.rounds, max_depth=case.depth, learning_rate=0.05, row_subsample=0.8, col_subsample=0.8, continuous_binning_max_bins=256, seed=seed, n_jobs=threads)
        params["multi_output_strategy"] = "native_joint"
        params["deterministic_applied"] = False
        params["deterministic_limitation"] = "joint bridge does not accept deterministic kwarg"
        params["policy_verification"] = "unavailable_joint_bridge"
        params["objective"] = "squared_error"
    else:
        raise ValueError(f"unsupported task: {case.task}")
    params["topology"] = "levelwise"
    started = time.perf_counter_ns()
    if case.task == "ranking":
        model.fit(X_train, case.y_train, group=case.group_train, **fit_kwargs)
    else:
        model.fit(X_train, case.y_train, **fit_kwargs)
    fit_seconds = _seconds(started, time.perf_counter_ns())
    started = time.perf_counter_ns()
    predictions = model.predict_proba(X_test)[:, 1] if case.task == "binary_classification" else model.predict(X_test)
    predict_seconds = _seconds(started, time.perf_counter_ns())
    resolved = getattr(model, "resolved_training_policy_", None)
    if resolved is not None:
        params["resolved_training_policy"] = resolved
        if isinstance(resolved, Mapping):
            for key in ("row_subsample", "col_subsample"):
                if key in resolved and not np.isclose(float(resolved[key]), 0.8, rtol=0.0, atol=1e-6):
                    raise RuntimeError(f"AlloyGBM resolved {key} drifted from explicit 0.8")
    rounds = _completed_rounds(model, case.rounds)
    return AdapterResult(np.asarray(predictions), prep_seconds, fit_seconds, predict_seconds, _rss_bytes(), "alloygbm", _version("alloygbm"), params, representation, rounds)


def _fit_lightgbm(case: DatasetCase, seed: int, threads: int) -> AdapterResult:
    lgb = _import_optional("lightgbm", "lightgbm")
    X_train, X_test, representation, prep_params, prep_seconds, _ = _prepare_input(case, "lightgbm")
    params = _base_params(case, seed, threads) | prep_params
    common = dict(n_estimators=case.rounds, max_depth=case.depth, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, num_leaves=2**case.depth, subsample_freq=1, max_bin=255, random_state=seed, n_jobs=threads, verbosity=-1)
    params.update({"num_leaves": 2**case.depth, "subsample_freq": 1, "max_bin": 255})
    if case.task == "regression":
        params.update({"objective": "regression", "topology": "leafwise"})
        model = lgb.LGBMRegressor(objective="regression", **common)
    elif case.task == "binary_classification":
        params.update({"objective": "binary", "topology": "leafwise"})
        model = lgb.LGBMClassifier(objective="binary", **common)
    elif case.task == "ranking":
        params.update({"objective": "lambdarank", "topology": "leafwise"})
        model = lgb.LGBMRanker(objective="lambdarank", **common)
    elif case.task == "multi_output_regression":
        from sklearn.multioutput import MultiOutputRegressor
        model = MultiOutputRegressor(lgb.LGBMRegressor(objective="regression", **common))
        params["multi_output_strategy"] = "independent_estimators"
        params.update({"objective": "regression", "topology": "leafwise"})
    else:
        raise ValueError(f"unsupported task: {case.task}")
    started = time.perf_counter_ns()
    if case.task == "ranking":
        model.fit(X_train, case.y_train, group=np.bincount(case.group_train))
    else:
        model.fit(X_train, case.y_train)
    fit_seconds = _seconds(started, time.perf_counter_ns())
    started = time.perf_counter_ns()
    predictions = model.predict_proba(X_test)[:, 1] if case.task == "binary_classification" else model.predict(X_test)
    predict_seconds = _seconds(started, time.perf_counter_ns())
    return AdapterResult(np.asarray(predictions), prep_seconds, fit_seconds, predict_seconds, _rss_bytes(), "lightgbm", _version("lightgbm"), params, representation, _completed_rounds(model, case.rounds))


def _fit_xgboost(case: DatasetCase, seed: int, threads: int) -> AdapterResult:
    xgb = _import_optional("xgboost", "xgboost")
    X_train, X_test, representation, prep_params, prep_seconds, _ = _prepare_input(case, "xgboost")
    params = _base_params(case, seed, threads) | prep_params
    common = dict(n_estimators=case.rounds, max_depth=case.depth, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, max_bin=256, tree_method="hist", random_state=seed, n_jobs=threads, enable_categorical=bool(case.categorical_feature_indices))
    params.update({"max_bin": 256, "tree_method": "hist"})
    if case.task == "regression":
        params.update({"objective": "reg:squarederror", "topology": "depthwise"})
        model = xgb.XGBRegressor(objective="reg:squarederror", **common)
    elif case.task == "binary_classification":
        params.update({"objective": "binary:logistic", "topology": "depthwise"})
        model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    elif case.task == "ranking":
        params.update({"objective": "rank:ndcg", "topology": "depthwise"})
        model = xgb.XGBRanker(objective="rank:ndcg", eval_metric="ndcg@10", **common)
    elif case.task == "multi_output_regression":
        model = xgb.XGBRegressor(objective="reg:squarederror", multi_strategy="multi_output_tree", **common)
        params.update({"multi_output_strategy": "multi_output_tree", "objective": "reg:squarederror", "topology": "depthwise"})
    else:
        raise ValueError(f"unsupported task: {case.task}")
    started = time.perf_counter_ns()
    if case.task == "ranking":
        model.fit(X_train, case.y_train, group=np.bincount(case.group_train))
    else:
        model.fit(X_train, case.y_train)
    fit_seconds = _seconds(started, time.perf_counter_ns())
    started = time.perf_counter_ns()
    predictions = model.predict_proba(X_test)[:, 1] if case.task == "binary_classification" else model.predict(X_test)
    predict_seconds = _seconds(started, time.perf_counter_ns())
    return AdapterResult(np.asarray(predictions), prep_seconds, fit_seconds, predict_seconds, _rss_bytes(), "xgboost", _version("xgboost"), params, representation, _completed_rounds(model, case.rounds))


def _fit_catboost(case: DatasetCase, seed: int, threads: int) -> AdapterResult:
    cat = _import_optional("catboost", "catboost")
    X_train, X_test, representation, prep_params, prep_seconds, _ = _prepare_input(case, "catboost")
    params = _base_params(case, seed, threads) | prep_params
    common = dict(iterations=case.rounds, depth=case.depth, learning_rate=0.05, rsm=0.8, random_seed=seed, thread_count=threads, bootstrap_type="Bernoulli", subsample=0.8, border_count=254, allow_writing_files=False, verbose=False)
    params.update({"bootstrap_type": "Bernoulli", "border_count": 254, "allow_writing_files": False})
    cat_features = list(case.categorical_feature_indices)
    if case.task == "regression":
        params.update({"objective": "RMSE", "topology": "symmetric_oblivious"})
        model = cat.CatBoostRegressor(loss_function="RMSE", **common)
    elif case.task == "binary_classification":
        params.update({"objective": "Logloss", "topology": "symmetric_oblivious"})
        model = cat.CatBoostClassifier(loss_function="Logloss", **common)
    elif case.task == "ranking":
        params.update({"objective": "YetiRank", "topology": "symmetric_oblivious"})
        model = cat.CatBoostRanker(loss_function="YetiRank", **common)
    elif case.task == "multi_output_regression":
        model = cat.CatBoostRegressor(loss_function="MultiRMSE", **common)
        params.update({"multi_output_strategy": "native_multi_rmse", "objective": "MultiRMSE", "topology": "symmetric_oblivious"})
    else:
        raise ValueError(f"unsupported task: {case.task}")
    started = time.perf_counter_ns()
    fit_kwargs = {"cat_features": cat_features} if cat_features else {}
    if case.task == "ranking":
        fit_kwargs["group_id"] = case.group_train
    model.fit(X_train, case.y_train, **fit_kwargs)
    fit_seconds = _seconds(started, time.perf_counter_ns())
    started = time.perf_counter_ns()
    predictions = model.predict_proba(X_test)[:, 1] if case.task == "binary_classification" else model.predict(X_test)
    predict_seconds = _seconds(started, time.perf_counter_ns())
    return AdapterResult(np.asarray(predictions), prep_seconds, fit_seconds, predict_seconds, _rss_bytes(), "catboost", _version("catboost"), params, representation, _completed_rounds(model, case.rounds))


ADAPTER_FACTORIES = {"alloygbm": _fit_alloy, "lightgbm": _fit_lightgbm, "xgboost": _fit_xgboost, "catboost": _fit_catboost}


class AlloyGBMAdapter:
    def fit_predict(self, case: DatasetCase, seed: int, threads: int) -> AdapterResult:
        return _fit_alloy(case, seed, threads)


class LightGBMAdapter:
    def fit_predict(self, case: DatasetCase, seed: int, threads: int) -> AdapterResult:
        return _fit_lightgbm(case, seed, threads)


class XGBoostAdapter:
    def fit_predict(self, case: DatasetCase, seed: int, threads: int) -> AdapterResult:
        return _fit_xgboost(case, seed, threads)


class CatBoostAdapter:
    def fit_predict(self, case: DatasetCase, seed: int, threads: int) -> AdapterResult:
        return _fit_catboost(case, seed, threads)


def load_adapters(libraries: list[str]) -> dict[str, Adapter]:
    """Load selected adapters; optional dependency failures remain explicit."""
    return {name: _FactoryAdapter(name, factory) for name, factory in ((name, ADAPTER_FACTORIES[name]) for name in libraries)}


class _FactoryAdapter:
    def __init__(self, name: str, factory) -> None:
        self.name, self.factory = name, factory

    def fit_predict(self, case: DatasetCase, seed: int, threads: int) -> AdapterResult:
        return self.factory(case, seed, threads)
