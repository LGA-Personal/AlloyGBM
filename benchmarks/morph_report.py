#!/usr/bin/env python3
"""Morph vs peers benchmark report.

Compares alloygbm (auto / morph / morph+cosine) against LightGBM, XGBoost,
and CatBoost on five sklearn datasets covering regression, binary
classification, multiclass classification, and ranking.

Usage::
    python benchmarks/morph_report.py [--quick] [--output results.csv]
"""

from __future__ import annotations

import argparse
import csv
import io
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_digits,
    load_wine,
)
from sklearn.metrics import accuracy_score, ndcg_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Optional imports — catch failures gracefully
# ---------------------------------------------------------------------------

try:
    from alloygbm import GBMClassifier, GBMRanker, GBMRegressor
    _ALLOYGBM_OK = True
except ImportError as _e:
    _ALLOYGBM_OK = False
    _ALLOYGBM_ERR = str(_e)

try:
    from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
    _LGBM_OK = True
except ImportError as _e:
    _LGBM_OK = False
    _LGBM_ERR = str(_e)

try:
    from xgboost import XGBClassifier, XGBRanker, XGBRegressor
    _XGB_OK = True
except ImportError as _e:
    _XGB_OK = False
    _XGB_ERR = str(_e)

try:
    from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor
    _CAT_OK = True
except ImportError as _e:
    _CAT_OK = False
    _CAT_ERR = str(_e)


# ---------------------------------------------------------------------------
# Benchmark profile
# ---------------------------------------------------------------------------

DEFAULT_N_ESTIMATORS = 300
QUICK_N_ESTIMATORS = 60
MAX_DEPTH = 6
LEARNING_RATE = 0.1
SEED = 42


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Row:
    dataset: str
    model: str
    metric1_name: str
    metric1: str        # formatted or "n/a" or "ERROR: ..."
    metric2_name: str
    metric2: str
    fit_sec: float


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def _build_california_housing(seed: int):
    """Regression: California Housing (RMSE + R²)."""
    from sklearn.metrics import mean_squared_error, r2_score as r2
    bunch = fetch_california_housing()
    X, y = bunch.data.astype(np.float32), bunch.target.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    n_total = len(X)
    n_train = len(X_tr)
    return dict(
        X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
        task="regression",
        n_total=n_total, n_train=n_train, n_features=X.shape[1],
        title="California Housing (regression)",
    )


def _build_breast_cancer(seed: int):
    """Binary classification: Breast Cancer (AUC + accuracy)."""
    bunch = load_breast_cancer()
    X, y = bunch.data.astype(np.float32), bunch.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    return dict(
        X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
        task="binary",
        n_total=len(X), n_train=len(X_tr), n_features=X.shape[1],
        title="Breast Cancer (binary classification)",
    )


def _build_wine(seed: int):
    """Multiclass classification: Wine (accuracy)."""
    bunch = load_wine()
    X, y = bunch.data.astype(np.float32), bunch.target
    n_classes = len(np.unique(y))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    return dict(
        X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
        task="multiclass",
        n_total=len(X), n_train=len(X_tr), n_features=X.shape[1],
        n_classes=n_classes,
        title="Wine (multiclass classification)",
    )


def _build_digits(seed: int):
    """Multiclass classification: Digits (accuracy)."""
    bunch = load_digits()
    X, y = bunch.data.astype(np.float32), bunch.target
    n_classes = len(np.unique(y))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    return dict(
        X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
        task="multiclass",
        n_total=len(X), n_train=len(X_tr), n_features=X.shape[1],
        n_classes=n_classes,
        title="Digits (multiclass classification)",
    )


def _build_california_ranking(seed: int):
    """Ranking: California Housing reframed as LTR (NDCG@10)."""
    bunch = fetch_california_housing(as_frame=True)
    frame = bunch.frame.copy()

    # Determine target column
    target_col = getattr(bunch, "target_names", ["target"])[0]
    if target_col not in frame.columns:
        frame["target"] = bunch.target
        target_col = "target"
    frame = frame.rename(columns={target_col: "median_house_value"})

    # Build query IDs: floor(Latitude) * 10 + floor(Longitude)
    lat_cell = np.floor(frame["Latitude"].values).astype(int)
    lon_cell = np.floor(frame["Longitude"].values).astype(int)
    raw_query_id = lat_cell * 10 + lon_cell
    frame["query_id"] = raw_query_id

    # Keep only queries with >= 10 docs
    group_sizes = frame.groupby("query_id").size()
    keep_ids = group_sizes[group_sizes >= 10].index
    frame = frame[frame["query_id"].isin(keep_ids)].copy()

    # Remap query IDs to contiguous integers
    sorted_ids = sorted(frame["query_id"].unique())
    id_remap = {old: new for new, old in enumerate(sorted_ids)}
    frame["query_id"] = frame["query_id"].map(id_remap)

    # Relevance: bin MedHouseVal into 5 quantile levels
    import pandas as pd
    frame["relevance"] = pd.qcut(
        frame["median_house_value"], q=5,
        labels=[0, 1, 2, 3, 4], duplicates="drop"
    ).astype(int)

    feature_cols = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                    "Population", "AveOccup", "Latitude", "Longitude"]

    # Sort by query_id, then split: 80% queries for train, 20% for test
    frame = frame.sort_values("query_id").reset_index(drop=True)
    unique_qids = frame["query_id"].unique()
    n_queries = len(unique_qids)
    split_q = int(n_queries * 0.8)
    train_qids = set(unique_qids[:split_q])
    test_qids = set(unique_qids[split_q:])

    train_mask = frame["query_id"].isin(train_qids)
    test_mask = frame["query_id"].isin(test_qids)

    X_tr = frame.loc[train_mask, feature_cols].values.astype(np.float32)
    y_tr = frame.loc[train_mask, "relevance"].values.astype(np.float32)
    group_tr = frame.loc[train_mask, "query_id"].values.astype(np.int32)

    X_te = frame.loc[test_mask, feature_cols].values.astype(np.float32)
    y_te = frame.loc[test_mask, "relevance"].values.astype(np.float32)
    group_te = frame.loc[test_mask, "query_id"].values.astype(np.int32)

    return dict(
        X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
        group_tr=group_tr, group_te=group_te,
        task="ranking",
        n_total=len(frame), n_train=len(X_tr), n_features=len(feature_cols),
        n_queries=n_queries,
        title="California Ranking (ranking — NDCG@10)",
    )


# ---------------------------------------------------------------------------
# NDCG@10 computation
# ---------------------------------------------------------------------------

def _compute_ndcg10(y_true_groups, scores_groups) -> float:
    """Compute mean NDCG@10 across all query groups."""
    ndcg_vals = []
    for yt, sc in zip(y_true_groups, scores_groups):
        if len(yt) < 2:
            continue
        yt_2d = np.asarray(yt, dtype=np.float64).reshape(1, -1)
        sc_2d = np.asarray(sc, dtype=np.float64).reshape(1, -1)
        ndcg_vals.append(ndcg_score(yt_2d, sc_2d, k=10))
    return float(np.mean(ndcg_vals)) if ndcg_vals else float("nan")


def _split_by_group(y, scores, group_ids):
    """Split y and scores into per-group lists."""
    unique_gids = sorted(set(group_ids))
    y_groups, s_groups = [], []
    for gid in unique_gids:
        mask = group_ids == gid
        y_groups.append(y[mask])
        s_groups.append(scores[mask])
    return y_groups, s_groups


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _make_models(n_est: int, n_classes: int | None, task: str) -> list[tuple[str, Any]]:
    """Return list of (name, model) for a given task."""
    D = MAX_DEPTH
    LR = LEARNING_RATE
    models = []

    # --- AlloyGBM ---
    if _ALLOYGBM_OK:
        common = dict(
            n_estimators=n_est, max_depth=D, learning_rate=LR,
            row_subsample=0.8, col_subsample=0.8, seed=SEED, deterministic=True,
        )
        if task == "regression":
            models += [
                ("alloygbm",              GBMRegressor(**common, training_mode="auto")),
                ("alloygbm_morph",        GBMRegressor(**common, training_mode="morph")),
                ("alloygbm_morph_cosine", GBMRegressor(**common, training_mode="morph",
                                                       lr_schedule="warmup_cosine",
                                                       lr_warmup_frac=0.1)),
            ]
        elif task == "binary":
            models += [
                ("alloygbm",              GBMClassifier(**common, training_mode="auto")),
                ("alloygbm_morph",        GBMClassifier(**common, training_mode="morph")),
                ("alloygbm_morph_cosine", GBMClassifier(**common, training_mode="morph",
                                                        lr_schedule="warmup_cosine",
                                                        lr_warmup_frac=0.1)),
            ]
        elif task == "multiclass":
            models += [
                ("alloygbm",              GBMClassifier(**common, training_mode="auto")),
                ("alloygbm_morph",        GBMClassifier(**common, training_mode="morph")),
                ("alloygbm_morph_cosine", GBMClassifier(**common, training_mode="morph",
                                                        lr_schedule="warmup_cosine",
                                                        lr_warmup_frac=0.1)),
            ]
        elif task == "ranking":
            models += [
                ("alloygbm",              GBMRanker(**common, training_mode="auto")),
                ("alloygbm_morph",        GBMRanker(**common, training_mode="morph")),
                ("alloygbm_morph_cosine", GBMRanker(**common, training_mode="morph",
                                                    lr_schedule="warmup_cosine",
                                                    lr_warmup_frac=0.1)),
            ]
    else:
        for name in ("alloygbm", "alloygbm_morph", "alloygbm_morph_cosine"):
            models.append((name, None))

    # --- LightGBM ---
    if _LGBM_OK:
        lgbm_common = dict(
            n_estimators=n_est, max_depth=D, learning_rate=LR,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED,
            n_jobs=1, verbose=-1,
        )
        if task == "regression":
            models.append(("lightgbm", LGBMRegressor(**lgbm_common)))
        elif task == "binary":
            models.append(("lightgbm", LGBMClassifier(**lgbm_common)))
        elif task == "multiclass":
            models.append(("lightgbm", LGBMClassifier(
                **lgbm_common, objective="multiclass", num_class=n_classes
            )))
        elif task == "ranking":
            models.append(("lightgbm", LGBMRanker(
                objective="lambdarank", **lgbm_common
            )))
    else:
        models.append(("lightgbm", None))

    # --- XGBoost ---
    if _XGB_OK:
        xgb_common = dict(
            n_estimators=n_est, max_depth=D, learning_rate=LR,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED,
            n_jobs=1, tree_method="hist", verbosity=0,
        )
        if task == "regression":
            models.append(("xgboost", XGBRegressor(**xgb_common)))
        elif task == "binary":
            models.append(("xgboost", XGBClassifier(**xgb_common)))
        elif task == "multiclass":
            models.append(("xgboost", XGBClassifier(
                **xgb_common, objective="multi:softprob", num_class=n_classes
            )))
        elif task == "ranking":
            models.append(("xgboost", XGBRanker(
                **xgb_common, objective="rank:ndcg"
            )))
    else:
        models.append(("xgboost", None))

    # --- CatBoost ---
    if _CAT_OK:
        cat_common = dict(
            iterations=n_est, depth=D, learning_rate=LR,
            random_seed=SEED, verbose=False,
            allow_writing_files=False, thread_count=1,
        )
        if task == "regression":
            models.append(("catboost", CatBoostRegressor(**cat_common)))
        elif task == "binary":
            models.append(("catboost", CatBoostClassifier(**cat_common)))
        elif task == "multiclass":
            models.append(("catboost", CatBoostClassifier(
                **cat_common, loss_function="MultiClass"
            )))
        elif task == "ranking":
            models.append(("catboost", CatBoostRanker(
                **cat_common, loss_function="YetiRank"
            )))
    else:
        models.append(("catboost", None))

    return models


# ---------------------------------------------------------------------------
# Run one dataset
# ---------------------------------------------------------------------------

def run_dataset(ds: dict, n_est: int) -> list[Row]:
    """Fit all models on a dataset and return result rows."""
    from sklearn.metrics import mean_squared_error, r2_score

    task = ds["task"]
    n_classes = ds.get("n_classes")
    X_tr, y_tr = ds["X_tr"], ds["y_tr"]
    X_te, y_te = ds["X_te"], ds["y_te"]
    group_tr = ds.get("group_tr")
    group_te = ds.get("group_te")

    models = _make_models(n_est, n_classes, task)
    rows = []

    for name, model in models:
        if model is None:
            err_msg = (
                _ALLOYGBM_ERR if name.startswith("alloygbm") else
                _LGBM_ERR if name == "lightgbm" else
                _XGB_ERR if name == "xgboost" else
                _CAT_ERR
            )
            short_err = f"ERROR: {err_msg[:50]}"
            rows.append(_error_row(ds["title"], name, task, short_err))
            continue

        try:
            t0 = time.perf_counter()

            if task == "regression":
                model.fit(X_tr, y_tr)
                fit_sec = time.perf_counter() - t0
                preds = np.asarray(model.predict(X_te), dtype=np.float64)
                rmse_val = float(np.sqrt(mean_squared_error(y_te, preds)))
                r2_val = float(r2_score(y_te, preds))
                rows.append(Row(
                    dataset=ds["title"], model=name,
                    metric1_name="RMSE", metric1=f"{rmse_val:.4f}",
                    metric2_name="R²",   metric2=f"{r2_val:.4f}",
                    fit_sec=fit_sec,
                ))

            elif task == "binary":
                model.fit(X_tr, y_tr)
                fit_sec = time.perf_counter() - t0
                if hasattr(model, "predict_proba"):
                    proba = np.asarray(model.predict_proba(X_te))
                    scores = proba[:, 1] if proba.ndim == 2 else proba
                elif hasattr(model, "predict"):
                    scores = np.asarray(model.predict(X_te), dtype=np.float64)
                else:
                    scores = np.asarray(model.predict(X_te), dtype=np.float64)
                preds = np.asarray(model.predict(X_te))
                auc_val = float(roc_auc_score(y_te, scores))
                acc_val = float(accuracy_score(y_te, preds))
                rows.append(Row(
                    dataset=ds["title"], model=name,
                    metric1_name="AUC",      metric1=f"{auc_val:.4f}",
                    metric2_name="Accuracy", metric2=f"{acc_val:.4f}",
                    fit_sec=fit_sec,
                ))

            elif task == "multiclass":
                model.fit(X_tr, y_tr)
                fit_sec = time.perf_counter() - t0
                preds = np.asarray(model.predict(X_te))
                # For XGBoost multiclass predict() may return floats
                preds = np.round(preds).astype(int)
                acc_val = float(accuracy_score(y_te, preds))
                rows.append(Row(
                    dataset=ds["title"], model=name,
                    metric1_name="Accuracy", metric1=f"{acc_val:.4f}",
                    metric2_name="n/a",      metric2="n/a",
                    fit_sec=fit_sec,
                ))

            elif task == "ranking":
                rows.append(_fit_ranking(ds, name, model, t0=t0))

        except Exception as exc:
            short_err = f"ERROR: {str(exc)[:60]}"
            rows.append(_error_row(ds["title"], name, task, short_err))

    return rows


def _fit_ranking(ds: dict, name: str, model: Any, t0: float | None = None) -> Row:
    """Fit a ranking model and compute NDCG@10."""
    X_tr, y_tr = ds["X_tr"], ds["y_tr"]
    X_te, y_te = ds["X_te"], ds["y_te"]
    group_tr = ds["group_tr"]
    group_te = ds["group_te"]

    if t0 is None:
        t0 = time.perf_counter()

    # Compute group sizes arrays for LightGBM (requires array of group sizes)
    # and per-sample group arrays for others
    unique_tr, counts_tr = np.unique(group_tr, return_counts=True)
    lgbm_group_sizes = counts_tr.tolist()

    if name == "lightgbm":
        model.fit(X_tr, y_tr.astype(int), group=lgbm_group_sizes)
    elif name == "xgboost":
        model.fit(X_tr, y_tr.astype(int), qid=group_tr)
    elif name == "catboost":
        model.fit(X_tr, y_tr.astype(int), group_id=group_tr)
    elif name.startswith("alloygbm"):
        model.fit(X_tr, y_tr, group=group_tr)
    else:
        model.fit(X_tr, y_tr)

    fit_sec = time.perf_counter() - t0

    scores = np.asarray(model.predict(X_te), dtype=np.float64)
    y_groups, s_groups = _split_by_group(y_te, scores, group_te)
    ndcg_val = _compute_ndcg10(y_groups, s_groups)

    return Row(
        dataset=ds["title"], model=name,
        metric1_name="NDCG@10", metric1=f"{ndcg_val:.4f}",
        metric2_name="n/a",     metric2="n/a",
        fit_sec=fit_sec,
    )


def _error_row(dataset: str, model: str, task: str, err: str) -> Row:
    if task == "regression":
        return Row(dataset=dataset, model=model,
                   metric1_name="RMSE", metric1=err,
                   metric2_name="R²",   metric2=err,
                   fit_sec=0.0)
    elif task == "binary":
        return Row(dataset=dataset, model=model,
                   metric1_name="AUC",      metric1=err,
                   metric2_name="Accuracy", metric2=err,
                   fit_sec=0.0)
    elif task == "multiclass":
        return Row(dataset=dataset, model=model,
                   metric1_name="Accuracy", metric1=err,
                   metric2_name="n/a",      metric2="n/a",
                   fit_sec=0.0)
    else:  # ranking
        return Row(dataset=dataset, model=model,
                   metric1_name="NDCG@10", metric1=err,
                   metric2_name="n/a",     metric2="n/a",
                   fit_sec=0.0)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_section(ds: dict, rows: list[Row]) -> None:
    task = ds["task"]
    title = ds["title"]
    n_total = ds["n_total"]
    n_train = ds["n_train"]
    n_features = ds["n_features"]

    print(f"\n## {title}")
    size_info = f"n={n_total}, train={n_train}, features={n_features}"
    if task == "ranking":
        size_info += f", queries={ds.get('n_queries', '?')}"
    print(f"  ({size_info})")
    print()

    # Table header
    if task == "regression":
        header = f"| {'Model':<23} | {'RMSE':>8} | {'R²':>8} | {'Fit (s)':>8} |"
        sep    = f"|{'-'*25}|{'-'*10}|{'-'*10}|{'-'*10}|"
    elif task == "binary":
        header = f"| {'Model':<23} | {'AUC':>8} | {'Accuracy':>8} | {'Fit (s)':>8} |"
        sep    = f"|{'-'*25}|{'-'*10}|{'-'*10}|{'-'*10}|"
    elif task == "multiclass":
        header = f"| {'Model':<23} | {'Accuracy':>8} | {'n/a':>8} | {'Fit (s)':>8} |"
        sep    = f"|{'-'*25}|{'-'*10}|{'-'*10}|{'-'*10}|"
    else:  # ranking
        header = f"| {'Model':<23} | {'NDCG@10':>8} | {'n/a':>8} | {'Fit (s)':>8} |"
        sep    = f"|{'-'*25}|{'-'*10}|{'-'*10}|{'-'*10}|"

    print(header)
    print(sep)
    for row in rows:
        m2 = row.metric2
        fit_str = f"{row.fit_sec:>8.1f}" if row.fit_sec > 0 else f"{'n/a':>8}"
        print(f"| {row.model:<23} | {row.metric1:>8} | {m2:>8} | {fit_str} |")
    print()


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _write_csv(all_rows: list[Row], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "model",
                         "metric1_name", "metric1",
                         "metric2_name", "metric2",
                         "fit_sec"])
        for r in all_rows:
            writer.writerow([r.dataset, r.model,
                              r.metric1_name, r.metric1,
                              r.metric2_name, r.metric2,
                              f"{r.fit_sec:.3f}"])
    print(f"[morph_report] CSV written to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Morph vs peers benchmark report"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help=f"Use {QUICK_N_ESTIMATORS} estimators instead of {DEFAULT_N_ESTIMATORS}"
    )
    parser.add_argument(
        "--output", metavar="FILE", default=None,
        help="Optional path to write CSV results (default: no CSV)"
    )
    args = parser.parse_args()

    n_est = QUICK_N_ESTIMATORS if args.quick else DEFAULT_N_ESTIMATORS
    mode_str = "quick" if args.quick else "full"
    print(f"\n# AlloyGBM Morph Report — {mode_str} mode (n_estimators={n_est})\n")

    dataset_builders = [
        _build_california_housing,
        _build_breast_cancer,
        _build_wine,
        _build_digits,
        _build_california_ranking,
    ]

    all_rows: list[Row] = []
    t_total = time.perf_counter()

    for builder in dataset_builders:
        ds = builder(seed=SEED)
        print(f"Running {ds['title']} ...", flush=True)
        t0 = time.perf_counter()
        rows = run_dataset(ds, n_est)
        elapsed = time.perf_counter() - t0
        _print_section(ds, rows)
        all_rows.extend(rows)
        print(f"  [dataset done in {elapsed:.1f}s]")

    total_elapsed = time.perf_counter() - t_total
    print(f"\n# Total elapsed: {total_elapsed:.1f}s")

    if args.output:
        _write_csv(all_rows, args.output)


if __name__ == "__main__":
    main()
