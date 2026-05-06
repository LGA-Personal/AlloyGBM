#!/usr/bin/env python3
"""
Numerai Tournament Benchmark — AlloyGBM vs LightGBM vs XGBoost vs CatBoost

Development benchmark for iterating on AlloyGBM training performance at
Numerai-scale data (~500K+ rows, 42-780 features, rank-normalised targets).

Downloads Numerai v5.2 data on first run (cached in benchmarks/data/numerai/).
Trains on benchmark-residualized targets, evaluates on validation with
standard Numerai metrics (numerai_corr, sharpe, MMC).

Usage:
    # Quick smoke test (42 features, 200 rounds, 400 eras)
    .venv/bin/python benchmarks/numerai_benchmark.py --fast

    # Medium feature set (780 features)
    .venv/bin/python benchmarks/numerai_benchmark.py --fast --feature-set medium

    # Just AlloyGBM for iteration speed
    .venv/bin/python benchmarks/numerai_benchmark.py --fast --arms alloygbm

    # Custom profile
    .venv/bin/python benchmarks/numerai_benchmark.py --fast --rounds 500 --learning-rate 0.03
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = Path(__file__).parent / "data" / "numerai"
ALL_ARMS = ["alloygbm", "alloygbm_morph", "alloygbm_morph_cosine", "lightgbm", "xgboost", "catboost"]

NUMERAI_DATASET_FILES = {
    "train": "v5.2/train.parquet",
    "validation": "v5.2/validation.parquet",
    "features": "v5.2/features.json",
    "train_benchmark": "v5.2/train_benchmark_models.parquet",
    "validation_benchmark": "v5.2/validation_benchmark_models.parquet",
    "meta_model": "v5.2/meta_model.parquet",
}


# ---------------------------------------------------------------------------
# Data management
# ---------------------------------------------------------------------------
def _local_path(key: str) -> Path:
    filename = NUMERAI_DATASET_FILES[key].split("/")[-1]
    return DATA_DIR / filename


def ensure_numerai_data() -> None:
    """Download Numerai v5.2 data files if not already present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    missing = [key for key in NUMERAI_DATASET_FILES if not _local_path(key).exists()]
    if not missing:
        logger.info("All Numerai data files present in %s", DATA_DIR)
        return

    from numerapi import NumerAPI

    api = NumerAPI()
    for key in missing:
        remote = NUMERAI_DATASET_FILES[key]
        local = _local_path(key)
        logger.info("Downloading %s -> %s", remote, local)
        api.download_dataset(remote, str(local))


def load_features_json() -> dict:
    with open(_local_path("features"), "r") as f:
        return json.load(f)


def resolve_feature_cols(feature_set: str) -> list[str]:
    """Resolve a named feature set from features.json."""
    features_json = load_features_json()
    feature_sets = features_json.get("feature_sets", {})
    if feature_set not in feature_sets:
        available = sorted(feature_sets.keys())
        raise ValueError(f"Unknown feature set '{feature_set}'. Available: {available}")
    return list(feature_sets[feature_set])


def load_parquet_subset(
    key: str,
    columns: list[str] | None = None,
    max_eras: int | None = None,
) -> pd.DataFrame:
    """Load a parquet file, optionally subsetting columns and eras."""
    path = _local_path(key)
    df = pd.read_parquet(path, columns=columns)
    if df.index.name == "id":
        df = df.reset_index()
    if max_eras is not None and "era" in df.columns:
        eras = sorted(df["era"].unique())
        if len(eras) > max_eras:
            keep = eras[-max_eras:]
            df = df[df["era"].isin(keep)]
    return df


# ---------------------------------------------------------------------------
# Scoring (self-contained, no external deps beyond numerapi)
# ---------------------------------------------------------------------------
def numerai_corr(predictions: pd.Series, targets: pd.Series) -> float:
    """Numerai correlation: Pearson of ranked predictions vs gaussianized targets."""
    ranked_preds = predictions.rank(pct=True, method="average")
    # Gaussianize via percent-point-function approximation
    from scipy.stats import norm

    gauss_preds = pd.Series(norm.ppf(ranked_preds.clip(0.001, 0.999)), index=predictions.index)
    gauss_targets = pd.Series(norm.ppf(targets.rank(pct=True, method="average").clip(0.001, 0.999)), index=targets.index)
    return float(gauss_preds.corr(gauss_targets))


def per_era_numerai_corr(
    predictions: pd.Series, targets: pd.Series, eras: pd.Series
) -> pd.Series:
    """Per-era Numerai correlations."""
    results = {}
    for era, idx in eras.groupby(eras).groups.items():
        p = predictions.loc[idx]
        t = targets.loc[idx]
        if len(p) > 10:
            results[era] = numerai_corr(p, t)
    return pd.Series(results, dtype=float)


def sharpe_ratio(per_era_scores: pd.Series) -> float:
    if len(per_era_scores) < 2 or per_era_scores.std() == 0:
        return 0.0
    return float(per_era_scores.mean() / per_era_scores.std())


def correlation_contribution(
    predictions: pd.Series, targets: pd.Series, meta_model: pd.Series
) -> float:
    """MMC: orthogonalized correlation contribution."""
    common = predictions.index.intersection(meta_model.index).intersection(targets.index)
    if len(common) < 100:
        return float("nan")
    p = predictions.loc[common].rank(pct=True, method="average")
    t = targets.loc[common]
    m = meta_model.loc[common].rank(pct=True, method="average")
    # Neutralize predictions against meta model
    beta = float(np.cov(p, m)[0, 1] / (np.var(m) + 1e-10))
    neutralized = p - beta * m
    return float(neutralized.corr(t))


def evaluate_predictions(
    preds: np.ndarray,
    ids: Any,
    eras: Any,
    targets: np.ndarray,
    *,
    meta_model: pd.Series | None = None,
    benchmark_proxy: pd.Series | None = None,
) -> dict[str, float]:
    """Compute standard Numerai metrics."""
    idx = pd.Index(list(ids), name="id")
    pred_s = pd.Series(np.asarray(preds), index=idx, name="prediction")
    target_s = pd.Series(np.asarray(targets), index=idx, name="target")
    era_s = pd.Series(np.asarray(eras), index=idx, name="era")

    per_era = per_era_numerai_corr(pred_s, target_s, era_s).dropna()
    metrics: dict[str, float] = {
        "numerai_corr": numerai_corr(pred_s, target_s),
        "mean_per_era_corr": float(per_era.mean()) if not per_era.empty else 0.0,
        "std_per_era_corr": float(per_era.std(ddof=0)) if len(per_era) > 1 else 0.0,
        "sharpe": sharpe_ratio(per_era),
        "positive_eras_pct": float((per_era > 0).mean() * 100.0) if not per_era.empty else 0.0,
        "n_eras": int(len(per_era)),
    }

    if meta_model is not None:
        metrics["mmc"] = correlation_contribution(pred_s, target_s, meta_model)

    if benchmark_proxy is not None:
        metrics["benchmark_mmc"] = correlation_contribution(pred_s, target_s, benchmark_proxy)

    return metrics


# ---------------------------------------------------------------------------
# Walk-forward CV
# ---------------------------------------------------------------------------
def build_cv_folds(
    eras: list[str], chunk_size: int = 156, purge_eras: int = 8
) -> list[tuple[list[str], list[str]]]:
    """Walk-forward era-based splits."""
    unique = sorted(set(eras))
    n = len(unique)
    min_needed = chunk_size + purge_eras + chunk_size
    if n < min_needed:
        mid = n // 2
        return [(unique[:mid], unique[mid:])]

    folds = []
    test_start = chunk_size + purge_eras
    while test_start + chunk_size <= n:
        train_end = test_start - purge_eras
        train = unique[:train_end]
        test = unique[test_start : test_start + chunk_size]
        folds.append((train, test))
        test_start += chunk_size
    return folds if folds else [(unique[: n // 2], unique[n // 2 :])]


# ---------------------------------------------------------------------------
# Benchmark residualization
# ---------------------------------------------------------------------------
def build_benchmark_avg(benchmark_df: pd.DataFrame) -> pd.Series:
    """Average all numeric benchmark columns, indexed by id."""
    num_cols = benchmark_df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        raise ValueError("No numeric columns found in benchmark data")
    avg = benchmark_df[num_cols].mean(axis=1)
    # id may be the index (Numerai default) or a column
    ids = benchmark_df.index if benchmark_df.index.name == "id" else benchmark_df["id"].values
    return pd.Series(avg.values, index=ids, name="benchmark_avg")


def residualize_targets(
    targets: np.ndarray, ids: np.ndarray, benchmark: pd.Series, weight: float
) -> tuple[np.ndarray, np.ndarray]:
    bench = benchmark.reindex(ids).values.astype(np.float64)
    valid = np.isfinite(bench)
    residual = targets.copy().astype(np.float64)
    residual[valid] = targets[valid] - weight * bench[valid]
    return residual, valid


def deresidualize_predictions(
    preds: np.ndarray, ids: np.ndarray, benchmark: pd.Series, weight: float
) -> np.ndarray:
    bench = benchmark.reindex(ids).fillna(0.0).values.astype(np.float64)
    return preds + weight * bench


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def fit_model(
    arm: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    rounds: int,
    learning_rate: float,
    max_depth: int,
    row_subsample: float,
    col_subsample: float,
    seed: int,
    binning_strategy: str,
    feature_names: list[str] | None = None,
) -> tuple[object, dict[str, float]]:
    """Train a model and return (model_object, fit_timing_dict)."""

    if arm in ("alloygbm", "alloygbm_morph", "alloygbm_morph_cosine"):
        from alloygbm import GBMRegressor

        morph_kwargs: dict = {}
        if arm == "alloygbm_morph":
            morph_kwargs = {"training_mode": "morph"}
        elif arm == "alloygbm_morph_cosine":
            morph_kwargs = {
                "training_mode": "morph",
                "lr_schedule": "warmup_cosine",
                "lr_warmup_frac": 0.1,
            }

        model = GBMRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=rounds,
            row_subsample=row_subsample,
            col_subsample=col_subsample,
            min_data_in_leaf=5000,
            lambda_l2=1.0,
            min_child_hessian=5000.0,
            seed=seed,
            deterministic=True,
            continuous_binning_strategy=binning_strategy,
            continuous_binning_max_bins=256,
            **morph_kwargs,
        )
        model.fit(X_train, y_train)
        timing = {k: float(v) for k, v in getattr(model, "fit_timing_", {}).items()}

    elif arm == "lightgbm":
        import lightgbm as lgb

        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": 2**max_depth,
            "bagging_fraction": row_subsample,
            "bagging_freq": 1,
            "feature_fraction": col_subsample,
            "min_data_in_leaf": 5000,
            "lambda_l2": 1.0,
            "verbose": -1,
            "seed": seed,
            "feature_pre_filter": False,
        }
        ds = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
        model = lgb.train(params, ds, num_boost_round=rounds)
        timing = {}
        del ds

    elif arm == "xgboost":
        import xgboost as xgb

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": row_subsample,
            "colsample_bytree": col_subsample,
            "min_child_weight": 5000.0,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "seed": seed,
            "verbosity": 0,
        }
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        model = (xgb.train(params, dtrain, num_boost_round=rounds), feature_names)
        timing = {}
        del dtrain

    elif arm == "catboost":
        import catboost as cb

        params = {
            "loss_function": "RMSE",
            "learning_rate": learning_rate,
            "depth": max_depth,
            "subsample": row_subsample,
            "rsm": col_subsample,
            "min_data_in_leaf": 5000,
            "l2_leaf_reg": 1.0,
            "iterations": rounds,
            "random_seed": seed,
            "verbose": 0,
            "task_type": "CPU",
            "bootstrap_type": "MVS",
        }
        pool = cb.Pool(X_train, label=y_train)
        model = cb.CatBoost(params)
        model.fit(pool, verbose=0)
        timing = {}
        del pool
    else:
        raise ValueError(f"Unknown arm: {arm}")

    gc.collect()
    return model, timing


def predict_model(arm: str, model: object, X_predict: np.ndarray) -> np.ndarray:
    """Predict using a fitted model."""
    if arm in ("alloygbm", "alloygbm_morph", "alloygbm_morph_cosine"):
        preds = model.predict(X_predict)
    elif arm == "lightgbm":
        preds = model.predict(X_predict)
    elif arm == "xgboost":
        import xgboost as xgb
        xgb_model, feature_names = model
        dpred = xgb.DMatrix(X_predict, feature_names=feature_names)
        preds = xgb_model.predict(dpred)
    elif arm == "catboost":
        preds = model.predict(X_predict)
    else:
        raise ValueError(f"Unknown arm: {arm}")
    return np.asarray(preds, dtype=np.float64)


# ---------------------------------------------------------------------------
# Benchmark record
# ---------------------------------------------------------------------------
@dataclass
class NumeraiBenchmarkRecord:
    arm: str
    phase: str  # "cv" or "validation"
    rounds: int
    learning_rate: float
    max_depth: int
    feature_set: str
    n_features: int
    train_rows: int
    predict_rows: int
    wall_seconds: float
    numerai_corr: float
    sharpe: float
    mean_per_era_corr: float
    positive_eras_pct: float
    mmc: float
    benchmark_mmc: float
    fit_timing: dict[str, float]


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_benchmark(args: argparse.Namespace) -> list[NumeraiBenchmarkRecord]:
    ensure_numerai_data()

    # Resolve features
    feature_cols = resolve_feature_cols(args.feature_set)
    logger.info("Feature set '%s': %d features", args.feature_set, len(feature_cols))

    # Load data
    meta_cols = ["id", "era", "target"]
    train_cols = meta_cols + feature_cols
    val_cols = meta_cols + feature_cols

    logger.info("Loading training data...")
    train_df = load_parquet_subset("train", columns=train_cols, max_eras=args.max_train_eras)
    # Downcast feature columns to float32 to halve memory usage
    for col in feature_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(np.float32)
    logger.info("Train: %d rows, %d eras", len(train_df), train_df["era"].nunique())

    # Load benchmarks
    logger.info("Loading benchmark models...")
    bench_train_df = load_parquet_subset("train_benchmark")
    bench_val_df = load_parquet_subset("validation_benchmark")
    benchmark_train = build_benchmark_avg(bench_train_df)
    benchmark_val = build_benchmark_avg(bench_val_df)
    del bench_train_df, bench_val_df

    # Load meta model
    meta_model_s = None
    try:
        meta_df = load_parquet_subset("meta_model")
        if "numerai_meta_model" in meta_df.columns:
            meta_model_s = pd.Series(
                meta_df["numerai_meta_model"].values,
                index=meta_df["id"].values,
                name="numerai_meta_model",
            )
        del meta_df
    except Exception:
        logger.warning("Could not load meta_model — MMC will be unavailable")

    # Build CV folds
    folds = build_cv_folds(
        train_df["era"].tolist(), chunk_size=args.chunk_size, purge_eras=args.purge_eras
    )
    logger.info("Walk-forward CV: %d folds (chunk=%d, purge=%d)", len(folds), args.chunk_size, args.purge_eras)

    records: list[NumeraiBenchmarkRecord] = []

    for arm in args.resolved_arms:
        logger.info("=" * 60)
        logger.info("ARM: %s", arm)
        logger.info("=" * 60)

        model_kwargs = dict(
            rounds=args.rounds,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            row_subsample=args.row_subsample,
            col_subsample=args.col_subsample,
            seed=args.seed,
            binning_strategy=args.binning_strategy,
            feature_names=feature_cols,
        )

        # --- Walk-forward CV ---
        oof_parts: list[pd.DataFrame] = []
        cv_wall = 0.0
        cv_stage_timing: dict[str, float] = {}

        for fold_idx, (train_eras, test_eras) in enumerate(folds, 1):
            fold_train = train_df[train_df["era"].isin(train_eras)]
            fold_test = train_df[train_df["era"].isin(test_eras)]

            train_ids = fold_train["id"].values
            test_ids = fold_test["id"].values
            y_raw = fold_train["target"].values.astype(np.float64)

            y_train, _valid = residualize_targets(y_raw, train_ids, benchmark_train, args.residual_weight)
            # Keep all rows — residualize_targets only adjusts rows with benchmark data,
            # leaving the rest as raw targets. Only filter out NaN targets.
            target_valid = np.isfinite(y_train)
            if not target_valid.all():
                fold_train = fold_train[target_valid]
                y_train = y_train[target_valid]

            X_train = fold_train[feature_cols].to_numpy(dtype=np.float32, copy=False)
            X_test = fold_test[feature_cols].to_numpy(dtype=np.float32, copy=False)

            t0 = time.time()
            model, fit_timing = fit_model(arm, X_train, y_train, **model_kwargs)
            del X_train, y_train
            gc.collect()
            preds = predict_model(arm, model, X_test)
            cv_wall += time.time() - t0
            del model
            for k, v in fit_timing.items():
                cv_stage_timing[k] = cv_stage_timing.get(k, 0.0) + v

            preds = deresidualize_predictions(preds, test_ids, benchmark_train, args.residual_weight)
            oof_parts.append(pd.DataFrame({
                "id": test_ids, "era": fold_test["era"].values,
                "target": fold_test["target"].values, "prediction": preds,
            }))
            del fold_train, fold_test, X_test, preds
            gc.collect()

        oof_df = pd.concat(oof_parts, ignore_index=True)
        oof_metrics = evaluate_predictions(
            oof_df["prediction"].values, ids=oof_df["id"].values,
            eras=oof_df["era"].values, targets=oof_df["target"].values,
            benchmark_proxy=benchmark_train,
        )
        records.append(NumeraiBenchmarkRecord(
            arm=arm, phase="cv", rounds=args.rounds,
            learning_rate=args.learning_rate, max_depth=args.max_depth,
            feature_set=args.feature_set, n_features=len(feature_cols),
            train_rows=len(train_df), predict_rows=len(oof_df),
            wall_seconds=round(cv_wall, 2),
            numerai_corr=oof_metrics["numerai_corr"],
            sharpe=oof_metrics["sharpe"],
            mean_per_era_corr=oof_metrics["mean_per_era_corr"],
            positive_eras_pct=oof_metrics["positive_eras_pct"],
            mmc=oof_metrics.get("mmc", float("nan")),
            benchmark_mmc=oof_metrics.get("benchmark_mmc", float("nan")),
            fit_timing=cv_stage_timing,
        ))

        # --- Full train -> validation ---
        # Extract training features/targets, then free train_df to reduce memory
        train_ids_all = train_df["id"].values
        y_all_raw = train_df["target"].values.astype(np.float64)
        y_all, _valid_all = residualize_targets(y_all_raw, train_ids_all, benchmark_train, args.residual_weight)
        target_valid_all = np.isfinite(y_all)
        train_filtered = train_df[target_valid_all] if not target_valid_all.all() else train_df
        y_all = y_all[target_valid_all] if not target_valid_all.all() else y_all
        val_train_rows = len(train_filtered)

        X_train_full = train_filtered[feature_cols].to_numpy(dtype=np.float32, copy=False)
        is_last_arm = (arm == args.resolved_arms[-1])
        del train_filtered, train_ids_all, y_all_raw, oof_df
        if is_last_arm:
            del train_df
        gc.collect()

        # Train with only X_train_full + y_all in memory (~6GB for 780 features)
        t0 = time.time()
        model, val_fit_timing = fit_model(arm, X_train_full, y_all, **model_kwargs)
        del X_train_full, y_all
        gc.collect()

        # Now load validation data — train_df is gone so peak memory stays low
        logger.info("Loading validation data for prediction...")
        val_df = load_parquet_subset("validation", columns=val_cols, max_eras=args.max_val_eras)
        for col in feature_cols:
            if col in val_df.columns:
                val_df[col] = val_df[col].astype(np.float32)
        val_predict_rows = len(val_df)

        val_ids = val_df["id"].values.copy()
        val_eras = val_df["era"].values.copy()
        val_targets = val_df["target"].values.copy()
        X_val = val_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        del val_df
        gc.collect()

        val_preds = predict_model(arm, model, X_val)
        val_wall = time.time() - t0
        del model, X_val
        gc.collect()

        val_preds = deresidualize_predictions(val_preds, val_ids, benchmark_val, args.residual_weight)

        val_metrics = evaluate_predictions(
            val_preds, ids=val_ids, eras=val_eras,
            targets=val_targets,
            meta_model=meta_model_s, benchmark_proxy=benchmark_val,
        )
        records.append(NumeraiBenchmarkRecord(
            arm=arm, phase="validation", rounds=args.rounds,
            learning_rate=args.learning_rate, max_depth=args.max_depth,
            feature_set=args.feature_set, n_features=len(feature_cols),
            train_rows=val_train_rows, predict_rows=val_predict_rows,
            wall_seconds=round(val_wall, 2),
            numerai_corr=val_metrics["numerai_corr"],
            sharpe=val_metrics["sharpe"],
            mean_per_era_corr=val_metrics["mean_per_era_corr"],
            positive_eras_pct=val_metrics["positive_eras_pct"],
            mmc=val_metrics.get("mmc", float("nan")),
            benchmark_mmc=val_metrics.get("benchmark_mmc", float("nan")),
            fit_timing=val_fit_timing,
        ))

        logger.info("  %s — CV: %.1fs, Val: %.1fs", arm, cv_wall, val_wall)
        del val_preds
        gc.collect()

    return records


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_results(records: list[NumeraiBenchmarkRecord]) -> None:
    val_records = [r for r in records if r.phase == "validation"]
    cv_records = [r for r in records if r.phase == "cv"]

    for label, subset in [("OOF (Walk-Forward CV)", cv_records), ("Validation", val_records)]:
        if not subset:
            continue
        print(f"\n{'=' * 88}")
        print(f"  {label}")
        print(f"{'=' * 88}")
        header = f"{'Arm':>12} | {'numerai_corr':>12} | {'sharpe':>8} | {'mmc':>8} | {'bench_mmc':>9} | {'pos_eras%':>9} | {'seconds':>8}"
        print(header)
        print("-" * 88)
        for r in subset:
            mmc_str = f"{r.mmc:>8.4f}" if np.isfinite(r.mmc) else f"{'N/A':>8}"
            bmmc_str = f"{r.benchmark_mmc:>9.4f}" if np.isfinite(r.benchmark_mmc) else f"{'N/A':>9}"
            print(
                f"{r.arm:>12} | {r.numerai_corr:>12.4f} | {r.sharpe:>8.4f} | "
                f"{mmc_str} | {bmmc_str} | {r.positive_eras_pct:>9.1f} | {r.wall_seconds:>8.1f}"
            )

    # Timing breakdown for AlloyGBM
    alloy_records = [r for r in records if r.arm == "alloygbm" and r.fit_timing]
    if alloy_records:
        print(f"\n{'=' * 60}")
        print("  AlloyGBM fit_timing_ breakdown (seconds)")
        print(f"{'=' * 60}")
        for r in alloy_records:
            print(f"  [{r.phase}]")
            for k, v in sorted(r.fit_timing.items()):
                print(f"    {k}: {v:.3f}")


def save_results(records: list[NumeraiBenchmarkRecord], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"numerai_benchmark_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "records": [asdict(r) for r in records],
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Numerai tournament benchmark for AlloyGBM")
    parser.add_argument("--fast", action="store_true", help="Subset eras and reduce rounds")
    parser.add_argument("--feature-set", default="small", help="Feature set from features.json (default: small)")
    parser.add_argument("--rounds", type=int, default=None, help="Override n_estimators")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--row-subsample", type=float, default=0.8)
    parser.add_argument("--col-subsample", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--residual-weight", type=float, default=0.5)
    parser.add_argument("--no-residualize", action="store_true")
    parser.add_argument("--binning-strategy", default="quantile", choices=["linear", "rank", "quantile"])
    parser.add_argument("--chunk-size", type=int, default=156)
    parser.add_argument("--purge-eras", type=int, default=8)
    parser.add_argument("--max-train-eras", type=int, default=None)
    parser.add_argument("--max-val-eras", type=int, default=None)
    parser.add_argument("--arms", nargs="+", default=None, choices=ALL_ARMS)
    parser.add_argument("--output-dir", default=None, help="Save JSON results here")
    args = parser.parse_args()

    if args.no_residualize:
        args.residual_weight = 0.0

    if args.fast:
        args.max_train_eras = args.max_train_eras or 400
        args.max_val_eras = args.max_val_eras or 400
        if args.rounds is None:
            args.rounds = 200
    if args.rounds is None:
        args.rounds = 1200

    # Resolve arms
    requested = args.arms or list(ALL_ARMS)
    try:
        from alloygbm import GBMRegressor  # noqa: F401
    except ImportError:
        alloy_arms = [a for a in requested if a.startswith("alloygbm")]
        if alloy_arms:
            logger.warning("alloygbm not installed — skipping %s", alloy_arms)
            requested = [a for a in requested if not a.startswith("alloygbm")]
    args.resolved_arms = requested

    if not args.resolved_arms:
        logger.error("No arms to run")
        return

    logger.info(
        "Config: arms=%s, rounds=%d, lr=%.3f, depth=%d, features=%s, residual_weight=%.2f",
        args.resolved_arms, args.rounds, args.learning_rate, args.max_depth,
        args.feature_set, args.residual_weight,
    )

    records = run_benchmark(args)
    print_results(records)

    output_dir = Path(args.output_dir) if args.output_dir else (REPO_ROOT / "benchmarks" / "results" / "numerai")
    save_results(records, output_dir)


if __name__ == "__main__":
    main()
