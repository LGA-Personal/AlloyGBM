#!/usr/bin/env python3
"""
Numerai at-scale benchmark: AlloyGBM vs LightGBM vs XGBoost vs CatBoost.

A special benchmark for large, low signal-to-noise regression. The OpenML
suites are small to medium tables; Numerai is ~3M labelled rows of noisy
stock-market features where a model needs many deep, heavily regularised
trees to find the signal. The full data does not fit in memory, so the
benchmark reads three chronological era chunks straight from the parquet
files (only the requested eras and columns are loaded):

    [ train chunk ] gap [ validation chunk ] gap [ test chunk ]

* train      - fits every library
* validation - early stopping (RMSE), shared by every library
* test       - held-out scoring: RMSE plus Numerai's per-era correlation

Chunks are contiguous runs of eras, never share an era, and are separated
by an embargo of ``--gap-eras`` eras (default 4) because the 20-day targets
of neighbouring weekly eras overlap. Eras are taken from ``train.parquet``
and ``validation.parquet`` as one timeline; eras without targets (live
rows) are skipped. By default the test chunk ends at the last labelled era.

Every library gets the same deep, regularised configuration (see
``PROFILES``); knobs a library cannot express are recorded per fit as
``unmapped`` so the comparison stays honest.

Usage:
    # Pipeline check: lr 0.05, up to 3,000 trees, small feature set
    .venv/bin/python benchmarks/numerai_scale.py --profile quick --feature-set small

    # The deep profile (lr 0.005, up to 30,000 trees, early stopping 500)
    .venv/bin/python benchmarks/numerai_scale.py --profile deep

    # One library at a time keeps peak memory down on a laptop
    .venv/bin/python benchmarks/numerai_scale.py --profile deep --arms alloygbm
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from numerai_benchmark import (  # noqa: E402
    _gather_reproducibility_manifest,
    _local_path,
    _log_alloygbm_build_info,
    ensure_numerai_data,
    numerai_corr,
    per_era_numerai_corr,
    resolve_feature_cols,
    sharpe_ratio,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
ALL_ARMS = ["alloygbm", "lightgbm", "xgboost", "catboost"]
DATA_KEYS = ("train", "validation", "features")


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Profile:
    """Library-neutral hyperparameters shared by every arm.

    ``num_leaves`` with ``max_depth`` means leaf-wise (best-first) growth
    capped at that depth. ``min_data_in_leaf`` is a row count; for squared
    error every row has hessian 1, so it is also the minimum hessian sum.
    ``subsample``/``colsample`` are per-tree row and feature fractions.
    """

    name: str
    description: str
    n_estimators: int
    learning_rate: float
    max_depth: int
    num_leaves: int
    colsample: float
    subsample: float
    min_data_in_leaf: int
    min_split_gain: float
    l1: float
    l2: float
    max_bin: int
    early_stopping: int


PROFILES = {
    # Translated from Logan's Numerai LightGBM config.
    "deep": Profile(
        "deep",
        "lr 0.005, up to 30,000 trees, depth 12 / 1024 leaves, 25% features, "
        "80% rows, 10,000 rows per leaf, early stopping 500",
        n_estimators=30_000, learning_rate=0.005, max_depth=12, num_leaves=1024,
        colsample=0.25, subsample=0.8, min_data_in_leaf=10_000, min_split_gain=0.001,
        l1=0.01, l2=0.05, max_bin=511, early_stopping=500,
    ),
    # Same tree shape and regularisation, 10x the learning rate: a run that
    # finishes in a sitting and checks the pipeline end to end.
    "quick": Profile(
        "quick",
        "lr 0.05, up to 3,000 trees, depth 12 / 1024 leaves, 25% features, "
        "80% rows, 10,000 rows per leaf, early stopping 100",
        n_estimators=3_000, learning_rate=0.05, max_depth=12, num_leaves=1024,
        colsample=0.25, subsample=0.8, min_data_in_leaf=10_000, min_split_gain=0.001,
        l1=0.01, l2=0.05, max_bin=511, early_stopping=100,
    ),
}


def library_params(profile: Profile, library: str, *, seed: int, threads: int,
                   catboost_grow_policy: str = "Lossguide",
                   lightgbm_bagging_freq: int = 1) -> tuple[dict, list[str]]:
    """Map a profile onto one library's constructor arguments.

    Returns the parameters and the profile knobs the library has no
    equivalent for.
    """
    p = profile
    unmapped: list[str] = []
    if library == "alloygbm":
        params = dict(
            n_estimators=p.n_estimators, learning_rate=p.learning_rate,
            max_depth=p.max_depth, tree_growth="leaf", max_leaves=p.num_leaves,
            col_subsample=p.colsample, row_subsample=p.subsample,
            min_data_in_leaf=p.min_data_in_leaf, min_split_gain=p.min_split_gain,
            lambda_l1=p.l1, lambda_l2=p.l2, continuous_binning_max_bins=p.max_bin,
            early_stopping_rounds=p.early_stopping,
            # The auto policy would add its own leaf regularisation on top.
            training_policy="manual",
            seed=seed, n_jobs=threads,
        )
    elif library == "lightgbm":
        params = dict(
            objective="regression", n_estimators=p.n_estimators,
            learning_rate=p.learning_rate, max_depth=p.max_depth, num_leaves=p.num_leaves,
            colsample_bytree=p.colsample, subsample=p.subsample,
            subsample_freq=lightgbm_bagging_freq,
            min_child_samples=p.min_data_in_leaf, min_split_gain=p.min_split_gain,
            reg_alpha=p.l1, reg_lambda=p.l2, max_bin=p.max_bin,
            random_state=seed, n_jobs=threads, verbose=-1,
        )
    elif library == "xgboost":
        params = dict(
            objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
            n_estimators=p.n_estimators, learning_rate=p.learning_rate,
            grow_policy="lossguide", max_depth=p.max_depth, max_leaves=p.num_leaves,
            colsample_bytree=p.colsample, subsample=p.subsample,
            # Squared error: hessian 1 per row, so this is a row minimum.
            min_child_weight=float(p.min_data_in_leaf), gamma=p.min_split_gain,
            reg_alpha=p.l1, reg_lambda=p.l2, max_bin=p.max_bin,
            early_stopping_rounds=p.early_stopping,
            random_state=seed, n_jobs=threads,
        )
    elif library == "catboost":
        params = dict(
            loss_function="RMSE", iterations=p.n_estimators,
            learning_rate=p.learning_rate, depth=p.max_depth,
            grow_policy=catboost_grow_policy,
            rsm=p.colsample, bootstrap_type="Bernoulli", subsample=p.subsample,
            l2_leaf_reg=p.l2, border_count=p.max_bin,
            early_stopping_rounds=p.early_stopping, use_best_model=True,
            random_seed=seed, thread_count=threads, verbose=0,
            allow_writing_files=False,
        )
        if catboost_grow_policy == "Lossguide":
            params.update(max_leaves=p.num_leaves, min_data_in_leaf=p.min_data_in_leaf)
        elif catboost_grow_policy == "Depthwise":
            params["min_data_in_leaf"] = p.min_data_in_leaf
            unmapped.append("num_leaves")
        else:
            # Symmetric (oblivious) trees ignore both leaf limits.
            unmapped += ["num_leaves", "min_data_in_leaf"]
        unmapped += ["l1", "min_split_gain"]
    else:
        raise ValueError(f"unknown library {library!r}")
    return params, unmapped


# ---------------------------------------------------------------------------
# Chronological chunks
# ---------------------------------------------------------------------------
def era_sort_key(era: object) -> tuple[int, object]:
    """Numerai eras are zero-padded numbers ("0001"); sort them as numbers."""
    try:
        return (0, int(str(era).removeprefix("era")))
    except ValueError:
        return (1, str(era))


@dataclass(frozen=True)
class EraInfo:
    era: str
    source: str  # "train" or "validation"
    rows: int


def labelled_era_timeline(sources: dict[str, Path], target: str) -> list[EraInfo]:
    """Every era with at least one labelled row, oldest first.

    Reads only the era and target columns. An era present in both files
    (should not happen) keeps the earlier source.
    """
    import pyarrow.parquet as pq

    seen: dict[str, EraInfo] = {}
    for source, path in sources.items():
        table = pq.read_table(path, columns=["era", target])
        frame = pd.DataFrame({
            "era": table.column("era").to_numpy(zero_copy_only=False).astype(str),
            "target": table.column(target).to_numpy(zero_copy_only=False),
        })
        del table
        counts = frame.dropna(subset=["target"]).groupby("era").size()
        for era, rows in counts.items():
            if era not in seen:
                seen[era] = EraInfo(str(era), source, int(rows))
        del frame
    return sorted(seen.values(), key=lambda e: era_sort_key(e.era))


@dataclass(frozen=True)
class ChunkPlan:
    train: list[str]
    valid: list[str]
    test: list[str]
    gap: int

    def describe(self) -> dict:
        def span(eras: list[str]) -> dict:
            return {"first": eras[0], "last": eras[-1], "eras": len(eras)}

        return {"train": span(self.train), "valid": span(self.valid),
                "test": span(self.test), "gap_eras": self.gap}


def plan_chunks(eras: list[str], *, train_eras: int, valid_eras: int, test_eras: int,
                gap_eras: int, end_offset: int = 0) -> ChunkPlan:
    """Split an oldest-first era list into train / valid / test chunks.

    The test chunk ends ``end_offset`` eras before the newest era; valid
    sits right before it and train before that, each separated by
    ``gap_eras`` embargoed eras that are used by nobody.
    """
    if min(train_eras, valid_eras, test_eras) < 1:
        raise ValueError("each chunk needs at least one era")
    if gap_eras < 0 or end_offset < 0:
        raise ValueError("gap_eras and end_offset must be non-negative")
    needed = train_eras + valid_eras + test_eras + 2 * gap_eras + end_offset
    if needed > len(eras):
        raise ValueError(
            f"plan needs {needed} eras (train {train_eras} + valid {valid_eras} + "
            f"test {test_eras} + 2 x gap {gap_eras} + offset {end_offset}) "
            f"but only {len(eras)} labelled eras exist"
        )
    test_end = len(eras) - end_offset
    test_start = test_end - test_eras
    valid_end = test_start - gap_eras
    valid_start = valid_end - valid_eras
    train_end = valid_start - gap_eras
    train_start = train_end - train_eras
    return ChunkPlan(
        train=list(eras[train_start:train_end]),
        valid=list(eras[valid_start:valid_end]),
        test=list(eras[test_start:test_end]),
        gap=gap_eras,
    )


@dataclass
class Chunk:
    X: np.ndarray  # float32, C order
    y: np.ndarray  # float64
    eras: np.ndarray  # str

    @property
    def rows(self) -> int:
        return int(self.X.shape[0])


def load_chunk(eras: list[str], timeline: list[EraInfo], sources: dict[str, Path],
               feature_cols: list[str], target: str) -> Chunk:
    """Read only ``eras`` x (era, target, features) from the parquet files.

    Features are written column by column into one preallocated float32
    matrix, so peak memory is about one copy of the chunk.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    wanted = set(eras)
    by_source: dict[str, list[str]] = {}
    for info in timeline:
        if info.era in wanted:
            by_source.setdefault(info.source, []).append(info.era)

    parts: list[Chunk] = []
    for source, source_eras in by_source.items():
        table = pq.read_table(
            sources[source], columns=["era", target, *feature_cols],
            filters=[("era", "in", source_eras)],
        )
        table = table.filter(pc.is_valid(table.column(target)))
        n = table.num_rows
        X = np.empty((n, len(feature_cols)), dtype=np.float32)
        for j, col in enumerate(feature_cols):
            X[:, j] = table.column(col).to_numpy(zero_copy_only=False)
        y = table.column(target).to_numpy(zero_copy_only=False).astype(np.float64)
        era_values = table.column("era").to_numpy(zero_copy_only=False).astype(str)
        del table
        parts.append(Chunk(X, y, era_values))
        gc.collect()

    if len(parts) == 1:
        chunk = parts[0]
    else:
        chunk = Chunk(
            np.concatenate([c.X for c in parts]),
            np.concatenate([c.y for c in parts]),
            np.concatenate([c.eras for c in parts]),
        )
    del parts
    gc.collect()
    # Chronological row order (pyarrow keeps file order, but be explicit).
    position = {info.era: i for i, info in enumerate(timeline)}
    order = np.argsort(np.array([position[e] for e in chunk.eras]), kind="stable")
    if not np.array_equal(order, np.arange(order.size)):
        chunk = Chunk(np.ascontiguousarray(chunk.X[order]), chunk.y[order], chunk.eras[order])
    return chunk


# ---------------------------------------------------------------------------
# Fit / predict
# ---------------------------------------------------------------------------
def fit_arm(arm: str, params: dict, train: Chunk, valid: Chunk, *,
            early_stopping: int, log_period: int):
    if arm == "alloygbm":
        from alloygbm import GBMRegressor

        model = GBMRegressor(**params)
        model.fit(train.X, train.y, eval_set=(valid.X, valid.y))
    elif arm == "lightgbm":
        import lightgbm as lgb

        model = lgb.LGBMRegressor(**params)
        callbacks = [lgb.early_stopping(early_stopping, verbose=False)]
        if log_period > 0:
            callbacks.append(lgb.log_evaluation(log_period))
        model.fit(train.X, train.y, eval_set=[(valid.X, valid.y)], eval_metric="rmse",
                  callbacks=callbacks)
    elif arm == "xgboost":
        import xgboost as xgb

        model = xgb.XGBRegressor(**params)
        model.fit(train.X, train.y, eval_set=[(valid.X, valid.y)],
                  verbose=log_period if log_period > 0 else False)
    elif arm == "catboost":
        import catboost as cb

        if log_period > 0:
            params = {**params, "verbose": log_period}
        model = cb.CatBoostRegressor(**params)
        model.fit(train.X, train.y, eval_set=(valid.X, valid.y))
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return model


def trees_used(model, arm: str) -> int | None:
    try:
        if arm == "alloygbm":
            best = getattr(model, "best_iteration_", None)
            done = getattr(model, "rounds_completed_", None)
            return int(best) + 1 if best is not None else (int(done) if done is not None else None)
        if arm == "lightgbm":
            best = getattr(model, "best_iteration_", None)
            return int(best) if best else int(model.booster_.current_iteration())
        if arm == "xgboost":
            best = getattr(model, "best_iteration", None)
            return int(best) + 1 if best is not None else int(model.get_booster().num_boosted_rounds())
        return int(model.tree_count_)
    except Exception:  # diagnostics only
        return None


def predict_arm(model, X: np.ndarray) -> np.ndarray:
    # Every sklearn wrapper here predicts with the early-stopped best round.
    return np.asarray(model.predict(X), dtype=np.float64)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(preds: np.ndarray, chunk: Chunk) -> dict[str, float]:
    resid = preds - chunk.y
    idx = pd.RangeIndex(chunk.rows)
    pred_s = pd.Series(preds, index=idx)
    target_s = pd.Series(chunk.y, index=idx)
    per_era = per_era_numerai_corr(pred_s, target_s, pd.Series(chunk.eras, index=idx)).dropna()
    return {
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "pearson": float(np.corrcoef(preds, chunk.y)[0, 1]) if np.std(preds) > 0 else 0.0,
        "numerai_corr": numerai_corr(pred_s, target_s),
        "mean_per_era_corr": float(per_era.mean()) if not per_era.empty else 0.0,
        "std_per_era_corr": float(per_era.std(ddof=0)) if len(per_era) > 1 else 0.0,
        "sharpe": sharpe_ratio(per_era),
        "positive_eras_pct": float((per_era > 0).mean() * 100.0) if not per_era.empty else 0.0,
        "n_eras": int(len(per_era)),
    }


def peak_rss_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    return peak / (1024 ** 3) if sys.platform == "darwin" else peak / (1024 ** 2)


@dataclass
class ArmRecord:
    arm: str
    profile: str
    params: dict
    unmapped: list[str]
    trees: int | None
    fit_seconds: float
    predict_seconds: float
    seconds_per_tree: float | None
    test: dict[str, float]
    valid: dict[str, float]
    peak_rss_gb_so_far: float
    error: str | None = None
    fit_timing: dict[str, float] = field(default_factory=dict)


def run_arm(arm: str, profile: Profile, args: argparse.Namespace,
            train: Chunk, valid: Chunk, test: Chunk) -> ArmRecord:
    params, unmapped = library_params(
        profile, arm, seed=args.seed, threads=args.threads,
        catboost_grow_policy=args.catboost_grow_policy,
        lightgbm_bagging_freq=args.lightgbm_bagging_freq,
    )
    logger.info("=" * 60)
    logger.info("ARM %s  params=%s  unmapped=%s", arm, params, unmapped)
    t0 = time.perf_counter()
    try:
        model = fit_arm(arm, params, train, valid, early_stopping=profile.early_stopping,
                        log_period=args.log_period)
    except Exception as exc:  # keep the other arms running
        logger.exception("%s failed", arm)
        return ArmRecord(arm, profile.name, params, unmapped, None,
                         time.perf_counter() - t0, 0.0, None, {}, {}, peak_rss_gb(),
                         error=f"{type(exc).__name__}: {exc}")
    fit_seconds = time.perf_counter() - t0
    trees = trees_used(model, arm)

    t1 = time.perf_counter()
    test_preds = predict_arm(model, test.X)
    predict_seconds = time.perf_counter() - t1
    valid_preds = predict_arm(model, valid.X)
    fit_timing = {k: float(v) for k, v in getattr(model, "fit_timing_", {}).items()} \
        if arm == "alloygbm" else {}
    del model
    gc.collect()

    record = ArmRecord(
        arm=arm, profile=profile.name, params=params, unmapped=unmapped, trees=trees,
        fit_seconds=round(fit_seconds, 2), predict_seconds=round(predict_seconds, 2),
        seconds_per_tree=round(fit_seconds / trees, 4) if trees else None,
        test=score(test_preds, test), valid=score(valid_preds, valid),
        peak_rss_gb_so_far=round(peak_rss_gb(), 2), fit_timing=fit_timing,
    )
    logger.info(
        "%s: %s trees in %.0fs | test rmse %.5f, per-era corr %.4f, sharpe %.3f",
        arm, trees, fit_seconds, record.test["rmse"], record.test["mean_per_era_corr"],
        record.test["sharpe"],
    )
    return record


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def markdown_report(records: list[ArmRecord], plan: ChunkPlan, sizes: dict,
                    profile: Profile, args: argparse.Namespace) -> str:
    d = plan.describe()
    lines = [
        f"# Numerai at-scale benchmark ({profile.name} profile)",
        "",
        f"Profile: {profile.description}. Feature set `{args.feature_set}` "
        f"({sizes['features']} features), target `{args.target}`, seed {args.seed}, "
        f"{args.threads} threads.",
        "",
        "| Chunk | Eras | Rows |",
        "|---|---|---:|",
    ]
    for name in ("train", "valid", "test"):
        lines.append(f"| {name} | {d[name]['first']} - {d[name]['last']} ({d[name]['eras']}) "
                     f"| {sizes[name]:,} |")
    lines += [
        "",
        f"Chunks are separated by {plan.gap} embargoed eras. Early stopping on validation RMSE.",
        "",
        "| Arm | Test RMSE | Per-era corr | Sharpe | Positive eras | numerai_corr | Trees | "
        "Fit (s) | s / tree | Unmapped |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in records:
        if r.error:
            lines.append(f"| {r.arm} | error: {r.error} | | | | | | {r.fit_seconds:.0f} | | |")
            continue
        t = r.test
        spt = f"{r.seconds_per_tree:.3f}" if r.seconds_per_tree else ""
        lines.append(
            f"| {r.arm} | {t['rmse']:.5f} | {t['mean_per_era_corr']:.4f} | {t['sharpe']:.3f} "
            f"| {t['positive_eras_pct']:.0f}% | {t['numerai_corr']:.4f} | {r.trees} "
            f"| {r.fit_seconds:.0f} | {spt} | {', '.join(r.unmapped) or '-'} |"
        )
    return "\n".join(lines) + "\n"


def save_results(records: list[ArmRecord], plan: ChunkPlan, sizes: dict, profile: Profile,
                 args: argparse.Namespace, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = output_dir / f"numerai_scale_{profile.name}_{timestamp}"
    payload = {
        "timestamp": timestamp,
        "profile": asdict(profile),
        "chunks": plan.describe(),
        "rows": sizes,
        "records": [asdict(r) for r in records],
        "reproducibility": _gather_reproducibility_manifest(args),
    }
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=str))
    stem.with_suffix(".md").write_text(markdown_report(records, plan, sizes, profile, args))
    logger.info("Results saved to %s.{json,md}", stem)
    return stem


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Numerai at-scale benchmark")
    parser.add_argument("--profile", default="deep", choices=sorted(PROFILES))
    parser.add_argument("--arms", nargs="+", default=list(ALL_ARMS), choices=ALL_ARMS)
    parser.add_argument("--feature-set", default="medium",
                        help="Feature set from features.json (default: medium, ~700 features)")
    parser.add_argument("--target", default="target")
    parser.add_argument("--train-eras", type=int, default=156, help="default: 3 years")
    parser.add_argument("--valid-eras", type=int, default=52, help="default: 1 year")
    parser.add_argument("--test-eras", type=int, default=104, help="default: 2 years")
    parser.add_argument("--gap-eras", type=int, default=4,
                        help="embargo between chunks (20-day targets overlap 4 weekly eras)")
    parser.add_argument("--end-offset", type=int, default=0,
                        help="end the test chunk this many eras before the newest labelled era")
    parser.add_argument("--seed", type=int, default=85)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--catboost-grow-policy", default="Lossguide",
                        choices=["Lossguide", "Depthwise", "SymmetricTree"],
                        help="Lossguide honours num_leaves and min_data_in_leaf (default)")
    parser.add_argument("--lightgbm-bagging-freq", type=int, default=1,
                        help="1 matches the other libraries' per-tree row sampling")
    parser.add_argument("--log-period", type=int, default=500,
                        help="peer libraries print validation RMSE every N trees (0: quiet)")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    profile = PROFILES[args.profile]

    ensure_numerai_data(keys=DATA_KEYS)
    sources = {"train": _local_path("train"), "validation": _local_path("validation")}
    feature_cols = resolve_feature_cols(args.feature_set)

    logger.info("Scanning eras...")
    timeline = labelled_era_timeline(sources, args.target)
    plan = plan_chunks(
        [e.era for e in timeline], train_eras=args.train_eras, valid_eras=args.valid_eras,
        test_eras=args.test_eras, gap_eras=args.gap_eras, end_offset=args.end_offset,
    )
    logger.info("Chunks: %s", plan.describe())
    _log_alloygbm_build_info()

    train = load_chunk(plan.train, timeline, sources, feature_cols, args.target)
    valid = load_chunk(plan.valid, timeline, sources, feature_cols, args.target)
    test = load_chunk(plan.test, timeline, sources, feature_cols, args.target)
    sizes = {"features": len(feature_cols), "train": train.rows, "valid": valid.rows,
             "test": test.rows}
    logger.info("Rows: %s | X train %.2f GB | peak RSS %.2f GB", sizes,
                train.X.nbytes / 1024 ** 3, peak_rss_gb())

    records = [run_arm(arm, profile, args, train, valid, test) for arm in args.arms]

    output_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "benchmarks" / "results" / "numerai_scale")
    stem = save_results(records, plan, sizes, profile, args, output_dir)
    print(stem.with_suffix(".md").read_text())


if __name__ == "__main__":
    main()
