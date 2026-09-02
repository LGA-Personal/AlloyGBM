"""Thread-scaling comparison at high depth and high tree count.

Complements `run_model_comparison.py`, which measures a curated accuracy suite
at one thread budget. This one sweeps the thread budget across deep, many-tree
configurations to show how each library's *scaling* behaves, not just its
single-point speed.

Fairness rules, identical to `run_model_comparison.py`:

  - one thread budget per run, applied to every library's own knob AND to the
    OpenMP/BLAS environment, so no runtime quietly grabs more cores than asked;
  - matched capacity: `num_leaves = 2**max_depth` for LightGBM, whose default
    of 31 would otherwise cap it far below the depth-wise peers;
  - matched sampling: row 0.8 / column 0.8 everywhere, with LightGBM's
    `subsample_freq=1` since it ignores `subsample` otherwise;
  - matched bin counts, learning rate, rounds, depth, and seed.

Because the thread budget must be pinned in the environment *before* the
libraries are imported, each budget runs in its own subprocess: this module
re-executes itself with `--threads N --worker`.

Results are appended to the JSON output as each cell completes, so a partial
run is still usable.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

LEARNING_RATE = 0.05
SEED = 20260902
ROW_SUBSAMPLE = 0.8
COL_SUBSAMPLE = 0.8
MAX_BINS = 256
REPEATS = 2

# (name, rows, features)
SHAPES = [
    ("tall", 500_000, 40),
    ("wide", 200_000, 320),
    ("large", 1_000_000, 60),
]

# (name, rounds, depth)
CONFIGS = [
    ("r200-d8", 200, 8),
    ("r500-d8", 500, 8),
    ("r200-d12", 200, 12),
]

# Ordered so the most informative budgets finish first: the run is long, and a
# partial sweep should already answer "how does each library scale". 10 and 4
# are quick, 1 is the expensive but essential baseline, the rest fill the curve.
THREAD_BUDGETS = [10, 4, 1, 2, 6, 8]


def make_dataset(rows: int, features: int):
    import numpy as np

    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(rows, features)).astype(np.float32)
    signal = x[:, :5] @ np.array([1.5, -2.0, 0.75, 1.0, -0.5], dtype=np.float32)
    interaction = 0.8 * x[:, 0] * x[:, 1]
    y = (signal + interaction + rng.normal(scale=0.1, size=rows)).astype(np.float32)
    split = int(rows * 0.8)
    return x[:split], y[:split], x[split:], y[split:]


def rmse(actual, predicted) -> float:
    import numpy as np

    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def build_factories(rounds: int, depth: int, threads: int):
    """One factory per library, with every comparable knob matched."""
    from alloygbm import GBMRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    factories = {
        "alloygbm": lambda: GBMRegressor(
            n_estimators=rounds,
            max_depth=depth,
            learning_rate=LEARNING_RATE,
            row_subsample=ROW_SUBSAMPLE,
            col_subsample=COL_SUBSAMPLE,
            continuous_binning_max_bins=MAX_BINS,
            seed=SEED,
            deterministic=True,
            n_jobs=threads,
        ),
        "lightgbm": lambda: LGBMRegressor(
            objective="regression",
            n_estimators=rounds,
            max_depth=depth,
            # Match depth-wise capacity; LightGBM's default of 31 leaves would
            # cap it far below 2**depth and make the comparison meaningless.
            num_leaves=2**depth,
            learning_rate=LEARNING_RATE,
            subsample=ROW_SUBSAMPLE,
            # LightGBM ignores `subsample` unless bagging_freq >= 1.
            subsample_freq=1,
            colsample_bytree=COL_SUBSAMPLE,
            max_bin=MAX_BINS - 1,
            random_state=SEED,
            n_jobs=threads,
            verbose=-1,
        ),
        "xgboost": lambda: XGBRegressor(
            objective="reg:squarederror",
            n_estimators=rounds,
            max_depth=depth,
            learning_rate=LEARNING_RATE,
            subsample=ROW_SUBSAMPLE,
            colsample_bytree=COL_SUBSAMPLE,
            max_bin=MAX_BINS,
            random_state=SEED,
            n_jobs=threads,
            tree_method="hist",
            verbosity=0,
        ),
    }
    try:
        from catboost import CatBoostRegressor
    except Exception:
        return factories

    # CatBoost caps depth at 16 and grows oblivious trees, so its capacity at a
    # given depth is not identical to the others'; recorded, not hidden.
    factories["catboost"] = lambda: CatBoostRegressor(
        loss_function="RMSE",
        iterations=rounds,
        depth=min(depth, 16),
        learning_rate=LEARNING_RATE,
        subsample=ROW_SUBSAMPLE,
        bootstrap_type="Bernoulli",
        rsm=COL_SUBSAMPLE,
        border_count=MAX_BINS - 2,
        random_seed=SEED,
        thread_count=threads,
        logging_level="Silent",
        allow_writing_files=False,
    )
    return factories


def check_alloy_sampling(model) -> dict:
    """Fairness gate: AlloyGBM's auto policy can override explicit sampling.

    Setting `row_subsample=1.0` is silently replaced by the auto policy's own
    value, so an explicitly matched setting is not proof of a matched run. Read
    back what the fit actually resolved and fail loudly if it drifted from the
    peers' 0.8 / 0.8 -- a mismatch here would let AlloyGBM do less histogram
    work than the libraries it is being compared against.
    """
    resolved = getattr(model, "resolved_training_policy_", None)
    if resolved is None:
        return {"checked": False}
    row = float(resolved["row_subsample"])
    col = float(resolved["col_subsample"])
    matched = abs(row - ROW_SUBSAMPLE) < 1e-6 and abs(col - COL_SUBSAMPLE) < 1e-6
    if not matched:
        # Recorded rather than raised: this runs unattended, and aborting would
        # throw away every cell after the first mismatch. Cells marked
        # `matched: false` are not comparable and must be excluded when the
        # results are read, not quietly averaged in.
        print(
            f"  !! FAIRNESS: alloygbm resolved row={row} col={col}, "
            f"peers run at {ROW_SUBSAMPLE}/{COL_SUBSAMPLE} -- cell not comparable",
            flush=True,
        )
    return {
        "checked": True,
        "matched": matched,
        "row_subsample": row,
        "col_subsample": col,
    }


def run_worker(threads: int, out_path: Path) -> int:
    x_train = y_train = x_test = y_test = None
    loaded_shape = None
    records = []

    for shape_name, rows, features in SHAPES:
        for config_name, rounds, depth in CONFIGS:
            if loaded_shape != shape_name:
                x_train, y_train, x_test, y_test = make_dataset(rows, features)
                loaded_shape = shape_name
            factories = build_factories(rounds, depth, threads)
            for library, factory in factories.items():
                best = None
                score = None
                sampling = None
                for _ in range(REPEATS):
                    model = factory()
                    started = time.perf_counter()
                    model.fit(x_train, y_train)
                    elapsed = time.perf_counter() - started
                    best = elapsed if best is None else min(best, elapsed)
                    if score is None:
                        score = rmse(y_test, model.predict(x_test))
                        if library == "alloygbm":
                            sampling = check_alloy_sampling(model)
                record = {
                    "shape": shape_name,
                    "rows": rows,
                    "features": features,
                    "config": config_name,
                    "rounds": rounds,
                    "depth": depth,
                    "threads": threads,
                    "library": library,
                    "fit_seconds": round(best, 4),
                    "test_rmse": round(score, 6),
                }
                if sampling:
                    record["alloygbm_resolved_sampling"] = sampling
                records.append(record)
                print(
                    f"[t={threads}] {shape_name:6s} {config_name:9s} "
                    f"{library:9s} {best:8.2f}s  rmse={score:.5f}",
                    flush=True,
                )
                # Written after every cell so an interrupted run is still usable.
                out_path.write_text(json.dumps(records, indent=2))
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.worker:
        return run_worker(args.threads, args.out_dir / f"deep_scaling_t{args.threads}.json")

    print(f"host: {platform.platform()} / {os.cpu_count()} logical CPUs", flush=True)
    for threads in THREAD_BUDGETS:
        env = dict(os.environ)
        for var in THREAD_ENV_VARS:
            env[var] = str(threads)
        print(f"\n=== thread budget {threads} ===", flush=True)
        completed = subprocess.run(
            [sys.executable, __file__, "--worker", "--threads", str(threads),
             "--out-dir", str(args.out_dir)],
            env=env,
        )
        if completed.returncode != 0:
            print(f"thread budget {threads} failed ({completed.returncode})", file=sys.stderr)
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
