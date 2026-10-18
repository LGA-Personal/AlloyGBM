"""Latest-periods holdout for ``n_estimators="auto"`` on time-ordered data.

With ``n_estimators="auto"``, a fit picks its round count on a 10% holdout.
Without ``time_index`` that holdout is random rows; with it, the holdout is
the newest periods. This benchmark compares the two on chronologically
ordered data.

Rolling-origin evaluation: each dataset's timeline is cut into ``--blocks``
equal blocks of periods; fold ``k`` trains on every block before block
``first_test + k`` and tests on that block, so every test set is the future
of its training set. Both arms use the defaults otherwise:

- ``random``: ``fit(X, y)``, the random-rows holdout.
- ``time``: ``fit(X, y, time_index=period)``, the latest-periods holdout.

The unit of comparison is the test period. Each period's mean loss (squared
error, or log loss for binary targets) is averaged over seeds, and a
moving-block bootstrap over test periods gives a 95% interval for the mean
paired difference, so serial correlation between neighbouring periods does
not inflate the confidence. Negative is better for ``time``.

Datasets are fetched once from GitHub-hosted mirrors into ``--cache-dir``:

- ``elec2``: NSW electricity price direction (Harries 1999), 45,312 half-hour
  rows, binary; period = day.
- ``weather``: NOAA daily rain (Elwell and Polikar 2011), 18,159 days, binary;
  period = 30 days.
- ``beijing_pm25``: hourly PM2.5 in Beijing 2010-2014 (Liang et al. 2015),
  log1p target, regression; period = day.
- ``household_power``: next-hour household active power (Hebrail and Berard
  2012), hourly aggregates, regression; period = week.

and synthetic panels regenerated per seed, with a known feature-target
relationship: ``synth_drift=<s>`` (each true coefficient takes a random-walk
step of ``s`` per period; ``s=0`` is i.i.d. across time) and ``synth_jump``
(one abrupt regime change at 80% of the timeline).

Example::

    python benchmarks/time_holdout_benchmark.py --seeds 5 --out results.json
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from alloygbm import GBMClassifier, GBMRegressor

_URLS = {
    "elec2": "https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/master/elec.csv",
    "weather": "https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/master/weather.csv",
    "beijing_pm25": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv",
    "household_power": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/household_power_consumption.zip",
}


def _fetch(name: str, cache_dir: Path) -> bytes:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / Path(_URLS[name]).name
    if not path.exists():
        with urllib.request.urlopen(_URLS[name], timeout=300) as response:
            path.write_bytes(response.read())
    return path.read_bytes()


def load_public(name: str, cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return ``(X, y, period, task)`` with rows in time order."""
    import pandas as pd

    raw = _fetch(name, cache_dir)
    if name == "elec2":
        frame = pd.read_csv(io.BytesIO(raw))
        day = np.arange(len(frame)) // 48
        X = frame.drop(columns=["class"]).assign(day_of_week=day % 7).to_numpy(np.float64)
        return X, frame["class"].to_numpy(np.int64), day, "binary"
    if name == "weather":
        frame = pd.read_csv(io.BytesIO(raw))
        X = frame.drop(columns=["target"]).to_numpy(np.float64)
        return X, frame["target"].to_numpy(np.int64), np.arange(len(frame)) // 30, "binary"
    if name == "beijing_pm25":
        frame = pd.read_csv(io.BytesIO(raw)).dropna(subset=["pm2.5"]).reset_index(drop=True)
        frame["cbwd"] = frame["cbwd"].astype("category").cat.codes
        stamp = pd.to_datetime(frame[["year", "month", "day", "hour"]])
        period = ((stamp - stamp.min()).dt.days).to_numpy(np.int64)
        X = frame[["month", "hour", "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]].assign(
            day_of_week=stamp.dt.dayofweek
        ).to_numpy(np.float64)
        return X, np.log1p(frame["pm2.5"].to_numpy(np.float64)), period, "regression"
    if name == "household_power":
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            text = archive.read("household_power_consumption.txt")
        frame = pd.read_csv(io.BytesIO(text), sep=";", na_values="?", low_memory=False)
        stamp = pd.to_datetime(frame["Date"] + " " + frame["Time"], dayfirst=True)
        values = frame.drop(columns=["Date", "Time"]).astype(np.float64)
        hourly = values.groupby(stamp.dt.floor("h")).mean().dropna()
        target = hourly["Global_active_power"].shift(-1)
        keep = target.notna().to_numpy()
        hourly, target = hourly[keep], target[keep]
        index = hourly.index
        X = hourly.assign(hour=index.hour, day_of_week=index.dayofweek, month=index.month)
        period = ((index - index.min()).days // 7).to_numpy(np.int64)
        return X.to_numpy(np.float64), target.to_numpy(np.float64), period, "regression"
    raise ValueError(f"unknown dataset {name!r}")



def synthetic_panel(
    seed: int, *, drift: float = 0.0, jump: bool = False, periods: int = 80, rows: int = 250
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """A regression panel whose feature-target relationship drifts by a known
    amount. Eight features enter through fixed nonlinear shapes with
    time-varying coefficients; noise has standard deviation 1."""
    rng = np.random.default_rng(10_000 + seed)
    n_features = 8
    beta = np.empty((periods, n_features))
    beta[0] = rng.normal(size=n_features)
    beta[1:] = beta[0] + np.cumsum(drift * rng.normal(size=(periods - 1, n_features)), axis=0)
    if jump:
        beta[int(0.8 * periods) :] = rng.normal(size=n_features)
    period = np.repeat(np.arange(periods), rows)
    X = rng.normal(size=(len(period), n_features + 2))
    shapes = np.column_stack(
        [
            X[:, 0],
            np.tanh(2 * X[:, 1]),
            X[:, 2] ** 2 - 1,
            np.sign(X[:, 3]),
            np.sin(2 * X[:, 4]),
            X[:, 5] * X[:, 6],
            np.maximum(X[:, 6], 0) - 0.4,
            np.abs(X[:, 7]) - 0.8,
        ]
    )
    y = 0.55 * (beta[period] * shapes).sum(axis=1) + rng.normal(size=len(period))
    return X, y, period, "regression"


def load(name: str, seed: int, cache_dir: Path):
    if name.startswith("synth_drift="):
        return synthetic_panel(seed, drift=float(name.split("=", 1)[1]))
    if name == "synth_jump":
        return synthetic_panel(seed, jump=True)
    return load_public(name, cache_dir)


def _period_losses(task, y_true, prediction, period) -> dict[int, float]:
    if task == "regression":
        losses = (y_true - prediction) ** 2
    else:
        p = np.clip(prediction, 1e-7, 1 - 1e-7)
        losses = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    keys, inverse = np.unique(period, return_inverse=True)
    sums = np.bincount(inverse, weights=losses)
    counts = np.bincount(inverse)
    return {int(k): float(s / c) for k, s, c in zip(keys, sums, counts)}


def run_dataset(name: str, args: argparse.Namespace, log) -> list[dict]:
    rows: list[dict] = []
    for seed in range(args.seeds):
        X, y, period, task = load(name, seed, args.cache_dir)
        edges = np.array_split(np.unique(period), args.blocks)
        estimator = GBMClassifier if task == "binary" else GBMRegressor
        for fold in range(args.blocks - args.first_test):
            block = edges[args.first_test + fold]
            train = period < block[0]
            test = (period >= block[0]) & (period <= block[-1])
            for arm in ("random", "time"):
                model = estimator(seed=seed)
                kwargs = {"time_index": period[train]} if arm == "time" else {}
                started = time.perf_counter()
                model.fit(X[train], y[train], **kwargs)
                seconds = time.perf_counter() - started
                if task == "binary":
                    prediction = model.predict_proba(X[test])[:, 1]
                else:
                    prediction = model.predict(X[test])
                period_loss = _period_losses(task, y[test], prediction, period[test])
                row = {
                    "dataset": name,
                    "task": task,
                    "seed": seed,
                    "fold": fold,
                    "arm": arm,
                    "train_rows": int(train.sum()),
                    "rounds": int(model.rounds_completed_),
                    "fit_seconds": seconds,
                    "loss": float(np.mean(list(period_loss.values()))),
                }
                log(json.dumps(row))
                rows.append({**row, "period_loss": period_loss})
    return rows


def block_bootstrap_ci(deltas: np.ndarray, *, draws: int = 4000, seed: int = 0):
    """95% interval for the mean of a serially correlated series, from a
    circular moving-block bootstrap with block length ``n ** (1/3)``."""
    n = len(deltas)
    block = max(1, int(round(n ** (1 / 3))))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(draws, -(-n // block)))
    index = ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]
    means = deltas[index].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def compare(rows: list[dict]) -> dict:
    """Paired comparison of the ``time`` arm against ``random``."""
    by_arm = {
        arm: {(r["seed"], r["fold"]): r for r in rows if r["arm"] == arm}
        for arm in ("random", "time")
    }
    pairs = sorted(by_arm["time"])
    per_period: dict[int, list[float]] = {}
    for key in pairs:
        a, b = by_arm["time"][key]["period_loss"], by_arm["random"][key]["period_loss"]
        for p in a:
            per_period.setdefault(int(p), []).append(a[p] - b[p])
    deltas = np.array([np.mean(per_period[p]) for p in sorted(per_period)])
    reference = float(np.mean([by_arm["random"][k]["loss"] for k in pairs]))
    low, high = block_bootstrap_ci(deltas)
    fold_deltas = [by_arm["time"][k]["loss"] - by_arm["random"][k]["loss"] for k in pairs]
    return {
        "random_loss": reference,
        "time_loss": float(np.mean([by_arm["time"][k]["loss"] for k in pairs])),
        "mean_delta": float(deltas.mean()),
        "relative": float(deltas.mean() / reference),
        "ci_low": low,
        "ci_high": high,
        "periods": int(len(deltas)),
        "period_win_rate": float(np.mean(deltas < 0)),
        "fold_wins": int(sum(d < 0 for d in fold_deltas)),
        "folds": len(pairs),
        "random_rounds": float(np.mean([by_arm["random"][k]["rounds"] for k in pairs])),
        "time_rounds": float(np.mean([by_arm["time"][k]["rounds"] for k in pairs])),
        "random_seconds": float(np.mean([by_arm["random"][k]["fit_seconds"] for k in pairs])),
        "time_seconds": float(np.mean([by_arm["time"][k]["fit_seconds"] for k in pairs])),
    }


def summarize(results: dict) -> str:
    lines = [
        "| Dataset | Metric | Random | Time | Delta | 95% CI | Periods won | Folds won "
        "| Trees (random / time) | Fit s (random / time) |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for name, rows in results["datasets"].items():
        c = compare(rows)
        metric = "MSE" if rows[0]["task"] == "regression" else "log loss"
        sig = " *" if c["ci_high"] < 0 or c["ci_low"] > 0 else ""
        lines.append(
            f"| {name} | {metric} | {c['random_loss']:.5f} | {c['time_loss']:.5f} | "
            f"{100 * c['relative']:+.2f}%{sig} | "
            f"[{100 * c['ci_low'] / c['random_loss']:+.2f}%, "
            f"{100 * c['ci_high'] / c['random_loss']:+.2f}%] | "
            f"{100 * c['period_win_rate']:.0f}% | {c['fold_wins']} / {c['folds']} | "
            f"{c['random_rounds']:.0f} / {c['time_rounds']:.0f} | "
            f"{c['random_seconds']:.1f} / {c['time_seconds']:.1f} |"
        )
    lines += ["", "`*`: the 95% interval excludes zero. Negative delta favours `time`."]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "synth_drift=0",
            "synth_drift=0.03",
            "synth_drift=0.1",
            "synth_jump",
            "elec2",
            "weather",
            "beijing_pm25",
            "household_power",
        ],
    )
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--first-test", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--cache-dir", type=Path, default=Path("benchmarks/data/time_holdout"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    log_path = args.out.with_suffix(".jsonl") if args.out else None

    def log(line: str) -> None:
        print(line, flush=True)
        if log_path is not None:
            with log_path.open("a") as handle:
                handle.write(line + "\n")

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    results: dict = {"config": config, "datasets": {}}
    for name in args.datasets:
        results["datasets"][name] = run_dataset(name, args, log)
        if args.out is not None:
            args.out.write_text(json.dumps(results))
    report = summarize(results)
    print(report)
    if args.out is not None:
        args.out.with_suffix(".md").write_text(report + "\n")


if __name__ == "__main__":
    main()
