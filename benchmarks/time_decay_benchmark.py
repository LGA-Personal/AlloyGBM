"""Time-decay weighting benchmark on chronologically ordered public data.

Rolling-origin evaluation: each dataset's timeline is cut into ``--blocks``
equal blocks of periods; fold ``k`` trains on every block before block
``first_test + k`` and tests on that block, so every test set is the future of
its training set. Every fit uses the defaults (``n_estimators="auto"``), so the
automatic round count is part of what is measured.

Arms:

- ``baseline``: no ``time_index``; the automatic holdout is a random 10%.
- ``time_holdout``: ``time_decay_half_life=inf``, the latest-periods holdout
  with every weight 1, which isolates the holdout from the decay itself.
- ``hl=<f>``: half-life of ``f`` times the fold's training span in periods.
- ``hl=<f>/random``: the same decay with the random holdout
  (``ALLOYGBM_EXPERIMENT_TIME_DECAY_RANDOM_HOLDOUT=1``), which isolates the
  decay from the holdout.

Datasets are fetched once from GitHub-hosted mirrors into ``--cache-dir``:

- ``elec2``: NSW electricity price direction (Harries 1999), 45,312 half-hour
  rows, binary; period = day.
- ``weather``: NOAA daily rain (Elwell and Polikar 2011), 18,159 days, binary;
  period = 30 days.
- ``beijing_pm25``: hourly PM2.5 in Beijing 2010-2014 (Liang et al. 2015),
  log1p target, regression; period = day.
- ``household_power``: next-hour household active power (Hebrail and Berard
  2012), hourly aggregates, regression; period = week.

Example::

    python benchmarks/time_decay_benchmark.py --seeds 3 --out results.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from alloygbm import GBMClassifier, GBMRegressor

_RANDOM_HOLDOUT_ENV = "ALLOYGBM_EXPERIMENT_TIME_DECAY_RANDOM_HOLDOUT"

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


def load_dataset(name: str, cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
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


def _score(task: str, y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if task == "regression":
        return {"rmse": float(np.sqrt(np.mean((y_true - prediction) ** 2)))}
    from sklearn.metrics import log_loss, roc_auc_score

    p = np.clip(prediction, 1e-7, 1 - 1e-7)
    return {"logloss": float(log_loss(y_true, p)), "auc": float(roc_auc_score(y_true, p))}


def run(args: argparse.Namespace) -> dict:
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    results: dict = {"config": config, "datasets": {}}
    for name in args.datasets:
        X, y, period, task = load_dataset(name, args.cache_dir)
        periods = np.unique(period)
        edges = np.array_split(periods, args.blocks)
        rows = []
        for fold in range(args.blocks - args.first_test):
            test_block = args.first_test + fold
            cut, end = edges[test_block][0], edges[test_block][-1]
            train = period < cut
            test = (period >= cut) & (period <= end)
            span = float(period[train].max() - period[train].min() + 1)
            arms: list[tuple[str, float | None, bool]] = [
                ("baseline", None, False),
                ("time_holdout", float("inf"), False),
            ]
            for f in args.half_life_fractions:
                arms += [(f"hl={f:g}", f * span, False), (f"hl={f:g}/random", f * span, True)]
            for seed in range(args.seeds):
                for arm, half_life, random_holdout in arms:
                    estimator = GBMClassifier if task == "binary" else GBMRegressor
                    model = estimator(seed=seed, time_decay_half_life=half_life)
                    kwargs = {} if half_life is None else {"time_index": period[train]}
                    if random_holdout:
                        os.environ[_RANDOM_HOLDOUT_ENV] = "1"
                    started = time.perf_counter()
                    try:
                        model.fit(X[train], y[train], **kwargs)
                    finally:
                        os.environ.pop(_RANDOM_HOLDOUT_ENV, None)
                    seconds = time.perf_counter() - started
                    if task == "binary":
                        prediction = model.predict_proba(X[test])[:, 1]
                    else:
                        prediction = model.predict(X[test])
                    decay = getattr(model, "time_decay_", None) or {}
                    row = {
                        "fold": fold,
                        "seed": seed,
                        "arm": arm,
                        "half_life": None if half_life in (None, float("inf")) else half_life,
                        "train_rows": int(train.sum()),
                        "test_rows": int(test.sum()),
                        "rounds": int(model.rounds_completed_),
                        "fit_seconds": seconds,
                        "effective_sample_size": decay.get("effective_sample_size"),
                        **_score(task, y[test], prediction),
                    }
                    rows.append(row)
                    print(json.dumps({"dataset": name, **row}), flush=True)
        results["datasets"][name] = {"task": task, "rows": rows}
    return results


def summarize(results: dict) -> str:
    """Markdown table: mean metric per arm, and paired deltas and wins against
    the random-holdout baseline and against the undecayed time holdout."""
    lines = []
    for name, payload in results["datasets"].items():
        metric = "rmse" if payload["task"] == "regression" else "logloss"
        rows = payload["rows"]

        def by_pair(arm: str) -> dict:
            return {(r["fold"], r["seed"]): r[metric] for r in rows if r["arm"] == arm}

        base, held = by_pair("baseline"), by_pair("time_holdout")
        lines += [
            f"### {name} ({metric}, lower is better)",
            "",
            "| Arm | Mean | vs baseline | Wins | vs time_holdout | Wins | Rounds | ESS share |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        arms = list(dict.fromkeys(r["arm"] for r in rows))
        for arm in arms:
            mine = [r for r in rows if r["arm"] == arm]
            deltas = [r[metric] - base[(r["fold"], r["seed"])] for r in mine]
            wins = sum(d < 0 for d in deltas)
            held_deltas = [r[metric] - held[(r["fold"], r["seed"])] for r in mine]
            held_wins = sum(d < 0 for d in held_deltas)
            shares = [
                r["effective_sample_size"] / r["train_rows"]
                for r in mine
                if r["effective_sample_size"] is not None
            ]
            share = f"{np.mean(shares):.2f}" if shares else "1.00"
            lines.append(
                f"| {arm} | {np.mean([r[metric] for r in mine]):.5f} | "
                f"{np.mean(deltas):+.5f} | {wins} / {len(deltas)} | "
                f"{np.mean(held_deltas):+.5f} | {held_wins} / {len(held_deltas)} | "
                f"{np.mean([r['rounds'] for r in mine]):.0f} | {share} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--datasets", nargs="+", default=["elec2", "weather", "beijing_pm25", "household_power"]
    )
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--first-test", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--half-life-fractions", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.125]
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("benchmarks/data/time_decay"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    results = run(args)
    report = summarize(results)
    print(report)
    if args.out is not None:
        args.out.write_text(json.dumps(results, indent=1))
        args.out.with_suffix(".md").write_text(report)


if __name__ == "__main__":
    main()
