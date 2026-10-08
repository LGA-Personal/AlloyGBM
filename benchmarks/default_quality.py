"""Out-of-the-box quality suite: how good is each library *at its defaults*?

Most of AlloyGBM's comparison tooling measures machinery: every library gets
the same `n_estimators`, `learning_rate`, `max_depth` and sampling, so the
comparison isolates the tree-building code. Users do not experience that.
They construct an estimator, call `fit`, and get whatever the defaults (and
AlloyGBM's `training_policy="auto"`) produce. This suite measures exactly
that, the way mature GBDT projects choose and defend their defaults:

* **Many datasets, several seeds.** Each dataset is repartitioned and refit
  per seed, so the seed-to-seed spread is part of the result rather than an
  invisible confound.
* **Proper scoring rules.** Log loss for classifiers (accuracy hides
  calibration), RMSE for regression, both on a held-out split.
* **Normalized loss.** Each dataset's loss is divided by the best arm's mean
  loss on that dataset, so datasets with different units can be averaged.
  1.0 means "as good as the best arm in this run".
* **Paired, pre-declared comparison.** ``compare`` pairs two arms (or two
  result files from two builds) per dataset and seed, reports the geometric
  mean loss ratio, a Wilcoxon signed-rank test across datasets, and the worst
  per-dataset regression, and applies a fail-closed gate.

The dataset set deliberately mixes small real datasets bundled with
scikit-learn (no network needed), scikit-learn's standard synthetic
benchmarks, and targeted fixtures for known failure modes (zero-inflated and
skewed-discrete features, tiny noisy wide data, class imbalance). Scale
variants train on a rescaled target and score in the original units: a
defaults policy that depends on the target's units shows up there as a
regression with no other cause.

Usage::

    # Each library at its own defaults, 5 seeds.
    python benchmarks/default_quality.py run --seeds 5 --output out.json

    # AlloyGBM only, two arms in one process.
    python benchmarks/default_quality.py run --arms alloy alloy-manual --output out.json

    # Paired comparison of two arms (same file or two builds' files).
    python benchmarks/default_quality.py compare out.json --baseline alloy \\
        --candidate alloy-manual --gate
    python benchmarks/default_quality.py compare base.json cand.json \\
        --baseline alloy --candidate alloy --gate
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
import warnings
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

REGRESSION = "regression"
BINARY = "binary"
MULTICLASS = "multiclass"
TASKS = (REGRESSION, BINARY, MULTICLASS)


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    name: str
    task: str
    stratum: str
    X: np.ndarray
    y: np.ndarray
    # Train on ``y * target_scale``; predictions are divided back before
    # scoring so the loss stays in the original units.
    target_scale: float = 1.0


def _sklearn_real() -> list[Dataset]:
    from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_wine

    out = []
    X, y = load_breast_cancer(return_X_y=True)
    out.append(Dataset("breast_cancer", BINARY, "real", X, y))
    X, y = load_diabetes(return_X_y=True)
    out.append(Dataset("diabetes", REGRESSION, "real", X, y))
    X, y = load_digits(return_X_y=True)
    out.append(Dataset("digits", MULTICLASS, "real", X, y))
    X, y = load_wine(return_X_y=True)
    out.append(Dataset("wine", MULTICLASS, "real", X, y))
    return out


def _sklearn_synthetic() -> list[Dataset]:
    from sklearn.datasets import make_classification, make_friedman1, make_hastie_10_2

    out = []
    X, y = make_friedman1(n_samples=5_000, n_features=10, noise=1.0, random_state=11)
    out.append(Dataset("friedman1", REGRESSION, "synthetic", X, y))
    X, y = make_classification(
        n_samples=6_000, n_features=20, n_informative=8, n_redundant=4,
        flip_y=0.05, class_sep=0.8, random_state=12,
    )
    out.append(Dataset("make_classification", BINARY, "synthetic", X, y))
    X, y = make_hastie_10_2(n_samples=6_000, random_state=13)
    out.append(Dataset("hastie_10_2", BINARY, "synthetic", X, (y > 0).astype(float)))
    X, y = make_classification(
        n_samples=6_000, n_features=20, n_informative=10, n_classes=5,
        n_clusters_per_class=1, flip_y=0.03, random_state=14,
    )
    out.append(Dataset("make_multiclass", MULTICLASS, "synthetic", X, y))
    return out


def _targeted() -> list[Dataset]:
    rng = np.random.default_rng(20261008)
    out = []
    n = 12_000
    others = rng.normal(size=(n, 7))

    x0 = np.where(rng.random(n) < 0.6, 0.0, rng.lognormal(0.0, 1.0, n))
    y = (
        2.0 * np.sin(2.0 * np.log1p(x0)) + others[:, 0]
        + 0.5 * others[:, 1] * others[:, 2] + rng.normal(0.0, 0.1, n)
    )
    out.append(Dataset("zero_inflated", REGRESSION, "targeted", np.column_stack([x0, others]), y))

    k = rng.zipf(1.6, n).clip(1, 400).astype(float)
    y = 1.5 * np.sin(0.7 * k) + others[:, 0] + rng.normal(0.0, 0.1, n)
    out.append(Dataset("skewed_discrete", REGRESSION, "targeted", np.column_stack([k, others]), y))

    Xs = rng.normal(size=(600, 64))
    y = Xs[:, :4] @ np.array([1.0, -0.8, 0.6, 0.4]) + rng.normal(0.0, 2.0, 600)
    out.append(Dataset("small_wide_noisy", REGRESSION, "targeted", Xs, y))

    Xl = rng.normal(size=(20_000, 20))
    y = (
        Xl[:, :5] @ np.array([1.0, 0.8, 0.6, 0.4, 0.2]) + Xl[:, 0] * Xl[:, 1]
        + rng.normal(0.0, 0.1, 20_000)
    )
    out.append(Dataset("linear_interaction", REGRESSION, "targeted", Xl, y))

    Xi = rng.normal(size=(8_000, 12))
    logits = Xi[:, 0] + 0.8 * Xi[:, 1] - 0.6 * Xi[:, 2] * Xi[:, 3] - 3.4
    yi = (rng.random(8_000) < 1.0 / (1.0 + np.exp(-logits))).astype(float)
    out.append(Dataset("imbalanced_binary", BINARY, "targeted", Xi, yi))
    return out


def _scale_variants(base: Iterable[Dataset]) -> list[Dataset]:
    out = []
    for ds in base:
        if ds.task != REGRESSION:
            continue
        for scale in (1e-3, 1e3):
            out.append(Dataset(
                f"{ds.name}@scale{scale:g}", ds.task, "scale", ds.X, ds.y, target_scale=scale,
            ))
    return out


def build_suite(quick: bool = False) -> list[Dataset]:
    real = _sklearn_real()
    synthetic = _sklearn_synthetic()
    targeted = _targeted()
    by_name = {ds.name: ds for ds in real + synthetic}
    by_name.update({ds.name: ds for ds in targeted})
    scale_bases = [by_name["diabetes"], by_name["friedman1"], by_name["linear_interaction"]]
    suite = real + synthetic + targeted + _scale_variants(scale_bases)
    if quick:
        keep = {"breast_cancer", "diabetes", "wine", "make_classification",
                "skewed_discrete", "diabetes@scale0.001"}
        suite = [ds for ds in suite if ds.name in keep]
    return suite


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    name: str
    library: str
    params: dict = field(default_factory=dict)


BUILTIN_ARMS: dict[str, Arm] = {
    "alloy": Arm("alloy", "alloygbm"),
    "alloy-manual": Arm("alloy-manual", "alloygbm", {"training_policy": "manual"}),
    "lightgbm": Arm("lightgbm", "lightgbm"),
    "xgboost": Arm("xgboost", "xgboost"),
    "catboost": Arm("catboost", "catboost"),
}
DEFAULT_ARMS = ("alloy", "lightgbm", "xgboost", "catboost")


def parse_arm(spec: str) -> Arm:
    """``name`` (builtin) or ``name=library[:key=value,key=value]``.

    Values are parsed as JSON when possible (``1``, ``0.5``, ``true``,
    ``"x"``), else kept as strings.
    """
    if "=" not in spec.split(":", 1)[0]:
        if spec not in BUILTIN_ARMS:
            raise ValueError(f"unknown arm {spec!r}; builtins: {sorted(BUILTIN_ARMS)}")
        return BUILTIN_ARMS[spec]
    name, rest = spec.split("=", 1)
    library, _, raw_params = rest.partition(":")
    if library not in {"alloygbm", "lightgbm", "xgboost", "catboost"}:
        raise ValueError(f"unknown library {library!r} in arm {spec!r}")
    params = {}
    for item in filter(None, raw_params.split(",")):
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"malformed parameter {item!r} in arm {spec!r}")
        try:
            params[key] = json.loads(value)
        except json.JSONDecodeError:
            params[key] = value
    return Arm(name, library, params)


def _make_estimator(arm: Arm, task: str, seed: int, threads: int):
    params = dict(arm.params)
    classifier = task in (BINARY, MULTICLASS)
    if arm.library == "alloygbm":
        from alloygbm import GBMClassifier, GBMRegressor

        cls = GBMClassifier if classifier else GBMRegressor
        return cls(seed=seed, n_jobs=threads, **params)
    if arm.library == "lightgbm":
        import lightgbm as lgb

        cls = lgb.LGBMClassifier if classifier else lgb.LGBMRegressor
        return cls(random_state=seed, n_jobs=threads, verbose=-1, **params)
    if arm.library == "xgboost":
        import xgboost as xgb

        cls = xgb.XGBClassifier if classifier else xgb.XGBRegressor
        return cls(random_state=seed, n_jobs=threads, **params)
    import catboost as cb

    cls = cb.CatBoostClassifier if classifier else cb.CatBoostRegressor
    return cls(random_seed=seed, thread_count=threads, verbose=0,
               allow_writing_files=False, **params)


def _library_version(library: str) -> str:
    try:
        module = __import__(library)
    except ImportError:
        return "missing"
    return str(getattr(module, "__version__", "unknown"))


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


def score(task: str, y_true: np.ndarray, prediction: np.ndarray, labels: np.ndarray) -> float:
    if task == REGRESSION:
        return float(np.sqrt(np.mean((np.asarray(prediction, dtype=float) - y_true) ** 2)))
    from sklearn.metrics import log_loss

    proba = np.clip(np.asarray(prediction, dtype=float), 1e-15, 1.0)
    return float(log_loss(y_true, proba, labels=labels))


def split(ds: Dataset, seed: int, test_size: float = 0.25):
    from sklearn.model_selection import train_test_split

    stratify = ds.y if ds.task in (BINARY, MULTICLASS) else None
    return train_test_split(ds.X, ds.y, test_size=test_size, random_state=seed, stratify=stratify)


def fit_and_score(arm: Arm, ds: Dataset, seed: int, threads: int) -> dict:
    X_train, X_test, y_train, y_test = split(ds, seed)
    model = _make_estimator(arm, ds.task, seed, threads)
    started = time.perf_counter()
    if ds.task == REGRESSION:
        model.fit(X_train, y_train * ds.target_scale)
        fit_seconds = time.perf_counter() - started
        prediction = np.asarray(model.predict(X_test), dtype=float) / ds.target_scale
    else:
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        prediction = model.predict_proba(X_test)
    labels = np.unique(ds.y)
    return {
        "loss": score(ds.task, y_test, prediction, labels),
        "fit_seconds": fit_seconds,
    }


def run(arms: list[Arm], suite: list[Dataset], seeds: list[int], threads: int,
        log: Callable[[str], None] = print) -> dict:
    warnings.filterwarnings("ignore")
    records = []
    for ds in suite:
        for seed in seeds:
            for arm in arms:
                try:
                    result = fit_and_score(arm, ds, seed, threads)
                    error = None
                except Exception as exc:  # recorded, and the gate fails closed on it
                    result, error = {"loss": None, "fit_seconds": None}, f"{type(exc).__name__}: {exc}"
                records.append({
                    "dataset": ds.name, "task": ds.task, "stratum": ds.stratum,
                    "seed": seed, "arm": arm.name, **result, "error": error,
                })
        log(f"  {ds.name}: done")
    return {
        "schema_version": SCHEMA_VERSION,
        "arms": [asdict(arm) | {"version": _library_version(arm.library)} for arm in arms],
        "seeds": seeds,
        "threads": threads,
        "datasets": [{"name": ds.name, "task": ds.task, "stratum": ds.stratum,
                      "rows": int(ds.X.shape[0]), "features": int(ds.X.shape[1]),
                      "target_scale": ds.target_scale} for ds in suite],
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "records": records,
    }


# --------------------------------------------------------------------------
# Summaries and paired comparison
# --------------------------------------------------------------------------


def _losses(records: list[dict], arm: str) -> dict[tuple[str, int], float]:
    return {
        (r["dataset"], r["seed"]): r["loss"]
        for r in records if r["arm"] == arm and r["loss"] is not None
    }


def summarize(result: dict) -> dict:
    """Per-dataset mean loss and normalized loss per arm, plus aggregates."""
    records = result["records"]
    arms = [a["name"] for a in result["arms"]]
    datasets = [d["name"] for d in result["datasets"]]
    means: dict[str, dict[str, float]] = {}
    for ds in datasets:
        means[ds] = {}
        for arm in arms:
            values = [r["loss"] for r in records
                      if r["dataset"] == ds and r["arm"] == arm and r["loss"] is not None]
            if values:
                means[ds][arm] = statistics.fmean(values)
    normalized = {
        ds: {arm: value / min(row.values()) for arm, value in row.items()}
        for ds, row in means.items() if row and min(row.values()) > 0
    }
    aggregate = {}
    for arm in arms:
        ratios = [row[arm] for row in normalized.values() if arm in row]
        if ratios:
            aggregate[arm] = {
                "geomean_normalized_loss": math.exp(statistics.fmean(math.log(v) for v in ratios)),
                "wins": sum(1 for row in normalized.values() if row.get(arm) == 1.0),
                "datasets": len(ratios),
            }
    return {"mean_loss": means, "normalized_loss": normalized, "aggregate": aggregate}


def _wilcoxon_p(log_ratios: list[float]) -> float | None:
    nonzero = [v for v in log_ratios if v != 0.0]
    if len(nonzero) < 5:
        return None
    from scipy.stats import wilcoxon

    return float(wilcoxon(nonzero).pvalue)


def compare(baseline_result: dict, candidate_result: dict, baseline: str, candidate: str,
            min_improvement: float = 0.005, max_dataset_regression: float = 0.03,
            alpha: float = 0.05) -> dict:
    """Pair two arms per (dataset, seed) and decide whether the candidate wins.

    The gate passes only if the candidate improves the geometric-mean loss
    ratio by at least ``min_improvement``, the improvement is significant
    under a Wilcoxon signed-rank test across datasets, no dataset gets worse
    by more than ``max_dataset_regression`` beyond its own seed noise, and no
    pair is missing or failed.
    """
    base = _losses(baseline_result["records"], baseline)
    cand = _losses(candidate_result["records"], candidate)
    expected = {(r["dataset"], r["seed"]) for r in baseline_result["records"]
                if r["arm"] == baseline} | {
               (r["dataset"], r["seed"]) for r in candidate_result["records"]
               if r["arm"] == candidate}
    missing = sorted(key for key in expected if key not in base or key not in cand)
    per_dataset: dict[str, list[float]] = {}
    for key in sorted(set(base) & set(cand)):
        if base[key] > 0 and cand[key] > 0:
            per_dataset.setdefault(key[0], []).append(math.log(cand[key] / base[key]))
    rows = []
    for ds, logs in per_dataset.items():
        mean_log = statistics.fmean(logs)
        spread = statistics.stdev(logs) / math.sqrt(len(logs)) if len(logs) > 1 else 0.0
        rows.append({
            "dataset": ds,
            "ratio": math.exp(mean_log),
            "seed_se_log": spread,
            "significant_regression": mean_log > math.log1p(max_dataset_regression)
                                      and mean_log - 2.0 * spread > 0.0,
        })
    rows.sort(key=lambda row: row["ratio"])
    dataset_logs = [math.log(row["ratio"]) for row in rows]
    geomean = math.exp(statistics.fmean(dataset_logs)) if dataset_logs else float("nan")
    p_value = _wilcoxon_p(dataset_logs)
    reasons = []
    if missing:
        reasons.append(f"{len(missing)} dataset/seed pairs missing or failed, e.g. {missing[:3]}")
    if not dataset_logs or geomean > 1.0 - min_improvement:
        reasons.append(f"geomean loss ratio {geomean:.4f} does not improve by {min_improvement:.1%}")
    if p_value is None or p_value >= alpha:
        reasons.append(f"Wilcoxon p={p_value} is not below {alpha}")
    regressions = [row["dataset"] for row in rows if row["significant_regression"]]
    if regressions:
        reasons.append(f"significant regressions over {max_dataset_regression:.0%}: {regressions}")
    return {
        "baseline": baseline, "candidate": candidate,
        "geomean_ratio": geomean, "wilcoxon_p": p_value,
        "improved_datasets": sum(1 for r in rows if r["ratio"] < 1.0),
        "regressed_datasets": sum(1 for r in rows if r["ratio"] > 1.0),
        "datasets": rows, "missing": missing,
        "gate_passed": not reasons, "gate_reasons": reasons,
    }


def render_summary(result: dict) -> str:
    summary = summarize(result)
    arms = [a["name"] for a in result["arms"]]
    lines = ["| Dataset | " + " | ".join(arms) + " |", "|---|" + "---:|" * len(arms)]
    for ds, row in summary["normalized_loss"].items():
        cells = [f"{row[a]:.3f}" if a in row else "n/a" for a in arms]
        lines.append(f"| `{ds}` | " + " | ".join(cells) + " |")
    agg = summary["aggregate"]
    lines.append("| **geomean** | " + " | ".join(
        f"**{agg[a]['geomean_normalized_loss']:.3f}**" if a in agg else "n/a" for a in arms) + " |")
    return "\n".join(lines)


def render_comparison(cmp: dict) -> str:
    lines = [
        f"{cmp['candidate']} vs {cmp['baseline']}: geomean loss ratio {cmp['geomean_ratio']:.4f}, "
        f"Wilcoxon p={cmp['wilcoxon_p']}, improved {cmp['improved_datasets']}, "
        f"regressed {cmp['regressed_datasets']}",
        "",
        "| Dataset | Loss ratio | Seed SE (log) | Significant regression |",
        "|---|---:|---:|---|",
    ]
    for row in cmp["datasets"]:
        lines.append(f"| `{row['dataset']}` | {row['ratio']:.4f} | {row['seed_se_log']:.4f} | "
                     f"{'yes' if row['significant_regression'] else ''} |")
    lines.append("")
    lines.append("Gate: " + ("passed" if cmp["gate_passed"] else "failed: " + "; ".join(cmp["gate_reasons"])))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="fit every arm on every dataset and seed")
    run_p.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    run_p.add_argument("--seeds", type=int, default=5)
    run_p.add_argument("--threads", type=int, default=1)
    run_p.add_argument("--quick", action="store_true")
    run_p.add_argument("--output", type=Path, required=True)
    cmp_p = sub.add_parser("compare", help="paired comparison of two arms")
    cmp_p.add_argument("files", type=Path, nargs="+", help="one result file, or baseline then candidate")
    cmp_p.add_argument("--baseline", required=True)
    cmp_p.add_argument("--candidate", required=True)
    cmp_p.add_argument("--min-improvement", type=float, default=0.005)
    cmp_p.add_argument("--max-dataset-regression", type=float, default=0.03)
    cmp_p.add_argument("--gate", action="store_true", help="exit 1 unless the gate passes")
    args = parser.parse_args(argv)

    if args.command == "run":
        arms = [parse_arm(spec) for spec in args.arms]
        suite = build_suite(quick=args.quick)
        result = run(arms, suite, list(range(args.seeds)), args.threads)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(render_summary(result))
        return 0

    if len(args.files) not in (1, 2):
        parser.error("compare takes one or two result files")
    baseline_result = json.loads(args.files[0].read_text())
    candidate_result = json.loads(args.files[-1].read_text())
    cmp = compare(baseline_result, candidate_result, args.baseline, args.candidate,
                  args.min_improvement, args.max_dataset_regression)
    print(render_comparison(cmp))
    return 0 if (cmp["gate_passed"] or not args.gate) else 1


if __name__ == "__main__":
    sys.exit(main())
