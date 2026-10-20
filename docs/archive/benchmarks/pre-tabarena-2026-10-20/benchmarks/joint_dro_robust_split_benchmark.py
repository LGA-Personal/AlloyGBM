#!/usr/bin/env python3
"""Evidence benchmark for the opt-in joint-DRO robust split-gain flag.

`MultiLabelGBMRanker(multi_label_mode="joint", leaf_solver="dro", ...,
dro_robust_split=True)` carries a per-bin `grad_sq` buffer alongside the
existing multi-output histogram so numeric split *selection* — not just leaf
values — is routed through the DRO effective-gradient formula. That buffer
costs roughly 1.5x the joint-histogram memory (see
`crates/engine/src/shared_histogram.rs::compute_multi_output_split_gain_dro`
and the `docs/reviews/2026-07-02-v0.12.10-special-modes-resolutions.md` §3.3
gap this closes).

This script is **evidence, not a gate**: it measures, for a handful of fixed-
seed fixtures (a homoscedastic control plus two heteroscedastic/outlier
scenarios), whether flipping `dro_robust_split` on changes held-out quality
(RMSE for regression fixtures, NDCG for the ranking fixture), fit wall time,
and peak resident memory — to substantiate or refute the ~1.5x memory-cost
hypothesis and inform whether the flag is worth shipping as a default, kept
opt-in, or deferred. It prints a Markdown report and never exits nonzero on
its own account; there is no `--gate` flag by design.

Each (fixture, flag) combination is fit in its own subprocess so the
peak-RSS reading (`resource.getrusage(...).ru_maxrss`) reflects only that
one fit + predict + eval, uncontaminated by the parent process or by
whichever combination ran before it in-process.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

QUICK_N_ESTIMATORS = 12
FULL_N_ESTIMATORS = 60
QUICK_N_QUERIES_REGRESSION = 96
FULL_N_QUERIES_REGRESSION = 512
QUICK_ITEMS_PER_QUERY = 6
FULL_ITEMS_PER_QUERY = 10
DRO_RADIUS = 0.4
SEED = 13


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    kind: Literal["regression", "ranking"]
    description: str


FIXTURES: tuple[FixtureSpec, ...] = (
    FixtureSpec(
        "homoscedastic_control",
        "regression",
        "Constant-variance Gaussian noise everywhere -- a control where "
        "robust split-gain should have little reason to help.",
    ),
    FixtureSpec(
        "heteroscedastic_regions",
        "regression",
        "Two feature regions with a 40x noise-variance ratio (one calm, "
        "one turbulent) -- the case robust split-gain targets.",
    ),
    FixtureSpec(
        "heteroscedastic_outliers_ranking",
        "ranking",
        "Ranking labels with a fraction of heavy-tailed relevance outliers "
        "injected in one feature region.",
    ),
)


def _n_estimators(quick: bool) -> int:
    return QUICK_N_ESTIMATORS if quick else FULL_N_ESTIMATORS


def _n_queries(quick: bool) -> int:
    return QUICK_N_QUERIES_REGRESSION if quick else FULL_N_QUERIES_REGRESSION


def _items_per_query(quick: bool) -> int:
    return QUICK_ITEMS_PER_QUERY if quick else FULL_ITEMS_PER_QUERY


def _make_regression_fixture(
    *, heteroscedastic: bool, quick: bool, seed: int
) -> dict[str, np.ndarray]:
    """Build a 2-output regression fixture.

    Both outputs share the same latent signal (a smooth function of 2 of 4
    features) plus additive noise. In the heteroscedastic variant, rows
    with `X[:, 0] > 0` ("turbulent" region) get noise with 40x the standard
    deviation of the "calm" region (`X[:, 0] <= 0`); the homoscedastic
    control uses the calm-region noise scale everywhere.
    """
    rng = np.random.default_rng(seed)
    n_queries = _n_queries(quick)
    items = _items_per_query(quick)
    n_rows = n_queries * items
    n_features = 4
    X = rng.standard_normal((n_rows, n_features)).astype(np.float32)

    signal = (
        1.5 * X[:, 1]
        + 0.75 * X[:, 1] * X[:, 2]
        - 0.5 * np.sin(2.0 * X[:, 3])
    )
    calm_std = 0.2
    turbulent_std = 8.0
    if heteroscedastic:
        noise_std = np.where(X[:, 0] > 0.0, turbulent_std, calm_std)
    else:
        noise_std = np.full(n_rows, calm_std)
    noise_0 = rng.normal(0.0, 1.0, size=n_rows) * noise_std
    noise_1 = rng.normal(0.0, 1.0, size=n_rows) * noise_std

    y0 = (signal + noise_0).astype(np.float32)
    y1 = (0.5 * signal + 2.0 + noise_1).astype(np.float32)
    y = np.column_stack([y0, y1]).astype(np.float32)
    return {"X": X, "y": y, "group": None}


def _make_ranking_fixture(*, quick: bool, seed: int) -> dict[str, np.ndarray]:
    """Build a 2-output ranking fixture with query groups and a fraction of
    heavy-tailed relevance-label outliers concentrated in one feature region
    (rows with `X[:, 0] > 0.5`)."""
    rng = np.random.default_rng(seed)
    n_queries = _n_queries(quick)
    items = _items_per_query(quick)
    n_rows = n_queries * items
    n_features = 4
    X = rng.standard_normal((n_rows, n_features)).astype(np.float32)
    group = np.full(n_queries, items, dtype=np.int64)

    base_relevance = 1.5 * X[:, 1] + 0.5 * X[:, 2]
    outlier_region = X[:, 0] > 0.5
    outlier_noise = rng.standard_normal(n_rows) * np.where(outlier_region, 6.0, 0.3)
    relevance_0 = base_relevance + outlier_noise
    relevance_1 = 0.6 * base_relevance + rng.standard_normal(n_rows) * np.where(
        outlier_region, 5.0, 0.3
    )
    # Bucket into small integer relevance grades (NDCG-friendly) without
    # letting the outlier tail escape the label range.
    y0 = np.clip(
        np.round((relevance_0 - relevance_0.min()) / (np.ptp(relevance_0) + 1e-9) * 4), 0, 4
    )
    y1 = np.clip(
        np.round((relevance_1 - relevance_1.min()) / (np.ptp(relevance_1) + 1e-9) * 4), 0, 4
    )
    y = np.column_stack([y0, y1]).astype(np.float32)
    return {"X": X, "y": y, "group": group}


def build_fixture(name: str, *, quick: bool, seed: int) -> dict[str, np.ndarray]:
    if name == "homoscedastic_control":
        return _make_regression_fixture(heteroscedastic=False, quick=quick, seed=seed)
    if name == "heteroscedastic_regions":
        return _make_regression_fixture(heteroscedastic=True, quick=quick, seed=seed)
    if name == "heteroscedastic_outliers_ranking":
        return _make_ranking_fixture(quick=quick, seed=seed)
    raise ValueError(f"unknown fixture {name!r}")


def _split_train_holdout(
    fixture: dict[str, np.ndarray], *, holdout_fraction: float = 0.25
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    X = fixture["X"]
    y = fixture["y"]
    group = fixture.get("group")
    if group is None:
        n_rows = X.shape[0]
        cut = int(n_rows * (1.0 - holdout_fraction))
        train = {"X": X[:cut], "y": y[:cut], "group": None}
        holdout = {"X": X[cut:], "y": y[cut:], "group": None}
        return train, holdout

    n_queries = len(group)
    cut_q = int(n_queries * (1.0 - holdout_fraction))
    cut_row = int(np.sum(group[:cut_q]))
    train = {"X": X[:cut_row], "y": y[:cut_row], "group": group[:cut_q]}
    holdout = {"X": X[cut_row:], "y": y[cut_row:], "group": group[cut_q:]}
    return train, holdout


def _peak_rss_mib() -> float | None:
    try:
        import resource

        raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, AttributeError):
        return None
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return raw / divisor


def _fit_and_evaluate(
    fixture_name: str, *, robust: bool, quick: bool
) -> dict[str, object]:
    """Fit one (fixture, robust) combination and return quality/timing
    metrics. Runs in-process -- callers wanting isolated peak-RSS should
    invoke this via the `--worker` subprocess entry point instead."""
    from alloygbm import MultiLabelGBMRanker
    from alloygbm.evaluation import ndcg, rmse

    spec = next(f for f in FIXTURES if f.name == fixture_name)
    fixture = build_fixture(fixture_name, quick=quick, seed=SEED)
    train, holdout = _split_train_holdout(fixture)

    common_kwargs: dict[str, object] = dict(
        n_estimators=_n_estimators(quick),
        learning_rate=0.1,
        max_depth=4,
        min_data_in_leaf=4,
        seed=SEED,
        multi_label_mode="joint",
        leaf_solver="dro",
        dro_radius=DRO_RADIUS,
        dro_metric="wasserstein",
        dro_robust_split=robust,
    )
    if spec.kind == "ranking":
        common_kwargs["ranking_objective"] = "rank:ndcg"
    else:
        common_kwargs["ranking_objective"] = "squared_error"

    model = MultiLabelGBMRanker(**common_kwargs)
    fit_kwargs: dict[str, object] = {}
    if train["group"] is not None:
        fit_kwargs["group"] = train["group"]

    started = time.perf_counter()
    model.fit(train["X"], train["y"], **fit_kwargs)
    fit_seconds = time.perf_counter() - started

    predictions = np.asarray(model.predict(holdout["X"]), dtype=np.float64)
    y_holdout = np.asarray(holdout["y"], dtype=np.float64)
    n_outputs = y_holdout.shape[1]

    if spec.kind == "ranking":
        # `ndcg()` wants per-row group IDs, but the fixture (mirroring
        # `MultiLabelGBMRanker.fit`'s convention) stores per-query group
        # *sizes* -- expand to per-row IDs.
        group_sizes = np.asarray(holdout["group"])
        row_group_ids = np.repeat(np.arange(len(group_sizes)), group_sizes)
        scores = [
            ndcg(y_holdout[:, k], predictions[:, k], group=row_group_ids)
            for k in range(n_outputs)
        ]
        quality_metric = "mean_ndcg"
        # Higher NDCG is better; report the mean across outputs.
        quality_value = float(np.mean(scores))
    else:
        scores = [
            rmse(y_holdout[:, k], predictions[:, k]) for k in range(n_outputs)
        ]
        quality_metric = "mean_rmse"
        quality_value = float(np.mean(scores))

    return {
        "fixture": fixture_name,
        "robust": robust,
        "quality_metric": quality_metric,
        "quality_value": quality_value,
        "per_output_scores": [float(s) for s in scores],
        "fit_seconds": float(fit_seconds),
        "peak_rss_mib": _peak_rss_mib(),
        "predictions_finite": bool(np.isfinite(predictions).all()),
    }


def _run_worker_subprocess(
    fixture_name: str, *, robust: bool, quick: bool
) -> dict[str, object]:
    """Run `_fit_and_evaluate` in a fresh subprocess so `peak_rss_mib`
    reflects only this one fit + predict + eval."""
    args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--fixture",
        fixture_name,
        "--robust",
        "1" if robust else "0",
    ]
    if quick:
        args.append("--quick")
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"worker failed for fixture={fixture_name!r} robust={robust}: "
            f"{result.stderr.strip()[-4000:]}"
        )
    line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if not line:
        raise RuntimeError(
            f"worker produced no output for fixture={fixture_name!r} robust={robust}: "
            f"stderr={result.stderr.strip()[-2000:]}"
        )
    return json.loads(line)


@dataclass(frozen=True)
class ComparisonRecord:
    fixture: str
    description: str
    quality_metric: str
    quality_off: float
    quality_on: float
    quality_delta: float
    fit_seconds_off: float
    fit_seconds_on: float
    time_ratio: float
    peak_rss_mib_off: float | None
    peak_rss_mib_on: float | None
    rss_ratio: float | None
    predictions_finite_off: bool
    predictions_finite_on: bool


def run_benchmark(*, quick: bool) -> dict[str, object]:
    records: list[ComparisonRecord] = []
    for spec in FIXTURES:
        off = _run_worker_subprocess(spec.name, robust=False, quick=quick)
        on = _run_worker_subprocess(spec.name, robust=True, quick=quick)

        rss_off = off.get("peak_rss_mib")
        rss_on = on.get("peak_rss_mib")
        rss_ratio = (
            float(rss_on) / float(rss_off)
            if isinstance(rss_off, (int, float))
            and isinstance(rss_on, (int, float))
            and float(rss_off) > 0
            else None
        )
        quality_off = float(off["quality_value"])
        quality_on = float(on["quality_value"])
        records.append(
            ComparisonRecord(
                fixture=spec.name,
                description=spec.description,
                quality_metric=str(off["quality_metric"]),
                quality_off=quality_off,
                quality_on=quality_on,
                quality_delta=quality_on - quality_off,
                fit_seconds_off=float(off["fit_seconds"]),
                fit_seconds_on=float(on["fit_seconds"]),
                time_ratio=(
                    float(on["fit_seconds"]) / float(off["fit_seconds"])
                    if float(off["fit_seconds"]) > 0
                    else float("nan")
                ),
                peak_rss_mib_off=rss_off,
                peak_rss_mib_on=rss_on,
                rss_ratio=rss_ratio,
                predictions_finite_off=bool(off["predictions_finite"]),
                predictions_finite_on=bool(on["predictions_finite"]),
            )
        )
    return {
        "quick": quick,
        "n_estimators": _n_estimators(quick),
        "dro_radius": DRO_RADIUS,
        "seed": SEED,
        "records": records,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }


def _fmt_rss(value: float | None) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "n/a"


def _fmt_ratio(value: float | None) -> str:
    return f"{value:.3f}x" if isinstance(value, (int, float)) else "n/a"


def render_markdown(report: dict[str, object]) -> str:
    records: list[ComparisonRecord] = report["records"]  # type: ignore[assignment]
    lines = [
        "# Joint-DRO Robust Split-Gain Evidence Benchmark",
        "",
        "This is an **evidence report, not a CI gate**. It measures whether "
        "`MultiLabelGBMRanker(multi_label_mode='joint', leaf_solver='dro', "
        "dro_robust_split=True)` improves held-out quality relative to the "
        "existing DRO-leaf-only path (`dro_robust_split=False`, the "
        "default), and at what fit-time and peak-memory cost. Lower is "
        "better for `mean_rmse`; higher is better for `mean_ndcg`.",
        "",
        "## Environment",
        "",
    ]
    environment = report.get("environment", {})
    if isinstance(environment, dict):
        for key, value in environment.items():
            lines.append(f"- {key}: {value}")
    lines.extend(
        [
            f"- quick: {report.get('quick')}",
            f"- n_estimators: {report.get('n_estimators')}",
            f"- dro_radius: {report.get('dro_radius')}",
            f"- seed: {report.get('seed')}",
            "- Each (fixture, flag) combination fit in its own subprocess; "
            "`peak_rss_mib` is that subprocess's `resource.getrusage(...).ru_maxrss`, "
            "so it is not diluted by anything else running in-process.",
            "",
            "## Results",
            "",
            "| Fixture | Metric | Off | On | Delta (on - off) | Fit s (off) | Fit s (on) | Time ratio | Peak RSS MiB (off) | Peak RSS MiB (on) | RSS ratio | Finite (off/on) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for record in records:
        lines.append(
            f"| {record.fixture} | {record.quality_metric} | "
            f"{record.quality_off:.6f} | {record.quality_on:.6f} | "
            f"{record.quality_delta:+.6f} | {record.fit_seconds_off:.4f} | "
            f"{record.fit_seconds_on:.4f} | {record.time_ratio:.3f}x | "
            f"{_fmt_rss(record.peak_rss_mib_off)} | {_fmt_rss(record.peak_rss_mib_on)} | "
            f"{_fmt_ratio(record.rss_ratio)} | "
            f"{record.predictions_finite_off}/{record.predictions_finite_on} |"
        )
    lines.append("")
    lines.append("## Fixture descriptions")
    lines.append("")
    for spec in FIXTURES:
        lines.append(f"- **{spec.name}**: {spec.description}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="use the compact fixture sizes")
    parser.add_argument("--output", type=str, help="write the Markdown report to this path")
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,  # internal: single (fixture, flag) subprocess entry point
    )
    parser.add_argument("--fixture", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--robust", type=str, choices=("0", "1"), help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.worker:
        if not args.fixture or args.robust is None:
            print("--worker requires --fixture and --robust", file=sys.stderr)
            return 2
        result = _fit_and_evaluate(
            args.fixture, robust=args.robust == "1", quick=args.quick
        )
        print(json.dumps(result))
        return 0

    report = run_benchmark(quick=args.quick)
    rendered = render_markdown(report)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
