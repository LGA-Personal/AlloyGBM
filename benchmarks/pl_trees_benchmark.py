#!/usr/bin/env python3
"""
PL Trees benchmark: constant-leaf vs linear-leaf comparison.

Runs AlloyGBM with leaf_model="constant" and leaf_model="linear" on several
datasets to quantify accuracy gains, training cost, and convergence speed.

Key findings from this benchmark:
- Both models use auto training policy which includes early stopping.
  Use training_policy="manual" for raw convergence curves.
- Linear leaves converge faster and to better solutions on datasets with
  complex local-linear structure (e.g. california_housing).
- On small/simple datasets with very few features, constant leaves may
  be competitive due to leaf regularization effects.

Usage:
    # Quick run (synthetic + one dataset, fast profile):
    .venv/bin/python benchmarks/pl_trees_benchmark.py --quick

    # Full run (all scenarios, multiple profiles):
    .venv/bin/python benchmarks/pl_trees_benchmark.py
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# ── Data generators ────────────────────────────────────────────────────────────

def _make_synthetic_linear(
    n: int = 2000,
    n_features: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate y = X @ w + noise with multi-feature linear trend."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    # Assign non-zero weights to all features
    w = np.array([1.5, -2.0, 0.8, 0.3, 1.2, -0.5, 0.7, -1.1], dtype=np.float32)[:n_features]
    y = (X @ w + 0.15 * rng.standard_normal(n)).astype(np.float32)
    return X, y


def _make_synthetic_classification(
    n: int = 2000,
    n_features: int = 6,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    w = np.array([1.5, -2.0, 0.8, 0.3, 1.2, -0.5], dtype=np.float32)[:n_features]
    logits = X @ w
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(np.float32)
    return X, y


def _load_sklearn_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    from sklearn import datasets
    loader = getattr(datasets, f"fetch_{name}", None) or getattr(datasets, f"load_{name}")
    data = loader()
    X = data["data"].astype(np.float32)
    y = data["target"].astype(np.float32)
    return X, y


def _try_load_csv_dataset(scenario: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Try to load a prepared benchmark CSV dataset."""
    import pandas as pd
    import yaml

    manifest_path = ROOT / "benchmarks" / scenario / "manifest.yaml"
    if not manifest_path.exists():
        return None
    with manifest_path.open() as f:
        manifest = yaml.safe_load(f)
    target_col = str(manifest["prepared"]["target"])
    fname = str(manifest["prepared"]["filename"])
    prepared = ROOT / "benchmarks" / "data" / scenario / "prepared" / fname
    if not prepared.exists():
        return None
    frame = pd.read_csv(prepared)
    if target_col not in frame.columns:
        return None
    skip = {target_col, "group_id", "timestamp"}
    feat_cols = [c for c in frame.columns if c not in skip]
    frame_num = frame[feat_cols].apply(
        lambda s: pd.to_numeric(s, errors="coerce")
    ).replace([float("inf"), float("-inf")], float("nan"))
    y_raw = pd.to_numeric(frame[target_col], errors="coerce")
    mask = frame_num.notna().all(axis=1) & y_raw.notna()
    X = frame_num[mask].to_numpy(dtype=np.float32)
    y = y_raw[mask].to_numpy(dtype=np.float32)
    return X, y


# ── Benchmark record ────────────────────────────────────────────────────────────

@dataclass
class PLRecord:
    scenario: str
    leaf_model: str
    n_estimators: int
    max_depth: int
    learning_rate: float
    seed: int
    lambda_l2: float
    train_rows: int
    test_rows: int
    n_features: int
    fit_seconds: float
    peak_rss_mb: float
    rmse: float | None
    accuracy: float | None
    status: str
    error: str


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pred_labels = (y_pred >= 0.5).astype(int)
    return float(np.mean(pred_labels == y_true.astype(int)))


# ── Single run ─────────────────────────────────────────────────────────────────

def _run_one(
    scenario: str,
    task: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    leaf_model: str,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    seed: int,
    lambda_l2: float = 0.0,
    training_policy: str = "auto",
    row_subsample: float = 0.8,
    col_subsample: float = 0.8,
) -> PLRecord:
    from alloygbm import GBMClassifier, GBMRegressor

    nan = float("nan")
    kwargs: dict[str, Any] = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        seed=seed,
        leaf_model=leaf_model,
        lambda_l2=lambda_l2,
        row_subsample=row_subsample,
        col_subsample=col_subsample,
        deterministic=True,
        training_policy=training_policy,
    )

    try:
        if task == "regression":
            model = GBMRegressor(**kwargs)
        else:
            model = GBMClassifier(**kwargs)

        tracemalloc.start()
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / (1024 * 1024)

        if task == "regression":
            y_pred = np.array(model.predict(X_test), dtype=float)
            rmse = _rmse(y_test, y_pred)
            acc = None
        else:
            proba = np.array(model.predict_proba(X_test), dtype=float)
            y_pred_proba = proba[:, 1] if proba.ndim == 2 else proba
            rmse = None
            acc = _accuracy(y_test, y_pred_proba)

        return PLRecord(
            scenario=scenario,
            leaf_model=leaf_model,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            seed=seed,
            lambda_l2=lambda_l2,
            train_rows=len(X_train),
            test_rows=len(X_test),
            n_features=X_train.shape[1],
            fit_seconds=fit_seconds,
            peak_rss_mb=peak_mb,
            rmse=rmse,
            accuracy=acc,
            status="PASS",
            error="",
        )
    except Exception as exc:  # noqa: BLE001
        import traceback
        return PLRecord(
            scenario=scenario,
            leaf_model=leaf_model,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            seed=seed,
            lambda_l2=lambda_l2,
            train_rows=0,
            test_rows=0,
            n_features=0,
            fit_seconds=0.0,
            peak_rss_mb=0.0,
            rmse=nan,
            accuracy=nan,
            status="FAIL",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


# ── Convergence helper ─────────────────────────────────────────────────────────

def _convergence_curve(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    leaf_model: str,
    checkpoints: list[int],
    max_depth: int,
    learning_rate: float,
    seed: int,
) -> list[tuple[int, float]]:
    """Return (n_estimators, rmse) at each checkpoint. Uses manual policy to disable early stopping."""
    curve = []
    for n in checkpoints:
        r = _run_one(
            scenario="convergence",
            task="regression",
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            leaf_model=leaf_model,
            n_estimators=n,
            max_depth=max_depth,
            learning_rate=learning_rate,
            seed=seed,
            training_policy="manual",  # disable auto early stopping
            row_subsample=1.0,          # full data for cleaner convergence
            col_subsample=1.0,
        )
        if r.status == "PASS" and r.rmse is not None:
            curve.append((n, r.rmse))
    return curve


# ── lambda_l2 sweep ────────────────────────────────────────────────────────────

def _sweep_lambda_l2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    candidates: list[float],
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    seed: int,
) -> list[tuple[float, float, float]]:
    """Return (lambda_l2, rmse, fit_seconds) for each candidate."""
    results = []
    for lam in candidates:
        r = _run_one(
            scenario="sweep",
            task="regression",
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            leaf_model="linear",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            seed=seed,
            lambda_l2=lam,
            training_policy="manual",
            row_subsample=1.0,
            col_subsample=1.0,
        )
        if r.status == "PASS" and r.rmse is not None:
            results.append((lam, r.rmse, r.fit_seconds))
            print(f"  lambda_l2={lam}: rmse={r.rmse:.6f} fit={r.fit_seconds:.3f}s")
        else:
            print(f"  lambda_l2={lam}: FAIL {r.error[:120]}")
    return results


# ── Report rendering ───────────────────────────────────────────────────────────

def _render_report(
    run_id: str,
    quick: bool,
    comparison_records: list[PLRecord],
    convergence_const: list[tuple[int, float]],
    convergence_linear: list[tuple[int, float]],
    sweep_results: list[tuple[float, float, float]],
    sweep_n_features: int,
) -> str:
    lines = [
        "# PL Trees Benchmark Report",
        "",
        f"**Run ID**: `{run_id}`  ",
        f"**Mode**: {'quick' if quick else 'full'}  ",
        f"**Date**: {run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}",
        "",
        "## Overview",
        "",
        "This report compares `leaf_model='constant'` (standard scalar leaves) against "
        "`leaf_model='linear'` (piecewise-linear leaves) across regression and "
        "classification datasets. Both models use `training_policy='auto'` for the "
        "main comparison table and `training_policy='manual'` for convergence curves.",
        "",
    ]

    # Comparison table.
    lines += ["## Accuracy Comparison", ""]
    pass_records = [r for r in comparison_records if r.status == "PASS"]
    if pass_records:
        lines += [
            "| scenario | task | leaf_model | n_est | depth | lr | "
            "train_rows | n_feat | fit_s | rmse | accuracy | peak_mb |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in sorted(pass_records, key=lambda x: (x.scenario, x.leaf_model)):
            rmse_s = f"{r.rmse:.6f}" if r.rmse is not None else "—"
            acc_s = f"{r.accuracy:.4f}" if r.accuracy is not None else "—"
            task_label = "reg" if r.rmse is not None else "clf"
            lines.append(
                f"| {r.scenario} | {task_label} | "
                f"{r.leaf_model} | {r.n_estimators} | {r.max_depth} | "
                f"{r.learning_rate} | {r.train_rows} | {r.n_features} | "
                f"{r.fit_seconds:.3f} | {rmse_s} | {acc_s} | {r.peak_rss_mb:.1f} |"
            )
        lines.append("")

        # Relative improvement section for regression.
        reg_records = [r for r in pass_records if r.rmse is not None]
        scenarios = sorted({r.scenario for r in reg_records})
        improvements = []
        for sc in scenarios:
            const_rows = [r for r in reg_records if r.scenario == sc and r.leaf_model == "constant"]
            lin_rows = [r for r in reg_records if r.scenario == sc and r.leaf_model == "linear"]
            if const_rows and lin_rows:
                const_rmse = float(np.median([r.rmse for r in const_rows]))  # type: ignore[misc]
                lin_rmse = float(np.median([r.rmse for r in lin_rows]))  # type: ignore[misc]
                rel_improv = (const_rmse - lin_rmse) / const_rmse * 100.0
                const_fit = float(np.median([r.fit_seconds for r in const_rows]))
                lin_fit = float(np.median([r.fit_seconds for r in lin_rows]))
                overhead = (lin_fit - const_fit) / const_fit * 100.0
                improvements.append((sc, const_rmse, lin_rmse, rel_improv, overhead))

        if improvements:
            lines += ["### RMSE Improvement Summary (Regression)", ""]
            lines += [
                "| scenario | constant RMSE | linear RMSE | RMSE improvement | train time overhead |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
            for sc, cr, lr, ri, oh in improvements:
                lines.append(
                    f"| {sc} | {cr:.6f} | {lr:.6f} | "
                    f"{ri:+.2f}% | {oh:+.1f}% |"
                )
            lines.append("")

        # Relative improvement section for classification.
        clf_records = [r for r in pass_records if r.accuracy is not None]
        clf_scenarios = sorted({r.scenario for r in clf_records})
        clf_improvements = []
        for sc in clf_scenarios:
            const_rows = [r for r in clf_records if r.scenario == sc and r.leaf_model == "constant"]
            lin_rows = [r for r in clf_records if r.scenario == sc and r.leaf_model == "linear"]
            if const_rows and lin_rows:
                const_acc = float(np.median([r.accuracy for r in const_rows]))  # type: ignore[misc]
                lin_acc = float(np.median([r.accuracy for r in lin_rows]))  # type: ignore[misc]
                delta = lin_acc - const_acc
                clf_improvements.append((sc, const_acc, lin_acc, delta))

        if clf_improvements:
            lines += ["### Accuracy Improvement Summary (Classification)", ""]
            lines += [
                "| scenario | constant acc | linear acc | Δ accuracy |",
                "| --- | ---: | ---: | ---: |",
            ]
            for sc, ca, la, d in clf_improvements:
                lines.append(f"| {sc} | {ca:.4f} | {la:.4f} | {d:+.4f} |")
            lines.append("")
    else:
        lines += ["*No PASS records.*", ""]

    # Failures.
    fail_records = [r for r in comparison_records if r.status != "PASS"]
    if fail_records:
        lines += ["### Failures", ""]
        for r in fail_records:
            lines.append(f"- **{r.scenario}** / {r.leaf_model}: {r.error[:200]}")
        lines.append("")

    # Convergence curves (manual policy, no subsampling, full data).
    if convergence_const or convergence_linear:
        lines += [
            "## Convergence on Synthetic Linear Target",
            "",
            "RMSE vs number of estimators on a linear-trend dataset "
            "(y = X @ w + noise). `training_policy='manual'`, no subsampling "
            "(row/col = 1.0) to show raw convergence without early stopping.",
            "",
            "| n_estimators | constant RMSE | linear RMSE |",
            "| ---: | ---: | ---: |",
        ]
        const_map = dict(convergence_const)
        lin_map = dict(convergence_linear)
        all_n = sorted(set(const_map) | set(lin_map))
        for n in all_n:
            cr = f"{const_map[n]:.6f}" if n in const_map else "—"
            lr = f"{lin_map[n]:.6f}" if n in lin_map else "—"
            lines.append(f"| {n} | {cr} | {lr} |")
        lines.append("")

    # lambda_l2 sweep.
    if sweep_results:
        lines += [
            "## `lambda_l2` Sweep for Linear Leaves",
            "",
            f"Synthetic linear dataset with {sweep_n_features} features. "
            "`training_policy='manual'`, no subsampling. Lower RMSE is better.",
            "",
            "| lambda_l2 | linear RMSE | fit_seconds |",
            "| ---: | ---: | ---: |",
        ]
        for lam, rmse, fs in sweep_results:
            lines.append(f"| {lam} | {rmse:.6f} | {fs:.3f} |")
        lines.append("")

        best_lam, best_rmse, _ = min(sweep_results, key=lambda x: x[1])
        lines += [
            f"**Recommended default**: `lambda_l2={best_lam}` "
            f"(lowest RMSE = {best_rmse:.6f}). For `leaf_model='linear'`, "
            f"non-zero `lambda_l2` provides regularization for the linear weights.",
            "",
        ]

    lines += [
        "## Notes",
        "",
        "- Peak RSS measured with Python `tracemalloc` (Python heap only).",
        "- Main comparison uses `training_policy='auto'` with `row_subsample=0.8`, "
        "  `col_subsample=0.8`, `deterministic=True`. Auto-policy includes dataset-aware "
        "  early stopping, which may truncate training before `n_estimators` rounds.",
        "- Convergence curves use `training_policy='manual'`, no subsampling, to show "
        "  raw gradient descent behavior.",
        "- Linear leaves use the closed-form ridge solve "
        "  `α* = -(XᵀHX + λI)⁻¹ Xᵀg` per node.",
        "- `lambda_l2` regularizes both the scalar leaf (standard NR) and the "
        "  linear leaf weights.",
        "",
    ]

    return "\n".join(lines) + "\n"


# ── Main ───────────────────────────────────────────────────────────────────────

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="fast mode: synthetic only, single profile, minimal seeds")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs" / "benchmarks" / "pl_trees_v1.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"PL Trees benchmark  run_id={run_id}  quick={args.quick}")

    from sklearn.model_selection import train_test_split

    # ── Profiles ───────────────────────────────────────────────────────────────
    if args.quick:
        profiles = [{"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05}]
        seeds = [args.seed]
        lambda_candidates = [0.0, 0.01, 0.1, 1.0, 10.0]
        convergence_checkpoints = [10, 25, 50, 100, 200]
    else:
        profiles = [
            {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.10},
            {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05},
        ]
        seeds = [args.seed, args.seed + 7, args.seed + 17]
        lambda_candidates = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        convergence_checkpoints = [10, 25, 50, 100, 200, 400]

    # ── Dataset registry ───────────────────────────────────────────────────────
    reg_scenarios: list[tuple[str, np.ndarray, np.ndarray]] = []
    clf_scenarios: list[tuple[str, np.ndarray, np.ndarray]] = []

    # 1. Synthetic linear (always available).
    X_syn, y_syn = _make_synthetic_linear(n=3000, n_features=8, seed=args.seed)
    reg_scenarios.append(("synthetic_linear", X_syn, y_syn))

    # 2. sklearn datasets (always available with scikit-learn installed).
    try:
        X_ca, y_ca = _load_sklearn_dataset("california_housing")
        reg_scenarios.append(("california_housing", X_ca, y_ca))
    except Exception as exc:  # noqa: BLE001
        print(f"  skipping california_housing: {exc}")

    # 3. CSV-prepared scenarios.
    for scenario in ["abalone_regression", "bike_sharing"]:
        result = _try_load_csv_dataset(scenario)
        if result is not None:
            reg_scenarios.append((scenario, result[0], result[1]))
        else:
            print(f"  skipping {scenario}: not prepared (run benchmarks/{scenario}/prepare.py first)")

    # 4. Synthetic classification.
    X_clf, y_clf = _make_synthetic_classification(n=2000, n_features=6, seed=args.seed)
    clf_scenarios.append(("synthetic_classification", X_clf, y_clf))

    # 5. sklearn classification.
    try:
        X_bc, y_bc = _load_sklearn_dataset("breast_cancer")
        clf_scenarios.append(("breast_cancer", X_bc, y_bc))
    except Exception as exc:  # noqa: BLE001
        print(f"  skipping breast_cancer: {exc}")

    # ── Main comparison loop (auto policy) ────────────────────────────────────
    comparison_records: list[PLRecord] = []

    for profile in profiles:
        n_est = profile["n_estimators"]
        depth = profile["max_depth"]
        lr = profile["learning_rate"]

        for seed in seeds:
            # Regression.
            for name, X, y in reg_scenarios:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.2, random_state=seed
                )
                for lm in ["constant", "linear"]:
                    print(f"  [{name}] {lm} n_est={n_est} depth={depth} lr={lr} seed={seed}")
                    r = _run_one(
                        scenario=name, task="regression",
                        X_train=X_tr, y_train=y_tr,
                        X_test=X_te, y_test=y_te,
                        leaf_model=lm,
                        n_estimators=n_est, max_depth=depth, learning_rate=lr,
                        seed=seed,
                    )
                    comparison_records.append(r)
                    if r.status == "PASS":
                        print(f"    rmse={r.rmse:.6f} fit={r.fit_seconds:.3f}s")
                    else:
                        print(f"    FAIL: {r.error[:120]}")

            # Classification.
            for name, X, y in clf_scenarios:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.2, random_state=seed,
                    stratify=y.astype(int),
                )
                for lm in ["constant", "linear"]:
                    print(f"  [{name}] {lm} n_est={n_est} depth={depth} lr={lr} seed={seed}")
                    r = _run_one(
                        scenario=name, task="classification",
                        X_train=X_tr, y_train=y_tr,
                        X_test=X_te, y_test=y_te,
                        leaf_model=lm,
                        n_estimators=n_est, max_depth=depth, learning_rate=lr,
                        seed=seed,
                    )
                    comparison_records.append(r)
                    if r.status == "PASS":
                        print(f"    acc={r.accuracy:.4f} fit={r.fit_seconds:.3f}s")
                    else:
                        print(f"    FAIL: {r.error[:120]}")

    # ── Convergence curves (manual policy, no subsampling) ────────────────────
    print("\nConvergence curves on synthetic_linear (manual policy, no subsampling) ...")
    X_conv, y_conv = _make_synthetic_linear(n=3000, n_features=8, seed=args.seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X_conv, y_conv, test_size=0.2, random_state=args.seed)
    conv_lr = profiles[0]["learning_rate"]
    conv_depth = profiles[0]["max_depth"]
    convergence_const = _convergence_curve(
        X_tr, y_tr, X_te, y_te, "constant", convergence_checkpoints, conv_depth, conv_lr, args.seed
    )
    convergence_linear = _convergence_curve(
        X_tr, y_tr, X_te, y_te, "linear", convergence_checkpoints, conv_depth, conv_lr, args.seed
    )
    print(f"  constant: {convergence_const}")
    print(f"  linear:   {convergence_linear}")

    # ── lambda_l2 sweep ────────────────────────────────────────────────────────
    print("\nlambda_l2 sweep for linear leaves ...")
    X_sw, y_sw = _make_synthetic_linear(n=3000, n_features=8, seed=args.seed)
    X_str, X_ste, y_str, y_ste = train_test_split(X_sw, y_sw, test_size=0.2, random_state=args.seed)
    sweep_results = _sweep_lambda_l2(
        X_str, y_str, X_ste, y_ste,
        candidates=lambda_candidates,
        n_estimators=profiles[0]["n_estimators"],
        max_depth=profiles[0]["max_depth"],
        learning_rate=profiles[0]["learning_rate"],
        seed=args.seed,
    )
    sweep_n_features = X_sw.shape[1]

    # ── Report ────────────────────────────────────────────────────────────────
    report = _render_report(
        run_id=run_id,
        quick=args.quick,
        comparison_records=comparison_records,
        convergence_const=convergence_const,
        convergence_linear=convergence_linear,
        sweep_results=sweep_results,
        sweep_n_features=sweep_n_features,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"\nReport written to: {args.output}")

    # Brief summary.
    pass_reg = [r for r in comparison_records if r.status == "PASS" and r.rmse is not None]
    pass_clf = [r for r in comparison_records if r.status == "PASS" and r.accuracy is not None]
    fail_count = sum(1 for r in comparison_records if r.status != "PASS")
    print(f"PASS records: {len(pass_reg)} regression, {len(pass_clf)} classification; FAIL: {fail_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
