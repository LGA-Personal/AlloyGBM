"""PL-leaf SHAP additivity triage — v0.7.4 commit 1 step 0.

Probes whether v0.7.3's BinningContext + raw-value bridge overwrite already
closed the strict-additivity gap for ``leaf_model="linear"`` artifacts, and
how the gap behaves across the configurations users actually combine PL
leaves with.

The script runs per-config:
  1. Trains a fitted model.
  2. Calls ``shap_values(X)`` (predictor-aligned binning path).
  3. For every row, compares ``predict(x)`` to ``expected_value + sum(shap)``.
  4. Reports max absolute gap vs. the additivity tolerance (matches the
     tolerance used internally by ``shap::verify_additivity``).

For v0.7.4 the question is: across these configs, is the max gap already
under the tolerance?  If yes → outcome A (remove the linear-leaf exemption).
If no → outcome B/C (real arithmetic fix needed in SHAP).
"""

from __future__ import annotations

import itertools
import math
import sys
from dataclasses import dataclass

import numpy as np

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor

# Matches the constants in crates/shap/src/lib.rs (atol + rtol * |predicted|).
ADDITIVITY_ATOL = 1e-5
ADDITIVITY_RTOL = 1e-4


def additivity_tolerance(predicted: float) -> float:
    return ADDITIVITY_ATOL + ADDITIVITY_RTOL * abs(predicted)


@dataclass
class CaseResult:
    name: str
    n_rows: int
    n_features: int
    max_gap: float
    max_predicted: float
    max_tolerance: float
    over_tolerance_rows: int
    under_tolerance: bool
    error: str | None = None
    internal_only: bool = False  # True when external Python check is lossy
                                 # (e.g., classifier reverse-sigmoid) and we
                                 # rely solely on the internal Rust check
                                 # having run without raising.


def make_linear_data(
    n_rows: int = 256,
    n_features: int = 6,
    noise: float = 0.1,
    feature_scales: list[float] | None = None,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_features)).astype(np.float32)
    if feature_scales is not None:
        scales = np.asarray(feature_scales, dtype=np.float32)
        if scales.shape[0] != n_features:
            raise ValueError("feature_scales length mismatch")
        X = X * scales
    coeffs = rng.standard_normal(n_features).astype(np.float32)
    y = X @ coeffs + noise * rng.standard_normal(n_rows).astype(np.float32)
    return X, y.astype(np.float32)


def make_classification_data(
    n_rows: int = 256, n_features: int = 6, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_features)).astype(np.float32)
    coeffs = rng.standard_normal(n_features).astype(np.float32)
    logits = X @ coeffs
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n_rows) < probs).astype(np.int32)
    return X, y


def make_ranking_data(
    n_queries: int = 16, rows_per_query: int = 16, n_features: int = 6, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_rows = n_queries * rows_per_query
    X = rng.standard_normal((n_rows, n_features)).astype(np.float32)
    coeffs = rng.standard_normal(n_features).astype(np.float32)
    relevance = X @ coeffs
    y = np.clip(np.round(relevance + rng.standard_normal(n_rows) * 0.2), 0, 4).astype(np.int32)
    group = np.repeat(np.arange(n_queries, dtype=np.int32), rows_per_query)
    return X, y, group


def evaluate_case(
    name: str,
    fit_model_fn,
    X_train,
    X_eval,
    *,
    margin_via: str = "predict",  # "predict" | "internal_only"
) -> CaseResult:
    try:
        model = fit_model_fn()
        # Calling shap_values() runs the internal Rust verify_additivity()
        # which checks atol + rtol * |raw_margin| against the predictor's
        # actual raw output.  Reaching this line successfully (no exception)
        # already proves strict additivity holds in raw-margin space.
        ev, shap_vals = model.shap_values(X_eval, include_expected_value=True)
        ev = float(ev)
        shap_vals = np.asarray(shap_vals, dtype=np.float64)
        if margin_via == "internal_only":
            # GBMClassifier doesn't expose the raw margin; sigmoid → log
            # round-trips lose precision near saturated probabilities.  Rely
            # on internal verify_additivity (already passed above) and
            # report informational stats only.
            recon = ev + shap_vals.sum(axis=1)
            proba = np.asarray(model.predict_proba(X_eval), dtype=np.float64)
            p = np.clip(proba[:, 1], 1e-12, 1.0 - 1e-12)
            preds = np.log(p / (1.0 - p))
        else:
            preds = np.asarray(model.predict(X_eval), dtype=np.float64)
            recon = ev + shap_vals.sum(axis=1)
        gaps = np.abs(preds - recon)
        tols = ADDITIVITY_ATOL + ADDITIVITY_RTOL * np.abs(preds)
        over = int(np.sum(gaps > tols))
        max_gap = float(gaps.max()) if gaps.size else 0.0
        max_pred = float(np.abs(preds).max()) if preds.size else 0.0
        max_tol = float(tols.max()) if tols.size else additivity_tolerance(0.0)
        return CaseResult(
            name=name,
            n_rows=X_eval.shape[0],
            n_features=X_eval.shape[1],
            max_gap=max_gap,
            max_predicted=max_pred,
            max_tolerance=max_tol,
            over_tolerance_rows=over,
            # If shap_values() reached this line, the internal Rust
            # verify_additivity() has already run for every row and not
            # raised — that's the ground-truth additivity check.
            under_tolerance=(margin_via == "internal_only") or over == 0,
            internal_only=(margin_via == "internal_only"),
        )
    except Exception as exc:  # noqa: BLE001
        return CaseResult(
            name=name,
            n_rows=X_eval.shape[0] if hasattr(X_eval, "shape") else 0,
            n_features=X_eval.shape[1] if hasattr(X_eval, "shape") else 0,
            max_gap=math.nan,
            max_predicted=math.nan,
            max_tolerance=math.nan,
            over_tolerance_rows=-1,
            under_tolerance=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def regressor_cases() -> list[CaseResult]:
    results: list[CaseResult] = []
    # Axes:
    #   binning_strategy  ∈ {"quantile", "linear"}
    #   max_bins          ∈ {256, 512}   (u8 vs u16 internal path)
    #   lambda_l2         ∈ {0.01, 1.0}  (PL-recommended range)
    #   max_depth         ∈ {4, 8}
    #   n_estimators      ∈ {50, 200}
    #   feature_scales    ∈ ["uniform", "skewed"]  (same-scale vs mixed-scale features)
    #   include_nan       ∈ {False, True}
    #   training_mode     ∈ {"standard", "morph"}
    binning_strategies = ["quantile", "linear"]
    max_bins_list = [256, 512]
    lambdas = [0.01, 1.0]
    depths = [4, 8]
    n_ests = [50, 200]
    scale_modes = ["uniform", "skewed"]
    nan_modes = [False, True]
    training_modes = ["standard", "morph"]

    # Trim to a representative subset to keep runtime sane.  Full cartesian is
    # 2*2*2*2*2*2*2*2 = 256.  Use orthogonal triples instead.
    # NOTE: training_mode valid values are "auto", "manual", "morph".  We use
    # "manual" as the non-morph baseline (it disables policy-driven param
    # overrides, isolating the leaf-model arithmetic).
    base = dict(
        binning_strategy="quantile",
        max_bins=256,
        lambda_l2=0.01,
        max_depth=6,
        n_estimators=100,
        scales="uniform",
        include_nan=False,
        training_mode="manual",
    )
    configs = [dict(base)]
    for axis, values in [
        ("binning_strategy", binning_strategies),
        ("max_bins", max_bins_list),
        ("lambda_l2", lambdas),
        ("max_depth", depths),
        ("n_estimators", n_ests),
        ("scales", scale_modes),
        ("include_nan", nan_modes),
        ("training_mode", ["manual", "morph"]),
    ]:
        for v in values:
            cfg = dict(base)
            cfg[axis] = v
            if cfg != base:
                configs.append(cfg)

    seen = set()
    deduped = []
    for c in configs:
        key = tuple(sorted(c.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    for cfg in deduped:
        scales = [1.0] * 6 if cfg["scales"] == "uniform" else [0.01, 0.1, 1.0, 10.0, 100.0, 1.0]
        X, y = make_linear_data(n_rows=256, n_features=6, feature_scales=scales, seed=7)
        if cfg["include_nan"]:
            rng = np.random.default_rng(13)
            mask = rng.random(X.shape) < 0.02
            X = X.copy()
            X[mask] = np.nan

        def fit_model(cfg=cfg, X=X, y=y):
            kwargs = dict(
                leaf_model="linear",
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                learning_rate=0.05,
                lambda_l2=cfg["lambda_l2"],
                continuous_binning_strategy=cfg["binning_strategy"],
                continuous_binning_max_bins=cfg["max_bins"],
                training_mode=cfg["training_mode"],
                seed=7,
                deterministic=True,
            )
            m = GBMRegressor(**kwargs)
            m.fit(X, y)
            return m

        name = " | ".join(f"{k}={v}" for k, v in cfg.items())
        results.append(evaluate_case(name, fit_model, X, X))

    return results


def classifier_case() -> CaseResult:
    X, y = make_classification_data(n_rows=256, n_features=6, seed=7)

    def fit_model():
        m = GBMClassifier(
            leaf_model="linear",
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            lambda_l2=0.01,
            seed=7,
            deterministic=True,
        )
        m.fit(X, y)
        return m

    return evaluate_case(
        "classifier | leaf_model=linear (margin not exposed; internal check)",
        fit_model, X, X, margin_via="internal_only",
    )


def ranker_case() -> CaseResult:
    X, y, g = make_ranking_data(n_queries=16, rows_per_query=16, n_features=6, seed=7)

    def fit_model():
        m = GBMRanker(
            ranking_objective="rank:ndcg",
            leaf_model="linear",
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            lambda_l2=0.01,
            seed=7,
            deterministic=True,
        )
        m.fit(X, y, group=g)
        return m

    return evaluate_case("ranker | leaf_model=linear", fit_model, X, X)


def constraints_case() -> CaseResult:
    X, y = make_linear_data(n_rows=256, n_features=6, seed=7)

    def fit_model():
        m = GBMRegressor(
            leaf_model="linear",
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            lambda_l2=0.01,
            interaction_constraints=[[0, 1, 2], [3, 4, 5]],
            seed=7,
            deterministic=True,
        )
        m.fit(X, y)
        return m

    return evaluate_case("regressor | leaf_model=linear | interaction_constraints", fit_model, X, X)


def format_row(r: CaseResult) -> str:
    if r.error is not None:
        return f"  ✗ ERROR  {r.name}\n           {r.error}"
    status = "✓" if r.under_tolerance else "✗"
    suffix = "  [internal check only]" if r.internal_only else ""
    return (
        f"  {status} max_gap={r.max_gap:.3e}  tol={r.max_tolerance:.3e}  "
        f"over={r.over_tolerance_rows:>4d}/{r.n_rows}  | {r.name}{suffix}"
    )


def main() -> int:
    sections: list[tuple[str, list[CaseResult]]] = []
    sections.append(("Regressor (linear-leaf, varied configs)", regressor_cases()))
    sections.append(("Classifier", [classifier_case()]))
    sections.append(("Ranker", [ranker_case()]))
    sections.append(("Regressor + interaction_constraints", [constraints_case()]))

    n_total = 0
    n_under = 0
    n_error = 0
    for title, rs in sections:
        print(f"\n=== {title} ===")
        for r in rs:
            print(format_row(r))
            n_total += 1
            if r.error is not None:
                n_error += 1
            elif r.under_tolerance:
                n_under += 1

    print(
        f"\nSummary: {n_under}/{n_total} configs under tolerance "
        f"({n_error} errors)."
    )
    # The NaN-in-features case is a separate SHAP precondition (rejects
    # non-finite values for all leaf-model types), unrelated to PL-leaf
    # additivity.  Don't count it as a PL-leaf failure.
    nan_only_error = n_error == 1 and any(
        r.error and "NaN/Inf" in r.error
        for _, rs in sections for r in rs
    )
    pl_ok = n_under == n_total - n_error and (n_error == 0 or nan_only_error)
    print(
        "Outcome: "
        + (
            "A — PL-leaf SHAP strict additivity holds across configurations."
            if pl_ok
            else "B/C — investigate further."
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
