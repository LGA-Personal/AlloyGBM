"""Frontier matrix: AlloyGBM against LightGBM, XGBoost and CatBoost on every axis.

The other comparison suites each answer one question. ``default_quality.py``
asks how good each library is at its own defaults on ~20 small datasets;
``competitiveness/`` times fixed, matched configurations; ``scale_comparison``
and ``deep_scaling_comparison`` sweep size for one regression target. None of
them says where AlloyGBM wins or loses as the *situation* changes: more rows,
more columns, another task, categorical or missing data, deep or shallow
trees, ten trees or three thousand, no regularization or a lot.

This suite does, the way published GBDT and tabular benchmarks are built:

* **Controlled axes, one factor at a time around a base point** (the design
  behind Grinsztajn et al.'s "medium-sized" benchmark and szilard's GBM-perf
  size sweeps). Every data scenario is the 10,000 x 20 base dataset with one
  thing changed, so a loss on the ``missing=mnar`` cell is about informative
  missingness and nothing else.
* **Hyperparameter regimes applied to every library through its own knobs**
  (GBM-perf and LightGBM's own experiments fix depth, trees and learning rate
  across libraries; CatBoost's benchmarks report defaults *and* tuned).
  ``defaults`` is what users get; the matched regimes isolate the tree
  learner; ``early_stopping`` is the practitioner protocol (a large round
  cap stopped on a shared validation split). Knobs a library has no
  equivalent for are recorded per fit, never silently dropped.
* **Many seeds, proper scoring rules, normalized loss** (OpenML-CC18/CTR23,
  AMLB and TabArena practice): RMSE, log loss, and 1 - NDCG@10 on held-out
  rows; each cell's loss is divided by the best arm's, and Alloy's ratio to
  the best peer is aggregated per axis with win/tie/loss counts that treat
  differences inside seed noise as ties.
* **Tiers** keep it practical: ``smoke`` (seconds, CI), ``quick`` (routine,
  every axis at modest size), ``full`` (the complete matrix, up to 1M rows
  and 1,000 columns, plus extra levels per axis).
* **Real data on official folds**: ``--openml-suite`` adds the tasks of
  OpenML benchmark suites (``cc18``, ``ctr23``, ``grinsztajn`` or any suite
  id), each scored on the task's own repeat/fold splits so per-fold numbers
  are comparable with published results. Needs network access and the
  ``openml`` package.

Usage::

    python benchmarks/frontier_matrix.py list --tier quick
    python benchmarks/frontier_matrix.py run --tier quick --output quick.json \\
        --report quick.md
    python benchmarks/frontier_matrix.py run --tier full --threads 8 \\
        --output full.json --report full.md
    python benchmarks/frontier_matrix.py run --tier quick --axes rows task \\
        --regimes defaults --arms alloy lightgbm --output part.json
    python benchmarks/frontier_matrix.py run --openml-only --openml-suite cc18 ctr23 \
        --regimes defaults --threads 8 --output openml.json --report openml.md
    python benchmarks/frontier_matrix.py report quick.json
    # Paired, gated comparison of two builds (same records contract as
    # default_quality.py: one "dataset" per cell).
    python benchmarks/frontier_matrix.py compare base.json cand.json \\
        --baseline alloy --candidate alloy --gate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from default_quality import (  # noqa: E402
    Arm,
    compare as paired_compare,
    parse_arm,
    render_comparison,
)

SCHEMA_VERSION = 1

REGRESSION = "regression"
BINARY = "binary"
MULTICLASS = "multiclass"
RANKING = "ranking"
TASKS = (REGRESSION, BINARY, MULTICLASS, RANKING)

TIERS = ("smoke", "quick", "full")
PEER_LIBRARIES = ("lightgbm", "xgboost", "catboost")
DEFAULT_ARMS = ("alloy", "lightgbm", "xgboost", "catboost")
DEFAULT_SEEDS = {"smoke": 1, "quick": 3, "full": 5}

THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS",
)

# Two arms whose seed-averaged losses differ by less than this (relative) are
# tied even when one seed's noise estimate is tighter: a 0.5% loss gap is
# below what any user would act on.
MIN_MEANINGFUL_GAP = 0.005


def _tier_rank(tier: str) -> int:
    return TIERS.index(tier)


# --------------------------------------------------------------------------
# Data scenarios
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DataSpec:
    """A synthetic dataset described by the factors the matrix varies.

    Features are generated from a standard-normal latent matrix ``Z``; the
    target depends only on the first ``informative`` latent columns (plus
    categorical effects). ``feature_kind`` changes how latent columns are
    *observed*: monotone transforms (``skewed``) leave the ideal trees
    unchanged and so test binning, while ``discrete`` and ``sparse`` destroy
    information the way real coarse or zero-heavy columns do.
    """

    name: str
    task: str
    axis: str
    level: str
    tier: str = "quick"
    rows: int = 10_000
    features: int = 20
    informative: int = 8
    n_classes: int = 2
    signal: str = "mixed"  # linear | additive | interaction | step | mixed
    noise: float = 0.5  # noise sd relative to the signal's sd
    feature_kind: str = "numeric"  # numeric | skewed | discrete | sparse | redundant
    cat_low: int = 0  # low-cardinality categorical columns (8 levels)
    cat_high: int = 0  # high-cardinality categorical columns (Zipf over 1,000 levels)
    missing_rate: float = 0.0
    missing_mechanism: str = "mcar"  # mcar | mnar
    positive_rate: float | None = None  # binary class balance; None = natural (~50%)
    group_size: int = 20  # ranking documents per query

    def data_seed(self) -> int:
        payload = json.dumps(asdict(self) | {"axis": "", "level": "", "tier": ""}, sort_keys=True)
        return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "little")


@dataclass
class Dataset:
    spec: DataSpec
    X: np.ndarray  # float64, NaN = missing, categorical columns hold integer codes
    y: np.ndarray
    categorical: tuple[int, ...]
    groups: np.ndarray | None  # contiguous query ids for ranking
    # Official (train, test) row indices, one pair per OpenML repeat/fold.
    # When set, the seed axis indexes these folds instead of drawing splits.
    folds: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None
    # Cap on training rows per fold (test folds stay whole), as in the
    # Grinsztajn benchmark. None keeps every training row.
    train_cap: int | None = None

    @property
    def task(self) -> str:
        return self.spec.task


def _univariate(z: np.ndarray, kind: int) -> np.ndarray:
    if kind == 0:
        return np.sin(1.5 * z)
    if kind == 1:
        return np.abs(z) - 0.8
    if kind == 2:
        return np.where(z > 0.3, 1.0, -0.5)
    return np.tanh(2.0 * z) * (z > -1.0)


def _signal(Z: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    n, k = Z.shape
    w = rng.normal(size=k) * np.linspace(1.0, 0.4, k)
    linear = Z @ w
    additive = sum(_univariate(Z[:, j], j % 4) * w[j] for j in range(k))
    pairs = [(j, (j + 1) % k) for j in range(0, k, 2)]
    interaction = sum(Z[:, a] * Z[:, b] * w[a] for a, b in pairs)
    if k >= 3:
        interaction = interaction + 0.7 * Z[:, 0] * Z[:, 1] * Z[:, 2]
    if kind == "linear":
        out = linear
    elif kind == "additive":
        out = additive
    elif kind == "interaction":
        out = interaction
    elif kind == "step":
        cuts = rng.normal(size=k) * 0.7
        out = sum(np.where(Z[:, j] > cuts[j], w[j], -w[j]) for j in range(k))
        out = out + np.where((Z[:, 0] > cuts[0]) & (Z[:, 1] > cuts[1]), 1.5, 0.0)
    elif kind == "mixed":
        out = (
            0.6 * linear / max(np.std(linear), 1e-12)
            + additive / max(np.std(additive), 1e-12)
            + 0.8 * interaction / max(np.std(interaction), 1e-12)
        )
    else:
        raise ValueError(f"unknown signal {kind!r}")
    return np.asarray(out, dtype=float)


def _observe(Z: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "numeric":
        return Z
    if kind == "skewed":
        # Lognormal with a zero-inflated half: monotone in z, so the ideal
        # trees are unchanged and only the binning is being tested.
        X = np.exp(1.5 * Z)
        X[:, ::2] = np.where(Z[:, ::2] < 0.0, 0.0, X[:, ::2])
        return X
    if kind == "discrete":
        return np.clip(np.round(Z * 2.0), -4, 4)
    if kind == "sparse":
        return np.where(np.abs(Z) > 1.65, Z, 0.0)  # ~90% zeros
    if kind == "redundant":
        return Z
    raise ValueError(f"unknown feature_kind {kind!r}")


def make_dataset(spec: DataSpec) -> Dataset:
    rng = np.random.default_rng(spec.data_seed())
    n = spec.rows
    n_numeric = spec.features - spec.cat_low - spec.cat_high
    if n_numeric < 1:
        raise ValueError(f"{spec.name}: needs at least one numeric feature")
    informative = min(spec.informative, n_numeric)
    outputs = spec.n_classes if spec.task == MULTICLASS else 1

    Z = rng.standard_normal((n, n_numeric))
    if spec.feature_kind == "redundant":
        # Half the uninformative columns are noisy copies of informative ones.
        for j in range(informative, n_numeric, 2):
            Z[:, j] = Z[:, j % informative] + 0.3 * rng.standard_normal(n)
    numeric = _observe(Z, spec.feature_kind, rng)

    signals = np.column_stack([_signal(Z[:, :informative], spec.signal, rng) for _ in range(outputs)])
    columns = [numeric]
    categorical: list[int] = []
    for c in range(spec.cat_low + spec.cat_high):
        high = c >= spec.cat_low
        levels = 1_000 if high else 8
        if high:
            weights = 1.0 / np.arange(1, levels + 1) ** 1.1
            codes = rng.choice(levels, size=n, p=weights / weights.sum())
            effect_scale = 0.8
        else:
            codes = rng.integers(0, levels, size=n)
            effect_scale = 1.0
        effects = rng.normal(0.0, effect_scale, size=(levels, outputs))
        signals = signals + effects[codes] * np.std(signals, axis=0)
        if not high and informative:
            # Low-cardinality categories also change a numeric slope: an
            # interaction one-hot or target encoding alone cannot express.
            slopes = rng.normal(0.0, 0.7, size=(levels, outputs))
            signals = signals + slopes[codes] * Z[:, [0]] * np.std(signals, axis=0)
        categorical.append(n_numeric + c)
        columns.append(codes.reshape(-1, 1).astype(float))
    X = np.column_stack(columns).astype(float)

    if spec.missing_rate > 0.0:
        if spec.missing_mechanism == "mcar":
            mask = rng.random(X.shape) < spec.missing_rate
        elif spec.missing_mechanism == "mnar":
            # Informative missingness: high latent values go missing more
            # often, so "is missing" itself predicts the target.
            score = np.column_stack([Z, rng.standard_normal((n, X.shape[1] - n_numeric))])
            prob = 1.0 / (1.0 + np.exp(-(2.0 * score - 1.0)))
            prob *= spec.missing_rate / max(prob.mean(), 1e-12)
            mask = rng.random(X.shape) < np.clip(prob, 0.0, 0.95)
        else:
            raise ValueError(f"unknown missing_mechanism {spec.missing_mechanism!r}")
        X[mask] = np.nan

    scale = np.std(signals, axis=0)
    scale[scale == 0.0] = 1.0
    f = signals / scale
    groups = None
    if spec.task == REGRESSION:
        y = f[:, 0] + rng.normal(0.0, spec.noise, n)
    elif spec.task == BINARY:
        logit = 2.5 * (f[:, 0] + rng.normal(0.0, spec.noise, n))
        if spec.positive_rate is not None:
            logit = logit - np.quantile(logit, 1.0 - spec.positive_rate)
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(float)
    elif spec.task == MULTICLASS:
        logits = 2.0 * (f + rng.normal(0.0, spec.noise, f.shape))
        logits -= logits.max(axis=1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(axis=1, keepdims=True)
        y = (prob.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1).astype(float)
    elif spec.task == RANKING:
        n_groups = n // spec.group_size
        n = n_groups * spec.group_size
        X, f = X[:n], f[:n]
        groups = np.repeat(np.arange(n_groups), spec.group_size)
        # Query-level offsets make raw scores incomparable across queries,
        # as in real search logs; relevance is graded 0-4 within each query.
        score = f[:, 0] + rng.normal(0.0, spec.noise, n) + np.repeat(rng.normal(0, 1, n_groups), spec.group_size)
        y = np.zeros(n)
        for g in range(n_groups):
            s = score[g * spec.group_size:(g + 1) * spec.group_size]
            ranks = s.argsort().argsort() / max(spec.group_size - 1, 1)
            y[g * spec.group_size:(g + 1) * spec.group_size] = np.floor(ranks ** 2 * 4.999)
    else:
        raise ValueError(f"unknown task {spec.task!r}")
    return Dataset(spec, X, y, tuple(categorical), groups)


def _base(task: str, **overrides) -> DataSpec:
    defaults = {"name": f"{task}_base", "task": task, "axis": "base", "level": "base"}
    if task == MULTICLASS:
        defaults["n_classes"] = 5
    return DataSpec(**(defaults | overrides))


def data_scenarios() -> list[DataSpec]:
    """Every data scenario, each tagged with the axis it sweeps and its tier.

    The ``*_base`` datasets (10,000 rows x 20 numeric features, 8
    informative, mixed signal) are the shared reference point of every sweep.
    """
    S = []
    S += [_base(REGRESSION, tier="smoke"), _base(BINARY, tier="smoke"), _base(MULTICLASS)]

    # rows: small data overfits, large data rewards histogram precision.
    for rows, label, tier in ((300, "300", "smoke"), (2_000, "2k", "quick"),
                              (100_000, "100k", "quick"), (1_000_000, "1M", "full")):
        for task in (REGRESSION, BINARY):
            if tier == "smoke" and task == BINARY:
                continue
            S.append(_base(task, name=f"{task}_rows{label}", axis="rows", level=label,
                           tier=tier, rows=rows))
    # columns: few, many, and wide-and-short (more columns than rows).
    for feats, label, tier in ((5, "5", "quick"), (200, "200", "quick"), (1_000, "1000", "full")):
        for task in (REGRESSION, BINARY):
            S.append(_base(task, name=f"{task}_cols{label}", axis="columns", level=label,
                           tier=tier, features=feats, informative=min(8, feats)))
    S.append(_base(REGRESSION, name="regression_wide_short", axis="columns", level="p>n",
                   tier="quick", rows=400, features=800))
    S.append(_base(BINARY, name="binary_wide_short", axis="columns", level="p>n",
                   tier="full", rows=400, features=800))

    # task variants beyond the three bases.
    S.append(_base(BINARY, name="binary_imbalanced", axis="task", level="binary 5% positive",
                   positive_rate=0.05))
    S.append(_base(MULTICLASS, name="multiclass_3", axis="task", level="multiclass 3",
                   n_classes=3, tier="full"))
    S.append(_base(MULTICLASS, name="multiclass_20", axis="task", level="multiclass 20",
                   n_classes=20, rows=20_000))
    S.append(_base(RANKING, name="ranking_base", axis="task", level="ranking", group_size=20))
    S.append(_base(RANKING, name="ranking_long_lists", axis="task", level="ranking 100/query",
                   group_size=100, rows=20_000, tier="full"))

    # feature types.
    for kind, tier in (("skewed", "quick"), ("discrete", "quick"), ("sparse", "quick"),
                       ("redundant", "full")):
        for task in (REGRESSION, BINARY):
            S.append(_base(task, name=f"{task}_{kind}", axis="feature type", level=kind,
                           tier=tier, feature_kind=kind))
    S.append(_base(BINARY, name="binary_irrelevant", axis="feature type",
                   level="95% irrelevant", features=100, informative=5))
    S.append(_base(REGRESSION, name="regression_irrelevant", axis="feature type",
                   level="95% irrelevant", features=100, informative=5, tier="full"))

    # categorical columns.
    for task in (REGRESSION, BINARY):
        S.append(_base(task, name=f"{task}_cat_low", axis="categorical", level="4 x 8 levels",
                       tier="smoke" if task == BINARY else "quick", cat_low=4, features=20))
        S.append(_base(task, name=f"{task}_cat_high", axis="categorical",
                       level="2 x 1000 levels (Zipf)", cat_high=2, features=20))
        S.append(_base(task, name=f"{task}_cat_mixed_missing", axis="categorical",
                       level="mixed + 15% missing", tier="full", cat_low=3, cat_high=1,
                       missing_rate=0.15))

    # missing values.
    for rate, mech, label, tier in ((0.2, "mcar", "20% random", "quick"),
                                    (0.3, "mnar", "30% informative", "quick"),
                                    (0.6, "mcar", "60% random", "full")):
        for task in (REGRESSION, BINARY):
            S.append(_base(task, name=f"{task}_missing_{mech}{int(rate * 100)}", axis="missing",
                           level=label, tier=tier, missing_rate=rate, missing_mechanism=mech))

    # signal shape and noise.
    for kind, tier in (("linear", "quick"), ("interaction", "quick"), ("additive", "full"),
                       ("step", "full")):
        S.append(_base(REGRESSION, name=f"regression_signal_{kind}", axis="signal", level=kind,
                       tier=tier, signal=kind))
    S.append(_base(BINARY, name="binary_signal_interaction", axis="signal", level="interaction",
                   tier="full", signal="interaction"))
    for noise, label, tier in ((0.05, "near noiseless", "full"), (3.0, "low SNR", "quick")):
        for task in (REGRESSION, BINARY):
            S.append(_base(task, name=f"{task}_noise_{label.replace(' ', '_')}", axis="noise",
                           level=label, tier=tier, noise=noise))
    names = [s.name for s in S]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise AssertionError(f"duplicate scenario names: {sorted(duplicates)}")
    return S


# --------------------------------------------------------------------------
# Hyperparameter regimes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Regime:
    """Library-neutral hyperparameters; ``None`` leaves a library's default.

    ``min_child_weight`` is the minimum hessian sum per leaf and
    ``min_data_in_leaf`` the minimum row count; ``subsample``/``colsample``
    are per-tree row and column fractions; ``num_leaves`` switches to
    leaf-wise (best-first) growth; ``early_stopping`` is the patience in
    rounds on a validation split shared by every library.
    """

    name: str
    description: str
    tier: str = "quick"
    n_estimators: int | None = None
    learning_rate: float | None = None
    max_depth: int | None = None
    num_leaves: int | None = None
    l2: float | None = None
    l1: float | None = None
    min_child_weight: float | None = None
    min_data_in_leaf: int | None = None
    subsample: float | None = None
    colsample: float | None = None
    early_stopping: int | None = None
    validation_fraction: float = 0.15
    # Alloy's auto policy adds leaf regularization unless told otherwise;
    # a regime that pins "no regularization" must turn it off to be honest.
    alloy_manual: bool = False

    @property
    def matched(self) -> bool:
        return self.name != "defaults"


REGIMES: tuple[Regime, ...] = (
    Regime("defaults", "every library at its own defaults (Alloy: auto policy, auto rounds)", tier="smoke"),
    Regime("standard", "depth 6, 300 trees, lr 0.05", tier="smoke",
           n_estimators=300, learning_rate=0.05, max_depth=6),
    Regime("stumps", "depth 1, 1000 trees, lr 0.1 (additive model)", tier="full",
           n_estimators=1_000, learning_rate=0.1, max_depth=1),
    Regime("shallow", "depth 2, 500 trees, lr 0.1",
           n_estimators=500, learning_rate=0.1, max_depth=2),
    Regime("deep", "depth 12, 300 trees, lr 0.05",
           n_estimators=300, learning_rate=0.05, max_depth=12),
    Regime("few_trees", "depth 6, 10 trees, lr 0.3",
           n_estimators=10, learning_rate=0.3, max_depth=6),
    Regime("many_trees", "depth 6, 2000 trees, lr 0.02",
           n_estimators=2_000, learning_rate=0.02, max_depth=6),
    Regime("very_many_trees", "depth 6, 5000 trees, lr 0.1 (overfit stress)", tier="full",
           n_estimators=5_000, learning_rate=0.1, max_depth=6),
    Regime("light_reg", "depth 8, 300 trees, lr 0.1, no L1/L2, min hessian 0, 1 row per leaf",
           n_estimators=300, learning_rate=0.1, max_depth=8, l2=0.0, l1=0.0,
           min_child_weight=0.0, min_data_in_leaf=1, alloy_manual=True),
    Regime("heavy_reg", "depth 8, 300 trees, lr 0.1, L2 10, L1 1, min hessian 10, "
           "100 rows per leaf, 70% row and column sampling",
           n_estimators=300, learning_rate=0.1, max_depth=8, l2=10.0, l1=1.0,
           min_child_weight=10.0, min_data_in_leaf=100, subsample=0.7, colsample=0.7),
    Regime("leafwise", "leaf-wise growth, 63 leaves, depth cap 12, 300 trees, lr 0.05",
           n_estimators=300, learning_rate=0.05, max_depth=12, num_leaves=63),
    Regime("early_stopping", "lr 0.05, depth 6, up to 5000 trees, stop after 50 rounds "
           "without improvement on a shared 15% validation split",
           n_estimators=5_000, learning_rate=0.05, max_depth=6, early_stopping=50),
)
REGIMES_BY_NAME = {r.name: r for r in REGIMES}

# Data scenarios run under these; the regime panel runs every regime.
DATA_REGIMES = ("defaults", "standard")


def library_params(regime: Regime, library: str, task: str) -> tuple[dict, list[str]]:
    """Map a regime onto one library's constructor arguments.

    Returns the parameters and the regime knobs the library has no
    equivalent for (recorded per fit so the comparison stays honest).
    """
    p: dict = {}
    unmapped: list[str] = []
    r = regime
    leafwise = r.num_leaves is not None
    if library == "alloygbm":
        if r.n_estimators is not None:
            p["n_estimators"] = r.n_estimators
        if r.learning_rate is not None:
            p["learning_rate"] = r.learning_rate
        if r.max_depth is not None:
            p["max_depth"] = r.max_depth
        if leafwise:
            p.update(tree_growth="leaf", max_leaves=r.num_leaves)
        if r.l2 is not None:
            p["lambda_l2"] = r.l2
        if r.l1 is not None:
            p["lambda_l1"] = r.l1
        if r.min_child_weight is not None:
            p["min_child_hessian"] = r.min_child_weight
        if r.min_data_in_leaf is not None:
            p["min_data_in_leaf"] = r.min_data_in_leaf
        if r.subsample is not None:
            p["row_subsample"] = r.subsample
        if r.colsample is not None:
            p["col_subsample"] = r.colsample
        if r.early_stopping is not None:
            p["early_stopping_rounds"] = r.early_stopping
        if r.alloy_manual:
            p["training_policy"] = "manual"
    elif library == "lightgbm":
        if r.n_estimators is not None:
            p["n_estimators"] = r.n_estimators
        if r.learning_rate is not None:
            p["learning_rate"] = r.learning_rate
        if leafwise:
            p.update(num_leaves=r.num_leaves, max_depth=r.max_depth or -1)
        elif r.max_depth is not None:
            # LightGBM grows leaf-wise capped by num_leaves (31 by default);
            # a matched depth must also lift the leaf cap or trees stay small.
            p.update(max_depth=r.max_depth, num_leaves=min(2 ** r.max_depth, 131_072))
        if r.l2 is not None:
            p["reg_lambda"] = r.l2
        if r.l1 is not None:
            p["reg_alpha"] = r.l1
        if r.min_child_weight is not None:
            p["min_child_weight"] = r.min_child_weight
        if r.min_data_in_leaf is not None:
            p["min_child_samples"] = r.min_data_in_leaf
        if r.subsample is not None:
            p.update(subsample=r.subsample, subsample_freq=1)
        if r.colsample is not None:
            p["colsample_bytree"] = r.colsample
    elif library == "xgboost":
        if r.n_estimators is not None:
            p["n_estimators"] = r.n_estimators
        if r.learning_rate is not None:
            p["learning_rate"] = r.learning_rate
        if leafwise:
            p.update(grow_policy="lossguide", max_leaves=r.num_leaves, max_depth=r.max_depth or 0)
        elif r.max_depth is not None:
            p["max_depth"] = r.max_depth
        if r.l2 is not None:
            p["reg_lambda"] = r.l2
        if r.l1 is not None:
            p["reg_alpha"] = r.l1
        if r.min_child_weight is not None:
            p["min_child_weight"] = r.min_child_weight
        if r.min_data_in_leaf is not None:
            unmapped.append("min_data_in_leaf")
        if r.subsample is not None:
            p["subsample"] = r.subsample
        if r.colsample is not None:
            p["colsample_bytree"] = r.colsample
        if r.early_stopping is not None:
            p["early_stopping_rounds"] = r.early_stopping
    elif library == "catboost":
        if r.n_estimators is not None:
            p["iterations"] = r.n_estimators
        if r.learning_rate is not None:
            p["learning_rate"] = r.learning_rate
        if r.max_depth is not None:
            # CatBoost caps depth at 16; every regime here stays below it.
            p["depth"] = r.max_depth
        if leafwise:
            p.update(grow_policy="Lossguide", max_leaves=r.num_leaves)
        if r.l2 is not None:
            p["l2_leaf_reg"] = r.l2
        if r.l1 is not None and r.l1 > 0.0:
            unmapped.append("l1")
        if r.min_child_weight is not None and r.min_child_weight > 0.0:
            unmapped.append("min_child_weight")
        if r.min_data_in_leaf is not None:
            if leafwise:
                p["min_data_in_leaf"] = r.min_data_in_leaf
            elif r.min_data_in_leaf > 1:
                # Symmetric (oblivious) trees ignore min_data_in_leaf.
                unmapped.append("min_data_in_leaf")
        if r.subsample is not None:
            p.update(bootstrap_type="Bernoulli", subsample=r.subsample)
        if r.colsample is not None:
            p["rsm"] = r.colsample
        if r.early_stopping is not None:
            p["early_stopping_rounds"] = r.early_stopping
    else:
        raise ValueError(f"unknown library {library!r}")
    return p, unmapped


# --------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One (dataset, regime) pair: the unit every arm is scored on."""

    spec: DataSpec
    regime: Regime
    source: str = "synthetic"

    @property
    def id(self) -> str:
        return f"{self.spec.name}|{self.regime.name}"

    @property
    def axis(self) -> str:
        return self.spec.axis

    @property
    def level(self) -> str:
        return self.spec.level


# The regime panel: base datasets that every regime (hyperparameter axis)
# is evaluated on. Quick runs the regression and binary bases; full adds
# multiclass and a categorical-with-missing binary task.
PANEL = (
    ("regression_base", "smoke"),
    ("binary_base", "quick"),
    ("multiclass_base", "full"),
    ("binary_cat_mixed_missing", "full"),
)


def build_cells(tier: str, axes: Sequence[str] | None = None,
                regimes: Sequence[str] | None = None,
                names: Sequence[str] | None = None) -> list[Cell]:
    limit = _tier_rank(tier)
    specs = {s.name: s for s in data_scenarios()}
    cells: dict[str, Cell] = {}

    def add(spec: DataSpec, regime: Regime) -> None:
        cell = Cell(spec, regime)
        cells.setdefault(cell.id, cell)

    data_regimes = [REGIMES_BY_NAME[n] for n in DATA_REGIMES]
    for spec in specs.values():
        if _tier_rank(spec.tier) > limit:
            continue
        for regime in data_regimes:
            if _tier_rank(regime.tier) <= limit:
                add(spec, regime)
    for name, panel_tier in PANEL:
        if _tier_rank(panel_tier) > limit:
            continue
        for regime in REGIMES:
            if _tier_rank(regime.tier) <= limit:
                add(specs[name], regime)

    out = list(cells.values())
    if axes:
        wanted = set(axes)
        out = [c for c in out if c.axis in wanted or ("hyperparameters" in wanted and c.regime.name not in DATA_REGIMES)]
    if regimes:
        out = [c for c in out if c.regime.name in set(regimes)]
    if names:
        out = [c for c in out if any(n in c.id for n in names)]
    return out


# --------------------------------------------------------------------------
# Splits, inputs, fitting
# --------------------------------------------------------------------------


@dataclass
class Split:
    train: np.ndarray
    test: np.ndarray
    valid: np.ndarray | None  # carved from train for early-stopping regimes


def _official_split(ds: Dataset, fold: int, validation_fraction: float | None) -> Split:
    from sklearn.model_selection import train_test_split

    if fold >= len(ds.folds):
        raise IndexError(f"{ds.spec.name} has {len(ds.folds)} folds, asked for fold {fold}")
    train, test = (np.asarray(a, dtype=np.int64) for a in ds.folds[fold])
    classify = ds.task in (BINARY, MULTICLASS)
    if ds.train_cap is not None and len(train) > ds.train_cap:
        # Deterministic per (dataset, fold); every arm sees the same rows.
        train, _ = train_test_split(train, train_size=ds.train_cap, random_state=fold,
                                    stratify=ds.y[train] if classify else None)
    valid = None
    if validation_fraction:
        train, valid = train_test_split(train, test_size=validation_fraction, random_state=fold,
                                        stratify=ds.y[train] if classify else None)
    return Split(np.sort(train), np.sort(test), None if valid is None else np.sort(valid))


def make_split(ds: Dataset, seed: int, validation_fraction: float | None) -> Split:
    if ds.folds is not None:
        return _official_split(ds, seed, validation_fraction)
    rng = np.random.default_rng(seed + 1_000_003 * (ds.spec.data_seed() % 997))
    n = len(ds.y)
    if ds.groups is not None:
        # Whole queries go to one side; rows stay contiguous by query.
        gids = np.unique(ds.groups)
        rng.shuffle(gids)
        n_test = max(1, len(gids) // 4)
        test_g = set(gids[:n_test].tolist())
        rest = gids[n_test:]
        valid_g: set = set()
        if validation_fraction:
            valid_g = set(rest[: max(1, int(len(rest) * validation_fraction))].tolist())
        test = np.flatnonzero(np.isin(ds.groups, list(test_g)))
        valid = np.flatnonzero(np.isin(ds.groups, list(valid_g))) if valid_g else None
        train = np.flatnonzero(~np.isin(ds.groups, list(test_g | valid_g)))
        return Split(train, test, valid)

    from sklearn.model_selection import train_test_split

    idx = np.arange(n)
    classify = ds.task in (BINARY, MULTICLASS)
    train, test = train_test_split(idx, test_size=0.25, random_state=seed,
                                   stratify=ds.y if classify else None)
    valid = None
    if validation_fraction:
        train, valid = train_test_split(train, test_size=validation_fraction, random_state=seed,
                                        stratify=ds.y[train] if classify else None)
    return Split(np.sort(train), np.sort(test), None if valid is None else np.sort(valid))


def _frame(X: np.ndarray, categorical: Sequence[int], library: str):
    """Hand categorical columns to a library in its native categorical form."""
    if not categorical:
        return X
    import pandas as pd

    frame = pd.DataFrame(X, columns=[f"f{j}" for j in range(X.shape[1])])
    for j in categorical:
        col = frame.iloc[:, j]
        if library == "catboost":
            frame[frame.columns[j]] = [
                "__missing__" if math.isnan(v) else str(int(v)) for v in col.to_numpy()
            ]
        else:
            # Integer category codes with a fixed category list: identical
            # codes in every split, NaN stays missing.
            levels = np.arange(int(np.nanmax(X[:, j])) + 1 if np.isfinite(X[:, j]).any() else 1)
            frame[frame.columns[j]] = pd.Categorical(col.to_numpy(), categories=levels)
    return frame


def _take(frame, rows: np.ndarray):
    return frame.iloc[rows] if hasattr(frame, "iloc") else frame[rows]


def _estimator(arm: Arm, task: str, params: dict, seed: int, threads: int):
    params = dict(params)
    if arm.library == "alloygbm":
        from alloygbm import GBMClassifier, GBMRanker, GBMRegressor

        cls = {REGRESSION: GBMRegressor, BINARY: GBMClassifier, MULTICLASS: GBMClassifier,
               RANKING: GBMRanker}[task]
        return cls(seed=seed, n_jobs=threads, **params)
    if arm.library == "lightgbm":
        import lightgbm as lgb

        params.pop("early_stopping_rounds", None)  # passed as a callback at fit time
        cls = {REGRESSION: lgb.LGBMRegressor, BINARY: lgb.LGBMClassifier,
               MULTICLASS: lgb.LGBMClassifier, RANKING: lgb.LGBMRanker}[task]
        return cls(random_state=seed, n_jobs=threads, verbose=-1, **params)
    if arm.library == "xgboost":
        import xgboost as xgb

        cls = {REGRESSION: xgb.XGBRegressor, BINARY: xgb.XGBClassifier,
               MULTICLASS: xgb.XGBClassifier, RANKING: xgb.XGBRanker}[task]
        # enable_categorical defaults to False before XGBoost 3.4.
        return cls(random_state=seed, n_jobs=threads, tree_method="hist",
                   enable_categorical=True, **params)
    import catboost as cb

    cls = {REGRESSION: cb.CatBoostRegressor, BINARY: cb.CatBoostClassifier,
           MULTICLASS: cb.CatBoostClassifier, RANKING: cb.CatBoostRanker}[task]
    return cls(random_seed=seed, thread_count=threads, verbose=0, allow_writing_files=False, **params)


def _fit(model, arm: Arm, ds: Dataset, X_all, split: Split, regime: Regime) -> None:
    X_tr, y_tr = _take(X_all, split.train), ds.y[split.train]
    kwargs: dict = {}
    eval_set = None
    if split.valid is not None:
        eval_set = (_take(X_all, split.valid), ds.y[split.valid])
    lib = arm.library
    if ds.task == RANKING:
        g_tr = ds.groups[split.train]
        if lib == "alloygbm":
            kwargs["group"] = g_tr
        elif lib == "lightgbm":
            kwargs["group"] = np.unique(g_tr, return_counts=True)[1]
        elif lib == "xgboost":
            kwargs["qid"] = g_tr
        else:
            kwargs["group_id"] = g_tr
    if lib == "catboost" and ds.categorical:
        kwargs["cat_features"] = list(ds.categorical)
    if eval_set is not None:
        if lib == "alloygbm":
            kwargs["eval_set"] = eval_set
            if ds.task == RANKING:
                kwargs["eval_group"] = ds.groups[split.valid]
        elif lib == "lightgbm":
            import lightgbm as lgb

            kwargs["eval_set"] = [eval_set]
            if ds.task == RANKING:
                kwargs["eval_group"] = [np.unique(ds.groups[split.valid], return_counts=True)[1]]
                kwargs["eval_at"] = [10]
            kwargs["callbacks"] = [lgb.early_stopping(regime.early_stopping, verbose=False)]
        elif lib == "xgboost":
            kwargs["eval_set"] = [eval_set]
            kwargs["verbose"] = False
            if ds.task == RANKING:
                kwargs["eval_qid"] = [ds.groups[split.valid]]
        else:
            from catboost import Pool

            kwargs["eval_set"] = Pool(
                eval_set[0], eval_set[1],
                cat_features=list(ds.categorical) or None,
                group_id=ds.groups[split.valid] if ds.task == RANKING else None,
            )
    model.fit(X_tr, y_tr, **kwargs)


def _trees_used(model, library: str) -> int | None:
    try:
        if library == "alloygbm":
            best = getattr(model, "best_iteration_", None)
            done = getattr(model, "rounds_completed_", None)
            return int(best) + 1 if best is not None else (int(done) if done is not None else None)
        if library == "lightgbm":
            best = getattr(model, "best_iteration_", None)
            return int(best) if best else int(model.booster_.current_iteration())
        if library == "xgboost":
            best = getattr(model, "best_iteration", None)
            return int(best) + 1 if best is not None else int(model.get_booster().num_boosted_rounds())
        return int(model.tree_count_)
    except Exception:  # diagnostics only
        return None


def ndcg_at_k(y: np.ndarray, score: np.ndarray, groups: np.ndarray, k: int = 10) -> float:
    values = []
    for g in np.unique(groups):
        m = groups == g
        rel, s = y[m], score[m]
        ideal = np.sort(rel)[::-1][:k]
        discounts = 1.0 / np.log2(np.arange(2, len(ideal) + 2))
        idcg = float(np.sum((2.0 ** ideal - 1.0) * discounts))
        if idcg <= 0.0:
            continue
        # Ties broken pessimistically so a constant scorer earns no credit.
        order = np.lexsort((rel, -s))[:k]
        dcg = float(np.sum((2.0 ** rel[order] - 1.0) * discounts[: len(order)]))
        values.append(dcg / idcg)
    return float(np.mean(values)) if values else float("nan")


def score_predictions(ds: Dataset, rows: np.ndarray, prediction: np.ndarray) -> dict:
    y = ds.y[rows]
    if ds.task == REGRESSION:
        rmse = float(np.sqrt(np.mean((np.asarray(prediction, dtype=float) - y) ** 2)))
        return {"loss": rmse, "metric": "rmse", "secondary": None, "secondary_metric": None}
    if ds.task == RANKING:
        ndcg = ndcg_at_k(y, np.asarray(prediction, dtype=float), ds.groups[rows])
        return {"loss": 1.0 - ndcg, "metric": "1-ndcg@10", "secondary": ndcg, "secondary_metric": "ndcg@10"}
    from sklearn.metrics import log_loss, roc_auc_score

    proba = np.clip(np.asarray(prediction, dtype=float), 1e-15, 1.0)
    labels = np.arange(ds.spec.n_classes if ds.task == MULTICLASS else 2)
    loss = float(log_loss(y, proba, labels=labels))
    if ds.task == BINARY:
        return {"loss": loss, "metric": "log_loss",
                "secondary": float(roc_auc_score(y, proba[:, 1])), "secondary_metric": "auc"}
    accuracy = float(np.mean(proba.argmax(axis=1) == y))
    return {"loss": loss, "metric": "log_loss", "secondary": accuracy, "secondary_metric": "accuracy"}


def _predict(model, task: str, X):
    if task in (BINARY, MULTICLASS):
        proba = np.asarray(model.predict_proba(X), dtype=float)
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        return proba
    return np.asarray(model.predict(X), dtype=float)


def fit_and_score(arm: Arm, cell: Cell, ds: Dataset, seed: int, threads: int) -> dict:
    regime = cell.regime
    split = make_split(ds, seed, regime.validation_fraction if regime.early_stopping else None)
    params, unmapped = library_params(regime, arm.library, ds.task)
    params = params | dict(arm.params)  # arm overrides win
    X_all = _frame(ds.X, ds.categorical, arm.library)
    model = _estimator(arm, ds.task, params, seed, threads)
    started = time.perf_counter()
    _fit(model, arm, ds, X_all, split, regime)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prediction = _predict(model, ds.task, _take(X_all, split.test))
    predict_seconds = time.perf_counter() - started
    if ds.task in (BINARY, MULTICLASS):
        classes = getattr(model, "classes_", None)
        expected = np.arange(ds.spec.n_classes if ds.task == MULTICLASS else 2)
        if classes is not None and not np.array_equal(np.asarray(classes, dtype=float), expected):
            raise RuntimeError(f"unexpected class order {classes!r}")
    out = score_predictions(ds, split.test, prediction)
    out.update(fit_seconds=fit_seconds, predict_seconds=predict_seconds,
               trees=_trees_used(model, arm.library), unmapped=unmapped,
               train_rows=int(len(split.train)), test_rows=int(len(split.test)))
    if arm.library == "alloygbm":
        out["alloy_auto_rounds_mode"] = (getattr(model, "auto_rounds_", None) or {}).get("mode")
    return out


# --------------------------------------------------------------------------
# Real datasets (optional)
# --------------------------------------------------------------------------

# Curated OpenML suites. CC18 and CTR23 are the standard classification and
# regression suites (Bischl et al. 2021; Fischer et al. 2023). 334-337 are
# the Grinsztajn et al. (NeurIPS 2022) "tabular-benchmark" suites: numerical
# and numerical+categorical features, for classification and regression.
OPENML_SUITES: dict[str, tuple[int, ...]] = {
    "cc18": (99,),
    "ctr23": (353,),
    "grinsztajn": (334, 335, 336, 337),
}


def _suite_ids(suite: str) -> tuple[int, ...]:
    key = suite.lower()
    if key in OPENML_SUITES:
        return OPENML_SUITES[key]
    return (int(key),)


def openml_cells(suites: Sequence[str], max_rows: int, regimes: Sequence[str],
                 max_folds: int | None = None, train_cap: int | None = None,
                 tasks: Sequence[int] | None = None,
                 log: Callable[[str], None] = print) -> list[tuple[Cell, Dataset]]:
    """Cells from OpenML benchmark suites, on each task's official splits.

    Every repeat/fold pair the task defines becomes one entry of the seed
    axis (``fold = repeat * n_folds + fold_index``), so per-fold scores line
    up with results other people publish on the same task. Tasks with more
    than ``max_rows`` rows are skipped rather than subsampled, which would
    break the official folds; ``train_cap`` instead caps training rows
    inside each fold and leaves the test folds whole. Categorical columns
    come from the dataset's ``categorical_indicator``.
    """
    try:
        import openml
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("--openml-suite needs the 'openml' package (pip install openml)") from exc
    out = []
    seen: set[int] = set()
    for suite in suites:
        for suite_id in _suite_ids(suite):
            benchmark = openml.study.get_suite(suite_id)
            for task_id in benchmark.tasks:
                if task_id in seen or (tasks and task_id not in set(tasks)):
                    continue
                seen.add(task_id)
                task = openml.tasks.get_task(task_id, download_splits=True)
                if task.task_type_id == openml.tasks.TaskType.SUPERVISED_REGRESSION:
                    kind = REGRESSION
                elif task.task_type_id == openml.tasks.TaskType.SUPERVISED_CLASSIFICATION:
                    kind = None  # binary or multiclass, decided after cleaning
                else:
                    continue
                dataset = task.get_dataset()
                X, y, is_cat, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
                if len(X) > max_rows:
                    log(f"  skip openml task {task_id} ({dataset.name}): {len(X):,} rows > {max_rows:,}")
                    continue
                repeats, n_folds, _samples = task.get_split_dimensions()
                folds = []
                for r in range(repeats):
                    for f in range(n_folds):
                        tr, te = task.get_train_test_split_indices(repeat=r, fold=f, sample=0)
                        folds.append((np.asarray(tr), np.asarray(te)))
                if max_folds is not None:
                    folds = folds[:max_folds]
                if kind is None:
                    kind = BINARY if y.nunique(dropna=True) == 2 else MULTICLASS
                ds = dataset_from_frame(f"openml_{task_id}_{dataset.name}", kind, X, y, is_cat,
                                        folds=folds, train_cap=train_cap)
                if kind != REGRESSION and ds.spec.n_classes == 2:
                    ds.spec = replace(ds.spec, task=BINARY)
                for name in regimes:
                    out.append((Cell(ds.spec, REGIMES_BY_NAME[name], source=f"openml:{suite}:{task_id}"), ds))
    return out


def dataset_from_frame(name: str, task: str, X, y, is_categorical: Sequence[bool],
                       max_rows: int | None = None,
                       folds: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
                       train_cap: int | None = None) -> Dataset:
    """Turn a pandas frame into a Dataset.

    With ``folds`` (row positions into ``X``), rows dropped while cleaning
    labels are removed from every fold and the rest re-indexed. Without
    folds, ``max_rows`` subsamples once, deterministically.
    """
    import pandas as pd

    if folds is None and max_rows is not None and len(X) > max_rows:
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little"))
        keep = np.sort(rng.choice(len(X), size=max_rows, replace=False))
        X, y = X.iloc[keep], y.iloc[keep]
    columns, categorical = [], []
    for j, (col, cat) in enumerate(zip(X.columns, is_categorical)):
        values = X[col]
        if cat or values.dtype == object or str(values.dtype) == "category":
            codes = pd.Categorical(values).codes.astype(float)
            codes[codes < 0] = np.nan
            columns.append(codes)
            categorical.append(j)
        else:
            columns.append(pd.to_numeric(values, errors="coerce").to_numpy(dtype=float))
    matrix = np.column_stack(columns)
    if task == REGRESSION:
        target = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
        n_classes = 2
    else:
        target = pd.Categorical(y).codes.astype(float)
        target[target < 0] = np.nan  # missing labels
        # Classes too rare to appear on both sides of a stratified split
        # (often left behind by row subsampling) are dropped, and the rest
        # re-coded to contiguous 0..K-1.
        labels, counts = np.unique(target[np.isfinite(target)], return_counts=True)
        target[~np.isin(target, labels[counts >= 4])] = np.nan
        kept = np.isfinite(target)
        target[kept] = np.unique(target[kept], return_inverse=True)[1]
        n_classes = int(target[kept].max()) + 1 if kept.any() else 2
    finite = np.isfinite(target)
    official = None
    if folds is not None:
        new_index = np.cumsum(finite) - 1
        official = tuple(
            (new_index[tr[finite[tr]]], new_index[te[finite[te]]])
            for tr, te in ((np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)) for a, b in folds)
        )
    spec = DataSpec(name=name, task=task, axis="real data", level="openml", tier="full",
                    rows=int(finite.sum()), features=matrix.shape[1],
                    n_classes=max(n_classes, 2), cat_low=0, cat_high=0)
    return Dataset(spec, matrix[finite], target[finite], tuple(categorical), None,
                   folds=official, train_cap=train_cap)


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent, check=True).stdout.strip()
    except Exception:
        return None


def _library_version(library: str) -> str:
    try:
        module = __import__(library)
    except ImportError:
        return "missing"
    return str(getattr(module, "__version__", "unknown"))


def run(arms: list[Arm], cells: list[Cell], seeds: list[int], threads: int, tier: str,
        datasets: dict[str, Dataset] | None = None,
        log: Callable[[str], None] = print,
        previous: Sequence[dict] | None = None,
        checkpoint: Callable[[dict], None] | None = None) -> dict:
    """Fit every arm on every cell and seed.

    ``previous`` holds records from an interrupted run of the same selection:
    any (cell, seed, arm) already recorded there is kept and not refit.
    ``checkpoint`` receives the partial result (``"complete": False``) after
    every cell, so a crash loses at most the cell in progress.
    """
    warnings.filterwarnings("ignore")
    cache: dict[str, Dataset] = dict(datasets or {})
    records = list(previous or [])
    done = {(r["dataset"], r["seed"], r["arm"]) for r in records}
    if records:
        log(f"  resuming: {len(records)} records already done")
    started = time.perf_counter()
    for index, cell in enumerate(cells, 1):
        ds = cache.get(cell.spec.name)
        if ds is None:
            ds = cache[cell.spec.name] = make_dataset(cell.spec)
        # Real-data cells run on their official folds: the seed is the fold.
        cell_seeds = seeds if ds.folds is None else list(range(len(ds.folds)))
        fitted = False
        for seed in cell_seeds:
            for arm in arms:
                if (cell.id, seed, arm.name) in done:
                    continue
                fitted = True
                try:
                    result = fit_and_score(arm, cell, ds, seed, threads)
                    error = None
                except Exception as exc:  # recorded; the report and gate surface it
                    result = {"loss": None, "fit_seconds": None}
                    error = f"{type(exc).__name__}: {exc}"
                records.append({
                    "dataset": cell.id, "scenario": cell.spec.name, "regime": cell.regime.name,
                    "axis": cell.axis, "level": cell.level, "task": ds.task, "source": cell.source,
                    "seed": seed, "arm": arm.name, "library": arm.library,
                    **result, "error": error,
                })
        log(f"  [{index}/{len(cells)}] {cell.id} ({time.perf_counter() - started:.0f}s)"
            + ("" if fitted else " (already done)"))
        if checkpoint is not None and fitted:
            checkpoint(_result_envelope(arms, cells[:index], seeds, threads, tier, cache,
                                        records, complete=False))
    return _result_envelope(arms, cells, seeds, threads, tier, cache, records, complete=True)


def _result_envelope(arms: list[Arm], cells: Sequence[Cell], seeds: list[int], threads: int,
                     tier: str, cache: dict[str, Dataset], records: list[dict],
                     complete: bool) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "suite": "frontier_matrix",
        "tier": tier,
        "complete": complete,
        "arms": [asdict(arm) | {"version": _library_version(arm.library)} for arm in arms],
        "seeds": seeds,
        "threads": threads,
        "git_commit": _git_commit(),
        "regimes": [asdict(r) for r in REGIMES],
        "datasets": [
            {"name": c.id, "scenario": c.spec.name, "regime": c.regime.name, "axis": c.axis,
             "level": c.level, "task": c.spec.task, "source": c.source,
             "rows": int(cache[c.spec.name].X.shape[0]), "features": int(cache[c.spec.name].X.shape[1]),
             "categorical": len(cache[c.spec.name].categorical),
             "folds": None if cache[c.spec.name].folds is None else len(cache[c.spec.name].folds)}
            for c in cells
        ],
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "cpu_count": os.cpu_count()},
        "records": [r for r in records if r["dataset"] in {c.id for c in cells}],
    }


def _write_json(path: Path, payload: dict) -> None:
    """Write via a temporary file so a crash never leaves half a file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def _valid(loss) -> bool:
    return loss is not None and math.isfinite(loss) and loss > 0.0


def _cell_table(result: dict) -> dict[str, dict]:
    """Per cell: mean loss and fit time per arm, and per-seed losses."""
    cells: dict[str, dict] = {}
    for meta in result["datasets"]:
        cells[meta["name"]] = {"meta": meta, "losses": {}, "fit": {}, "errors": {}}
    for r in result["records"]:
        cell = cells.setdefault(r["dataset"], {"meta": {"name": r["dataset"]}, "losses": {}, "fit": {}, "errors": {}})
        if r.get("error"):
            cell["errors"].setdefault(r["arm"], r["error"])
        elif _valid(r.get("loss")):
            cell["losses"].setdefault(r["arm"], {})[r["seed"]] = r["loss"]
            if r.get("fit_seconds") is not None:
                cell["fit"].setdefault(r["arm"], []).append(r["fit_seconds"])
        else:
            cell["errors"].setdefault(r["arm"], f"invalid loss {r.get('loss')!r}")
    return cells


def _paired(a: dict[int, float], b: dict[int, float]) -> tuple[float, float, int] | None:
    """Mean and standard error of log(a/b) over shared seeds."""
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    logs = [math.log(a[s] / b[s]) for s in seeds]
    mean = statistics.fmean(logs)
    se = statistics.stdev(logs) / math.sqrt(len(logs)) if len(logs) > 1 else 0.0
    return mean, se, len(logs)


def outcome(mean_log: float, se: float) -> str:
    """'win' / 'loss' for the first arm, or 'tie' inside seed noise."""
    margin = max(2.0 * se, math.log1p(MIN_MEANINGFUL_GAP))
    if mean_log < -margin:
        return "win"
    if mean_log > margin:
        return "loss"
    return "tie"


def summarize(result: dict, primary: str = "alloy") -> dict:
    arms = [a["name"] for a in result["arms"]]
    peers = [a["name"] for a in result["arms"] if a["library"] in PEER_LIBRARIES]
    cells = _cell_table(result)
    rows = []
    for cell_id, cell in cells.items():
        meta = cell["meta"]
        means = {arm: statistics.fmean(v.values()) for arm, v in cell["losses"].items() if v}
        row = {
            "cell": cell_id, "scenario": meta.get("scenario"), "regime": meta.get("regime"),
            "axis": meta.get("axis"), "level": meta.get("level"), "task": meta.get("task"),
            "mean_loss": means, "errors": cell["errors"],
            "fit_seconds": {arm: statistics.median(v) for arm, v in cell["fit"].items()},
        }
        if means:
            best = min(means.values())
            row["normalized"] = {arm: value / best for arm, value in means.items()}
        best_peer = None
        if primary in means and any(p in means for p in peers):
            best_peer = min((p for p in peers if p in means), key=lambda p: means[p])
        best_stats = _paired(cell["losses"][primary], cell["losses"][best_peer]) if best_peer else None
        if best_stats:
            # Paired over shared seeds, like every per-peer ratio and outcome.
            row["best_peer"] = best_peer
            row["ratio_vs_best_peer"] = math.exp(best_stats[0])
            row["vs_best_peer"] = outcome(best_stats[0], best_stats[1])
            row["vs"] = {}
            for peer in peers:
                if peer in means:
                    stats = _paired(cell["losses"][primary], cell["losses"][peer])
                    if stats:
                        row["vs"][peer] = {"ratio": math.exp(stats[0]), "outcome": outcome(stats[0], stats[1])}
            fits = row["fit_seconds"]
            peer_fits = [fits[p] for p in peers if p in fits]
            if primary in fits and peer_fits and min(peer_fits) > 0:
                row["fit_ratio_vs_fastest_peer"] = fits[primary] / min(peer_fits)
        rows.append(row)

    def aggregate(selected: list[dict]) -> dict:
        scored = [r for r in selected if "ratio_vs_best_peer" in r]
        agg = {"cells": len(selected), "scored": len(scored),
               "errors": sum(1 for r in selected if primary in r["errors"])}
        if scored:
            agg["geomean_ratio_vs_best_peer"] = math.exp(statistics.fmean(math.log(r["ratio_vs_best_peer"]) for r in scored))
            agg["worst_ratio_vs_best_peer"] = max(r["ratio_vs_best_peer"] for r in scored)
            agg["best_or_tied"] = sum(1 for r in scored if r["vs_best_peer"] in ("win", "tie"))
            for peer in peers:
                outcomes = [r["vs"][peer]["outcome"] for r in scored if peer in r.get("vs", {})]
                agg[f"vs_{peer}"] = {k: outcomes.count(k) for k in ("win", "tie", "loss")}
            fit = [r["fit_ratio_vs_fastest_peer"] for r in scored if "fit_ratio_vs_fastest_peer" in r]
            if fit:
                agg["median_fit_ratio_vs_fastest_peer"] = statistics.median(fit)
        return agg

    by_axis: dict[str, dict] = {}
    for regime in sorted({r["regime"] for r in rows if r["regime"] in DATA_REGIMES}, key=DATA_REGIMES.index):
        groups: dict[tuple[str, str], list[dict]] = {}
        for r in rows:
            if r["regime"] != regime:
                continue
            if r["axis"] == "base":
                # The base datasets are the reference level of every sweep.
                groups.setdefault(("base", "base"), []).append(r)
            else:
                groups.setdefault((r["axis"], r["level"]), []).append(r)
        by_axis[regime] = {f"{axis} = {level}": aggregate(sel) for (axis, level), sel in groups.items()}
    by_regime = {}
    panel = {name for name, _ in PANEL}
    for regime in REGIMES:
        sel = [r for r in rows if r["regime"] == regime.name and r["scenario"] in panel]
        if sel:
            by_regime[regime.name] = aggregate(sel)
    by_task = {}
    for task in TASKS:
        sel = [r for r in rows if r["task"] == task]
        if sel:
            by_task[task] = aggregate(sel)
    return {
        "primary": primary, "arms": arms, "peers": peers,
        "overall": aggregate(rows),
        "by_task": by_task,
        "by_axis": by_axis,
        "by_regime": by_regime,
        "cells": rows,
    }


def _fmt_wtl(counts: dict | None) -> str:
    if not counts:
        return "n/a"
    return f"{counts['win']}/{counts['tie']}/{counts['loss']}"


def _agg_row(label: str, agg: dict, peers: list[str]) -> str:
    if not agg.get("scored"):
        return f"| {label} | {agg['cells']} | n/a | n/a | " + " | ".join("n/a" for _ in peers) + f" | n/a | {agg['errors']} |"
    fit = agg.get("median_fit_ratio_vs_fastest_peer")
    return (
        f"| {label} | {agg['scored']} | {agg['geomean_ratio_vs_best_peer']:.3f} | "
        f"{agg['best_or_tied']}/{agg['scored']} | "
        + " | ".join(_fmt_wtl(agg.get(f"vs_{p}")) for p in peers)
        + f" | {'n/a' if fit is None else f'{fit:.2f}x'} | {agg['errors']} |"
    )


def render_report(result: dict, primary: str = "alloy", worst: int = 15) -> str:
    s = summarize(result, primary)
    peers = s["peers"]
    header = ("| Slice | Cells | Loss vs best peer (geomean) | Best or tied | "
              + " | ".join(f"W/T/L vs {p}" for p in peers)
              + " | Fit time vs fastest peer | Errors |")
    sep = "|---|---:|---:|---:|" + "---:|" * len(peers) + "---:|---:|"
    versions = ", ".join(f"{a['name']} {a.get('version', '?')}" for a in result["arms"])
    official = [d for d in result["datasets"] if d.get("folds")]
    fold_note = (f" (OpenML cells: official folds, {len({d['scenario'] for d in official})} tasks)"
                 if official else "")
    lines = [
        f"# Frontier matrix ({result.get('tier')} tier)",
        "",
        f"Arms: {versions}. Seeds: {len(result['seeds'])}{fold_note}. Threads: {result['threads']}. "
        f"Commit: `{(result.get('git_commit') or 'unknown')[:12]}`.",
        "",
        *(["**Partial result:** the run stopped before every cell finished "
           "(rerun with `--resume` to complete it).", ""]
          if result.get("complete") is False else []),
        f"Loss ratios compare `{primary}` with the best peer in each cell (below 1.0 means "
        f"{primary} is better). Win/tie/loss is per cell, paired by seed; a difference inside "
        f"two seed standard errors or under {MIN_MEANINGFUL_GAP:.1%} is a tie.",
        "",
        "## Overall",
        "", header, sep, _agg_row("all cells", s["overall"], peers),
    ]
    for task, agg in s["by_task"].items():
        lines.append(_agg_row(f"task: {task}", agg, peers))
    for regime, groups in s["by_axis"].items():
        lines += ["", f"## Data axes, `{regime}` regime ({REGIMES_BY_NAME[regime].description})", "",
                  header, sep]
        for label in sorted(groups, key=lambda k: (k != "base = base", k)):
            lines.append(_agg_row(label, groups[label], peers))
    if s["by_regime"]:
        lines += ["", "## Hyperparameter regimes (regime panel datasets)", "", header, sep]
        for name, agg in s["by_regime"].items():
            lines.append(_agg_row(f"{name}: {REGIMES_BY_NAME[name].description}", agg, peers))
    scored = [r for r in s["cells"] if "ratio_vs_best_peer" in r]
    if scored:
        lines += ["", f"## Largest gaps ({primary} vs best peer)", "",
                  "| Cell | Task | Loss ratio | Best peer | Outcome |", "|---|---|---:|---|---|"]
        for r in sorted(scored, key=lambda r: -r["ratio_vs_best_peer"])[:worst]:
            lines.append(f"| `{r['cell']}` | {r['task']} | {r['ratio_vs_best_peer']:.3f} | "
                         f"{r['best_peer']} | {r['vs_best_peer']} |")
        lines += ["", f"## Largest leads ({primary} vs best peer)", "",
                  "| Cell | Task | Loss ratio | Best peer | Outcome |", "|---|---|---:|---|---|"]
        for r in sorted((r for r in scored if r["ratio_vs_best_peer"] < 1.0),
                        key=lambda r: r["ratio_vs_best_peer"])[:worst]:
            lines.append(f"| `{r['cell']}` | {r['task']} | {r['ratio_vs_best_peer']:.3f} | "
                         f"{r['best_peer']} | {r['vs_best_peer']} |")
    errors = [(r["cell"], arm, msg) for r in s["cells"] for arm, msg in r["errors"].items()]
    if errors:
        lines += ["", "## Failed fits", "", "| Cell | Arm | Error |", "|---|---|---|"]
        for cell, arm, msg in errors:
            lines.append(f"| `{cell}` | {arm} | {msg[:160].replace('|', '/')} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _pin_threads(threads: int) -> None:
    for name in THREAD_ENVIRONMENT:
        os.environ[name] = str(threads)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_selection(p: argparse.ArgumentParser) -> None:
        p.add_argument("--tier", choices=TIERS, default="quick")
        p.add_argument("--axes", nargs="+", help="only these data axes ('hyperparameters' = regime panel)")
        p.add_argument("--regimes", nargs="+", choices=sorted(REGIMES_BY_NAME))
        p.add_argument("--cells", nargs="+", help="only cells whose id contains one of these strings")

    list_p = sub.add_parser("list", help="print the cells a selection would run")
    add_selection(list_p)

    run_p = sub.add_parser("run", help="fit every arm on every selected cell and seed")
    add_selection(run_p)
    run_p.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS),
                       help="builtin arm names or name=library[:key=value,...]")
    run_p.add_argument("--seeds", type=int, help="default: 1 smoke, 3 quick, 5 full")
    run_p.add_argument("--threads", type=int, default=1)
    run_p.add_argument("--openml-suite", nargs="+",
                       help="add tasks from OpenML suites (cc18, ctr23, grinsztajn or suite ids), "
                            "scored on each task's official folds")
    run_p.add_argument("--openml-max-rows", type=int, default=100_000,
                       help="skip OpenML tasks with more rows than this")
    run_p.add_argument("--openml-folds", type=int,
                       help="use only the first N official folds of each task (default: all)")
    run_p.add_argument("--openml-train-cap", type=int,
                       help="cap training rows per fold (test folds stay whole)")
    run_p.add_argument("--openml-tasks", type=int, nargs="+", help="only these OpenML task ids")
    run_p.add_argument("--openml-only", action="store_true", help="skip the synthetic cells")
    run_p.add_argument("--dev-suite", type=Path, metavar="CACHE",
                       help="add the dev/calibration suite from a dev_suite.py export cache "
                            "(disjoint from CC18/CTR23); implies skipping the synthetic cells")
    run_p.add_argument("--dev-tasks", type=int, nargs="+", help="only these dev-suite task ids")
    run_p.add_argument("--output", type=Path, required=True,
                       help="result JSON; rewritten after every cell, so a crash keeps what finished")
    run_p.add_argument("--resume", action="store_true",
                       help="keep the error-free records already in --output and fit only the rest")
    run_p.add_argument("--report", type=Path, help="also write the Markdown report here")
    run_p.add_argument("--primary", default="alloy")

    rep_p = sub.add_parser("report", help="render the Markdown report of a result file")
    rep_p.add_argument("file", type=Path)
    rep_p.add_argument("--primary", default="alloy")
    rep_p.add_argument("--output", type=Path)

    cmp_p = sub.add_parser("compare", help="paired, gated comparison of two arms or builds")
    cmp_p.add_argument("files", type=Path, nargs="+")
    cmp_p.add_argument("--baseline", required=True)
    cmp_p.add_argument("--candidate", required=True)
    cmp_p.add_argument("--min-improvement", type=float, default=0.0)
    cmp_p.add_argument("--max-dataset-regression", type=float, default=0.03)
    cmp_p.add_argument("--gate", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "list":
        cells = build_cells(args.tier, args.axes, args.regimes, args.cells)
        for c in cells:
            s = c.spec
            print(f"{c.id:55s} {s.task:10s} {s.rows:>9,} x {s.features:<5} axis={c.axis!r} level={c.level!r}")
        print(f"{len(cells)} cells")
        return 0

    if args.command == "run":
        if args.threads < 1:
            parser.error("--threads must be positive")
        _pin_threads(args.threads)
        arms = [parse_arm(spec) for spec in args.arms]
        seeds = list(range(args.seeds if args.seeds is not None else DEFAULT_SEEDS[args.tier]))
        skip_synthetic = args.openml_only or args.dev_suite is not None
        cells = [] if skip_synthetic else build_cells(args.tier, args.axes, args.regimes, args.cells)
        extra: dict[str, Dataset] = {}
        if args.dev_suite is not None:
            import dev_suite

            for ds in dev_suite.load(args.dev_suite, tasks=args.dev_tasks):
                if args.cells and not any(c in ds.spec.name for c in args.cells):
                    continue
                extra[ds.spec.name] = ds
                for name in args.regimes or DATA_REGIMES:
                    cells.append(Cell(ds.spec, REGIMES_BY_NAME[name], source="dev"))
        if args.openml_only and not args.openml_suite:
            parser.error("--openml-only needs --openml-suite")
        if args.openml_suite:
            for cell, ds in openml_cells(args.openml_suite, args.openml_max_rows,
                                         args.regimes or DATA_REGIMES, max_folds=args.openml_folds,
                                         train_cap=args.openml_train_cap, tasks=args.openml_tasks):
                cells.append(cell)
                extra[ds.spec.name] = ds
        if not cells:
            parser.error("the selection is empty")
        previous = None
        if args.resume and args.output.exists():
            prior = json.loads(args.output.read_text())
            previous = [r for r in prior.get("records", []) if r.get("error") is None]
        print(f"{len(cells)} cells x {len(seeds)} seeds x {len(arms)} arms")
        result = run(arms, cells, seeds, args.threads, args.tier, datasets=extra,
                     previous=previous, checkpoint=lambda partial: _write_json(args.output, partial))
        _write_json(args.output, result)
        report = render_report(result, args.primary)
        if args.report:
            args.report.write_text(report)
        print(report)
        return 0

    if args.command == "report":
        report = render_report(json.loads(args.file.read_text()), args.primary)
        if args.output:
            args.output.write_text(report)
        print(report)
        return 0

    if len(args.files) not in (1, 2):
        parser.error("compare takes one or two result files")
    base = json.loads(args.files[0].read_text())
    cand = json.loads(args.files[-1].read_text())
    cmp = paired_compare(base, cand, args.baseline, args.candidate,
                         args.min_improvement, args.max_dataset_regression)
    print(render_comparison(cmp))
    return 0 if (cmp["gate_passed"] or not args.gate) else 1


if __name__ == "__main__":
    sys.exit(main())
