"""Deterministic synthetic fixtures for the architecture benchmark cases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fixture:
    X: np.ndarray
    y: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    dimensions: dict[str, int]
    metadata: dict[str, object]


def _regression_fixture(rows: int, features: int, seed: int) -> Fixture:
    rng = np.random.default_rng(seed)
    total = rows + max(256, rows // 5)
    X = rng.standard_normal(size=(total, features), dtype=np.float32)
    weights = rng.standard_normal(size=features, dtype=np.float32) * np.float32(0.25)
    noise = rng.standard_normal(size=total, dtype=np.float32) * np.float32(0.15)
    y = X @ weights + np.float32(0.35) * X[:, 0] * X[:, 1] + noise
    return Fixture(
        X=X[:rows],
        y=y[:rows],
        X_test=X[rows:],
        y_test=y[rows:],
        dimensions={"rows": rows, "features": features, "test_rows": total - rows},
        metadata={},
    )


def _node_fixture(profile: str, seed: int) -> Fixture:
    del seed
    rows = 1 << (14 if profile == "quick" else 20)
    features = 10 if profile == "quick" else 14
    row_ids = np.arange(rows, dtype=np.uint32)
    X = np.column_stack([((row_ids >> bit) & 1).astype(np.float32) for bit in range(features)])
    weights = np.power(2.0, -np.arange(features, dtype=np.float32))
    y = (X @ weights + 0.01 * X[:, 0] * X[:, 1]).astype(np.float32)
    test_rows = min(4096, rows)
    return Fixture(X, y, X[:test_rows].copy(), y[:test_rows].copy(), {"rows": rows, "features": features, "test_rows": test_rows}, {})


def _efb_fixture(case: str, profile: str, seed: int) -> Fixture:
    rng = np.random.default_rng(seed)
    rows = 4_000 if profile == "quick" else 80_000
    groups = 8 if profile == "quick" else 32
    width = 8 if profile == "quick" else 16
    test_rows = max(512, rows // 5)
    total = rows + test_rows
    if case == "dense_control":
        return _regression_fixture(rows, groups * width, seed)
    X = np.zeros((total, groups * width), dtype=np.float32)
    active = np.empty((total, groups), dtype=np.int32)
    row_ids = np.arange(total)
    for group in range(groups):
        categories = ((2 * group + 1) * row_ids + 3 * group) % width
        active[:, group] = categories
        X[row_ids, group * width + categories] = 1.0
        if case == "controlled_conflict":
            conflict_count = max(1, int(total * 0.02))
            conflict_rows = rng.choice(total, size=conflict_count, replace=False)
            second = (categories[conflict_rows] + 1) % width
            X[conflict_rows, group * width + second] = 1.0
    group_weights = np.linspace(0.1, 1.0, groups, dtype=np.float32)
    y = (active * group_weights).sum(axis=1).astype(np.float32)
    y += rng.normal(scale=0.1, size=total).astype(np.float32)
    return Fixture(X[:rows], y[:rows], X[rows:], y[rows:], {"rows": rows, "features": groups * width, "test_rows": test_rows}, {"conflict_rate": 0.02 if case == "controlled_conflict" else 0.0})


def _quantile_fixture(profile: str, seed: int) -> Fixture:
    rng = np.random.default_rng(seed)
    rows = 25_000 if profile == "quick" else 1_000_000
    features = 8 if profile == "quick" else 16
    test_rows = max(1_000, rows // 20)
    total = rows + test_rows
    columns = []
    for index in range(features):
        kind = index % 6
        if kind == 0:
            values = rng.lognormal(mean=0.0, sigma=1.3, size=total)
        elif kind == 1:
            values = rng.exponential(scale=2.0, size=total)
        elif kind == 2:
            values = rng.standard_t(df=3.0, size=total)
        elif kind == 3:
            choose = rng.random(total) < 0.85
            values = np.where(choose, rng.normal(-1.0, 0.2, total), rng.normal(4.0, 1.0, total))
        elif kind == 4:
            values = np.round(rng.normal(size=total), 1)
        else:
            values = rng.normal(size=total)
            values[rng.random(total) < 0.03] = np.nan
        columns.append(values.astype(np.float32))
    X = np.column_stack(columns)
    clean = np.nan_to_num(X, nan=0.0)
    y = (np.log1p(np.abs(clean[:, 0])) + 0.4 * clean[:, 1] - 0.2 * clean[:, 2] + rng.normal(scale=0.2, size=total)).astype(np.float32)
    return Fixture(X[:rows], y[:rows], X[rows:], y[rows:], {"rows": rows, "features": features, "test_rows": test_rows}, {})


def make_fixture(scenario: str, case: str, profile: str, *, seed: int) -> Fixture:
    if profile not in {"quick", "full"}:
        raise ValueError(f"unknown profile {profile!r}")
    if scenario == "node_parallelism":
        return _node_fixture(profile, seed)
    if scenario == "efb":
        return _efb_fixture(case, profile, seed)
    if scenario == "quantile_sketches":
        return _quantile_fixture(profile, seed)
    if scenario == "compact_nodes":
        rows = 5_000 if profile == "quick" else 100_000
        X = np.ones((rows, 1), dtype=np.float32)
        return Fixture(X, np.zeros(rows, dtype=np.float32), X, np.zeros(rows, dtype=np.float32), {"rows": rows, "features": 1, "test_rows": rows}, {})
    shapes = {
        ("soa_histograms", "standard_wide"): ((8_000, 16) if profile == "quick" else (100_000, 128)),
        ("soa_histograms", "standard_deep"): ((8_000, 12) if profile == "quick" else (200_000, 24)),
        ("soa_histograms", "dro_wide"): ((6_000, 12) if profile == "quick" else (75_000, 48)),
        ("soa_histograms", "linear_leaf"): ((2_000, 8) if profile == "quick" else (30_000, 16)),
        ("duplicate_bins", "wide_shallow_u8"): ((2_000, 64) if profile == "quick" else (30_000, 512)),
        ("duplicate_bins", "wide_shallow_u16"): ((2_000, 64) if profile == "quick" else (30_000, 512)),
        ("duplicate_bins", "tall_narrow_u8"): ((20_000, 8) if profile == "quick" else (600_000, 16)),
        ("duplicate_bins", "tall_narrow_u16"): ((20_000, 8) if profile == "quick" else (600_000, 16)),
    }
    try:
        rows, features = shapes[(scenario, case)]
    except KeyError as exc:
        raise ValueError(f"unknown fixture {scenario}/{case}") from exc
    return _regression_fixture(rows, features, seed)
