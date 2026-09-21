#!/usr/bin/env python3
"""Shared generator for the controlled accuracy-at-depth scenario grid.

The grid exists to separate three things the fixed benchmark suite confounds:
sample count, feature count, and the *fraction of features that carry signal*.
`synthetic_classification` has 32 features of which only 8 matter, so any result
measured on it cannot distinguish "deep trees overfit" from "deep trees chase
noise columns". Varying one axis at a time makes that distinguishable.

Signal strength is held constant across variants by scaling the weighted sum by
1/sqrt(signal_count). Without that, adding signal features would also raise the
signal-to-noise ratio, and a comparison across `signal_count` would confound the
two effects it is meant to separate.
"""

from __future__ import annotations

import csv
import math
import random
import statistics
from pathlib import Path

PREPARED_FILENAME = "prepared.csv"


def _feature_value(row_index: int, feature_index: int, rng: random.Random) -> float:
    """Mirror synthetic_classification's feature mix so results stay comparable."""
    # Only index 2 may use the deterministic row pattern. An earlier version
    # cycled these shapes with `% 8`, which made f10 an exact duplicate of the
    # signal-carrying f2 -- so the "noise" features were copies of signal and the
    # noise-fraction axis measured nothing.
    if feature_index == 0:
        return round(rng.random() * 8.0) / 8.0
    if feature_index == 1:
        return math.exp(rng.gauss(0.0, 1.0))
    if feature_index == 2:
        return 0.0 if (row_index % 9) else 1.0
    if feature_index % 5 == 0:
        return round(rng.uniform(-3.0, 3.0), 2)
    return rng.uniform(-1.0, 1.0)


def _score(features: list[float], signal_count: int, rng: random.Random) -> float:
    weighted = 0.0
    for index in range(signal_count):
        weight = 0.7 - (index % 8) * 0.05
        weighted += features[index] * weight
    # Constant signal scale regardless of how many features carry it.
    weighted /= math.sqrt(signal_count)
    nonlinear = (
        math.sin(features[0] * 3.0) + math.log1p(abs(features[1]))
    ) / math.sqrt(signal_count)
    return weighted + nonlinear + rng.gauss(0.0, 0.3)


def write_grid_dataset(
    scenario: str,
    *,
    rows: int,
    feature_count: int,
    signal_count: int,
    seed: int,
    repo_root: Path | None = None,
) -> Path:
    """Write one grid cell, class-balanced by thresholding at the median score.

    A fixed zero threshold produced 1-3% positive rates here, because the
    nonlinear term has a positive mean while the sqrt normalisation shrinks the
    weighted sum. That would have made each cell differ in class balance as well
    as in the axis under test, so the cells would not have isolated anything.
    Splitting at the median fixes the balance at 50/50 for every cell, leaving
    the intended axis as the only thing that varies.
    """
    if signal_count > feature_count:
        raise ValueError("signal_count cannot exceed feature_count")
    if signal_count < 2:
        raise ValueError("signal_count must be at least 2 (the nonlinear term uses f0,f1)")
    root = repo_root or Path(__file__).resolve().parents[2]
    prepared_path = root / "benchmarks" / "data" / scenario / "prepared" / PREPARED_FILENAME
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    generated: list[tuple[list[float], float]] = []
    for row_index in range(rows):
        features = [
            _feature_value(row_index, index, rng) for index in range(feature_count)
        ]
        generated.append((features, _score(features, signal_count, rng)))
    threshold = statistics.median(score for _, score in generated)
    fieldnames = [f"f{i}" for i in range(feature_count)] + ["target"]
    with prepared_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for features, score in generated:
            record = {f"f{i}": features[i] for i in range(feature_count)}
            record["target"] = 1 if score > threshold else 0
            writer.writerow(record)
    return prepared_path


def write_multiclass_grid_dataset(
    scenario: str,
    *,
    rows: int,
    feature_count: int,
    signal_count: int,
    class_count: int,
    seed: int,
    separation: float = 1.1,
    repo_root: Path | None = None,
) -> Path:
    """Write a class-count grid cell with exact class balance.

    Real multiclass datasets confound class count with sample count, feature
    count and class balance, so none of them can measure how a regularisation
    constant should scale with K. Here every cell is identical except K: rows are
    assigned round-robin so balance is exactly 1/K, signal features are drawn
    class-conditionally around per-class means, and the remaining features are
    drawn from the same distribution for every class so they carry no signal.

    `separation` scales the per-class means. It is held fixed across cells, so
    the Bayes error drifts upward with K -- unavoidable when K is the only thing
    that may vary, and the reason these cells calibrate a *scaling law* rather
    than supply comparable absolute accuracies.
    """
    if signal_count > feature_count:
        raise ValueError("signal_count cannot exceed feature_count")
    if class_count < 2:
        raise ValueError("class_count must be at least 2")
    root = repo_root or Path(__file__).resolve().parents[2]
    prepared_path = root / "benchmarks" / "data" / scenario / "prepared" / PREPARED_FILENAME
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    means = [
        [
            separation * math.sin((klass + 1) * (index + 1) * 0.9)
            for index in range(signal_count)
        ]
        for klass in range(class_count)
    ]
    fieldnames = [f"f{i}" for i in range(feature_count)] + ["target"]
    with prepared_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index in range(rows):
            klass = row_index % class_count
            record: dict[str, float | int] = {}
            for index in range(feature_count):
                if index < signal_count:
                    record[f"f{index}"] = rng.gauss(means[klass][index], 1.0)
                else:
                    record[f"f{index}"] = rng.gauss(0.0, 1.0)
            record["target"] = klass
            writer.writerow(record)
    return prepared_path
