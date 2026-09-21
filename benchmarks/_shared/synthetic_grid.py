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


def _bayes_accuracy(means: list[list[float]], rng: random.Random, trials: int = 4000) -> float:
    """Monte-Carlo accuracy of the nearest-mean (Bayes) rule for these means."""
    class_count = len(means)
    correct = 0
    for trial in range(trials):
        true_class = trial % class_count
        point = [rng.gauss(mu, 1.0) for mu in means[true_class]]
        best_class, best_distance = 0, None
        for klass, mean in enumerate(means):
            distance = sum((point[i] - mean[i]) ** 2 for i in range(len(mean)))
            if best_distance is None or distance < best_distance:
                best_class, best_distance = klass, distance
        correct += best_class == true_class
    return correct / trials


def _class_means(class_count: int, signal_count: int, separation: float) -> list[list[float]]:
    return [
        [separation * math.sin((klass + 1) * (index + 1) * 0.9) for index in range(signal_count)]
        for klass in range(class_count)
    ]


def separation_for_target_accuracy(
    class_count: int, signal_count: int, target_accuracy: float, seed: int = 0
) -> float:
    """Find the mean separation giving a target Bayes accuracy for this K.

    Holding `separation` fixed while varying K makes the task progressively
    harder: the first kgrid sweep reached 0.913 of the random-guess log-loss at
    K=20, i.e. nearly unlearnable. On a near-noise problem maximal shrinkage
    always wins because the best prediction is the prior, so that sweep measured
    task difficulty rather than class count. Solving for separation per K holds
    difficulty fixed so K is genuinely the only thing that varies.
    """
    low, high = 0.05, 40.0
    for _ in range(40):
        mid = (low + high) / 2.0
        rng = random.Random(seed)
        accuracy = _bayes_accuracy(_class_means(class_count, signal_count, mid), rng)
        if accuracy < target_accuracy:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def write_multiclass_grid_dataset(
    scenario: str,
    *,
    rows: int,
    feature_count: int,
    signal_count: int,
    class_count: int,
    seed: int,
    separation: float | None = None,
    target_accuracy: float | None = 0.80,
    repo_root: Path | None = None,
) -> Path:
    """Write a class-count grid cell with exact class balance.

    Real multiclass datasets confound class count with sample count, feature
    count and class balance, so none of them can measure how a regularisation
    constant should scale with K. Here every cell is identical except K: rows are
    assigned round-robin so balance is exactly 1/K, signal features are drawn
    class-conditionally around per-class means, and the remaining features are
    drawn from the same distribution for every class so they carry no signal.

    Difficulty is held constant across cells by solving for the mean separation
    that gives `target_accuracy` under the Bayes (nearest-mean) rule at each K.
    A fixed separation does *not* work: it made the K=20 cell reach 0.913 of the
    random-guess log-loss, so that sweep compared task difficulty rather than
    class count. Pass an explicit `separation` only to reproduce that earlier,
    confounded behaviour.
    """
    if signal_count > feature_count:
        raise ValueError("signal_count cannot exceed feature_count")
    if class_count < 2:
        raise ValueError("class_count must be at least 2")
    root = repo_root or Path(__file__).resolve().parents[2]
    prepared_path = root / "benchmarks" / "data" / scenario / "prepared" / PREPARED_FILENAME
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    if separation is None:
        if target_accuracy is None:
            raise ValueError("pass either separation or target_accuracy")
        separation = separation_for_target_accuracy(
            class_count, signal_count, target_accuracy
        )
    rng = random.Random(seed)
    means = _class_means(class_count, signal_count, separation)
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
