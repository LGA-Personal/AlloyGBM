#!/usr/bin/env python3
"""Generate synthetic multi-class classification benchmark data."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

PREPARED_FILENAME = "prepared.csv"


def _generate_features(rng: random.Random, feature_count: int) -> list[float]:
    return [rng.gauss(0.0, 1.0) for _ in range(feature_count)]


def _generate_target(features: list[float], n_classes: int, rng: random.Random) -> int:
    """Assign class based on feature combinations with noise.

    Uses a simple scoring scheme: each class has a linear combination of
    features that gives it a 'vote'. The class with the highest vote wins.
    """
    scores = []
    for c in range(n_classes):
        score = 0.0
        for i in range(min(len(features), 6)):
            # Each class weights different feature subsets
            weight = math.sin((c + 1) * (i + 1) * 0.7) * (1.0 / (i + 1))
            score += features[i] * weight
        # Add class-specific interaction term
        if len(features) > c + 6:
            score += features[c + 6] * 0.3
        score += rng.gauss(0.0, 0.2)
        scores.append(score)
    return scores.index(max(scores))


def _write_dataset(
    prepared_path: Path,
    rows: int,
    feature_count: int,
    n_classes: int,
    seed: int,
) -> None:
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    fieldnames = [f"f{i}" for i in range(feature_count)] + ["target"]

    with prepared_path.open("w", encoding="utf-8", newline="") as prepared_file:
        writer = csv.DictWriter(prepared_file, fieldnames=fieldnames)
        writer.writeheader()
        for _ in range(rows):
            features = _generate_features(rng, feature_count)
            target = _generate_target(features, n_classes, rng)
            row = {f"f{idx}": round(value, 6) for idx, value in enumerate(features)}
            row["target"] = target
            writer.writerow(row)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "synthetic_multiclass",
        help="directory for prepared benchmark outputs",
    )
    parser.add_argument("--rows", type=int, default=10000, help="row count to generate")
    parser.add_argument(
        "--features", type=int, default=20, help="feature count to generate"
    )
    parser.add_argument("--classes", type=int, default=5, help="number of classes")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args(argv)

    if args.rows <= 0:
        raise ValueError("rows must be greater than 0")
    if args.features <= 0:
        raise ValueError("features must be greater than 0")
    if args.classes < 2:
        raise ValueError("classes must be at least 2")

    prepared_path = args.output_dir / "prepared" / PREPARED_FILENAME
    _write_dataset(
        prepared_path=prepared_path,
        rows=args.rows,
        feature_count=args.features,
        n_classes=args.classes,
        seed=args.seed,
    )
    print(
        "[synthetic_multiclass] prepared dataset written to "
        f"{prepared_path} (rows={args.rows}, features={args.features}, "
        f"classes={args.classes}, seed={args.seed})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
