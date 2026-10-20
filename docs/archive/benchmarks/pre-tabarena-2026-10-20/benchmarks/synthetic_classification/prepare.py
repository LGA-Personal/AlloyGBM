#!/usr/bin/env python3
"""Generate synthetic binary classification benchmark data."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

PREPARED_FILENAME = "prepared.csv"


def _generate_feature(row_index: int, feature_index: int, rng: random.Random) -> float:
    if feature_index == 0:
        return round(rng.random() * 8.0) / 8.0
    if feature_index == 1:
        return math.exp(rng.gauss(0.0, 1.0))
    if feature_index == 2:
        return 0.0 if (row_index % 9) else 1.0
    if feature_index % 5 == 0:
        return round(rng.uniform(-3.0, 3.0), 2)
    return rng.uniform(-1.0, 1.0)


def _generate_target(features: list[float], rng: random.Random) -> int:
    weighted = 0.0
    for index, value in enumerate(features[:8]):
        weighted += value * (0.7 - (index * 0.05))
    nonlinear = math.sin(features[0] * 3.0) + math.log1p(abs(features[1]))
    noise = rng.gauss(0.0, 0.3)
    score = weighted + nonlinear + noise
    return 1 if score > 0.0 else 0


def _write_dataset(
    prepared_path: Path, rows: int, feature_count: int, seed: int
) -> None:
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    fieldnames = [f"f{i}" for i in range(feature_count)] + ["target"]

    with prepared_path.open("w", encoding="utf-8", newline="") as prepared_file:
        writer = csv.DictWriter(prepared_file, fieldnames=fieldnames)
        writer.writeheader()
        for row_index in range(rows):
            features = [
                _generate_feature(row_index, feature_index, rng)
                for feature_index in range(feature_count)
            ]
            target = _generate_target(features, rng)
            row = {f"f{idx}": value for idx, value in enumerate(features)}
            row["target"] = target
            writer.writerow(row)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "synthetic_classification",
        help="directory for prepared benchmark outputs",
    )
    parser.add_argument("--rows", type=int, default=50000, help="row count to generate")
    parser.add_argument(
        "--features", type=int, default=32, help="feature count to generate"
    )
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    args = parser.parse_args(argv)

    if args.rows <= 0:
        raise ValueError("rows must be greater than 0")
    if args.features <= 0:
        raise ValueError("features must be greater than 0")

    prepared_path = args.output_dir / "prepared" / PREPARED_FILENAME
    _write_dataset(
        prepared_path=prepared_path,
        rows=args.rows,
        feature_count=args.features,
        seed=args.seed,
    )
    print(
        "[synthetic_classification] prepared dataset written to "
        f"{prepared_path} (rows={args.rows}, features={args.features}, seed={args.seed})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
