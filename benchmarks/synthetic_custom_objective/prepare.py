#!/usr/bin/env python3
"""Generate synthetic regression data for custom objective benchmarking.

Produces a CSV with numeric features and a continuous target suitable for
comparing built-in vs custom MSE-equivalent objectives.
"""

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


def _generate_target(features: list[float], rng: random.Random) -> float:
    """Linear combination of features with noise."""
    target = 0.0
    for i, f in enumerate(features[:6]):
        weight = math.sin((i + 1) * 0.7) * (1.0 / (i + 1))
        target += f * weight
    # Add interaction terms
    if len(features) > 6:
        target += features[0] * features[1] * 0.3
    if len(features) > 10:
        target += features[6] * features[7] * 0.2
    target += rng.gauss(0.0, 0.2)
    return target


def _write_dataset(
    prepared_path: Path,
    rows: int,
    feature_count: int,
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
            target = _generate_target(features, rng)
            row = {f"f{idx}": round(value, 6) for idx, value in enumerate(features)}
            row["target"] = round(target, 6)
            writer.writerow(row)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "synthetic_custom_objective",
        help="directory for prepared benchmark outputs",
    )
    parser.add_argument("--rows", type=int, default=10000, help="row count to generate")
    parser.add_argument(
        "--features", type=int, default=20, help="feature count to generate"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
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
        "[synthetic_custom_objective] prepared dataset written to "
        f"{prepared_path} (rows={args.rows}, features={args.features}, "
        f"seed={args.seed})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
