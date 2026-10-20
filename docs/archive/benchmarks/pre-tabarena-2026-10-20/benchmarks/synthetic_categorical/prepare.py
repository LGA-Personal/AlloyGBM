#!/usr/bin/env python3
"""Generate synthetic data with categorical features for benchmarking.

Produces a CSV with continuous and categorical features where the target
depends strongly on categorical membership.  This scenario favors native
categorical splits over target encoding.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

PREPARED_FILENAME = "prepared.csv"

# Categorical feature cardinalities
CAT_CARDINALITIES = [3, 8, 16, 32, 64]


def _generate_row(
    rng: random.Random,
    n_continuous: int,
    cat_cardinalities: list[int],
) -> tuple[list[float], list[str], float]:
    """Generate one row of continuous features, categorical features, and target."""
    continuous = [rng.gauss(0.0, 1.0) for _ in range(n_continuous)]

    categories: list[str] = []
    for k in cat_cardinalities:
        categories.append(f"cat_{rng.randint(0, k - 1)}")

    # Target depends on categorical interactions + continuous features
    target = 0.0
    # Continuous contribution
    for i, f in enumerate(continuous[:4]):
        target += f * (0.5 / (i + 1))

    # Categorical contribution — categories with low IDs get high target
    for i, cat_val in enumerate(categories):
        cat_id = int(cat_val.split("_")[1])
        k = cat_cardinalities[i]
        # Categories below median contribute positively, above median negatively
        if cat_id < k // 2:
            target += 1.5
        else:
            target -= 1.5

    # Interaction between first two categorical features
    cat0_id = int(categories[0].split("_")[1])
    cat1_id = int(categories[1].split("_")[1])
    if cat0_id == 0 and cat1_id < 4:
        target += 3.0

    target += rng.gauss(0.0, 0.5)
    return continuous, categories, target


def _write_dataset(
    prepared_path: Path,
    rows: int,
    n_continuous: int,
    cat_cardinalities: list[int],
    seed: int,
) -> None:
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    cont_names = [f"cont_{i}" for i in range(n_continuous)]
    cat_names = [f"cat_feat_{i}" for i in range(len(cat_cardinalities))]
    fieldnames = cont_names + cat_names + ["target"]

    with prepared_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _ in range(rows):
            continuous, categories, target = _generate_row(
                rng, n_continuous, cat_cardinalities
            )
            row: dict[str, object] = {}
            for idx, val in enumerate(continuous):
                row[f"cont_{idx}"] = round(val, 6)
            for idx, val in enumerate(categories):
                row[f"cat_feat_{idx}"] = val
            row["target"] = round(target, 6)
            writer.writerow(row)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "synthetic_categorical",
        help="directory for prepared benchmark outputs",
    )
    parser.add_argument("--rows", type=int, default=10000, help="row count")
    parser.add_argument(
        "--n-continuous", type=int, default=10, help="number of continuous features"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args(argv)

    if args.rows <= 0:
        raise ValueError("rows must be greater than 0")

    prepared_path = args.output_dir / "prepared" / PREPARED_FILENAME
    _write_dataset(
        prepared_path=prepared_path,
        rows=args.rows,
        n_continuous=args.n_continuous,
        cat_cardinalities=CAT_CARDINALITIES,
        seed=args.seed,
    )
    print(
        f"[synthetic_categorical] prepared dataset written to {prepared_path} "
        f"(rows={args.rows}, n_continuous={args.n_continuous}, seed={args.seed})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
