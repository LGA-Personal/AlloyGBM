#!/usr/bin/env python3
"""Generate synthetic learning-to-rank benchmark data."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

PREPARED_FILENAME = "prepared.csv"


def _generate_feature(rng: random.Random, feature_index: int) -> float:
    if feature_index == 0:
        return round(rng.random() * 8.0) / 8.0
    if feature_index == 1:
        return math.exp(rng.gauss(0.0, 0.5))
    if feature_index % 4 == 0:
        return round(rng.uniform(-2.0, 2.0), 2)
    return rng.uniform(-1.0, 1.0)


def _generate_relevance(features: list[float], rng: random.Random) -> int:
    weighted = 0.0
    for index, value in enumerate(features[:6]):
        weighted += value * (0.5 - (index * 0.06))
    nonlinear = math.sin(features[0] * 2.0) + 0.5 * math.log1p(abs(features[1]))
    noise = rng.gauss(0.0, 0.4)
    score = weighted + nonlinear + noise
    if score < -0.8:
        return 0
    elif score < -0.2:
        return 1
    elif score < 0.4:
        return 2
    elif score < 1.0:
        return 3
    else:
        return 4


def _write_dataset(
    prepared_path: Path,
    queries: int,
    docs_per_query: int,
    feature_count: int,
    seed: int,
) -> None:
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    fieldnames = ["query_id"] + [f"f{i}" for i in range(feature_count)] + ["relevance"]

    with prepared_path.open("w", encoding="utf-8", newline="") as prepared_file:
        writer = csv.DictWriter(prepared_file, fieldnames=fieldnames)
        writer.writeheader()
        for query_index in range(queries):
            for _doc_index in range(docs_per_query):
                features = [
                    _generate_feature(rng, feature_index)
                    for feature_index in range(feature_count)
                ]
                relevance = _generate_relevance(features, rng)
                row = {"query_id": query_index}
                row.update({f"f{idx}": value for idx, value in enumerate(features)})
                row["relevance"] = relevance
                writer.writerow(row)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "synthetic_ranking",
        help="directory for prepared benchmark outputs",
    )
    parser.add_argument("--queries", type=int, default=200, help="number of queries")
    parser.add_argument(
        "--docs-per-query", type=int, default=25, help="documents per query"
    )
    parser.add_argument(
        "--features", type=int, default=16, help="feature count to generate"
    )
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    args = parser.parse_args(argv)

    if args.queries <= 0:
        raise ValueError("queries must be greater than 0")
    if args.docs_per_query <= 0:
        raise ValueError("docs_per_query must be greater than 0")
    if args.features <= 0:
        raise ValueError("features must be greater than 0")

    prepared_path = args.output_dir / "prepared" / PREPARED_FILENAME
    _write_dataset(
        prepared_path=prepared_path,
        queries=args.queries,
        docs_per_query=args.docs_per_query,
        feature_count=args.features,
        seed=args.seed,
    )
    total_rows = args.queries * args.docs_per_query
    print(
        "[synthetic_ranking] prepared dataset written to "
        f"{prepared_path} (queries={args.queries}, docs_per_query={args.docs_per_query}, "
        f"features={args.features}, total_rows={total_rows}, seed={args.seed})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
