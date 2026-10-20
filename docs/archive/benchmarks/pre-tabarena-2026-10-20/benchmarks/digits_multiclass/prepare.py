#!/usr/bin/env python3
"""Prepare Digits multi-class classification benchmark data via scikit-learn."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from sklearn.datasets import load_digits

PREPARED_FILENAME = "prepared.csv"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "digits_multiclass",
        help="directory for prepared benchmark outputs",
    )
    args = parser.parse_args(argv)

    prepared_dir = args.output_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = prepared_dir / PREPARED_FILENAME

    bunch = load_digits()
    n_features = bunch.data.shape[1]
    feature_names = [f"pixel_{i}" for i in range(n_features)]
    fieldnames = feature_names + ["target"]

    with prepared_path.open("w", encoding="utf-8", newline="") as prepared_file:
        writer = csv.DictWriter(prepared_file, fieldnames=fieldnames)
        writer.writeheader()
        for features, target in zip(bunch.data, bunch.target):
            row = {name: float(value) for name, value in zip(feature_names, features)}
            row["target"] = int(target)
            writer.writerow(row)

    print(f"[digits_multiclass] prepared dataset written to {prepared_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
