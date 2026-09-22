#!/usr/bin/env python3
"""Prepare UCI Wine Quality (white) regression benchmark data.

Distinct from `dense_numeric`, which uses the *red* wine file (1,599 rows).
"""

from __future__ import annotations

import argparse
import csv
import urllib.request
from pathlib import Path

RAW_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-white.csv"
)
RAW_FILENAME = "winequality-white.csv"
PREPARED_FILENAME = "prepared.csv"
SCENARIO = "wine_quality_white"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    raw_path = root / "benchmarks" / "data" / SCENARIO / "raw" / RAW_FILENAME
    prepared_path = root / "benchmarks" / "data" / SCENARIO / "prepared" / PREPARED_FILENAME
    if not raw_path.exists() or args.force_download:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {RAW_URL}", flush=True)
        with urllib.request.urlopen(RAW_URL, timeout=120) as response:
            raw_path.write_bytes(response.read())
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=";")
        header = next(reader)
        columns = [name.strip().replace(" ", "_") for name in header]
        rows = [row for row in reader if len(row) == len(columns)]
    if not rows:
        raise ValueError("wine_quality_white produced no rows")
    with prepared_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=columns[:-1] + ["quality"])
        writer.writeheader()
        for row in rows:
            record = {columns[i]: float(row[i]) for i in range(len(columns) - 1)}
            record["quality"] = float(row[-1])
            writer.writerow(record)
    print(f"wrote {prepared_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
