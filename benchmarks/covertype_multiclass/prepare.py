#!/usr/bin/env python3
"""Prepare the covertype_multiclass benchmark from its UCI source."""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
import urllib.request
from pathlib import Path

RAW_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
RAW_FILENAME = "covtype.data.gz"
PREPARED_FILENAME = "prepared.csv"
SCENARIO = "covertype_multiclass"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _download(raw_path: Path, force: bool) -> None:
    if raw_path.exists() and not force:
        return
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {RAW_URL}", flush=True)
    with urllib.request.urlopen(RAW_URL, timeout=120) as response:
        raw_path.write_bytes(response.read())


SUBSAMPLE_ROWS = 60000
SUBSAMPLE_SEED = 20260920


def _write(raw_path: Path, prepared_path: Path) -> int:
    """7-class forest cover type, 54 features.

    The full source is 581,012 rows, which would dominate every sweep's wall
    time. A deterministic reservoir subsample keeps the real-data character and
    the class mix while staying comparable in cost to the other large scenarios.
    """
    import random

    rng = random.Random(SUBSAMPLE_SEED)
    reservoir: list[list[str]] = []
    seen = 0
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            parts = [piece.strip() for piece in line.strip().split(",")]
            if len(parts) != 55:
                continue
            seen += 1
            if len(reservoir) < SUBSAMPLE_ROWS:
                reservoir.append(parts)
            else:
                index = rng.randrange(seen)
                if index < SUBSAMPLE_ROWS:
                    reservoir[index] = parts
    if not reservoir:
        raise ValueError("covertype_multiclass produced no rows")
    fieldnames = [f"f{i}" for i in range(54)] + ["target"]
    with prepared_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for parts in reservoir:
            record = {f"f{i}": int(parts[i]) for i in range(54)}
            record["target"] = int(parts[54]) - 1
            writer.writerow(record)
    return len(reservoir)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    root = _repo_root()
    raw_path = root / "benchmarks" / "data" / SCENARIO / "raw" / RAW_FILENAME
    prepared_path = root / "benchmarks" / "data" / SCENARIO / "prepared" / PREPARED_FILENAME
    _download(raw_path, args.force_download)
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _write(raw_path, prepared_path)
    print(f"wrote {prepared_path} ({rows} rows)")


if __name__ == "__main__":
    main()
