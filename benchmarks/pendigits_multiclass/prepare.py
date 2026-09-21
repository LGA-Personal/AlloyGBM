#!/usr/bin/env python3
"""Prepare pendigits_multiclass (HOLDOUT -- not consulted while calibrating the auto policy)."""

from __future__ import annotations

import argparse
import csv
import urllib.request
from pathlib import Path

RAW_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra"
RAW_FILENAME = "pendigits.tra"
PREPARED_FILENAME = "prepared.csv"
SCENARIO = "pendigits_multiclass"
FEATURE_COUNT = 16
DELIMITER = ','


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    root = _repo_root()
    raw_path = root / "benchmarks" / "data" / SCENARIO / "raw" / RAW_FILENAME
    prepared_path = root / "benchmarks" / "data" / SCENARIO / "prepared" / PREPARED_FILENAME
    if not raw_path.exists() or args.force_download:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {RAW_URL}", flush=True)
        with urllib.request.urlopen(RAW_URL, timeout=180) as response:
            raw_path.write_bytes(response.read())
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    labels: dict[str, int] = {}
    written = 0
    fieldnames = [f"f{i}" for i in range(FEATURE_COUNT)] + ["target"]
    with raw_path.open("r", encoding="utf-8") as handle, prepared_path.open(
        "w", encoding="utf-8", newline=""
    ) as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for line in handle:
            parts = line.strip().split(DELIMITER) if DELIMITER else line.split()
            parts = [piece.strip() for piece in parts if piece.strip()]
            if len(parts) != FEATURE_COUNT + 1:
                continue
            record = {f"f{i}": float(parts[i]) for i in range(FEATURE_COUNT)}
            raw_label = parts[FEATURE_COUNT]
            if raw_label not in labels:
                labels[raw_label] = len(labels)
            record["target"] = labels[raw_label]
            writer.writerow(record)
            written += 1
    if written == 0:
        raise ValueError(f"{SCENARIO} produced no rows")
    print(f"wrote {prepared_path} ({written} rows, {len(labels)} classes)")


if __name__ == "__main__":
    main()
