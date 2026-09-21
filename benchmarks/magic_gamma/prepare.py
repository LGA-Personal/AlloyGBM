#!/usr/bin/env python3
"""Prepare the magic_gamma benchmark from its UCI source."""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
import urllib.request
from pathlib import Path

RAW_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"
RAW_FILENAME = "magic04.data"
PREPARED_FILENAME = "prepared.csv"
SCENARIO = "magic_gamma"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _download(raw_path: Path, force: bool) -> None:
    if raw_path.exists() and not force:
        return
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {RAW_URL}", flush=True)
    with urllib.request.urlopen(RAW_URL, timeout=120) as response:
        raw_path.write_bytes(response.read())


def _write(raw_path: Path, prepared_path: Path) -> int:
    """Binary gamma/hadron telescope classification: 10 continuous features."""
    fieldnames = [f"f{i}" for i in range(10)] + ["target"]
    written = 0
    with raw_path.open("r", encoding="utf-8") as handle, prepared_path.open(
        "w", encoding="utf-8", newline=""
    ) as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for line in handle:
            parts = [piece.strip() for piece in line.strip().split(",")]
            if len(parts) != 11 or parts[-1] not in {"g", "h"}:
                continue
            record = {f"f{i}": float(parts[i]) for i in range(10)}
            record["target"] = 1 if parts[-1] == "g" else 0
            writer.writerow(record)
            written += 1
    if written == 0:
        raise ValueError("magic_gamma produced no rows")
    return written



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
