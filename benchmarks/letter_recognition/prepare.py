#!/usr/bin/env python3
"""Prepare the letter_recognition benchmark from its UCI source."""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
import urllib.request
from pathlib import Path

RAW_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
RAW_FILENAME = "letter-recognition.data"
PREPARED_FILENAME = "prepared.csv"
SCENARIO = "letter_recognition"


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
    """26-class letter recognition: 16 integer features, class is the letter."""
    fieldnames = [f"f{i}" for i in range(16)] + ["target"]
    written = 0
    with raw_path.open("r", encoding="utf-8") as handle, prepared_path.open(
        "w", encoding="utf-8", newline=""
    ) as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for line in handle:
            parts = [piece.strip() for piece in line.strip().split(",")]
            if len(parts) != 17 or not parts[0]:
                continue
            record = {f"f{i}": int(parts[i + 1]) for i in range(16)}
            record["target"] = ord(parts[0].upper()) - ord("A")
            writer.writerow(record)
            written += 1
    if written == 0:
        raise ValueError("letter_recognition produced no rows")
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
