#!/usr/bin/env python3
"""Prepare dense numeric benchmark data from UCI Wine Quality."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

RAW_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
RAW_FILENAME = "winequality-red.csv"
PREPARED_FILENAME = "prepared.csv"


def _download(url: str, destination: Path) -> None:
    try:
        urllib.request.urlretrieve(url, destination)
        return
    except Exception as first_error:
        for command in (
            ["curl", "-fL", url, "-o", str(destination)],
            ["wget", "-O", str(destination), url],
        ):
            if shutil.which(command[0]) is None:
                continue
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                return
            except subprocess.CalledProcessError:
                continue
        raise RuntimeError(
            f"failed to download {url}; urllib and curl/wget fallback failed"
        ) from first_error


def _normalize_csv(raw_path: Path, prepared_path: Path) -> None:
    with raw_path.open("r", encoding="utf-8", newline="") as raw_file:
        reader = csv.DictReader(raw_file, delimiter=";")
        fieldnames = list(reader.fieldnames or [])
        if "quality" not in fieldnames:
            raise ValueError("expected 'quality' target column in dense_numeric source")
        rows = list(reader)

    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    with prepared_path.open("w", encoding="utf-8", newline="") as prepared_file:
        writer = csv.DictWriter(prepared_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "dense_numeric",
        help="directory for raw/ and prepared/ benchmark outputs",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="re-download raw file even if already present",
    )
    args = parser.parse_args(argv)

    raw_dir = args.output_dir / "raw"
    prepared_dir = args.output_dir / "prepared"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / RAW_FILENAME
    prepared_path = prepared_dir / PREPARED_FILENAME

    if args.force_download or not raw_path.exists():
        _download(RAW_URL, raw_path)

    _normalize_csv(raw_path, prepared_path)
    print(f"[dense_numeric] prepared dataset written to {prepared_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
