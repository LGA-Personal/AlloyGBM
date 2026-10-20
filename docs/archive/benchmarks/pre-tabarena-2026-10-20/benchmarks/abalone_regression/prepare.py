#!/usr/bin/env python3
"""Prepare UCI Abalone age-regression benchmark data."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

RAW_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
)
RAW_FILENAME = "abalone.data"
PREPARED_FILENAME = "prepared.csv"

RAW_COLUMNS = [
    "sex", "length", "diameter", "height",
    "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings",
]
SEX_ENCODING = {"F": 0, "I": 1, "M": 2}


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


def _normalize(raw_path: Path, prepared_path: Path) -> None:
    feature_names = [c for c in RAW_COLUMNS if c != "rings"]
    fieldnames = feature_names + ["rings"]

    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with (
        raw_path.open("r", encoding="utf-8") as rf,
        prepared_path.open("w", encoding="utf-8", newline="") as wf,
    ):
        reader = csv.reader(rf)
        writer = csv.DictWriter(wf, fieldnames=fieldnames)
        writer.writeheader()
        for parts in reader:
            if len(parts) != len(RAW_COLUMNS):
                continue
            raw_row = dict(zip(RAW_COLUMNS, [p.strip() for p in parts]))
            out: dict[str, object] = {
                "sex": SEX_ENCODING.get(raw_row["sex"], -1),
                "length": float(raw_row["length"]),
                "diameter": float(raw_row["diameter"]),
                "height": float(raw_row["height"]),
                "whole_weight": float(raw_row["whole_weight"]),
                "shucked_weight": float(raw_row["shucked_weight"]),
                "viscera_weight": float(raw_row["viscera_weight"]),
                "shell_weight": float(raw_row["shell_weight"]),
                "rings": int(raw_row["rings"]),
            }
            writer.writerow(out)
            rows_written += 1

    print(f"[abalone_regression] {rows_written} rows written")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "abalone_regression",
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

    _normalize(raw_path, prepared_path)
    print(f"[abalone_regression] prepared dataset written to {prepared_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
