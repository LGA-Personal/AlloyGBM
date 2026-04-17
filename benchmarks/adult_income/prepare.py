#!/usr/bin/env python3
"""Prepare UCI Adult Income binary classification benchmark data."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

RAW_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
RAW_FILENAME = "adult.data"
PREPARED_FILENAME = "prepared.csv"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]

CATEGORICAL_COLUMNS = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]

FEATURE_COLUMNS = [c for c in COLUMNS if c not in ("income", "fnlwgt")]


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


def _build_encodings(raw_path: Path) -> dict[str, dict[str, int]]:
    encodings: dict[str, set[str]] = {col: set() for col in CATEGORICAL_COLUMNS}
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) != len(COLUMNS):
                continue
            row = dict(zip(COLUMNS, parts))
            if "?" in row.values():
                continue
            for col in CATEGORICAL_COLUMNS:
                encodings[col].add(row[col])
    return {col: {val: idx for idx, val in enumerate(sorted(vals))}
            for col, vals in encodings.items()}


def _normalize(raw_path: Path, prepared_path: Path) -> None:
    encodings = _build_encodings(raw_path)
    fieldnames = FEATURE_COLUMNS + ["income"]

    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    rows_skipped = 0

    with (
        raw_path.open("r", encoding="utf-8") as rf,
        prepared_path.open("w", encoding="utf-8", newline="") as wf,
    ):
        writer = csv.DictWriter(wf, fieldnames=fieldnames)
        writer.writeheader()
        for line in rf:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) != len(COLUMNS):
                continue
            raw_row = dict(zip(COLUMNS, parts))
            if "?" in raw_row.values():
                rows_skipped += 1
                continue
            out: dict[str, object] = {}
            for col in FEATURE_COLUMNS:
                if col in CATEGORICAL_COLUMNS:
                    out[col] = encodings[col][raw_row[col]]
                else:
                    out[col] = float(raw_row[col])
            income_raw = raw_row["income"].rstrip(".")
            out["income"] = 1 if income_raw == ">50K" else 0
            writer.writerow(out)
            rows_written += 1

    print(
        f"[adult_income] {rows_written} rows written, {rows_skipped} skipped (missing values)"
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "adult_income",
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
    print(f"[adult_income] prepared dataset written to {prepared_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
