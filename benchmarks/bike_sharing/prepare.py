#!/usr/bin/env python3
"""Prepare hourly bike sharing benchmark data from UCI Bike Sharing."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

RAW_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/"
    "Bike-Sharing-Dataset.zip"
)
RAW_FILENAME = "Bike-Sharing-Dataset.zip"
PREPARED_FILENAME = "prepared.csv"

OUTPUT_FIELDS = [
    "group_id",
    "timestamp",
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "cnt",
]


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


def _prepare_rows(raw_zip_path: Path, prepared_path: Path) -> int:
    rows_written = 0
    prepared_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(raw_zip_path, "r") as archive:
        member_name = None
        for name in archive.namelist():
            if name.endswith("hour.csv"):
                member_name = name
                break
        if member_name is None:
            raise ValueError("hour.csv missing from bike_sharing source archive")

        with archive.open(member_name, "r") as raw_member, prepared_path.open(
            "w", encoding="utf-8", newline=""
        ) as prepared_file:
            raw_lines = (line.decode("utf-8") for line in raw_member)
            reader = csv.DictReader(raw_lines)
            writer = csv.DictWriter(prepared_file, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()

            for row in reader:
                try:
                    timestamp = f"{row['dteday']}T{int(row['hr']):02d}:00:00"
                    output_row = {
                        "group_id": "bike_sharing_hourly",
                        "timestamp": timestamp,
                        "season": int(row["season"]),
                        "yr": int(row["yr"]),
                        "mnth": int(row["mnth"]),
                        "hr": int(row["hr"]),
                        "holiday": int(row["holiday"]),
                        "weekday": int(row["weekday"]),
                        "workingday": int(row["workingday"]),
                        "weathersit": int(row["weathersit"]),
                        "temp": float(row["temp"]),
                        "atemp": float(row["atemp"]),
                        "hum": float(row["hum"]),
                        "windspeed": float(row["windspeed"]),
                        "cnt": int(row["cnt"]),
                    }
                except (KeyError, ValueError):
                    continue

                writer.writerow(output_row)
                rows_written += 1

    return rows_written


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "bike_sharing",
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

    rows_written = _prepare_rows(raw_path, prepared_path)
    print(f"[bike_sharing] prepared dataset written to {prepared_path} (rows={rows_written})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
