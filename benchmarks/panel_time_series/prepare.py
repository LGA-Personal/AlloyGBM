#!/usr/bin/env python3
"""Prepare panel/time-series benchmark data from UCI Air Quality dataset."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

RAW_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
RAW_FILENAME = "AirQualityUCI.zip"
PREPARED_FILENAME = "prepared.csv"

SOURCE_FIELDS = [
    "Date",
    "Time",
    "CO(GT)",
    "PT08.S1(CO)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]

OUTPUT_FIELDS = [
    "group_id",
    "timestamp",
    "co_gt",
    "pt08_s1_co",
    "c6h6_gt",
    "pt08_s2_nmhc",
    "nox_gt",
    "pt08_s3_nox",
    "no2_gt",
    "pt08_s4_no2",
    "pt08_s5_o3",
    "temperature_c",
    "relative_humidity",
    "absolute_humidity",
    "target_co_gt",
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


def _parse_timestamp(date_text: str, time_text: str) -> str:
    parsed = dt.datetime.strptime(f"{date_text} {time_text}", "%d/%m/%Y %H.%M.%S")
    return parsed.isoformat()


def _to_float(raw: str) -> float:
    cleaned = raw.strip()
    if cleaned in {"", "?", "-200"}:
        raise ValueError("missing numeric value")
    return float(cleaned.replace(",", "."))


def _prepare_rows(raw_zip_path: Path, prepared_path: Path, max_rows: int) -> tuple[int, int]:
    prepared_rows: list[dict[str, object]] = []
    dropped_no_future_target = 0
    prepared_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(raw_zip_path, "r") as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_members:
            raise ValueError("no CSV member found in panel_time_series source archive")
        source_name = csv_members[0]

        with archive.open(source_name, "r") as raw_member, prepared_path.open(
            "w", encoding="utf-8", newline=""
        ) as prepared_file:
            raw_lines = (line.decode("latin1") for line in raw_member)
            reader = csv.DictReader(raw_lines, delimiter=";")
            for row in reader:
                try:
                    timestamp = _parse_timestamp(row["Date"], row["Time"])
                    values = {_field: _to_float(row[_field]) for _field in SOURCE_FIELDS[2:]}
                except (KeyError, ValueError):
                    continue

                prepared_rows.append(
                    {
                        "group_id": "air_quality_station_1",
                        "timestamp": timestamp,
                        "co_gt": values["CO(GT)"],
                        "pt08_s1_co": values["PT08.S1(CO)"],
                        "c6h6_gt": values["C6H6(GT)"],
                        "pt08_s2_nmhc": values["PT08.S2(NMHC)"],
                        "nox_gt": values["NOx(GT)"],
                        "pt08_s3_nox": values["PT08.S3(NOx)"],
                        "no2_gt": values["NO2(GT)"],
                        "pt08_s4_no2": values["PT08.S4(NO2)"],
                        "pt08_s5_o3": values["PT08.S5(O3)"],
                        "temperature_c": values["T"],
                        "relative_humidity": values["RH"],
                        "absolute_humidity": values["AH"],
                    }
                )

    prepared_rows.sort(key=lambda row: str(row["timestamp"]))
    finalized_rows: list[dict[str, object]] = []
    for idx, row in enumerate(prepared_rows):
        next_idx = idx + 1
        while next_idx < len(prepared_rows) and (
            str(prepared_rows[next_idx]["timestamp"]) <= str(row["timestamp"])
        ):
            next_idx += 1
        if next_idx >= len(prepared_rows):
            dropped_no_future_target += 1
            continue

        out_row = dict(row)
        out_row["target_co_gt"] = float(prepared_rows[next_idx]["co_gt"])
        finalized_rows.append(out_row)

    if max_rows > 0:
        finalized_rows = finalized_rows[:max_rows]

    with prepared_path.open("w", encoding="utf-8", newline="") as prepared_file:
        writer = csv.DictWriter(prepared_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in finalized_rows:
            writer.writerow(row)

    return len(finalized_rows), dropped_no_future_target


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "panel_time_series",
        help="directory for raw/ and prepared/ benchmark outputs",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200000,
        help="maximum prepared rows to emit (0 means all rows)",
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

    kept_rows, dropped_no_future_target = _prepare_rows(
        raw_path, prepared_path, max_rows=args.max_rows
    )
    print(
        "[panel_time_series] prepared dataset written to "
        f"{prepared_path} (rows={kept_rows}, dropped_no_future_target={dropped_no_future_target})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
