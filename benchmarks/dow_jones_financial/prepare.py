#!/usr/bin/env python3
"""Prepare a finance-oriented benchmark from UCI Dow Jones Index data."""

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

RAW_URL = "https://archive.ics.uci.edu/static/public/312/dow+jones+index.zip"
RAW_FILENAME = "dow_jones_index.zip"
PREPARED_FILENAME = "prepared.csv"

OUTPUT_FIELDS = [
    "group_id",
    "timestamp",
    "quarter",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume",
    "percent_change_price",
    "percent_change_volume_over_last_wk",
    "previous_weeks_volume",
    "days_to_next_dividend",
    "percent_return_next_dividend",
    "target_percent_change_next_weeks_price",
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


def _to_float(raw: str) -> float:
    cleaned = raw.strip()
    if cleaned in {"", "?", "NA", "N/A"}:
        raise ValueError("missing numeric value")
    normalized = cleaned.replace("$", "").replace(",", "")
    return float(normalized)


def _parse_date(value: str) -> str:
    parsed = dt.datetime.strptime(value.strip(), "%m/%d/%Y")
    return parsed.isoformat()


def _row_target_percent_change(row: dict[str, str], close_price: float) -> tuple[float, bool]:
    raw_percent = row.get("percent_change_next_weeks_price", "")
    if raw_percent.strip() not in {"", "?", "NA", "N/A"}:
        return _to_float(raw_percent), False

    # Fallback path when percent target is unavailable in source row.
    next_weeks_close = _to_float(row["next_weeks_close"])
    return next_weeks_close, True


def _collect_prepared_rows(raw_zip_path: Path) -> tuple[list[dict[str, object]], int, int]:
    rows: list[dict[str, object]] = []
    dropped = 0
    fallback_targets = 0

    with zipfile.ZipFile(raw_zip_path, "r") as archive:
        if "dow_jones_index.data" not in archive.namelist():
            raise ValueError("expected dow_jones_index.data in source archive")

        with archive.open("dow_jones_index.data", "r") as raw_member:
            raw_lines = (line.decode("utf-8", errors="ignore") for line in raw_member)
            reader = csv.DictReader(raw_lines)
            for source_row in reader:
                try:
                    timestamp = _parse_date(source_row["date"])
                    close_price = _to_float(source_row["close"])
                    target_value, used_fallback = _row_target_percent_change(
                        source_row, close_price
                    )
                    if used_fallback:
                        fallback_targets += 1

                    prepared_row = {
                        "group_id": source_row["stock"].strip(),
                        "timestamp": timestamp,
                        "quarter": _to_float(source_row["quarter"]),
                        "open_price": _to_float(source_row["open"]),
                        "high_price": _to_float(source_row["high"]),
                        "low_price": _to_float(source_row["low"]),
                        "close_price": close_price,
                        "volume": _to_float(source_row["volume"]),
                        "percent_change_price": _to_float(source_row["percent_change_price"]),
                        "percent_change_volume_over_last_wk": _to_float(
                            source_row["percent_change_volume_over_last_wk"]
                        ),
                        "previous_weeks_volume": _to_float(source_row["previous_weeks_volume"]),
                        "days_to_next_dividend": _to_float(source_row["days_to_next_dividend"]),
                        "percent_return_next_dividend": _to_float(
                            source_row["percent_return_next_dividend"]
                        ),
                        "target_percent_change_next_weeks_price": target_value,
                    }
                except (KeyError, ValueError):
                    dropped += 1
                    continue

                rows.append(prepared_row)

    rows.sort(key=lambda row: (str(row["timestamp"]), str(row["group_id"])))
    return rows, dropped, fallback_targets


def _write_prepared_rows(prepared_path: Path, rows: list[dict[str, object]], max_rows: int) -> int:
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    limited_rows = rows[:max_rows] if max_rows > 0 else rows

    with prepared_path.open("w", encoding="utf-8", newline="") as prepared_file:
        writer = csv.DictWriter(prepared_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in limited_rows:
            writer.writerow(row)

    return len(limited_rows)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "dow_jones_financial",
        help="directory for raw/ and prepared/ benchmark outputs",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="maximum prepared rows to emit (0 means all prepared rows)",
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

    prepared_rows, dropped_rows, fallback_targets = _collect_prepared_rows(raw_path)
    kept_rows = _write_prepared_rows(prepared_path, prepared_rows, max_rows=args.max_rows)

    print(
        "[dow_jones_financial] prepared dataset written to "
        f"{prepared_path} (kept_rows={kept_rows}, dropped_rows={dropped_rows}, "
        f"fallback_targets={fallback_targets})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
