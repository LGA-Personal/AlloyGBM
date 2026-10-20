#!/usr/bin/env python3
"""Prepare California Housing benchmark data via scikit-learn."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing

RAW_FILENAME = "california_housing.csv"
PREPARED_FILENAME = "prepared.csv"


def _load_frame() -> pd.DataFrame:
    bunch = fetch_california_housing(as_frame=True)
    frame = bunch.frame.copy()
    target_column = getattr(bunch, "target_names", ["target"])[0]
    if target_column not in frame.columns:
        frame["target"] = bunch.target
        target_column = "target"
    frame = frame.rename(columns={target_column: "median_house_value"})
    return frame


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "california_housing",
        help="directory for raw/ and prepared/ benchmark outputs",
    )
    args = parser.parse_args(argv)

    raw_dir = args.output_dir / "raw"
    prepared_dir = args.output_dir / "prepared"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / RAW_FILENAME
    prepared_path = prepared_dir / PREPARED_FILENAME

    frame = _load_frame()
    frame.to_csv(raw_path, index=False)
    frame.to_csv(prepared_path, index=False)
    print(f"[california_housing] prepared dataset written to {prepared_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
