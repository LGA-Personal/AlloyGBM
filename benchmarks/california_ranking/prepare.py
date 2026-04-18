#!/usr/bin/env python3
"""Prepare California Housing ranking benchmark data via scikit-learn.

The dataset is reframed as a learning-to-rank task: 1-degree lat/lon grid
cells act as query groups, and each house is ranked within its cell by
median_house_value bucketed into 5 graded relevance levels (0–4).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

PREPARED_FILENAME = "prepared.csv"
DEFAULT_GRID_DEGREES = 1.0
N_RELEVANCE_LEVELS = 5
MIN_DOCS_PER_QUERY = 10

FEATURE_COLS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def _build_ranking_frame(grid_degrees: float) -> pd.DataFrame:
    """Load California Housing and convert to a ranking frame."""
    bunch = fetch_california_housing(as_frame=True)
    frame = bunch.frame.copy()

    # Normalise target column name.
    target_col = getattr(bunch, "target_names", ["target"])[0]
    if target_col not in frame.columns:
        frame["target"] = bunch.target
        target_col = "target"
    frame = frame.rename(columns={target_col: "median_house_value"})

    # Build geographic grid cells as query groups.
    lat_cell = np.floor(frame["Latitude"] / grid_degrees).astype(int)
    lon_cell = np.floor(frame["Longitude"] / grid_degrees).astype(int)
    lat_min = int(lat_cell.min())
    lon_min = int(lon_cell.min())
    lon_range = int(lon_cell.max() - lon_cell.min() + 1)
    raw_query_id = (lat_cell - lat_min) * lon_range + (lon_cell - lon_min)
    frame["query_id"] = raw_query_id

    # Drop queries that are too small to be useful for ranking.
    group_sizes = frame.groupby("query_id").size()
    keep_ids = group_sizes[group_sizes >= MIN_DOCS_PER_QUERY].index
    frame = frame[frame["query_id"].isin(keep_ids)].copy()

    # Re-index query IDs to contiguous integers starting at 0.
    sorted_ids = sorted(frame["query_id"].unique())
    id_remap = {old: new for new, old in enumerate(sorted_ids)}
    frame["query_id"] = frame["query_id"].map(id_remap)

    # Bucket target into graded relevance levels using quantiles.
    frame["relevance"] = pd.qcut(
        frame["median_house_value"],
        q=N_RELEVANCE_LEVELS,
        labels=list(range(N_RELEVANCE_LEVELS)),
        duplicates="drop",
    ).astype(int)

    out = frame[["query_id"] + FEATURE_COLS + ["relevance"]].reset_index(drop=True)
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "california_ranking",
        help="directory for prepared benchmark outputs",
    )
    parser.add_argument(
        "--grid-degrees",
        type=float,
        default=DEFAULT_GRID_DEGREES,
        help="size of geographic grid cells in degrees (default 1.0)",
    )
    args = parser.parse_args(argv)

    prepared_dir = args.output_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = prepared_dir / PREPARED_FILENAME

    frame = _build_ranking_frame(args.grid_degrees)
    frame.to_csv(prepared_path, index=False)

    n_queries = int(frame["query_id"].nunique())
    avg_docs = len(frame) / n_queries if n_queries > 0 else 0.0
    print(
        f"[california_ranking] prepared dataset written to {prepared_path} "
        f"(queries={n_queries}, avg_docs_per_query={avg_docs:.1f}, "
        f"total_rows={len(frame)}, relevance_levels={N_RELEVANCE_LEVELS})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
