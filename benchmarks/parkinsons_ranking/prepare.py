#!/usr/bin/env python3
"""Prepare a real learning-to-rank benchmark from UCI Parkinsons Telemonitoring.

Query groups are the 42 patients (`subject#`) -- a grouping that exists in the
data rather than one imposed on it, which is why this is worth adding alongside
`california_ranking`, whose groups are geographic cells constructed for the
purpose. Relevance is `total_UPDRS` bucketed into 5 graded levels.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

RAW_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "parkinsons/telemonitoring/parkinsons_updrs.data"
)
RAW_FILENAME = "parkinsons_updrs.data"
PREPARED_FILENAME = "prepared.csv"
SCENARIO = "parkinsons_ranking"
N_RELEVANCE_LEVELS = 5
MIN_DOCS_PER_QUERY = 10
# Dropped so the label cannot be read off a feature: motor_UPDRS is the other
# half of the same clinical score, and subject#/test_time identify the row.
LEAK_COLUMNS = ["subject#", "motor_UPDRS", "total_UPDRS", "test_time"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    raw_path = root / "benchmarks" / "data" / SCENARIO / "raw" / RAW_FILENAME
    prepared_path = root / "benchmarks" / "data" / SCENARIO / "prepared" / PREPARED_FILENAME
    if not raw_path.exists() or args.force_download:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {RAW_URL}", flush=True)
        with urllib.request.urlopen(RAW_URL, timeout=120) as response:
            raw_path.write_bytes(response.read())
    frame = pd.read_csv(raw_path)
    frame.columns = [str(c).strip() for c in frame.columns]

    sizes = frame.groupby("subject#").size()
    keep = sizes[sizes >= MIN_DOCS_PER_QUERY].index
    frame = frame[frame["subject#"].isin(keep)].copy()

    remap = {old: new for new, old in enumerate(sorted(frame["subject#"].unique()))}
    frame["query_id"] = frame["subject#"].map(remap)
    frame["relevance"] = pd.qcut(
        frame["total_UPDRS"].rank(method="first"),
        N_RELEVANCE_LEVELS,
        labels=list(range(N_RELEVANCE_LEVELS)),
    ).astype(int)

    feature_cols = [c for c in frame.columns
                    if c not in LEAK_COLUMNS + ["query_id", "relevance"]]
    out = frame[["query_id"] + feature_cols + ["relevance"]].sort_values(
        "query_id", kind="mergesort"
    ).reset_index(drop=True)
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(prepared_path, index=False)
    print(
        f"wrote {prepared_path} (rows={len(out)}, queries={out['query_id'].nunique()}, "
        f"features={len(feature_cols)})"
    )


if __name__ == "__main__":
    main()
