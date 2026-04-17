#!/usr/bin/env python3
"""
Prepare a learning-to-rank benchmark dataset.

PLACEHOLDER — this script is not yet implemented. To use:

1. Choose a dataset source:

   Option A — UCI Online News Popularity (tabular, multivariate, regression/ranking):
     - URL: https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip
     - 39644 rows, 58 features
     - Groups: binary `data_channel_is_*` columns (news topic channel), assign group_id by channel
     - Target (relevance): bin `shares` into 5 levels (0-4) using quintile thresholds
     - SCENARIO_NAME = "news_ranking"
     - TARGET_COLUMN = "relevance"  # binned shares
     - GROUP_COLUMN = "query_id"    # assigned channel ID

   Option B — MSLR-WEB10K (tabular, multivariate, ranking):
     - Requires free registration: https://www.microsoft.com/en-us/research/project/mslr/
     - Standard LTR benchmark, 5-level relevance, query-document feature vectors
     - SCENARIO_NAME = "mslr_ranking"
     - TARGET_COLUMN = "relevance"
     - GROUP_COLUMN = "query_id"

2. Update manifest.yaml with the actual URL, target, and group_column.

3. Implement the download + normalization logic following the same pattern as
   benchmarks/bike_sharing/prepare.py (UCI CSV/ZIP download) or
   benchmarks/synthetic_ranking/prepare.py (CSV output format).

4. Required output columns: [GROUP_COLUMN, ...feature_cols..., TARGET_COLUMN]
   - GROUP_COLUMN: integer query/group IDs, contiguous and sorted (required by AlloyGBM ranker)
   - TARGET_COLUMN: integer relevance labels 0-4 (or 0-N for your dataset)

5. Add "news_ranking" (or your chosen SCENARIO_NAME) to AVAILABLE_SCENARIOS in
   benchmarks/run_model_comparison.py.
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    print(
        "[news_ranking] PLACEHOLDER — prepare.py not yet implemented.\n"
        "See module docstring for instructions on choosing and wiring a ranking dataset."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
