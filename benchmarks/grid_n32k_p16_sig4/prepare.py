#!/usr/bin/env python3
"""Generate the grid_n32k_p16_sig4 controlled-grid cell."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _shared.synthetic_grid import write_grid_dataset  # noqa: E402

ROWS = 32000
FEATURE_COUNT = 16
SIGNAL_COUNT = 4
SEED = 20260920


def main() -> None:
    path = write_grid_dataset(
        "grid_n32k_p16_sig4",
        rows=ROWS,
        feature_count=FEATURE_COUNT,
        signal_count=SIGNAL_COUNT,
        seed=SEED,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
