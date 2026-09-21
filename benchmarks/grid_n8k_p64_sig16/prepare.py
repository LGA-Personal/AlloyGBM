#!/usr/bin/env python3
"""Generate the grid_n8k_p64_sig16 controlled-grid cell."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _shared.synthetic_grid import write_grid_dataset  # noqa: E402

ROWS = 8000
FEATURE_COUNT = 64
SIGNAL_COUNT = 16
SEED = 20260920


def main() -> None:
    path = write_grid_dataset(
        "grid_n8k_p64_sig16",
        rows=ROWS,
        feature_count=FEATURE_COUNT,
        signal_count=SIGNAL_COUNT,
        seed=SEED,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
