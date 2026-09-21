#!/usr/bin/env python3
"""Generate the kgrid_n8k_p16_k10 class-count grid cell (calibration)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _shared.synthetic_grid import write_multiclass_grid_dataset  # noqa: E402


def main() -> None:
    path = write_multiclass_grid_dataset(
        "kgrid_n8k_p16_k10",
        rows=8000,
        feature_count=16,
        signal_count=8,
        class_count=10,
        seed=20260921,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
