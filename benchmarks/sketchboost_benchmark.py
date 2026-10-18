"""SketchBoost (``multiclass_sketch``) against unsketched vector leaves.

Scores many-class tasks (10+ classes) with ``multiclass_dev_suite.py``'s
splits, regimes and scoring: helena, volkert and micro-mass from the shared
dev-suite ``.npz`` cache, and mnist, krkopt, movement_libras, soybean and
led24 from PMLB. Arms are ``--config`` items as in ``multiclass_dev_suite.py``.

Usage::

    python benchmarks/sketchboost_benchmark.py run --npz DIR --cache PMLB_DIR \\
        --regime defaults --config base \\
        --config 'rp1=multiclass_sketch="random_projection",multiclass_sketch_dim=1' \\
        --output out.json
    python benchmarks/multiclass_dev_suite.py report out.json --baseline base
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multiclass_dev_suite as suite  # noqa: E402

NPZ_TASKS = {"helena", "volkert", "micro-mass"}
PMLB_TASKS = ["mnist", "krkopt", "movement_libras", "soybean", "led24"]


def load(npz_dir: Path, pmlb_dir: Path, only: list[str] | None):
    out = []
    for file in sorted(npz_dir.glob("*.npz")):
        z = np.load(file, allow_pickle=False)
        if str(z["name"]) not in NPZ_TASKS:
            continue
        X, y = suite._relabel(z["X"].astype(np.float64), z["y"].astype(float))
        out.append((f"dev_{int(z['task_id'])}_{str(z['name'])}", X, y))
    out += suite.load(pmlb_dir, PMLB_TASKS)
    if only:
        out = [d for d in out if any(name in d[0] for name in only)]
    return out


def run(args) -> None:
    arms = [suite.parse_config(text) for text in args.config or ["base"]]
    output = Path(args.output)
    records = json.loads(output.read_text())["records"] if output.exists() and args.resume else []
    done = {(r["dataset"], r["arm"], r["split"]) for r in records}
    for name, X, y in load(Path(args.npz), Path(args.cache), args.datasets):
        repeats = args.repeats if len(y) < 1000 else 1
        for split, tr, te in suite.splits(y, repeats):
            if args.max_splits is not None and split >= args.max_splits:
                continue
            for arm, params in arms:
                if (name, arm, split) in done:
                    continue
                try:
                    proba, fit_s, extra = suite.fit_predict(
                        "alloygbm", params, args.regime, X[tr], y[tr], X[te], split, args.threads
                    )
                    ll, err = suite.score(y[te], proba)
                    error = None
                except Exception as exc:  # noqa: BLE001 - record and continue
                    ll = err = fit_s = float("nan")
                    extra, error = {}, f"{type(exc).__name__}: {exc}"
                records.append({"dataset": name, "rows": int(len(y)), "features": int(X.shape[1]),
                                "classes": int(len(np.unique(y))), "arm": arm, "split": split,
                                "log_loss": ll, "error_rate": err, "fit_seconds": fit_s,
                                "error": error, **extra})
                output.write_text(json.dumps({"regime": args.regime, "records": records}))
        print(f"{name}: done ({len(y)} rows)", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--npz", required=True, help="dev_suite.py .npz cache")
    p.add_argument("--cache", required=True, help="PMLB cache (multiclass_dev_suite.py download)")
    p.add_argument("--regime", choices=("defaults", "matched"), default="defaults")
    p.add_argument("--config", action="append")
    p.add_argument("--datasets", nargs="*")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--max-splits", type=int)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
