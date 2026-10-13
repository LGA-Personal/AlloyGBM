"""Paired ablation summary for AlloyGBM's special modes.

Reads one or more ``frontier_matrix.py`` result files and compares every
AlloyGBM arm with a baseline arm, cell by cell, paired by seed (or official
fold). For each arm it reports, overall and per task:

* the geometric mean of the per-cell loss ratio (candidate / baseline);
* win / tie / loss counts, where a cell is a tie when the mean log-ratio is
  inside two seed standard errors or under 0.5% (the frontier-matrix rule);
* the geometric mean fit-time ratio, so accuracy and cost are read together.

With ``--peers`` it also reports each arm's loss against the best peer
library in each cell (the frontier matrix's headline number), taking the
peer records from that file; use it to place a mode next to the published
pilot without refitting the peers.

Usage::

    python benchmarks/special_modes.py RESULT.json [MORE.json] --baseline alloy \\
        [--peers PILOT.json] [--output report.md]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

PEER_LIBRARIES = ("lightgbm", "xgboost", "catboost")
TIE_RELATIVE = 0.005


def _records(paths: Sequence[Path]) -> list[dict]:
    out = []
    for p in paths:
        out += [r for r in json.loads(Path(p).read_text())["records"]]
    return out


def _index(records: Sequence[dict]) -> dict[str, dict[str, dict[int, dict]]]:
    """cell -> arm -> seed -> record (error-free records only)."""
    idx: dict = defaultdict(lambda: defaultdict(dict))
    for r in records:
        if r.get("error") is None and r.get("loss") is not None and math.isfinite(r["loss"]):
            idx[r["dataset"]][r["arm"]][int(r["seed"])] = r
    return idx


def _outcome(log_ratios: list[float]) -> str:
    mean = statistics.fmean(log_ratios)
    se = statistics.stdev(log_ratios) / math.sqrt(len(log_ratios)) if len(log_ratios) > 1 else 0.0
    if abs(mean) < math.log1p(TIE_RELATIVE) or abs(mean) <= 2 * se:
        return "tie"
    return "win" if mean < 0 else "loss"


def compare_arms(records: Sequence[dict], baseline: str, peers: Sequence[dict] = ()) -> dict:
    idx = _index(records)
    peer_idx = _index(peers) if peers else idx
    arms = sorted({a for cell in idx.values() for a in cell} - {baseline}
                  - {a for cell in idx.values() for a, recs in cell.items()
                     if next(iter(recs.values()))["library"] in PEER_LIBRARIES})
    out: dict = {}
    for arm in arms:
        rows = []
        for cell, by_arm in idx.items():
            base, cand = by_arm.get(baseline), by_arm.get(arm)
            if not base or not cand:
                continue
            seeds = sorted(set(base) & set(cand))
            if not seeds:
                continue
            lr = [math.log(max(cand[s]["loss"], 1e-12) / max(base[s]["loss"], 1e-12)) for s in seeds]
            tr = [math.log(max(cand[s]["fit_seconds"], 1e-6) / max(base[s]["fit_seconds"], 1e-6))
                  for s in seeds]
            peer_best = None
            pcell = peer_idx.get(cell, {})
            peer_means = [statistics.fmean(rr["loss"] for rr in recs.values())
                          for a, recs in pcell.items()
                          if recs and next(iter(recs.values()))["library"] in PEER_LIBRARIES]
            if peer_means:
                peer_best = min(peer_means)
            cand_mean = statistics.fmean(cand[s]["loss"] for s in seeds)
            base_mean = statistics.fmean(base[s]["loss"] for s in seeds)
            rows.append({
                "cell": cell,
                "task": next(iter(cand.values()))["task"],
                "log_ratio": statistics.fmean(lr),
                "time_log_ratio": statistics.fmean(tr),
                "outcome": _outcome(lr),
                "vs_peer": (cand_mean / peer_best) if peer_best else None,
                "base_vs_peer": (base_mean / peer_best) if peer_best else None,
            })
        out[arm] = rows
    return out


def _summary(rows: list[dict]) -> dict:
    if not rows:
        return {}
    w = sum(r["outcome"] == "win" for r in rows)
    t = sum(r["outcome"] == "tie" for r in rows)
    l_ = sum(r["outcome"] == "loss" for r in rows)
    vp = [r["vs_peer"] for r in rows if r["vs_peer"]]
    bp = [r["base_vs_peer"] for r in rows if r["base_vs_peer"]]
    return {
        "cells": len(rows),
        "loss_ratio": math.exp(statistics.fmean(r["log_ratio"] for r in rows)),
        "wtl": f"{w}/{t}/{l_}",
        "time_ratio": math.exp(statistics.fmean(r["time_log_ratio"] for r in rows)),
        "vs_peer": math.exp(statistics.fmean(math.log(v) for v in vp)) if vp else None,
        "base_vs_peer": math.exp(statistics.fmean(math.log(v) for v in bp)) if bp else None,
    }


def render(cmp: dict, baseline: str, worst: int = 5) -> str:
    lines = [f"Loss and fit-time ratios are arm / `{baseline}` (below 1.0 = arm better or faster), "
             "geometric means over cells. W/T/L is per cell, paired by seed.", "",
             "| Arm | Slice | Cells | Loss ratio | W/T/L | Fit-time ratio | Arm vs best peer | "
             "Baseline vs best peer |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda v: "" if v is None else f"{v:.3f}"  # noqa: E731
    for arm, rows in cmp.items():
        slices = [("all", rows)] + [(t, [r for r in rows if r["task"] == t])
                                    for t in sorted({r["task"] for r in rows})]
        for name, sub in slices:
            s = _summary(sub)
            if not s:
                continue
            lines.append(f"| `{arm}` | {name} | {s['cells']} | {s['loss_ratio']:.3f} | {s['wtl']} | "
                         f"{s['time_ratio']:.2f}x | {fmt(s['vs_peer'])} | {fmt(s['base_vs_peer'])} |")
    lines.append("")
    for arm, rows in cmp.items():
        ranked = sorted(rows, key=lambda r: r["log_ratio"])
        best = ", ".join(f"{r['cell'].split('|')[0]} {math.exp(r['log_ratio']):.3f}"
                         for r in ranked[:worst])
        worst_ = ", ".join(f"{r['cell'].split('|')[0]} {math.exp(r['log_ratio']):.3f}"
                           for r in ranked[::-1][:worst])
        lines.append(f"- `{arm}` best: {best}")
        lines.append(f"- `{arm}` worst: {worst_}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--baseline", default="alloy")
    parser.add_argument("--peers", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", type=Path, help="write per-cell rows here")
    args = parser.parse_args(argv)
    cmp = compare_arms(_records(args.files), args.baseline, _records(args.peers))
    text = render(cmp, args.baseline)
    if args.output:
        args.output.write_text(text)
    if args.json:
        args.json.write_text(json.dumps(cmp, indent=1))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
