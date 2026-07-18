"""CLI orchestrator for the deferred-architecture benchmark pack."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .common import (
    SCHEMA_VERSION,
    BenchmarkReport,
    CaseResult,
    compare_reports,
    environment_manifest,
    quality_gates,
    render_markdown,
    validate_report,
)
from .scenarios import SCENARIO_CASES, run_case


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "full"), required=True)
    parser.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    parser.add_argument(
        "--scenario", action="append", choices=tuple(SCENARIO_CASES)
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--repetition", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def _worker(args: argparse.Namespace) -> int:
    if len(args.scenario or ()) != 1 or not args.case:
        raise ValueError("worker requires exactly one --scenario and one --case")
    result = run_case(
        scenario=args.scenario[0],
        case=args.case,
        profile=args.profile,
        mode=args.mode,
        repetition=args.repetition,
        seed=args.seed,
    )
    print(json.dumps(result.__dict__, sort_keys=True))
    return 0


def _run_worker(
    *,
    scenario: str,
    case: str,
    profile: str,
    mode: str,
    repetition: int,
    seed: int,
) -> CaseResult:
    env = os.environ.copy()
    if scenario == "node_parallelism":
        env["RAYON_NUM_THREADS"] = "1" if case == "threads_1" else "8"
    command = [
        sys.executable,
        "-m",
        "benchmarks.architectural_backlog.run",
        "--worker",
        "--profile",
        profile,
        "--mode",
        mode,
        "--scenario",
        scenario,
        "--case",
        case,
        "--repetition",
        str(repetition),
        "--seed",
        str(seed),
    ]
    completed = subprocess.run(
        command, env=env, check=True, capture_output=True, text=True
    )
    return CaseResult.from_dict(json.loads(completed.stdout))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker:
        return _worker(args)
    if args.baseline and args.mode != "candidate":
        raise ValueError("--baseline is only valid with --mode candidate")

    selected = args.scenario or list(SCENARIO_CASES)
    repetitions = 1 if args.profile == "quick" else 3
    results = []
    for repetition in range(repetitions):
        for scenario in selected:
            for case in SCENARIO_CASES[scenario]:
                results.append(
                    _run_worker(
                        scenario=scenario,
                        case=case,
                        profile=args.profile,
                        mode=args.mode,
                        repetition=repetition,
                        seed=args.seed + repetition,
                    )
                )
    report = BenchmarkReport(
        schema_version=SCHEMA_VERSION,
        profile=args.profile,
        mode=args.mode,
        environment=environment_manifest(),
        results=tuple(results),
    )
    validate_report(report)
    if args.baseline:
        baseline = BenchmarkReport.from_json(args.baseline.read_text(encoding="utf-8"))
        gates = compare_reports(baseline, report)
    else:
        gates = quality_gates(report)
    markdown = render_markdown(report, gates)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report.to_json(), encoding="utf-8")
        args.output.with_suffix(".md").write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")
    return 1 if args.gate and any(not gate.passed for gate in gates) else 0


if __name__ == "__main__":
    raise SystemExit(main())
