#!/usr/bin/env python3
"""Parse and gate MorphBoost scanner and fit-time benchmark evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


MORPH_CASES = (
    "best_split_morph_16",
    "best_split_morph_64",
    "best_split_morph_255",
)
REQUIRED_SPEEDUP_CASES = MORPH_CASES[1:]
MIN_SAMPLES = 5
_ROW = re.compile(
    r"^(?P<name>[A-Za-z0-9_]+): total_ms=(?P<total>[0-9.eE+-]+) "
    r"iterations=(?P<iterations>[0-9]+) ns_per_iter=(?P<ns>[0-9.eE+-]+)$"
)


@dataclass(frozen=True)
class MorphPerfGate:
    passed: bool
    speedups: dict[str, float]
    regressions: dict[str, float]
    end_to_end_improvement: float
    reasons: tuple[str, ...]


def parse_benchmark_text(text: str) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {}
    for line in text.splitlines():
        stripped = line.strip()
        match = _ROW.fullmatch(stripped)
        if match is None:
            if stripped.startswith("best_split_"):
                raise ValueError(f"malformed benchmark row: {stripped}")
            continue
        value = float(match.group("ns"))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"malformed benchmark value: {stripped}")
        samples.setdefault(match.group("name"), []).append(value)
    return samples


def validate_samples(samples: dict[str, list[float]]) -> None:
    missing = [name for name in MORPH_CASES if name not in samples]
    if missing:
        raise ValueError(f"missing benchmark names: {', '.join(missing)}")
    too_short = [name for name in MORPH_CASES if len(samples[name]) < MIN_SAMPLES]
    if too_short:
        raise ValueError(
            f"at least {MIN_SAMPLES} samples required for: {', '.join(too_short)}"
        )


def aggregate_medians(samples: dict[str, list[float]]) -> dict[str, float]:
    return {name: statistics.median(values) for name, values in samples.items()}


def evaluate_perf_gate(
    baseline: dict[str, float],
    candidate: dict[str, float],
    *,
    end_to_end_improvement: float,
) -> MorphPerfGate:
    missing = [name for name in MORPH_CASES if name not in baseline or name not in candidate]
    if missing:
        raise ValueError(f"missing benchmark medians: {', '.join(missing)}")
    if not math.isfinite(end_to_end_improvement):
        raise ValueError("end-to-end improvement must be finite")

    speedups = {name: baseline[name] / candidate[name] for name in MORPH_CASES}
    regressions = {name: candidate[name] / baseline[name] - 1.0 for name in MORPH_CASES}
    reasons: list[str] = []
    scanner_target = all(speedups[name] >= 1.5 for name in REQUIRED_SPEEDUP_CASES)
    if not scanner_target and end_to_end_improvement < 0.15:
        reasons.append(
            "64/255-bin speedups missed 1.5x and end-to-end improvement missed 15%"
        )
    regressed = [name for name, value in regressions.items() if value > 0.05]
    if regressed:
        reasons.append(f"shape regression exceeded 5%: {', '.join(regressed)}")
    return MorphPerfGate(
        passed=not reasons,
        speedups=speedups,
        regressions=regressions,
        end_to_end_improvement=end_to_end_improvement,
        reasons=tuple(reasons),
    )


def fit_time_improvement(baseline_path: Path, candidate_path: Path) -> float:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

    def keyed(payload: dict[str, object]) -> dict[tuple[object, ...], float]:
        keyed_records: dict[tuple[object, ...], float] = {}
        for record in payload.get("records", []):
            if record["arm"] != "morph_current":
                continue
            key = (
                record["dataset"],
                record["task_family"],
                record["shape"],
                record["seed"],
                record["primary_metric"],
            )
            if key in keyed_records:
                raise ValueError(f"duplicate fit-time record: {key}")
            keyed_records[key] = float(record["fit_seconds"])
        return keyed_records

    baseline_records = keyed(baseline)
    candidate_records = keyed(candidate)
    if baseline_records.keys() != candidate_records.keys() or not baseline_records:
        raise ValueError("baseline and candidate fit records must form identical non-empty pairs")
    paired = [
        (baseline_records[key] - candidate_records[key]) / baseline_records[key]
        for key in baseline_records
    ]
    return statistics.median(paired)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-fit", type=Path, required=True)
    parser.add_argument("--candidate-fit", type=Path, required=True)
    args = parser.parse_args()

    baseline_samples = parse_benchmark_text(args.baseline.read_text(encoding="utf-8"))
    candidate_samples = parse_benchmark_text(args.candidate.read_text(encoding="utf-8"))
    validate_samples(baseline_samples)
    validate_samples(candidate_samples)
    fit_improvement = fit_time_improvement(args.baseline_fit, args.candidate_fit)
    gate = evaluate_perf_gate(
        aggregate_medians(baseline_samples),
        aggregate_medians(candidate_samples),
        end_to_end_improvement=fit_improvement,
    )
    print(json.dumps(asdict(gate), indent=2, sort_keys=True))
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
