"""Aggregate immutable benchmark repetitions and render provenance-rich reports."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .schema import (
    BenchmarkRecordV1,
    BenchmarkSummaryV1,
    METRIC_DIRECTIONS,
    SCHEMA_VERSION,
    load_records,
    validate_record,
    validate_summary,
)


def median_mad(values: Iterable[float]) -> tuple[float, float]:
    """Return the ordinary median and unscaled MAD."""

    data = [float(value) for value in values]
    if not data:
        raise ValueError("at least one value is required")
    median = statistics.median(data)
    mad = statistics.median(abs(value - median) for value in data)
    return float(median), float(mad)


def _canonical(value: object) -> str:
    if isinstance(value, Mapping):
        value = {str(key): json.loads(_canonical(child)) for key, child in value.items()}
    elif isinstance(value, (tuple, list)):
        value = [json.loads(_canonical(child)) for child in value]
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _group_key(record: BenchmarkRecordV1) -> tuple[object, ...]:
    return tuple(getattr(record, field) for field in (
        "run_id", "scenario", "task", "library", "library_version", "threads",
        "dataset_sha256", "input_representation", "metric_name",
    ))


def _attach_metadata(summary: BenchmarkSummaryV1, records: Sequence[BenchmarkRecordV1]) -> BenchmarkSummaryV1:
    # The v1 summary contract intentionally has no effective_params or machine
    # columns. Keep these provenance values available to gate evaluation while
    # leaving the serialized contract stable.
    object.__setattr__(summary, "_effective_params", _canonical(records[0].effective_params))
    object.__setattr__(summary, "_machine", _canonical(records[0].machine))
    object.__setattr__(summary, "_raw_lines", {item.repetition: index + 1 for index, item in enumerate(records)})
    return summary


def aggregate_records(
    records: Sequence[BenchmarkRecordV1], *, minimum_repetitions: int = 1
) -> list[BenchmarkSummaryV1]:
    """Validate and aggregate records by the schema's grouping keys."""

    if minimum_repetitions <= 0:
        raise ValueError("minimum_repetitions must be positive")
    groups: dict[tuple[object, ...], list[BenchmarkRecordV1]] = {}
    seen: set[tuple[object, ...]] = set()
    versions: dict[tuple[str, str], str] = {}
    for record in records:
        validate_record(record)
        version_key = (record.run_id, record.library)
        previous_version = versions.setdefault(version_key, record.library_version)
        if previous_version != record.library_version:
            raise ValueError(
                f"library_version mismatch for run_id={record.run_id!r}, library={record.library!r}"
            )
        key = (
            record.run_id, record.library, record.scenario, record.threads,
            record.repetition,
        )
        if key in seen:
            raise ValueError(f"duplicate raw repetition provenance: {key!r}")
        seen.add(key)
        groups.setdefault(_group_key(record), []).append(record)

    summaries: list[BenchmarkSummaryV1] = []
    for key, population in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        params = {_canonical(item.effective_params) for item in population}
        if len(params) != 1:
            raise ValueError(f"effective_params mismatch in summary group {key!r}")
        machines = {_canonical(item.machine) for item in population}
        if len(machines) != 1:
            raise ValueError(f"machine metadata mismatch in summary group {key!r}")
        if len({item.library_version for item in population}) != 1:
            raise ValueError(f"library_version mismatch in summary group {key!r}")
        metric_median, metric_mad = median_mad(item.metric_value for item in population)
        prep_median, prep_mad = median_mad(item.preprocessing_seconds for item in population)
        fit_median, fit_mad = median_mad(item.fit_seconds for item in population)
        pred_median, pred_mad = median_mad(item.predict_seconds for item in population)
        rss_median, rss_mad = median_mad(item.peak_rss_bytes for item in population)
        first = population[0]
        summary = BenchmarkSummaryV1(
            schema=SCHEMA_VERSION,
            run_id=first.run_id,
            scenario=first.scenario,
            task=first.task,
            library=first.library,
            library_version=first.library_version,
            threads=first.threads,
            dataset_sha256=first.dataset_sha256,
            input_representation=first.input_representation,
            metric_name=first.metric_name,
            metric_median=metric_median,
            metric_mad=metric_mad,
            preprocessing_median_seconds=prep_median,
            preprocessing_mad_seconds=prep_mad,
            fit_median_seconds=fit_median,
            fit_mad_seconds=fit_mad,
            predict_median_seconds=pred_median,
            predict_mad_seconds=pred_mad,
            peak_rss_median_bytes=rss_median,
            peak_rss_mad_bytes=rss_mad,
            raw_repetition_ids=tuple(sorted(item.repetition for item in population)),
            metric_direction=METRIC_DIRECTIONS[first.metric_name],
        )
        validate_summary(summary)
        _attach_metadata(summary, population)
        summaries.append(summary)
    return summaries


def summarize_file(path: str | Path, *, minimum_repetitions: int = 1) -> list[BenchmarkSummaryV1]:
    return aggregate_records(load_records(path), minimum_repetitions=minimum_repetitions)


# Descriptive alias used by callers that already have records in memory.
summarize_records = aggregate_records


def render_markdown(
    summaries: Sequence[BenchmarkSummaryV1], *, raw_path: str | Path | None = None, gate: object | None = None
) -> str:
    """Render summaries with links to every raw repetition behind each row."""

    raw_label = str(raw_path or "raw.jsonl")
    lines = [
        "# Competitiveness benchmark summary", "",
        "| Scenario | Task | Library | Threads | Metric (median ± MAD) | Fit seconds (median ± MAD) | Raw repetitions |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        line_by_repetition = getattr(item, "_raw_lines", {})
        raw_ids = ", ".join(
            f"[{rep}]({raw_label}#L{line_by_repetition.get(rep, rep + 1)})"
            for rep in item.raw_repetition_ids
        )
        lines.append(
            f"| {item.scenario} | {item.task} | {item.library} | {item.threads} | "
            f"{item.metric_median:.8g} ± {item.metric_mad:.8g} | "
            f"{item.fit_median_seconds:.8g} ± {item.fit_mad_seconds:.8g} | {raw_ids} |"
        )
    if gate is not None:
        value = gate.to_dict() if hasattr(gate, "to_dict") else gate
        lines.extend(["", "## Gate", "", f"**{value.get('claim', value.get('claim_type', 'comparison'))}: {value['status']}**", ""])
        for reason in value.get("reasons", ()):
            lines.append(f"- {reason}")
    return "\n".join(lines) + "\n"


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw")
    parser.add_argument("--baseline")
    parser.add_argument("--minimum-repetitions", type=int, default=5)
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    parser.add_argument("--claim", choices=("speed", "quality", "default-policy"))
    parser.add_argument("--quality-first", action="store_true")
    args = parser.parse_args(argv)
    current = summarize_file(args.raw, minimum_repetitions=1)
    baseline = summarize_file(args.baseline, minimum_repetitions=1) if args.baseline else []
    from .gates import evaluate_claim
    gate = evaluate_claim(
        args.claim, current, baseline, minimum_repetitions=args.minimum_repetitions,
        quality_first=args.quality_first,
    ) if args.claim else None
    status = gate.status if gate is not None else "insufficient-data" if not args.baseline else "pass"
    payload: dict[str, object] = {
        "schema": SCHEMA_VERSION,
        "status": status,
        "claim": args.claim,
        "summaries": [item.to_dict() for item in current],
    }
    if gate is not None:
        payload["gate"] = gate.to_dict()
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True, indent=2) + "\n"
    if args.json_output:
        Path(args.json_output).write_text(encoded)
    else:
        print(encoded, end="")
    markdown = render_markdown(current, raw_path=args.raw, gate=gate)
    if args.markdown_output:
        Path(args.markdown_output).write_text(markdown)
    else:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
