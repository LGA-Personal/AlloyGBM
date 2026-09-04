"""Contracts and manifests for reproducible AlloyGBM competitiveness runs."""

from .schema import (
    INPUT_REPRESENTATIONS,
    METRIC_DIRECTIONS,
    PROFILE_STAGE_LABELS,
    SCHEMA_VERSION,
    TREE_STAGE_LABELS,
    BenchmarkRecordV1,
    BenchmarkSummaryV1,
    ProfileRecordV1,
    RunMetadataV1,
    load_records,
    load_run_metadata,
    validate_profile,
    validate_record,
    validate_summary,
)
__all__ = [
    "METRIC_DIRECTIONS",
    "INPUT_REPRESENTATIONS",
    "PROFILE_STAGE_LABELS",
    "SCHEMA_VERSION",
    "TREE_STAGE_LABELS",
    "BenchmarkRecordV1",
    "BenchmarkSummaryV1",
    "ProfileRecordV1",
    "RunMetadataV1",
    "load_records",
    "load_run_metadata",
    "validate_profile",
    "validate_record",
    "validate_summary",
    "GateResult",
    "aggregate_records",
    "median_mad",
    "render_markdown",
    "summarize_file",
    "summarize_records",
]


def __getattr__(name: str):
    """Load Task 4 helpers lazily so ``python -m ...summarize`` stays quiet."""

    if name == "GateResult":
        from .gates import GateResult
        return GateResult
    if name in {"aggregate_records", "median_mad", "render_markdown", "summarize_file", "summarize_records"}:
        from . import summarize
        return getattr(summarize, name)
    raise AttributeError(name)
