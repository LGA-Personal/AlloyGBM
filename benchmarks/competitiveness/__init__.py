"""Contracts and manifests for reproducible AlloyGBM competitiveness runs."""

from .schema import (
    METRIC_DIRECTIONS,
    PROFILE_STAGE_LABELS,
    SCHEMA_VERSION,
    TREE_STAGE_LABELS,
    BenchmarkRecordV1,
    BenchmarkSummaryV1,
    ProfileRecordV1,
    load_records,
    validate_profile,
    validate_record,
    validate_summary,
)

__all__ = [
    "METRIC_DIRECTIONS",
    "PROFILE_STAGE_LABELS",
    "SCHEMA_VERSION",
    "TREE_STAGE_LABELS",
    "BenchmarkRecordV1",
    "BenchmarkSummaryV1",
    "ProfileRecordV1",
    "load_records",
    "validate_profile",
    "validate_record",
    "validate_summary",
]
