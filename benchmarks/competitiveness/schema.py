"""Versioned records emitted by the competitiveness benchmark suite.

The schema deliberately keeps raw repetitions as first-class records.  A
later summarizer may calculate medians and MADs, but it must retain the IDs of
the observations used for that calculation.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

SCHEMA_VERSION = "alloygbm-competitiveness/v1"
SchemaVersion = Literal["alloygbm-competitiveness/v1"]

METRIC_DIRECTIONS: dict[str, Literal["minimize", "maximize"]] = {
    "rmse": "minimize",
    "mae": "minimize",
    "log_loss": "minimize",
    "error_rate": "minimize",
    "r2": "maximize",
    "accuracy": "maximize",
    "roc_auc": "maximize",
    "ndcg_at_10": "maximize",
}
INPUT_REPRESENTATIONS = frozenset({"dense", "native_categorical", "csr", "csc", "dense_fallback"})


def harness_tree_sha256(root: str | Path) -> str:
    """Hash the loaded top-level benchmark harness source deterministically.

    Only ``*.py`` files directly under ``root`` participate. Each canonical
    relative filename and its exact bytes are length-delimited in filename
    order, so edits to tracked or untracked harness source cannot hide behind
    an unchanged git commit.
    """

    source_root = Path(root)
    paths = sorted(
        path for path in source_root.glob("*.py") if path.is_file()
    )
    digest = hashlib.sha256()
    for path in paths:
        relative_name = path.relative_to(source_root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative_name).to_bytes(8, "big"))
        digest.update(relative_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()

# Keep these in lockstep with Stage::label and report_tree_stages in
# crates/engine/src/profiling.rs. Structured v1 profiles always include every
# key, including stages with zero observations.
PROFILE_STAGE_LABELS = frozenset(
    {
        "gradients",
        "row_sampling",
        "feature_tiles",
        "prediction_copy",
        "tree_build",
        "prediction_update",
        "loss",
        "validation",
    }
)
TREE_STAGE_LABELS = frozenset({"histogram_build", "split_find", "partition"})


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _positive_int(value: object, name: str) -> None:
    if not _is_int(value) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(value: object, name: str) -> None:
    if not _is_int(value) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _nonempty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _validate_machine_metadata(value: object, name: str, *, allow_none: bool = False) -> None:
    if allow_none and value is None:
        return
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a nonempty mapping")
    if not all(
        isinstance(key, str) and bool(key.strip())
        and isinstance(item, str) and bool(item.strip())
        for key, item in value.items()
    ):
        raise ValueError(f"{name} keys and values must be nonempty strings")
    hostname = value.get("hostname")
    if not isinstance(hostname, str) or not hostname.strip():
        raise ValueError(f"{name} must contain a nonempty hostname")


def _deep_freeze(value: object) -> object:
    """Copy JSON-like containers into immutable equivalents."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(child) for key, child in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(child) for child in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(child) for child in value)
    return value


def _deep_thaw(value: object) -> object:
    """Convert frozen containers back to JSON-compatible containers."""

    if isinstance(value, Mapping):
        return {key: _deep_thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(child) for child in value]
    if isinstance(value, frozenset):
        return [_deep_thaw(child) for child in value]
    return value


def _validate_ns_map(
    values: Mapping[str, object], name: str, labels: frozenset[str]
) -> None:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    actual = set(values)
    if actual != labels:
        missing = sorted(labels - actual)
        unknown = sorted(actual - labels)
        raise ValueError(
            f"{name} must contain exactly the stage labels; missing={missing}, unknown={unknown}"
        )
    for label, duration in values.items():
        _nonnegative_int(duration, f"{name}[{label!r}]")


@dataclass(frozen=True)
class ProfileRecordV1:
    """Structured, process-sequential profile data for one fit."""

    rows: int
    features: int
    rounds: int
    threads: int
    loop_wall_ns: int
    untimed_ns: int
    stage_ns: dict[str, int]
    tree_stage_ns: dict[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_ns", _deep_freeze(self.stage_ns))
        object.__setattr__(self, "tree_stage_ns", _deep_freeze(self.tree_stage_ns))

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "features": self.features,
            "rounds": self.rounds,
            "threads": self.threads,
            "loop_wall_ns": self.loop_wall_ns,
            "untimed_ns": self.untimed_ns,
            "stage_ns": dict(self.stage_ns),
            "tree_stage_ns": dict(self.tree_stage_ns),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ProfileRecordV1:
        return cls(
            rows=value["rows"],  # type: ignore[arg-type]
            features=value["features"],  # type: ignore[arg-type]
            rounds=value["rounds"],  # type: ignore[arg-type]
            threads=value["threads"],  # type: ignore[arg-type]
            loop_wall_ns=value["loop_wall_ns"],  # type: ignore[arg-type]
            untimed_ns=value["untimed_ns"],  # type: ignore[arg-type]
            stage_ns=dict(value["stage_ns"]),  # type: ignore[arg-type]
            tree_stage_ns=dict(value["tree_stage_ns"]),  # type: ignore[arg-type]
        )

    @classmethod
    def from_json(cls, value: str) -> ProfileRecordV1:
        decoded = json.loads(value)
        if not isinstance(decoded, Mapping):
            raise ValueError("profile JSON must contain an object")
        profile = cls.from_dict(decoded)
        validate_profile(profile)
        return profile


@dataclass(frozen=True)
class BenchmarkRecordV1:
    """One immutable timed repetition and its quality measurement."""

    schema: SchemaVersion
    run_id: str
    repetition: int
    dataset_sha256: str
    scenario: str
    task: str
    library: str
    library_version: str
    git_sha: str | None
    seed: int
    threads: int
    effective_params: dict[str, object]
    input_representation: str
    preprocessing_seconds: float
    fit_seconds: float
    predict_seconds: float
    peak_rss_bytes: int
    metric_name: str
    metric_value: float
    rounds_completed: int
    machine: dict[str, str]
    profile: ProfileRecordV1 | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "effective_params", _deep_freeze(self.effective_params))
        object.__setattr__(self, "machine", _deep_freeze(self.machine))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "repetition": self.repetition,
            "dataset_sha256": self.dataset_sha256,
            "scenario": self.scenario,
            "task": self.task,
            "library": self.library,
            "library_version": self.library_version,
            "git_sha": self.git_sha,
            "seed": self.seed,
            "threads": self.threads,
            "effective_params": _deep_thaw(self.effective_params),
            "input_representation": self.input_representation,
            "preprocessing_seconds": self.preprocessing_seconds,
            "fit_seconds": self.fit_seconds,
            "predict_seconds": self.predict_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "rounds_completed": self.rounds_completed,
            "machine": _deep_thaw(self.machine),
            "profile": self.profile.to_dict() if self.profile is not None else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> BenchmarkRecordV1:
        profile_value = value.get("profile")
        if profile_value is not None and not isinstance(profile_value, Mapping):
            raise ValueError("profile must be an object or null")
        return cls(
            schema=value["schema"],  # type: ignore[arg-type]
            run_id=value["run_id"],  # type: ignore[arg-type]
            repetition=value["repetition"],  # type: ignore[arg-type]
            dataset_sha256=value["dataset_sha256"],  # type: ignore[arg-type]
            scenario=value["scenario"],  # type: ignore[arg-type]
            task=value["task"],  # type: ignore[arg-type]
            library=value["library"],  # type: ignore[arg-type]
            library_version=value["library_version"],  # type: ignore[arg-type]
            git_sha=value.get("git_sha"),  # type: ignore[arg-type]
            seed=value["seed"],  # type: ignore[arg-type]
            threads=value["threads"],  # type: ignore[arg-type]
            effective_params=dict(value["effective_params"]),  # type: ignore[arg-type]
            input_representation=value["input_representation"],  # type: ignore[arg-type]
            preprocessing_seconds=value["preprocessing_seconds"],  # type: ignore[arg-type]
            fit_seconds=value["fit_seconds"],  # type: ignore[arg-type]
            predict_seconds=value["predict_seconds"],  # type: ignore[arg-type]
            peak_rss_bytes=value["peak_rss_bytes"],  # type: ignore[arg-type]
            metric_name=value["metric_name"],  # type: ignore[arg-type]
            metric_value=value["metric_value"],  # type: ignore[arg-type]
            rounds_completed=value["rounds_completed"],  # type: ignore[arg-type]
            machine=dict(value["machine"]),  # type: ignore[arg-type]
            profile=(
                ProfileRecordV1.from_dict(profile_value)
                if profile_value is not None
                else None
            ),
        )

    @classmethod
    def from_json(cls, value: str) -> BenchmarkRecordV1:
        decoded = json.loads(value)
        if not isinstance(decoded, Mapping):
            raise ValueError("benchmark record JSON must contain an object")
        record = cls.from_dict(decoded)
        validate_record(record)
        return record


@dataclass(frozen=True)
class RunMetadataV1:
    """Durable provenance for one raw benchmark run."""

    schema: SchemaVersion
    run_id: str
    measured_git_sha: str | None
    git_sha_semantics: str
    harness_git_sha: str | None
    harness_tree_sha256: str
    harness_source_path: str
    manifest_sha256: str
    manifest_identifier: str
    manifest_path: str
    libraries: tuple[str, ...]
    scenarios: tuple[str, ...]
    seed: int
    threads: int
    repetitions: int
    warmups: int
    smoke: bool
    profile_alloy: bool
    raw_sha256: str
    raw_record_count: int
    created_at_utc: str
    working_directory: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "libraries", tuple(self.libraries))
        object.__setattr__(self, "scenarios", tuple(self.scenarios))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "measured_git_sha": self.measured_git_sha,
            "git_sha_semantics": self.git_sha_semantics,
            "harness_git_sha": self.harness_git_sha,
            "harness_tree_sha256": self.harness_tree_sha256,
            "harness_source_path": self.harness_source_path,
            "manifest_sha256": self.manifest_sha256,
            "manifest_identifier": self.manifest_identifier,
            "manifest_path": self.manifest_path,
            "libraries": list(self.libraries),
            "scenarios": list(self.scenarios),
            "seed": self.seed,
            "threads": self.threads,
            "repetitions": self.repetitions,
            "warmups": self.warmups,
            "smoke": self.smoke,
            "profile_alloy": self.profile_alloy,
            "raw_sha256": self.raw_sha256,
            "raw_record_count": self.raw_record_count,
            "created_at_utc": self.created_at_utc,
            "working_directory": self.working_directory,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RunMetadataV1:
        libraries_value = value["libraries"]
        scenarios_value = value["scenarios"]
        if not isinstance(libraries_value, Sequence) or isinstance(libraries_value, (str, bytes)):
            raise ValueError("libraries must be a sequence")
        if not isinstance(scenarios_value, Sequence) or isinstance(scenarios_value, (str, bytes)):
            raise ValueError("scenarios must be a sequence")
        metadata = cls(
            schema=value["schema"],  # type: ignore[arg-type]
            run_id=value["run_id"],  # type: ignore[arg-type]
            measured_git_sha=value.get("measured_git_sha"),  # type: ignore[arg-type]
            git_sha_semantics=value["git_sha_semantics"],  # type: ignore[arg-type]
            harness_git_sha=value.get("harness_git_sha"),  # type: ignore[arg-type]
            harness_tree_sha256=value["harness_tree_sha256"],  # type: ignore[arg-type]
            harness_source_path=value["harness_source_path"],  # type: ignore[arg-type]
            manifest_sha256=value["manifest_sha256"],  # type: ignore[arg-type]
            manifest_identifier=value["manifest_identifier"],  # type: ignore[arg-type]
            manifest_path=value["manifest_path"],  # type: ignore[arg-type]
            libraries=tuple(libraries_value),  # type: ignore[arg-type]
            scenarios=tuple(scenarios_value),  # type: ignore[arg-type]
            seed=value["seed"],  # type: ignore[arg-type]
            threads=value["threads"],  # type: ignore[arg-type]
            repetitions=value["repetitions"],  # type: ignore[arg-type]
            warmups=value["warmups"],  # type: ignore[arg-type]
            smoke=value["smoke"],  # type: ignore[arg-type]
            profile_alloy=value["profile_alloy"],  # type: ignore[arg-type]
            raw_sha256=value["raw_sha256"],  # type: ignore[arg-type]
            raw_record_count=value["raw_record_count"],  # type: ignore[arg-type]
            created_at_utc=value["created_at_utc"],  # type: ignore[arg-type]
            working_directory=value["working_directory"],  # type: ignore[arg-type]
        )
        validate_run_metadata(metadata)
        return metadata

    @classmethod
    def from_json(cls, value: str) -> RunMetadataV1:
        decoded = json.loads(value)
        if not isinstance(decoded, Mapping):
            raise ValueError("run metadata JSON must contain an object")
        return cls.from_dict(decoded)


@dataclass(frozen=True)
class BenchmarkSummaryV1:
    """Median/MAD aggregate retaining the raw repetitions behind it."""

    schema: SchemaVersion
    run_id: str
    scenario: str
    task: str
    library: str
    library_version: str
    threads: int
    dataset_sha256: str
    input_representation: str
    metric_name: str
    metric_median: float
    metric_mad: float
    preprocessing_median_seconds: float
    preprocessing_mad_seconds: float
    fit_median_seconds: float
    fit_mad_seconds: float
    predict_median_seconds: float
    predict_mad_seconds: float
    peak_rss_median_bytes: float
    peak_rss_mad_bytes: float
    raw_repetition_ids: tuple[int, ...] = ()
    metric_direction: str | None = None
    # These fields were added in Task 4 so summaries remain safely comparable
    # after being written to and loaded from JSON. ``None`` is retained as a
    # compatibility representation for hand-authored legacy summaries; gate
    # APIs treat missing provenance as insufficient-data.
    effective_params: dict[str, object] | None = None
    machine: dict[str, str] | None = None
    raw_line_numbers: tuple[int, ...] | None = None
    profiled: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_repetition_ids", tuple(self.raw_repetition_ids))
        if self.effective_params is not None:
            object.__setattr__(self, "effective_params", _deep_freeze(self.effective_params))
        if self.machine is not None:
            object.__setattr__(self, "machine", _deep_freeze(self.machine))
        if self.raw_line_numbers is not None:
            object.__setattr__(self, "raw_line_numbers", tuple(self.raw_line_numbers))

    @property
    def grouping_keys(self) -> tuple[str, ...]:
        """Names identifying the population summarized by this row."""

        return (
            "run_id",
            "scenario",
            "task",
            "library",
            "library_version",
            "threads",
            "dataset_sha256",
            "input_representation",
            "metric_name",
        )

    def to_dict(self) -> dict[str, object]:
        result = {
            "schema": self.schema,
            "run_id": self.run_id,
            "scenario": self.scenario,
            "task": self.task,
            "library": self.library,
            "library_version": self.library_version,
            "threads": self.threads,
            "dataset_sha256": self.dataset_sha256,
            "input_representation": self.input_representation,
            "metric_name": self.metric_name,
            "metric_median": self.metric_median,
            "metric_mad": self.metric_mad,
            "preprocessing_median_seconds": self.preprocessing_median_seconds,
            "preprocessing_mad_seconds": self.preprocessing_mad_seconds,
            "fit_median_seconds": self.fit_median_seconds,
            "fit_mad_seconds": self.fit_mad_seconds,
            "predict_median_seconds": self.predict_median_seconds,
            "predict_mad_seconds": self.predict_mad_seconds,
            "peak_rss_median_bytes": self.peak_rss_median_bytes,
            "peak_rss_mad_bytes": self.peak_rss_mad_bytes,
            "metric_direction": self.metric_direction,
            "effective_params": (
                _deep_thaw(self.effective_params) if self.effective_params is not None else None
            ),
            "machine": _deep_thaw(self.machine) if self.machine is not None else None,
            "raw_line_numbers": (
                list(self.raw_line_numbers) if self.raw_line_numbers is not None else None
            ),
            "profiled": self.profiled,
        }
        result["raw_repetition_ids"] = list(self.raw_repetition_ids)
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> BenchmarkSummaryV1:
        return cls(
            schema=value["schema"],  # type: ignore[arg-type]
            run_id=value["run_id"],  # type: ignore[arg-type]
            scenario=value["scenario"],  # type: ignore[arg-type]
            task=value["task"],  # type: ignore[arg-type]
            library=value["library"],  # type: ignore[arg-type]
            library_version=value["library_version"],  # type: ignore[arg-type]
            threads=value["threads"],  # type: ignore[arg-type]
            dataset_sha256=value["dataset_sha256"],  # type: ignore[arg-type]
            input_representation=value["input_representation"],  # type: ignore[arg-type]
            metric_name=value["metric_name"],  # type: ignore[arg-type]
            metric_median=value["metric_median"],  # type: ignore[arg-type]
            metric_mad=value["metric_mad"],  # type: ignore[arg-type]
            preprocessing_median_seconds=value["preprocessing_median_seconds"],
            preprocessing_mad_seconds=value["preprocessing_mad_seconds"],  # type: ignore[arg-type]
            fit_median_seconds=value["fit_median_seconds"],  # type: ignore[arg-type]
            fit_mad_seconds=value["fit_mad_seconds"],  # type: ignore[arg-type]
            predict_median_seconds=value["predict_median_seconds"],  # type: ignore[arg-type]
            predict_mad_seconds=value["predict_mad_seconds"],  # type: ignore[arg-type]
            peak_rss_median_bytes=value["peak_rss_median_bytes"],  # type: ignore[arg-type]
            peak_rss_mad_bytes=value["peak_rss_mad_bytes"],  # type: ignore[arg-type]
            raw_repetition_ids=tuple(value.get("raw_repetition_ids", ())),  # type: ignore[arg-type]
            metric_direction=value.get("metric_direction"),  # type: ignore[arg-type]
            effective_params=(
                dict(value["effective_params"])
                if value.get("effective_params") is not None
                else None
            ),  # type: ignore[arg-type]
            machine=(dict(value["machine"]) if value.get("machine") is not None else None),  # type: ignore[arg-type]
            raw_line_numbers=(
                tuple(value["raw_line_numbers"])
                if value.get("raw_line_numbers") is not None
                else None
            ),  # type: ignore[arg-type]
            profiled=value.get("profiled"),  # type: ignore[arg-type]
        )

    @classmethod
    def from_json(cls, value: str) -> BenchmarkSummaryV1:
        decoded = json.loads(value)
        if not isinstance(decoded, Mapping):
            raise ValueError("benchmark summary JSON must contain an object")
        summary = cls.from_dict(decoded)
        validate_summary(summary)
        return summary


def validate_profile(profile: ProfileRecordV1) -> None:
    if not isinstance(profile, ProfileRecordV1):
        raise ValueError("profile must be a ProfileRecordV1")
    _positive_int(profile.rows, "profile.rows")
    _positive_int(profile.features, "profile.features")
    _positive_int(profile.rounds, "profile.rounds")
    _positive_int(profile.threads, "profile.threads")
    _positive_int(profile.loop_wall_ns, "profile.loop_wall_ns")
    _nonnegative_int(profile.untimed_ns, "profile.untimed_ns")
    _validate_ns_map(profile.stage_ns, "profile.stage_ns", PROFILE_STAGE_LABELS)
    _validate_ns_map(profile.tree_stage_ns, "profile.tree_stage_ns", TREE_STAGE_LABELS)


def validate_record(record: BenchmarkRecordV1) -> None:
    """Validate one record, raising ``ValueError`` with a useful field name."""

    if not isinstance(record, BenchmarkRecordV1):
        raise ValueError("record must be a BenchmarkRecordV1")
    if record.schema != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema: {record.schema!r}")
    for field in ("run_id", "scenario", "task", "library", "library_version"):
        _nonempty_string(getattr(record, field), field)
    if (
        not isinstance(record.dataset_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", record.dataset_sha256) is None
    ):
        raise ValueError("dataset_sha256 must be exactly 64 lowercase hexadecimal characters")
    _nonnegative_int(record.repetition, "repetition")
    _nonnegative_int(record.seed, "seed")
    _positive_int(record.threads, "threads")
    if not isinstance(record.effective_params, Mapping):
        raise ValueError("effective_params must be a mapping")
    _validate_finite_values(record.effective_params, "effective_params")
    _nonempty_string(record.input_representation, "input_representation")
    if record.input_representation not in INPUT_REPRESENTATIONS:
        raise ValueError(f"input_representation must be one of {sorted(INPUT_REPRESENTATIONS)}")
    for field in ("preprocessing_seconds", "fit_seconds", "predict_seconds"):
        value = _finite(getattr(record, field), field)
        if value <= 0:
            raise ValueError(f"{field} duration must be positive")
    _positive_int(record.peak_rss_bytes, "peak_rss_bytes")
    if record.metric_name not in METRIC_DIRECTIONS:
        raise ValueError(f"unknown metric: {record.metric_name!r}")
    _finite(record.metric_value, "metric_value")
    _positive_int(record.rounds_completed, "rounds_completed")
    if record.git_sha is not None:
        _nonempty_string(record.git_sha, "git_sha")
    _validate_machine_metadata(record.machine, "machine")
    if record.profile is not None:
        validate_profile(record.profile)


def _validate_summary_number(value: object, name: str, *, positive: bool = False) -> None:
    number = _finite(value, name)
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and number < 0:
        raise ValueError(f"{name} must be nonnegative")


def validate_summary(summary: BenchmarkSummaryV1) -> None:
    """Validate a summary and ensure its provenance is not discarded."""

    if not isinstance(summary, BenchmarkSummaryV1):
        raise ValueError("summary must be a BenchmarkSummaryV1")
    if summary.schema != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema: {summary.schema!r}")
    for field in (
        "run_id",
        "scenario",
        "task",
        "library",
        "library_version",
        "input_representation",
    ):
        _nonempty_string(getattr(summary, field), field)
    if summary.input_representation not in INPUT_REPRESENTATIONS:
        raise ValueError(f"input_representation must be one of {sorted(INPUT_REPRESENTATIONS)}")
    if (
        not isinstance(summary.dataset_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", summary.dataset_sha256) is None
    ):
        raise ValueError("dataset_sha256 must be exactly 64 lowercase hexadecimal characters")
    _positive_int(summary.threads, "threads")
    if summary.metric_name not in METRIC_DIRECTIONS:
        raise ValueError(f"unknown metric: {summary.metric_name!r}")
    if (
        summary.metric_direction is not None
        and summary.metric_direction != METRIC_DIRECTIONS[summary.metric_name]
    ):
        raise ValueError("metric_direction does not match metric_name")
    _finite(summary.metric_median, "metric_median")
    _validate_summary_number(summary.metric_mad, "metric_mad")
    for field in (
        "preprocessing_median_seconds",
        "fit_median_seconds",
        "predict_median_seconds",
    ):
        _validate_summary_number(getattr(summary, field), field, positive=True)
    for field in (
        "preprocessing_mad_seconds",
        "fit_mad_seconds",
        "predict_mad_seconds",
        "peak_rss_mad_bytes",
    ):
        _validate_summary_number(getattr(summary, field), field)
    _validate_summary_number(summary.peak_rss_median_bytes, "peak_rss_median_bytes", positive=True)
    if not isinstance(summary.raw_repetition_ids, Sequence) or isinstance(
        summary.raw_repetition_ids, (str, bytes)
    ):
        raise ValueError("raw_repetition_ids must be a sequence")
    if not summary.raw_repetition_ids:
        raise ValueError("raw_repetition_ids must be nonempty")
    if len(set(summary.raw_repetition_ids)) != len(summary.raw_repetition_ids):
        raise ValueError("raw_repetition_ids must be unique")
    for repetition in summary.raw_repetition_ids:
        _nonnegative_int(repetition, "raw_repetition_ids entry")
    if summary.effective_params is not None:
        if not isinstance(summary.effective_params, Mapping):
            raise ValueError("effective_params must be a mapping or null")
        _validate_finite_values(summary.effective_params, "effective_params")
    _validate_machine_metadata(summary.machine, "machine", allow_none=True)
    if summary.profiled is not None and not isinstance(summary.profiled, bool):
        raise ValueError("profiled must be a boolean or null")
    if summary.raw_line_numbers is not None:
        if not isinstance(summary.raw_line_numbers, Sequence) or isinstance(
            summary.raw_line_numbers, (str, bytes)
        ):
            raise ValueError("raw_line_numbers must be a sequence or null")
        if len(summary.raw_line_numbers) != len(summary.raw_repetition_ids):
            raise ValueError("raw_line_numbers must align with raw_repetition_ids")
        if len(set(summary.raw_line_numbers)) != len(summary.raw_line_numbers):
            raise ValueError("raw_line_numbers must be unique")
        for line in summary.raw_line_numbers:
            _positive_int(line, "raw_line_numbers entry")


def validate_run_metadata(metadata: RunMetadataV1) -> None:
    """Validate the sidecar that binds a raw run to its execution context."""

    if not isinstance(metadata, RunMetadataV1):
        raise ValueError("metadata must be a RunMetadataV1")
    if metadata.schema != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema: {metadata.schema!r}")
    for field in (
        "run_id", "git_sha_semantics", "harness_tree_sha256", "harness_source_path", "manifest_sha256",
        "manifest_identifier", "manifest_path", "raw_sha256", "created_at_utc",
        "working_directory",
    ):
        _nonempty_string(getattr(metadata, field), field)
    for field in ("manifest_sha256", "harness_tree_sha256", "raw_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", getattr(metadata, field)) is None:
            raise ValueError(f"{field} must be exactly 64 lowercase hexadecimal characters")
    for field in ("measured_git_sha", "harness_git_sha"):
        value = getattr(metadata, field)
        if value is not None:
            _nonempty_string(value, field)
            if re.fullmatch(r"[0-9a-f]{40}", value) is None:
                raise ValueError(f"{field} must be a full lowercase git SHA")
    if not metadata.libraries or not all(isinstance(value, str) and value.strip() for value in metadata.libraries):
        raise ValueError("libraries must be a nonempty sequence of strings")
    if not metadata.scenarios or not all(isinstance(value, str) and value.strip() for value in metadata.scenarios):
        raise ValueError("scenarios must be a nonempty sequence of strings")
    if len(set(metadata.libraries)) != len(metadata.libraries):
        raise ValueError("libraries must be unique")
    if len(set(metadata.scenarios)) != len(metadata.scenarios):
        raise ValueError("scenarios must be unique")
    _nonnegative_int(metadata.seed, "seed")
    _positive_int(metadata.threads, "threads")
    _positive_int(metadata.repetitions, "repetitions")
    _nonnegative_int(metadata.warmups, "warmups")
    if not isinstance(metadata.smoke, bool):
        raise ValueError("smoke must be a boolean")
    if not isinstance(metadata.profile_alloy, bool):
        raise ValueError("profile_alloy must be a boolean")
    _positive_int(metadata.raw_record_count, "raw_record_count")


def _validate_finite_values(value: object, name: str) -> None:
    """Validate JSON-like effective parameter values and finite numbers."""

    if value is None or isinstance(value, (bool, str)):
        return
    if _is_int(value):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} mapping keys must be strings")
            _validate_finite_values(child, f"{name}[{key!r}]")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _validate_finite_values(child, f"{name}[{index}]")
        return
    raise ValueError(f"{name} must contain only JSON-like values")


def _decode_records(text: str) -> list[Mapping[str, object]]:
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        decoded = json.loads(stripped)
        if not isinstance(decoded, list):
            raise ValueError("records JSON must contain an array")
        return decoded
    if stripped.startswith("{"):
        # A single wrapper object is accepted, while a file whose first line
        # is a JSON object is treated as JSONL when decoding the whole text
        # reports trailing data.
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            decoded = None
        if decoded is not None:
            if isinstance(decoded, Mapping) and "records" in decoded:
                decoded = decoded["records"]
            elif isinstance(decoded, Mapping):
                return [decoded]
            if not isinstance(decoded, list):
                raise ValueError("records JSON must contain an array")
            return decoded
    records: list[Mapping[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        decoded = json.loads(line)
        if not isinstance(decoded, Mapping):
            raise ValueError(f"record line {line_number} must contain an object")
        records.append(decoded)
    return records


def load_records(path: str | Path) -> list[BenchmarkRecordV1]:
    """Load and validate a JSON array, JSON wrapper, or JSONL record file."""

    records = [
        BenchmarkRecordV1.from_dict(item)
        for item in _decode_records(Path(path).read_text())
    ]
    for item in records:
        validate_record(item)
    versions: dict[tuple[str, str], str] = {}
    keys: set[tuple[str, str, str, int, int]] = set()
    for item in records:
        version_key = (item.run_id, item.library)
        previous = versions.setdefault(version_key, item.library_version)
        if previous != item.library_version:
            raise ValueError(
                "mismatched library_version for "
                f"run_id={item.run_id!r}, library={item.library!r}"
            )
        duplicate_key = (
            item.run_id,
            item.library,
            item.scenario,
            item.threads,
            item.repetition,
        )
        if duplicate_key in keys:
            raise ValueError(f"duplicate record key: {duplicate_key!r}")
        keys.add(duplicate_key)
    return records


def load_run_metadata(path: str | Path) -> RunMetadataV1:
    """Load and validate a run metadata JSON sidecar."""

    metadata = RunMetadataV1.from_json(Path(path).read_text())
    validate_run_metadata(metadata)
    return metadata


def load_run_bundle(
    raw_path: str | Path, metadata_path: str | Path | None = None
) -> tuple[RunMetadataV1, list[BenchmarkRecordV1]]:
    """Load a raw run and fail closed unless its metadata sidecar binds it.

    The default sidecar location is adjacent to ``raw_path``. The run UUID,
    record count, and exact raw-file SHA-256 are checked before returning.
    Scenario and library declarations are compared as sets (their order is
    not a metadata contract); seed, thread count, git SHA, and every declared
    cohort's repetition population are exact metadata contracts.
    """

    raw_file = Path(raw_path)
    if metadata_path is not None:
        metadata_file = Path(metadata_path)
    else:
        candidates = (
            raw_file.with_name("run-metadata.json"),
            raw_file.with_suffix(".run-metadata.json"),
        )
        existing = [candidate for candidate in candidates if candidate.exists()]
        if len(existing) != 1:
            raise ValueError(
                "expected exactly one adjacent run metadata sidecar, found "
                f"{len(existing)}"
            )
        metadata_file = existing[0]
    metadata = load_run_metadata(metadata_file)
    records = load_records(raw_file)
    if len(records) != metadata.raw_record_count:
        raise ValueError(
            "metadata raw_record_count does not match raw records: "
            f"{metadata.raw_record_count} != {len(records)}"
        )
    raw_sha256 = hashlib.sha256(raw_file.read_bytes()).hexdigest()
    if metadata.raw_sha256 != raw_sha256:
        raise ValueError("metadata raw checksum does not match raw records")
    if any(record.run_id != metadata.run_id for record in records):
        raise ValueError("metadata run_id does not match raw records")
    observed_scenarios = {record.scenario for record in records}
    if observed_scenarios != set(metadata.scenarios):
        raise ValueError("metadata scenarios do not match raw records")
    observed_libraries = {record.library for record in records}
    if observed_libraries != set(metadata.libraries):
        raise ValueError("metadata libraries do not match raw records")
    observed_threads = {record.threads for record in records}
    if observed_threads != {metadata.threads}:
        raise ValueError("metadata threads do not match raw records")
    observed_seeds = {record.seed for record in records}
    if observed_seeds != {metadata.seed}:
        raise ValueError("metadata seed does not match raw records")
    observed_git_shas = {record.git_sha for record in records}
    if observed_git_shas != {metadata.measured_git_sha}:
        raise ValueError("metadata git_sha does not match raw records")

    if metadata.profile_alloy and set(metadata.libraries) != {"alloygbm"}:
        raise ValueError("profile_alloy requires an AlloyGBM-only bundle")
    for record in records:
        if record.library == "alloygbm":
            if (record.profile is not None) != metadata.profile_alloy:
                raise ValueError("metadata profile_alloy does not match AlloyGBM record profile presence")
        elif record.profile is not None:
            raise ValueError("non-Alloy records may not contain AlloyGBM profiling")

    cohort_repetitions: dict[tuple[str, str, int], set[int]] = {}
    for record in records:
        cohort_key = (record.scenario, record.library, record.threads)
        cohort_repetitions.setdefault(cohort_key, set()).add(record.repetition)
    expected_cohorts = {
        (scenario, library, metadata.threads)
        for scenario in metadata.scenarios
        for library in metadata.libraries
    }
    if set(cohort_repetitions) != expected_cohorts:
        raise ValueError("metadata cohorts do not match raw records")
    expected_repetitions = set(range(metadata.repetitions))
    if any(
        repetitions != expected_repetitions
        for repetitions in cohort_repetitions.values()
    ):
        raise ValueError("cohort repetition IDs do not match metadata repetitions")
    expected_count = len(expected_cohorts) * metadata.repetitions
    if len(records) != expected_count:
        raise ValueError(
            "metadata cohort population does not match raw records: "
            f"{expected_count} != {len(records)}"
        )
    return metadata, records
