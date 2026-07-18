"""Shared schema, measurement, comparison, and rendering utilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
SCENARIO_CASES: dict[str, tuple[str, ...]] = {
    "soa_histograms": (
        "standard_wide",
        "standard_deep",
        "dro_wide",
        "linear_leaf",
    ),
    "node_parallelism": ("threads_1", "threads_8"),
    "duplicate_bins": (
        "wide_shallow_u8",
        "wide_shallow_u16",
        "tall_narrow_u8",
        "tall_narrow_u16",
    ),
    "compact_nodes": ("sparse_spines", "shallow_control"),
    "efb": ("exclusive_one_hot", "controlled_conflict", "dense_control"),
    "quantile_sketches": ("large_skewed",),
}
ENVIRONMENT_KEYS = (
    "platform",
    "machine",
    "logical_cpus",
    "python_major_minor",
    "rayon_num_threads",
)
REQUIRED_METRICS = {
    "soa_histograms": {
        "fit_seconds",
        "native_train_seconds",
        "fit_peak_rss_mb",
        "rmse",
        "prediction_digest",
    },
    "node_parallelism": {
        "fit_seconds",
        "native_train_seconds",
        "fit_peak_rss_mb",
        "rmse",
        "prediction_digest",
        "stump_count",
    },
    "duplicate_bins": {
        "fit_seconds",
        "native_bridge_prepare_seconds",
        "native_train_seconds",
        "fit_peak_rss_mb",
        "prediction_digest",
    },
    "compact_nodes": {
        "load_seconds",
        "predict_seconds_per_row",
        "fit_peak_rss_mb",
        "artifact_digest",
        "prediction_digest",
    },
    "efb": {
        "fit_seconds",
        "fit_peak_rss_mb",
        "rmse",
        "artifact_digest",
        "prediction_digest",
        "candidate_active",
    },
    "quantile_sketches": {
        "fit_seconds",
        "native_bridge_prepare_seconds",
        "fit_peak_rss_mb",
        "rmse",
        "mean_rank_error",
        "p99_rank_error",
        "max_rank_error",
        "candidate_active",
    },
}
STRING_METRICS = {"artifact_digest", "prediction_digest"}
BOOLEAN_METRICS = {"candidate_active"}
NULLABLE_NUMERIC_METRICS = {"fit_peak_rss_mb", "peak_rss_mb"}


@dataclass(frozen=True)
class CaseResult:
    scenario: str
    case: str
    repetition: int
    metrics: dict[str, Any]
    dimensions: dict[str, int]
    parameters: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CaseResult":
        return cls(
            scenario=str(payload["scenario"]),
            case=str(payload["case"]),
            repetition=int(payload["repetition"]),
            metrics=dict(payload["metrics"]),
            dimensions={str(k): int(v) for k, v in dict(payload["dimensions"]).items()},
            parameters=dict(payload["parameters"]),
        )


@dataclass(frozen=True)
class BenchmarkReport:
    schema_version: int
    profile: str
    mode: str
    environment: dict[str, Any]
    results: tuple[CaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "mode": self.mode,
            "environment": self.environment,
            "results": [asdict(result) for result in self.results],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkReport":
        report = cls(
            schema_version=int(payload["schema_version"]),
            profile=str(payload["profile"]),
            mode=str(payload["mode"]),
            environment=dict(payload["environment"]),
            results=tuple(CaseResult.from_dict(item) for item in payload["results"]),
        )
        validate_report(report)
        return report

    @classmethod
    def from_json(cls, payload: str) -> "BenchmarkReport":
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True)
class GateResult:
    name: str
    category: str
    passed: bool
    detail: str


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_manifest() -> dict[str, Any]:
    try:
        import alloygbm

        alloy_version = getattr(alloygbm, "__version__", "unknown")
        extension_path = None
        try:
            from alloygbm import _alloygbm

            extension_path = str(Path(_alloygbm.__file__).resolve())
        except (ImportError, AttributeError):
            pass
    except ImportError:
        alloy_version = "unavailable"
        extension_path = None
    return {
        "git_sha": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(_git_output("status", "--porcelain")),
        "alloygbm_version": alloy_version,
        "extension_path": extension_path,
        "python": platform.python_version(),
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": np.__version__,
        "platform": sys.platform,
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
    }


def environment_mismatches(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[str]:
    mismatches = []
    for key in ENVIRONMENT_KEYS:
        if baseline.get(key) != candidate.get(key):
            mismatches.append(
                f"{key}: baseline={baseline.get(key)} candidate={candidate.get(key)}"
            )
    return mismatches


def normalize_max_rss_mb(raw_value: float, platform_name: str) -> float | None:
    if platform_name == "darwin":
        return float(raw_value) / (1024.0 * 1024.0)
    if platform_name.startswith("linux"):
        return float(raw_value) / 1024.0
    return None


def peak_rss_mb() -> float | None:
    try:
        import resource

        raw_value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, AttributeError):
        return None
    return normalize_max_rss_mb(raw_value, sys.platform)


def current_rss_mb() -> float | None:
    if sys.platform.startswith("linux"):
        try:
            resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)
        except (OSError, ValueError, IndexError):
            return None
    if sys.platform == "darwin":
        try:
            rss_kib = int(
                subprocess.check_output(
                    ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True
                ).strip()
            )
            return rss_kib / 1024.0
        except (OSError, subprocess.CalledProcessError, ValueError):
            return None
    return None


def rss_delta_mb(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return max(0.0, after - before)


def prediction_digest(values: Sequence[float] | np.ndarray) -> str:
    array = np.asarray(values, dtype="<f4")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    actual = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def validate_report(report: BenchmarkReport) -> None:
    if report.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {report.schema_version}")
    if report.profile not in {"quick", "full"}:
        raise ValueError(f"unknown profile {report.profile!r}")
    if report.mode not in {"baseline", "candidate"}:
        raise ValueError(f"unknown mode {report.mode!r}")
    if not report.results:
        raise ValueError("benchmark report has no results")
    seen: set[tuple[str, str, int]] = set()
    repetitions: dict[tuple[str, str], set[int]] = {}
    for result in report.results:
        key = (result.scenario, result.case, result.repetition)
        if key in seen:
            raise ValueError(f"duplicate result key {key!r}")
        seen.add(key)
        repetitions.setdefault((result.scenario, result.case), set()).add(
            result.repetition
        )
        required = REQUIRED_METRICS.get(result.scenario)
        if required is None:
            raise ValueError(f"unknown scenario {result.scenario!r}")
        if result.case not in SCENARIO_CASES[result.scenario]:
            raise ValueError(
                f"unknown case {result.scenario}/{result.case}"
            )
        missing_metrics = sorted(required - set(result.metrics))
        if missing_metrics:
            raise ValueError(f"missing metrics {missing_metrics!r} for {key!r}")
        for metric_name in required:
            value = result.metrics[metric_name]
            if metric_name in STRING_METRICS:
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"metric {metric_name!r} for {key!r} must be a string"
                    )
            elif metric_name in BOOLEAN_METRICS:
                if not isinstance(value, bool):
                    raise ValueError(
                        f"metric {metric_name!r} for {key!r} must be boolean"
                    )
            elif value is None and metric_name in NULLABLE_NUMERIC_METRICS:
                continue
            elif not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    f"metric {metric_name!r} for {key!r} must be numeric"
                )
        for metric_name, value in result.metrics.items():
            if isinstance(value, bool) or value is None or isinstance(value, str):
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"metric {metric_name!r} for {key!r} is not finite")
            if float(value) < 0.0:
                raise ValueError(f"metric {metric_name!r} for {key!r} is negative")
    expected_repetitions = set(range(1 if report.profile == "quick" else 3))
    for case_key, observed in repetitions.items():
        if observed != expected_repetitions:
            raise ValueError(
                f"repetitions for {case_key!r} are {sorted(observed)!r}; "
                f"expected {sorted(expected_repetitions)!r}"
            )


def select_scenarios(
    report: BenchmarkReport, scenarios: Iterable[str]
) -> BenchmarkReport:
    selected = set(scenarios)
    filtered = replace(
        report,
        results=tuple(
            result for result in report.results if result.scenario in selected
        ),
    )
    validate_report(filtered)
    return filtered


def _group_results(report: BenchmarkReport) -> dict[tuple[str, str], list[CaseResult]]:
    grouped: dict[tuple[str, str], list[CaseResult]] = {}
    for result in report.results:
        grouped.setdefault((result.scenario, result.case), []).append(result)
    return grouped


def aggregate_report(report: BenchmarkReport) -> dict[tuple[str, str], dict[str, Any]]:
    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    for key, results in _group_results(report).items():
        metric_names = set().union(*(result.metrics for result in results))
        metrics: dict[str, Any] = {}
        for metric_name in metric_names:
            values = [result.metrics.get(metric_name) for result in results]
            numeric = [
                float(value)
                for value in values
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            if len(numeric) == len(values):
                metrics[metric_name] = statistics.median(numeric)
            elif all(value == values[0] for value in values):
                metrics[metric_name] = values[0]
            else:
                metrics[metric_name] = values[-1]
        aggregated[key] = metrics
    return aggregated


def _ratio(candidate: Mapping[str, Any], baseline: Mapping[str, Any], metric: str) -> float:
    baseline_value = float(baseline[metric])
    if baseline_value == 0.0:
        return 1.0 if float(candidate[metric]) == 0.0 else math.inf
    return float(candidate[metric]) / baseline_value


def _gate(name: str, category: str, passed: bool, detail: str) -> GateResult:
    return GateResult(name=name, category=category, passed=bool(passed), detail=detail)


def compare_reports(
    baseline: BenchmarkReport, candidate: BenchmarkReport
) -> list[GateResult]:
    validate_report(baseline)
    validate_report(candidate)
    gates: list[GateResult] = []
    gates.append(
        _gate(
            "report: modes",
            "contract",
            baseline.mode == "baseline" and candidate.mode == "candidate",
            f"baseline={baseline.mode} candidate={candidate.mode}",
        )
    )
    if baseline.profile != candidate.profile:
        return [_gate("environment: profile", "contract", False, "profiles differ")]
    mismatches = environment_mismatches(baseline.environment, candidate.environment)
    gates.append(
        _gate(
            "environment: compatibility",
            "contract",
            not mismatches,
            "; ".join(mismatches) if mismatches else "compatible",
        )
    )
    baseline_results = {
        (result.scenario, result.case, result.repetition): result
        for result in baseline.results
    }
    candidate_results = {
        (result.scenario, result.case, result.repetition): result
        for result in candidate.results
    }
    result_keys_match = set(baseline_results) == set(candidate_results)
    gates.append(
        _gate(
            "report: repetition coverage",
            "contract",
            result_keys_match,
            "exact match"
            if result_keys_match
            else (
                f"missing={sorted(set(baseline_results) - set(candidate_results))} "
                f"extra={sorted(set(candidate_results) - set(baseline_results))}"
            ),
        )
    )
    if not result_keys_match:
        return gates

    invariant_scenarios = {
        "soa_histograms",
        "node_parallelism",
        "duplicate_bins",
        "compact_nodes",
    }
    for result_key in sorted(baseline_results):
        baseline_result = baseline_results[result_key]
        candidate_result = candidate_results[result_key]
        inputs_match = (
            baseline_result.dimensions == candidate_result.dimensions
            and all(
                candidate_result.parameters.get(name) == value
                for name, value in baseline_result.parameters.items()
            )
        )
        gates.append(
            _gate(
                f"report: {result_key} inputs",
                "contract",
                inputs_match,
                "dimensions and baseline parameters",
            )
        )
        if baseline_result.scenario in invariant_scenarios:
            parity = candidate_result.metrics.get(
                "prediction_digest"
            ) == baseline_result.metrics.get("prediction_digest")
            gates.append(
                _gate(
                    f"{baseline_result.scenario}: {baseline_result.case} "
                    f"repetition {baseline_result.repetition} parity",
                    "quality",
                    parity,
                    "prediction digest",
                )
            )
        if baseline_result.scenario == "compact_nodes" or (
            baseline_result.scenario == "efb"
            and baseline_result.case
            in {"exclusive_one_hot", "controlled_conflict"}
        ):
            parity = candidate_result.metrics.get(
                "artifact_digest"
            ) == baseline_result.metrics.get("artifact_digest")
            gates.append(
                _gate(
                    f"{baseline_result.scenario}: {baseline_result.case} "
                    f"repetition {baseline_result.repetition} artifact parity",
                    "quality",
                    parity,
                    "artifact digest",
                )
            )
    base = aggregate_report(baseline)
    cand = aggregate_report(candidate)
    missing = sorted(set(base) - set(cand))
    gates.append(
        _gate(
            "report: case coverage",
            "contract",
            not missing,
            f"missing={missing}" if missing else "all baseline cases present",
        )
    )
    if missing:
        return gates

    for key in sorted(base):
        scenario, case = key
        if scenario in {
            "soa_histograms",
            "node_parallelism",
            "duplicate_bins",
            "compact_nodes",
        } and "prediction_digest" in base[key]:
            gates.append(
                _gate(
                    f"{scenario}: {case} parity",
                    "quality",
                    cand[key].get("prediction_digest") == base[key].get("prediction_digest"),
                    "prediction digest",
                )
            )
        if scenario in {"soa_histograms", "node_parallelism"}:
            delta = abs(float(cand[key]["rmse"]) - float(base[key]["rmse"]))
            gates.append(
                _gate(
                    f"{scenario}: {case} held-out quality",
                    "quality",
                    delta <= 1e-7,
                    f"absolute_delta={delta:.3g}",
                )
            )
        if scenario == "node_parallelism":
            same_count = cand[key].get("stump_count") == base[key].get("stump_count")
            minimum = 200 if baseline.profile == "quick" else 3500
            actual = int(cand[key].get("stump_count", 0))
            gates.extend(
                [
                    _gate(
                        f"node_parallelism: {case} tree-shape parity",
                        "quality",
                        same_count,
                        f"stump_count={actual}",
                    ),
                    _gate(
                        f"node_parallelism: {case} workload depth",
                        "quality",
                        actual >= minimum,
                        f"stump_count={actual} minimum={minimum}",
                    ),
                ]
            )

    candidate_groups = _group_results(candidate)
    for key in sorted(candidate_groups):
        if key[0] != "node_parallelism":
            continue
        digests = {
            result.metrics.get("prediction_digest")
            for result in candidate_groups[key]
        }
        gates.append(
            _gate(
                f"node_parallelism: {key[1]} repeated determinism",
                "quality",
                len(digests) == 1,
                f"distinct_prediction_digests={len(digests)}",
            )
        )

    for case in ("exclusive_one_hot", "dense_control"):
        key = ("efb", case)
        if key not in base:
            continue
        delta = abs(float(cand[key]["rmse"]) - float(base[key]["rmse"]))
        gates.append(
            _gate(
                f"efb: {case} held-out quality",
                "quality",
                delta <= 1e-6,
                f"absolute_delta={delta:.3g}",
            )
        )
    controlled = ("efb", "controlled_conflict")
    if controlled in base:
        active = bool(cand[controlled].get("candidate_active"))
        parity = cand[controlled].get("artifact_digest") == base[controlled].get(
            "artifact_digest"
        )
        gates.extend(
            [
                _gate(
                    "efb: conflict fallback",
                    "quality",
                    not active,
                    f"active={active}",
                ),
                _gate(
                    "efb: conflict artifact parity",
                    "quality",
                    parity,
                    "artifact digest",
                ),
            ]
        )

    exclusive = ("efb", "exclusive_one_hot")
    if exclusive in base:
        parity = cand[exclusive].get("artifact_digest") == base[exclusive].get(
            "artifact_digest"
        )
        gates.append(
            _gate(
                "efb: exclusive artifact parity",
                "quality",
                parity,
                "artifact digest",
            )
        )

    for key in sorted(base):
        if key[0] != "compact_nodes":
            continue
        parity = cand[key].get("artifact_digest") == base[key].get("artifact_digest")
        gates.append(
            _gate(
                f"compact_nodes: {key[1]} artifact parity",
                "quality",
                parity,
                "artifact digest",
            )
        )

    if ("efb", "exclusive_one_hot") in cand:
        metrics = cand[("efb", "exclusive_one_hot")]
        gates.append(
            _gate(
                "efb: activation",
                "quality",
                bool(metrics.get("candidate_active")),
                f"active={metrics.get('candidate_active')}",
            )
        )

    if ("quantile_sketches", "large_skewed") in cand:
        key = ("quantile_sketches", "large_skewed")
        metrics = cand[key]
        rmse_ratio = _ratio(metrics, base[key], "rmse")
        gates.extend(
            [
                _gate(
                    "quantile_sketches: activation",
                    "quality",
                    bool(metrics.get("candidate_active")),
                    f"active={metrics.get('candidate_active')}",
                ),
                _gate(
                    "quantile_sketches: mean rank error",
                    "quality",
                    float(metrics["mean_rank_error"]) <= 0.0025,
                    f"value={metrics['mean_rank_error']:.6f}",
                ),
                _gate(
                    "quantile_sketches: p99 rank error",
                    "quality",
                    float(metrics["p99_rank_error"]) <= 0.0075,
                    f"value={metrics['p99_rank_error']:.6f}",
                ),
                _gate(
                    "quantile_sketches: max rank error",
                    "quality",
                    float(metrics["max_rank_error"]) <= 0.01,
                    f"value={metrics['max_rank_error']:.6f}",
                ),
                _gate(
                    "quantile_sketches: held-out quality",
                    "quality",
                    rmse_ratio <= 1.01,
                    f"ratio={rmse_ratio:.3f}",
                ),
            ]
        )

    if baseline.profile != "full":
        return gates

    if mismatches:
        return gates

    for key in sorted(base):
        if key[0] != "soa_histograms":
            continue
        fit_ratio = _ratio(cand[key], base[key], "fit_seconds")
        gates.append(
            _gate(
                f"soa_histograms: {key[1]} fit regression",
                "performance",
                fit_ratio <= 1.05,
                f"ratio={fit_ratio:.3f}",
            )
        )
        if base[key].get("fit_peak_rss_mb") is not None and cand[key].get(
            "fit_peak_rss_mb"
        ) is not None:
            rss_ratio = _ratio(cand[key], base[key], "fit_peak_rss_mb")
            gates.append(
                _gate(
                    f"soa_histograms: {key[1]} memory regression",
                    "performance",
                    rss_ratio <= 1.05,
                    f"ratio={rss_ratio:.3f}",
                )
            )
    if ("soa_histograms", "standard_wide") in base:
        ratio = _ratio(
            cand[("soa_histograms", "standard_wide")],
            base[("soa_histograms", "standard_wide")],
            "native_train_seconds",
        )
        gates.append(_gate("soa_histograms: standard speed", "performance", ratio <= 0.90, f"ratio={ratio:.3f}"))

    if ("node_parallelism", "threads_8") in base:
        one_key = ("node_parallelism", "threads_1")
        eight_key = ("node_parallelism", "threads_8")
        ratio = _ratio(cand[eight_key], base[eight_key], "native_train_seconds")
        one_ratio = _ratio(cand[one_key], base[one_key], "fit_seconds")
        one = float(cand[one_key]["native_train_seconds"])
        eight = float(cand[eight_key]["native_train_seconds"])
        gates.append(_gate("node_parallelism: one-thread regression", "performance", one_ratio <= 1.05, f"ratio={one_ratio:.3f}"))
        gates.append(_gate("node_parallelism: eight-thread speed", "performance", ratio <= 0.85, f"ratio={ratio:.3f}"))
        gates.append(_gate("node_parallelism: scaling", "performance", one / max(eight, 1e-12) >= 1.25, f"speedup={one / max(eight, 1e-12):.3f}"))
        for key in (one_key, eight_key):
            if base[key].get("fit_peak_rss_mb") is not None and cand[key].get(
                "fit_peak_rss_mb"
            ) is not None:
                rss_ratio = _ratio(cand[key], base[key], "fit_peak_rss_mb")
                gates.append(_gate(f"node_parallelism: {key[1]} memory", "performance", rss_ratio <= 1.25, f"ratio={rss_ratio:.3f}"))

    for key in sorted(base):
        scenario, case = key
        if scenario != "duplicate_bins":
            continue
        train_ratio = _ratio(cand[key], base[key], "native_train_seconds")
        bridge_ratio = _ratio(
            cand[key], base[key], "native_bridge_prepare_seconds"
        )
        gates.extend(
            [
                _gate(f"duplicate_bins: {case} training", "performance", train_ratio <= 1.03, f"ratio={train_ratio:.3f}"),
                _gate(f"duplicate_bins: {case} bridge preparation", "performance", bridge_ratio <= 0.95, f"ratio={bridge_ratio:.3f}"),
            ]
        )
        if base[key].get("fit_peak_rss_mb") is not None and cand[key].get(
            "fit_peak_rss_mb"
        ) is not None:
            ratio = _ratio(cand[key], base[key], "fit_peak_rss_mb")
            gates.append(_gate(f"duplicate_bins: {case} memory", "performance", ratio <= 0.80, f"ratio={ratio:.3f}"))

    if ("compact_nodes", "sparse_spines") in base:
        key = ("compact_nodes", "sparse_spines")
        predict_ratio = _ratio(cand[key], base[key], "predict_seconds_per_row")
        load_ratio = _ratio(cand[key], base[key], "load_seconds")
        if base[key].get("fit_peak_rss_mb") is not None and cand[key].get(
            "fit_peak_rss_mb"
        ) is not None:
            rss_ratio = _ratio(cand[key], base[key], "fit_peak_rss_mb")
            gates.append(_gate("compact_nodes: sparse memory", "performance", rss_ratio <= 0.25, f"ratio={rss_ratio:.3f}"))
        gates.append(_gate("compact_nodes: sparse load", "performance", load_ratio <= 1.10, f"ratio={load_ratio:.3f}"))
        gates.append(_gate("compact_nodes: sparse throughput", "performance", predict_ratio <= 0.85, f"ratio={predict_ratio:.3f}"))
    shallow = ("compact_nodes", "shallow_control")
    if shallow in base:
        ratio = _ratio(cand[shallow], base[shallow], "predict_seconds_per_row")
        gates.append(_gate("compact_nodes: shallow control", "performance", ratio <= 1.05, f"ratio={ratio:.3f}"))

    if ("efb", "exclusive_one_hot") in base:
        key = ("efb", "exclusive_one_hot")
        active = bool(cand[key].get("candidate_active"))
        time_ratio = _ratio(cand[key], base[key], "fit_seconds")
        rss_base = base[key].get("fit_peak_rss_mb")
        rss_ratio = (
            _ratio(cand[key], base[key], "fit_peak_rss_mb")
            if rss_base is not None
            else 1.0
        )
        gates.append(_gate("efb: material benefit", "performance", time_ratio <= 0.85 or rss_ratio <= 0.80, f"time={time_ratio:.3f} rss={rss_ratio:.3f}"))
    dense = ("efb", "dense_control")
    if dense in base:
        ratio = _ratio(cand[dense], base[dense], "fit_seconds")
        gates.append(_gate("efb: dense control", "performance", ratio <= 1.03, f"ratio={ratio:.3f}"))

    if ("quantile_sketches", "large_skewed") in base:
        key = ("quantile_sketches", "large_skewed")
        bridge_ratio = _ratio(
            cand[key], base[key], "native_bridge_prepare_seconds"
        )
        fit_ratio = _ratio(cand[key], base[key], "fit_seconds")
        gates.extend(
            [
                _gate(
                    "quantile_sketches: bridge preparation",
                    "performance",
                    bridge_ratio <= 0.60,
                    f"ratio={bridge_ratio:.3f}",
                ),
                _gate(
                    "quantile_sketches: total fit",
                    "performance",
                    fit_ratio <= 1.05,
                    f"ratio={fit_ratio:.3f}",
                ),
            ]
        )
        if base[key].get("fit_peak_rss_mb") is not None and cand[key].get(
            "fit_peak_rss_mb"
        ) is not None:
            baseline_rss = float(base[key]["fit_peak_rss_mb"])
            candidate_rss = float(cand[key]["fit_peak_rss_mb"])
            ratio = candidate_rss / baseline_rss
            reduction = baseline_rss - candidate_rss
            gates.append(
                _gate(
                    "quantile_sketches: memory",
                    "performance",
                    ratio <= 0.90 and reduction >= 32.0,
                    f"ratio={ratio:.3f} reduction_mib={reduction:.2f}",
                )
            )
    return gates


def quality_gates(report: BenchmarkReport) -> list[GateResult]:
    validate_report(report)
    gates = []
    for key, metrics in sorted(aggregate_report(report).items()):
        finite = all(
            value is None
            or isinstance(value, (str, bool))
            or (isinstance(value, (int, float)) and math.isfinite(float(value)))
            for value in metrics.values()
        )
        gates.append(_gate(f"{key[0]}: {key[1]} finite metrics", "quality", finite, "finite" if finite else "non-finite"))
        if key[0] == "node_parallelism":
            minimum = 200 if report.profile == "quick" else 3500
            actual = int(metrics.get("stump_count", 0))
            gates.append(
                _gate(
                    f"node_parallelism: {key[1]} workload depth",
                    "quality",
                    actual >= minimum,
                    f"stump_count={actual} minimum={minimum}",
                )
            )
    return gates


def render_markdown(report: BenchmarkReport, gates: Sequence[GateResult]) -> str:
    title_names = {
        "soa_histograms": "SoA Histograms",
        "node_parallelism": "Node-Level Parallelism",
        "duplicate_bins": "Duplicate Bin Storage",
        "compact_nodes": "Compact Predictor Nodes",
        "efb": "Exclusive Feature Bundling",
        "quantile_sketches": "Approximate Quantile Sketches",
    }
    lines = [
        "# Architectural Backlog Benchmark",
        "",
        f"Profile: `{report.profile}`. Mode: `{report.mode}`.",
    ]
    aggregated = aggregate_report(report)
    for scenario in title_names:
        rows = [(key, value) for key, value in aggregated.items() if key[0] == scenario]
        if not rows:
            continue
        lines.extend(
            [
                "",
                f"## {title_names[scenario]}",
                "",
                "| Case | Fit (s) | Incremental peak RSS (MiB) | Quality |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for (_, case), metrics in sorted(rows):
            fit = metrics.get("fit_seconds")
            rss = metrics.get("fit_peak_rss_mb")
            quality = metrics.get("rmse", metrics.get("max_rank_error", "n/a"))
            fit_text = "n/a" if fit is None else f"{float(fit):.6f}"
            rss_text = "n/a" if rss is None else f"{float(rss):.2f}"
            quality_text = quality if isinstance(quality, str) else f"{float(quality):.6f}"
            lines.append(f"| `{case}` | {fit_text} | {rss_text} | {quality_text} |")
    lines.extend(["", "## Gates", ""])
    for gate in gates:
        lines.append(f"- {'pass' if gate.passed else 'FAIL'} [{gate.category}]: {gate.name} ({gate.detail})")
    return "\n".join(lines) + "\n"


def replace_metric(
    report: BenchmarkReport,
    *,
    scenario: str,
    case: str,
    metric: str,
    value: Any,
    repetition: int | None = None,
) -> BenchmarkReport:
    results = []
    for result in report.results:
        if (
            result.scenario == scenario
            and result.case == case
            and (repetition is None or result.repetition == repetition)
        ):
            metrics = dict(result.metrics)
            metrics[metric] = value
            result = replace(result, metrics=metrics)
        results.append(result)
    return replace(report, results=tuple(results))


def synthetic_report_for_tests(*, mode: str, profile: str) -> BenchmarkReport:
    cases = {
        "soa_histograms": ["standard_wide"],
        "node_parallelism": ["threads_1", "threads_8"],
        "duplicate_bins": ["wide_shallow_u8"],
        "compact_nodes": ["sparse_spines"],
        "efb": ["exclusive_one_hot"],
        "quantile_sketches": ["large_skewed"],
    }
    results = []
    repetition_count = 1 if profile == "quick" else 3
    for repetition in range(repetition_count):
        for scenario, scenario_cases in cases.items():
            for case in scenario_cases:
                is_candidate = mode == "candidate"
                metrics: dict[str, Any] = {
                    "fit_seconds": 0.8 if is_candidate else 1.0,
                    "load_seconds": 0.8 if is_candidate else 1.0,
                    "native_bridge_prepare_seconds": (
                        0.5 if is_candidate else 1.0
                    ),
                    "native_train_seconds": 0.7 if is_candidate else 1.0,
                    "fit_peak_rss_mb": 20.0 if is_candidate else 100.0,
                    "peak_rss_mb": 50.0 if is_candidate else 130.0,
                    "predict_seconds_per_row": 0.7 if is_candidate else 1.0,
                    "stump_count": 4000,
                    "rmse": 1.0,
                    "prediction_digest": "a" * 64,
                    "artifact_digest": "b" * 64,
                    "candidate_active": is_candidate,
                    "mean_rank_error": 0.001,
                    "p99_rank_error": 0.003,
                    "max_rank_error": 0.005,
                }
                if scenario == "node_parallelism" and case == "threads_1":
                    metrics["native_train_seconds"] = 1.0
                if scenario == "node_parallelism" and case == "threads_8":
                    metrics["native_train_seconds"] = (
                        0.6 if is_candidate else 1.0
                    )
                results.append(
                    CaseResult(
                        scenario,
                        case,
                        repetition,
                        metrics,
                        {"rows": 1},
                        {"seed": 17 + repetition},
                    )
                )
    environment = {
        "platform": "darwin",
        "machine": "arm64",
        "logical_cpus": 10,
        "python_major_minor": "3.13",
        "rayon_num_threads": None,
    }
    return BenchmarkReport(SCHEMA_VERSION, profile, mode, environment, tuple(results))
