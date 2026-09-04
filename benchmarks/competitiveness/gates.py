"""Direction-aware acceptance gates for competitiveness benchmark summaries."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

from .schema import BenchmarkSummaryV1, METRIC_DIRECTIONS, validate_summary

STATUSES = frozenset({"pass", "defer", "reject", "insufficient-data"})
_BOUNDARY_TOLERANCE = 1e-12


def _strictly_greater(value: float, threshold: float) -> bool:
    """Strict ``>`` with decimal-boundary floating-point noise ignored."""

    return value > threshold and not math.isclose(
        value, threshold, rel_tol=_BOUNDARY_TOLERANCE, abs_tol=_BOUNDARY_TOLERANCE
    )


def _at_least(value: float, threshold: float) -> bool:
    return value > threshold or math.isclose(
        value, threshold, rel_tol=_BOUNDARY_TOLERANCE, abs_tol=_BOUNDARY_TOLERANCE
    )


def _canonical(value: object) -> str:
    if isinstance(value, Mapping):
        value = {str(key): json.loads(_canonical(child)) for key, child in value.items()}
    elif isinstance(value, (list, tuple)):
        value = [json.loads(_canonical(child)) for child in value]
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(child) for child in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return value


@dataclass(frozen=True)
class GateResult:
    claim_type: str
    status: str
    reasons: tuple[str, ...] = ()
    evidence: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown gate status: {self.status!r}")
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "evidence", _freeze(self.evidence or {}))

    @property
    def claim(self) -> str:
        return self.claim_type

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_type": self.claim_type,
            "claim": self.claim_type,
            "status": self.status,
            "reasons": list(self.reasons),
            "evidence": _thaw(self.evidence),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)


def relative_metric_regression(metric_name: str, candidate: float, reference: float) -> float:
    """Positive values mean the candidate is worse, direction-aware."""

    epsilon = max(abs(reference), 1e-12)
    if METRIC_DIRECTIONS[metric_name] == "minimize":
        return (candidate - reference) / epsilon
    return (reference - candidate) / epsilon


def relative_metric_improvement(metric_name: str, candidate: float, reference: float) -> float:
    return -relative_metric_regression(metric_name, candidate, reference)


def fit_time_improvement(candidate: float, reference: float) -> float:
    return 1.0 - candidate / reference


def _slice_key(item: BenchmarkSummaryV1, *, include_library: bool = False) -> tuple[object, ...]:
    fields = ["scenario", "task", "threads", "dataset_sha256", "input_representation", "metric_name"]
    if include_library:
        fields.append("library")
    return tuple(getattr(item, field) for field in fields)


def _allowed_keys(values: Sequence[str] | None) -> frozenset[str]:
    if isinstance(values, str):
        raise ValueError("allowed_param_differences must be a sequence, not a string")
    keys = frozenset(values or ())
    if any(not isinstance(key, str) or not key.strip() for key in keys):
        raise ValueError("allowed_param_differences must contain nonempty top-level keys")
    return keys


def _normalize_summary_input(value: object, label: str) -> tuple[tuple[object, ...], list[str]]:
    """Turn an outer cohort argument into a safe tuple or an explicit reason."""

    if value is None or isinstance(value, (str, bytes, Mapping, BenchmarkSummaryV1)):
        return (), [f"{label} must be a sequence of summaries"]
    try:
        return tuple(value), []  # type: ignore[arg-type]
    except TypeError as exc:
        return (), [f"{label} must be a sequence of summaries: {exc}"]


def _normalize_targets(value: object) -> tuple[set[str] | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, (str, bytes, Mapping)):
        return None, "target_scenarios must be a sequence of nonempty strings"
    try:
        targets = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        return None, f"target_scenarios must be a sequence of nonempty strings: {exc}"
    if any(not isinstance(item, str) or not item.strip() for item in targets):
        return None, "target_scenarios must be a sequence of nonempty strings"
    return set(targets), None


def _parameter_reason(
    candidate: BenchmarkSummaryV1,
    reference: BenchmarkSummaryV1,
    allowed: frozenset[str],
    *,
    allow_candidate: bool,
) -> str | None:
    candidate_params, reference_params = candidate.effective_params, reference.effective_params
    if candidate_params is None or reference_params is None:
        return "missing effective parameter provenance"
    if not allow_candidate or not allowed:
        allowed = frozenset()
    keys = set(candidate_params) | set(reference_params)
    missing = object()
    mismatched = set()
    for key in keys:
        left, right = candidate_params.get(key, missing), reference_params.get(key, missing)
        differs = (left is missing) != (right is missing)
        if not differs and left is not missing:
            differs = _canonical(left) != _canonical(right)
        if differs and key not in allowed:
            mismatched.add(key)
    return f"effective_params mismatch (keys={sorted(mismatched)!r})" if mismatched else None


def _pairs(
    current: Sequence[BenchmarkSummaryV1], baseline: Sequence[BenchmarkSummaryV1],
    *, candidate_library: str = "alloygbm", minimum_repetitions: int = 5,
    allowed_param_differences: frozenset[str] = frozenset(),
) -> tuple[list[tuple[BenchmarkSummaryV1, BenchmarkSummaryV1]], list[str]]:
    current_by_key: dict[tuple[object, ...], BenchmarkSummaryV1] = {}
    baseline_by_key: dict[tuple[object, ...], BenchmarkSummaryV1] = {}
    reasons: list[str] = []
    for label, items, target in (("current", current, current_by_key), ("baseline", baseline, baseline_by_key)):
        for item in items:
            if item.library != candidate_library:
                continue
            key = _slice_key(item)
            if key in target:
                reasons.append(f"duplicate {label} summary slice/provenance: {key!r}")
            else:
                target[key] = item
    keys = sorted(set(current_by_key) | set(baseline_by_key), key=str)
    pairs: list[tuple[BenchmarkSummaryV1, BenchmarkSummaryV1]] = []
    for key in keys:
        candidate = current_by_key.get(key)
        reference = baseline_by_key.get(key)
        if candidate is None or reference is None:
            reasons.append(f"missing paired scenario/thread slice: {key!r}")
            continue
        if len(set(candidate.raw_repetition_ids)) < minimum_repetitions or len(set(reference.raw_repetition_ids)) < minimum_repetitions:
            reasons.append(f"insufficient timed repetitions for slice: {key!r}")
            continue
        if candidate.library_version != reference.library_version:
            reasons.append(f"library_version mismatch for slice: {key!r}")
            continue
        if candidate.machine is None or reference.machine is None:
            reasons.append(f"missing machine metadata for slice: {key!r}")
            continue
        if candidate.machine != reference.machine:
            reasons.append(f"machine metadata mismatch for slice: {key!r}")
            continue
        parameter_reason = _parameter_reason(
            candidate, reference, allowed_param_differences, allow_candidate=True
        )
        if parameter_reason is not None:
            reasons.append(f"{parameter_reason} for slice: {key!r}")
            continue
        pairs.append((candidate, reference))
    return pairs, reasons


def _insufficient(claim: str, reasons: Sequence[str], evidence: Mapping[str, object] | None = None, *, allowed: frozenset[str] = frozenset()) -> GateResult:
    merged = dict(evidence or {})
    merged.setdefault("allowed_param_differences", sorted(allowed))
    return GateResult(claim, "insufficient-data", tuple(reasons) or ("no comparable slices",), merged)


def _validate_gate_inputs(
    current: object, baseline: object,
    minimum_repetitions: int, allowed_param_differences: Sequence[str],
    *, require_durable_provenance: bool = True,
) -> tuple[frozenset[str], list[str], tuple[object, ...], tuple[object, ...]]:
    if minimum_repetitions <= 0:
        raise ValueError("minimum_repetitions must be positive")
    allowed = _allowed_keys(allowed_param_differences)
    current_items, current_reasons = _normalize_summary_input(current, "current")
    baseline_items, baseline_reasons = _normalize_summary_input(baseline, "baseline")
    reasons: list[str] = current_reasons + baseline_reasons
    for side, summaries in (("current", current), ("baseline", baseline)):
        items = current_items if side == "current" else baseline_items
        for index, summary in enumerate(items):
            try:
                validate_summary(summary)
            except (TypeError, ValueError) as exc:
                reasons.append(f"invalid {side} summary at index {index}: {exc}")
                continue
            if require_durable_provenance and summary.machine is None:
                reasons.append(f"missing machine provenance for {side} summary at index {index}")
            if require_durable_provenance and summary.effective_params is None:
                reasons.append(f"missing effective parameter provenance for {side} summary at index {index}")
    if reasons:
        return allowed, reasons, current_items, baseline_items
    try:
        current_runs = {item.run_id for item in current_items}
        baseline_runs = {item.run_id for item in baseline_items}
    except (AttributeError, TypeError) as exc:
        return allowed, [f"invalid summary provenance: {exc}"], current_items, baseline_items
    if len(current_runs) != 1:
        reasons.append(f"current cohort must contain exactly one run_id, found {sorted(current_runs)!r}")
    if len(baseline_runs) != 1:
        reasons.append(f"baseline cohort must contain exactly one run_id, found {sorted(baseline_runs)!r}")
    return allowed, reasons, current_items, baseline_items


def _candidate_gate_inputs(
    current: object, baseline: object, minimum_repetitions: int,
    allowed_param_differences: Sequence[str],
) -> tuple[frozenset[str], list[str], tuple[object, ...], tuple[object, ...], bool]:
    """Safely isolate AlloyGBM rows for catastrophe detection."""

    if minimum_repetitions <= 0:
        raise ValueError("minimum_repetitions must be positive")
    allowed = _allowed_keys(allowed_param_differences)
    current_items, current_outer_reasons = _normalize_summary_input(current, "current")
    baseline_items, baseline_outer_reasons = _normalize_summary_input(baseline, "baseline")
    reasons = current_outer_reasons + baseline_outer_reasons
    candidates: list[list[BenchmarkSummaryV1]] = [[], []]
    for side_index, items in enumerate((current_items, baseline_items)):
        side = "current" if side_index == 0 else "baseline"
        for index, item in enumerate(items):
            if not isinstance(item, BenchmarkSummaryV1):
                # An unidentifiable malformed row cannot be known to be the
                # candidate; retain it as evidence without blocking AlloyGBM.
                reasons.append(f"invalid {side} non-candidate summary at index {index}")
                continue
            try:
                validate_summary(item)
            except (TypeError, ValueError) as exc:
                if item.library == "alloygbm":
                    reasons.append(f"invalid {side} candidate summary at index {index}: {exc}")
                else:
                    reasons.append(f"invalid {side} competitor summary at index {index}: {exc}")
                continue
            if item.library == "alloygbm":
                candidates[side_index].append(item)
            elif item.machine is None or item.effective_params is None:
                reasons.append(f"missing durable provenance for {side} competitor summary at index {index}")
    candidate_allowed, candidate_reasons, candidate_current, candidate_baseline = _validate_gate_inputs(
        candidates[0], candidates[1], minimum_repetitions, allowed,
        require_durable_provenance=False,
    )
    return (
        candidate_allowed,
        reasons + candidate_reasons,
        candidate_current,
        candidate_baseline,
        bool(candidate_reasons),
    )


def _compatibility_reasons(
    current: Sequence[BenchmarkSummaryV1], baseline: Sequence[BenchmarkSummaryV1],
    *, allowed_param_differences: frozenset[str] = frozenset(),
) -> list[str]:
    """Check all paired library rows, including rank-context competitors."""

    left: dict[tuple[object, ...], BenchmarkSummaryV1] = {}
    right: dict[tuple[object, ...], BenchmarkSummaryV1] = {}
    reasons: list[str] = []
    for label, items, target in (("current", current, left), ("baseline", baseline, right)):
        for item in items:
            key = _slice_key(item, include_library=True)
            if key in target:
                reasons.append(f"duplicate {label} summary slice/provenance: {key!r}")
            else:
                target[key] = item
    for key in sorted(set(left) & set(right), key=str):
        candidate, reference = left[key], right[key]
        if candidate.library_version != reference.library_version:
            reasons.append(f"library_version mismatch for paired library slice: {key!r}")
        if candidate.machine is None or reference.machine is None:
            reasons.append(f"missing machine metadata for paired library slice: {key!r}")
        elif candidate.machine != reference.machine:
            reasons.append(f"machine metadata mismatch for paired library slice: {key!r}")
        if candidate.effective_params is None or reference.effective_params is None:
            reasons.append(f"missing effective parameter provenance for paired library slice: {key!r}")
        else:
            parameter_reason = _parameter_reason(
                candidate, reference, allowed_param_differences,
                allow_candidate=candidate.library == "alloygbm",
            )
            if parameter_reason is not None:
                reasons.append(f"{parameter_reason} for paired library slice: {key!r}")
    # A baseline may use a compact reference arm name. Those rows are paired
    # by _pairs, so do not call their differing library labels a mismatch.
    return reasons


def _rank_context_reasons(
    current: Sequence[BenchmarkSummaryV1], baseline: Sequence[BenchmarkSummaryV1], *, minimum_repetitions: int = 5,
    allowed_param_differences: frozenset[str] = frozenset(),
) -> list[str]:
    """Require identical competitor populations for a default-policy rank."""

    def by_slice(items: Sequence[BenchmarkSummaryV1]) -> dict[tuple[object, ...], list[BenchmarkSummaryV1]]:
        result: dict[tuple[object, ...], list[BenchmarkSummaryV1]] = {}
        for item in items:
            result.setdefault(_slice_key(item), []).append(item)
        return result

    left, right = by_slice(current), by_slice(baseline)
    reasons: list[str] = []
    for key in sorted(set(left) | set(right), key=str):
        current_rows, baseline_rows = left.get(key, []), right.get(key, [])
        current_libraries = {item.library for item in current_rows}
        baseline_libraries = {item.library for item in baseline_rows}
        if len(current_rows) != len(current_libraries) or len(baseline_rows) != len(baseline_libraries):
            reasons.append(f"duplicate rank summary slice/provenance: {key!r}")
            continue
        # “reference”/“baseline” is a compact single-arm baseline accepted by
        # speed and quality gates, but cannot provide rank context.
        if current_libraries != baseline_libraries:
            reasons.append(f"missing paired competitor context for slice: {key!r}")
            continue
        if any(
            len(item.raw_repetition_ids) < minimum_repetitions
            for item in (*current_rows, *baseline_rows)
        ):
            reasons.append(f"insufficient timed repetitions for rank-context slice: {key!r}")
            continue
        current_machines = {_canonical(item.machine) for item in current_rows if item.machine is not None}
        baseline_machines = {_canonical(item.machine) for item in baseline_rows if item.machine is not None}
        if len(current_machines) != 1 or len(baseline_machines) != 1:
            reasons.append(f"ranked libraries must share identical machine metadata for slice: {key!r}")
            continue
        current_by_library = {item.library: item for item in current_rows}
        baseline_by_library = {item.library: item for item in baseline_rows}
        for library in sorted(current_libraries):
            candidate, reference = current_by_library[library], baseline_by_library[library]
            if candidate.library_version != reference.library_version:
                reasons.append(f"library_version mismatch for rank-context library {library!r} in slice: {key!r}")
            if candidate.machine is None or reference.machine is None:
                reasons.append(f"missing machine metadata for rank-context library {library!r} in slice: {key!r}")
            elif candidate.machine != reference.machine:
                reasons.append(f"machine metadata mismatch for rank-context library {library!r} in slice: {key!r}")
            parameter_reason = _parameter_reason(
                candidate, reference, allowed_param_differences,
                allow_candidate=library == "alloygbm",
            )
            if parameter_reason is not None:
                reasons.append(f"{parameter_reason} for rank-context library {library!r} in slice: {key!r}")
    return reasons


def evaluate_speed(
    current: Sequence[BenchmarkSummaryV1], baseline: Sequence[BenchmarkSummaryV1], *,
    target_scenarios: Sequence[str] | None = None, minimum_repetitions: int = 5,
    allowed_param_differences: Sequence[str] = (),
) -> GateResult:
    allowed, cohort_reasons, current, baseline = _validate_gate_inputs(current, baseline, minimum_repetitions, allowed_param_differences)
    if cohort_reasons:
        return _insufficient("speed", cohort_reasons, allowed=allowed)
    wanted, target_reason = _normalize_targets(target_scenarios)
    if target_reason is not None:
        return _insufficient("speed", [target_reason], allowed=allowed)
    if wanted is not None:
        missing_current = wanted - {item.scenario for item in current}
        missing_baseline = wanted - {item.scenario for item in baseline}
        missing = sorted(missing_current | missing_baseline)
        if missing:
            return _insufficient(
                "speed", [f"requested target scenario missing from current/baseline: {name!r}" for name in missing],
                allowed=allowed,
            )
        current = [item for item in current if item.scenario in wanted]
        baseline = [item for item in baseline if item.scenario in wanted]
    compatibility = _compatibility_reasons(current, baseline, allowed_param_differences=allowed)
    pairs, reasons = _pairs(current, baseline, minimum_repetitions=minimum_repetitions, allowed_param_differences=allowed)
    reasons = compatibility + reasons
    if not pairs:
        return _insufficient("speed", reasons, allowed=allowed)
    if reasons:
        return _insufficient("speed", reasons, allowed=allowed)
    evidence: dict[str, object] = {"allowed_param_differences": sorted(allowed)}
    guardrail = False
    missed = False
    for candidate, reference in pairs:
        improvement = fit_time_improvement(candidate.fit_median_seconds, reference.fit_median_seconds)
        regression = relative_metric_regression(candidate.metric_name, candidate.metric_median, reference.metric_median)
        evidence[f"{candidate.scenario}|threads={candidate.threads}"] = {"fit_improvement": improvement, "metric_regression": regression}
        guardrail |= _strictly_greater(regression, 0.01)
        # Tolerate the one-ulp representation of the exact 10% boundary.
        missed |= not _at_least(improvement, 0.10)
    if guardrail:
        return GateResult("speed", "reject", tuple(reasons) + ("metric regression exceeds the 1% speed guardrail",), evidence)
    if missed:
        return GateResult("speed", "defer", tuple(reasons) + ("one or more slices miss the 10% fit improvement threshold",), evidence)
    return GateResult("speed", "pass", tuple(reasons), evidence)


def evaluate_quality(
    current: Sequence[BenchmarkSummaryV1], baseline: Sequence[BenchmarkSummaryV1], *,
    target_scenarios: Sequence[str] | None = None, quality_first: bool = False,
    minimum_repetitions: int = 5, allowed_param_differences: Sequence[str] = (),
) -> GateResult:
    allowed, cohort_reasons, current, baseline = _validate_gate_inputs(current, baseline, minimum_repetitions, allowed_param_differences)
    if cohort_reasons:
        return _insufficient("quality", cohort_reasons, allowed=allowed)
    wanted, target_reason = _normalize_targets(target_scenarios)
    if target_reason is not None:
        return _insufficient("quality", [target_reason], allowed=allowed)
    if wanted is not None:
        missing_current = wanted - {item.scenario for item in current}
        missing_baseline = wanted - {item.scenario for item in baseline}
        missing = sorted(missing_current | missing_baseline)
        if missing:
            return _insufficient(
                "quality", [f"requested target scenario missing from current/baseline: {name!r}" for name in missing],
                allowed=allowed,
            )
        current = [item for item in current if item.scenario in wanted]
        baseline = [item for item in baseline if item.scenario in wanted]
    compatibility = _compatibility_reasons(current, baseline, allowed_param_differences=allowed)
    pairs, reasons = _pairs(current, baseline, minimum_repetitions=minimum_repetitions, allowed_param_differences=allowed)
    reasons = compatibility + reasons
    if len(pairs) < 2:
        return _insufficient("quality", reasons + ["quality requires at least two comparable scenarios"], allowed=allowed)
    if reasons:
        return _insufficient("quality", reasons, allowed=allowed)
    if len({candidate.scenario for candidate, _ in pairs}) < 2:
        return _insufficient("quality", ["quality requires at least two distinct scenarios"], allowed=allowed)
    evidence: dict[str, object] = {"allowed_param_differences": sorted(allowed)}
    meaningful_scenarios: set[str] = set()
    fit_regressions: list[float] = []
    for candidate, reference in pairs:
        improvement = relative_metric_improvement(candidate.metric_name, candidate.metric_median, reference.metric_median)
        absolute_improvement = (reference.metric_median - candidate.metric_median) if METRIC_DIRECTIONS[candidate.metric_name] == "minimize" else (candidate.metric_median - reference.metric_median)
        noise = candidate.metric_mad + reference.metric_mad
        floor = 0.005 * abs(reference.metric_median)
        fit_regression = candidate.fit_median_seconds / reference.fit_median_seconds - 1.0
        fit_regressions.append(fit_regression)
        clears_noise = _strictly_greater(absolute_improvement, noise) and _strictly_greater(absolute_improvement, floor)
        if clears_noise and _strictly_greater(improvement, 0.005):
            meaningful_scenarios.add(candidate.scenario)
        evidence[f"{candidate.scenario}|threads={candidate.threads}"] = {"metric_improvement": improvement, "absolute_improvement": absolute_improvement, "noise": noise, "relative_floor": floor, "fit_regression": fit_regression, "clears_noise": clears_noise}
    median_fit_regression = float(statistics.median(fit_regressions))
    evidence["median_fit_regression"] = median_fit_regression
    if not quality_first and _strictly_greater(median_fit_regression, 0.10):
        return GateResult("quality", "reject", tuple(reasons) + ("median fit-time regression exceeds the 10% quality guardrail",), evidence)
    if len(meaningful_scenarios) < 2:
        return GateResult("quality", "defer", tuple(reasons) + ("fewer than two scenarios clear noise and the 0.5% relative improvement floor",), evidence)
    return GateResult("quality", "pass", tuple(reasons), evidence)


def normalized_ranks(summaries: Sequence[BenchmarkSummaryV1]) -> dict[str, float]:
    """Return each library's median normalized rank over comparable slices."""

    items, outer_reasons = _normalize_summary_input(summaries, "summaries")
    if outer_reasons:
        return {}
    for summary in items:
        try:
            validate_summary(summary)
        except (TypeError, ValueError):
            return {}
        # Legacy summaries may decode with null provenance, but rank
        # comparisons cannot establish a trustworthy execution context without
        # both durable machine and effective-parameter metadata.
        if summary.machine is None or summary.effective_params is None:
            return {}
    if len({summary.run_id for summary in items}) > 1:
        return {}
    slices: dict[tuple[object, ...], list[BenchmarkSummaryV1]] = {}
    for item in items:
        slices.setdefault(_slice_key(item), []).append(item)
    per_library: dict[str, list[float]] = {}
    for population in slices.values():
        libraries = {item.library for item in population}
        if len(libraries) < 2 or len(libraries) != len(population):
            return {}
        # Every library in a ranked slice must have run on the same machine;
        # this is a fairness invariant, not a current-vs-baseline check.
        if len({_canonical(item.machine) for item in population}) != 1:
            return {}
        direction = METRIC_DIRECTIONS[population[0].metric_name]
        ordered = sorted(population, key=lambda item: item.metric_median, reverse=direction == "maximize")
        ranks: dict[str, float] = {}
        position = 1
        index = 0
        while index < len(ordered):
            end = index + 1
            while end < len(ordered) and ordered[end].metric_median == ordered[index].metric_median:
                end += 1
            average = (position + position + (end - index) - 1) / 2.0
            for tied in ordered[index:end]:
                ranks[tied.library] = average
            position += end - index
            index = end
        divisor = len(ordered) - 1
        for library, rank in ranks.items():
            per_library.setdefault(library, []).append((rank - 1.0) / divisor)
    return {library: float(statistics.median(values)) for library, values in per_library.items()}


def catastrophic_regressions(
    current: Sequence[BenchmarkSummaryV1], baseline: Sequence[BenchmarkSummaryV1], *, minimum_repetitions: int = 5,
    allowed_param_differences: Sequence[str] = (),
) -> list[GateResult]:
    allowed, isolation_reasons, current, baseline, invalid_candidate_cohort = _candidate_gate_inputs(
        current, baseline, minimum_repetitions, allowed_param_differences
    )
    if invalid_candidate_cohort:
        return [_insufficient("catastrophic-regression", isolation_reasons, allowed=allowed)]
    pairs, pair_reasons = _pairs(current, baseline, minimum_repetitions=minimum_repetitions, allowed_param_differences=allowed)
    reasons = isolation_reasons + pair_reasons
    if not pairs:
        return [_insufficient("catastrophic-regression", reasons, allowed=allowed)]
    results: list[GateResult] = []
    for candidate, reference in pairs:
        metric_regression = relative_metric_regression(candidate.metric_name, candidate.metric_median, reference.metric_median)
        fit_ratio = candidate.fit_median_seconds / reference.fit_median_seconds
        # The specification's strict boundaries are mathematical values; the
        # tiny tolerance avoids rejecting decimal inputs such as 1.05 due to
        # binary floating-point rounding.
        status = "reject" if _strictly_greater(metric_regression, 0.05) or _strictly_greater(fit_ratio, 2.0) else "pass"
        reason = (f"catastrophic regression on {candidate.scenario}",) if status == "reject" else ()
        results.append(GateResult("catastrophic-regression", status, reason, {"allowed_param_differences": sorted(allowed), f"{candidate.scenario}|threads={candidate.threads}": {"metric_regression": metric_regression, "fit_ratio": fit_ratio}}))
    if reasons and not any(result.status == "reject" for result in results):
        return [_insufficient("catastrophic-regression", reasons, allowed=allowed)]
    if reasons:
        results.append(_insufficient("catastrophic-regression", reasons, allowed=allowed))
    return results


def evaluate_default_policy(
    current: Sequence[BenchmarkSummaryV1], baseline: Sequence[BenchmarkSummaryV1], *,
    minimum_repetitions: int = 5, allowed_param_differences: Sequence[str] = (),
) -> GateResult:
    allowed = _allowed_keys(allowed_param_differences)
    catastrophe = catastrophic_regressions(current, baseline, minimum_repetitions=minimum_repetitions, allowed_param_differences=allowed)
    if any(result.status == "reject" for result in catastrophe):
        return GateResult(
            "default-policy", "reject", ("catastrophic regression in a protected fixture",),
            {"allowed_param_differences": sorted(allowed), "catastrophic": [item.to_dict() for item in catastrophe]},
        )
    if any(result.status == "insufficient-data" for result in catastrophe):
        return _insufficient("default-policy", [reason for item in catastrophe for reason in item.reasons], allowed=allowed)
    allowed, cohort_reasons, current, baseline = _validate_gate_inputs(
        current, baseline, minimum_repetitions, allowed_param_differences
    )
    if cohort_reasons:
        return _insufficient("default-policy", cohort_reasons, allowed=allowed)
    compatibility = _compatibility_reasons(current, baseline, allowed_param_differences=allowed) + _rank_context_reasons(
        current, baseline, minimum_repetitions=minimum_repetitions, allowed_param_differences=allowed
    )
    if compatibility:
        return _insufficient("default-policy", compatibility, allowed=allowed)
    current_rank = normalized_ranks(current)
    baseline_rank = normalized_ranks(baseline)
    if "alloygbm" not in current_rank or "alloygbm" not in baseline_rank:
        return _insufficient("default-policy", ["no comparable competitor context for normalized rank"], allowed=allowed)
    evidence = {"allowed_param_differences": sorted(allowed), "candidate_median_normalized_rank": current_rank["alloygbm"], "baseline_median_normalized_rank": baseline_rank["alloygbm"], "candidate_ranks": current_rank, "baseline_ranks": baseline_rank}
    if current_rank["alloygbm"] < baseline_rank["alloygbm"]:
        return GateResult("default-policy", "pass", (), evidence)
    return GateResult("default-policy", "defer", ("candidate normalized rank is not strictly better than baseline",), evidence)


def evaluate_claim(
    claim: str | None, current: Sequence[BenchmarkSummaryV1], baseline: Sequence[BenchmarkSummaryV1], *,
    minimum_repetitions: int = 5, quality_first: bool = False, allowed_param_differences: Sequence[str] = (),
) -> GateResult:
    if claim == "speed":
        return evaluate_speed(current, baseline, minimum_repetitions=minimum_repetitions, allowed_param_differences=allowed_param_differences)
    if claim == "quality":
        return evaluate_quality(current, baseline, minimum_repetitions=minimum_repetitions, quality_first=quality_first, allowed_param_differences=allowed_param_differences)
    if claim == "default-policy":
        return evaluate_default_policy(current, baseline, minimum_repetitions=minimum_repetitions, allowed_param_differences=allowed_param_differences)
    return _insufficient("comparison", ["a claim and baseline are required for gate evaluation"])


def evaluate_catastrophic_regression(
    current: Sequence[BenchmarkSummaryV1], baseline: Sequence[BenchmarkSummaryV1], *, minimum_repetitions: int = 5,
    allowed_param_differences: Sequence[str] = (),
) -> GateResult:
    """Collapse protected-fixture results into one gate result."""

    results = catastrophic_regressions(current, baseline, minimum_repetitions=minimum_repetitions, allowed_param_differences=allowed_param_differences)
    allowed = _allowed_keys(allowed_param_differences)
    if any(item.status == "reject" for item in results):
        return GateResult("catastrophic-regression", "reject", tuple(reason for item in results for reason in item.reasons), {"allowed_param_differences": sorted(allowed), "fixtures": [item.to_dict() for item in results]})
    if any(item.status == "insufficient-data" for item in results):
        return _insufficient("catastrophic-regression", [reason for item in results for reason in item.reasons], allowed=allowed)
    return GateResult("catastrophic-regression", "pass", (), {"allowed_param_differences": sorted(allowed), "fixtures": [item.to_dict() for item in results]})


compute_normalized_ranks = normalized_ranks
