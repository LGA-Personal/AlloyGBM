"""Direction-aware acceptance gates for competitiveness benchmark summaries."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .schema import BenchmarkSummaryV1, METRIC_DIRECTIONS

STATUSES = frozenset({"pass", "defer", "reject", "insufficient-data"})


@dataclass(frozen=True)
class GateResult:
    claim_type: str
    status: str
    reasons: tuple[str, ...] = ()
    evidence: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown gate status: {self.status!r}")
        if self.evidence is None:
            object.__setattr__(self, "evidence", {})

    @property
    def claim(self) -> str:
        return self.claim_type

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_type": self.claim_type,
            "claim": self.claim_type,
            "status": self.status,
            "reasons": list(self.reasons),
            "evidence": dict(self.evidence),
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


def _metadata(item: BenchmarkSummaryV1, name: str, default: object = None) -> object:
    return getattr(item, name, default)


def _pairs(
    current: Sequence[BenchmarkSummaryV1], baseline: Sequence[BenchmarkSummaryV1],
    *, candidate_library: str = "alloygbm", minimum_repetitions: int = 5,
) -> tuple[list[tuple[BenchmarkSummaryV1, BenchmarkSummaryV1]], list[str]]:
    current_by_key = {
        _slice_key(item): item for item in current if item.library == candidate_library
    }
    baseline_candidates = [item for item in baseline if item.library == candidate_library]
    if not baseline_candidates:
        # A compact baseline may call the reference arm “baseline”. Accept it
        # only when there is exactly one arm per slice.
        baseline_candidates = [item for item in baseline if item.library in {"baseline", "reference"}]
    baseline_by_key = {_slice_key(item): item for item in baseline_candidates}
    keys = sorted(set(current_by_key) | set(baseline_by_key), key=str)
    pairs: list[tuple[BenchmarkSummaryV1, BenchmarkSummaryV1]] = []
    reasons: list[str] = []
    for key in keys:
        candidate = current_by_key.get(key)
        reference = baseline_by_key.get(key)
        if candidate is None or reference is None:
            reasons.append(f"missing paired scenario/thread slice: {key!r}")
            continue
        if len(candidate.raw_repetition_ids) < minimum_repetitions or len(reference.raw_repetition_ids) < minimum_repetitions:
            reasons.append(f"insufficient timed repetitions for slice: {key!r}")
            continue
        if candidate.library_version != reference.library_version:
            reasons.append(f"library_version mismatch for slice: {key!r}")
            continue
        for field, label in (("_machine", "machine metadata"), ("_effective_params", "effective_params")):
            left, right = _metadata(candidate, field), _metadata(reference, field)
            if left is not None and right is not None and left != right:
                reasons.append(f"{label} mismatch for slice: {key!r}")
                break
        else:
            pairs.append((candidate, reference))
    return pairs, reasons


def _insufficient(claim: str, reasons: Sequence[str], evidence: Mapping[str, object] | None = None) -> GateResult:
    return GateResult(claim, "insufficient-data", tuple(reasons) or ("no comparable slices",), evidence or {})


def _compatibility_reasons(
    current: Sequence[BenchmarkSummaryV1], baseline: Sequence[BenchmarkSummaryV1]
) -> list[str]:
    """Check all paired library rows, including rank-context competitors."""

    def index(items: Sequence[BenchmarkSummaryV1]) -> dict[tuple[object, ...], BenchmarkSummaryV1]:
        return {_slice_key(item, include_library=True): item for item in items}

    left, right = index(current), index(baseline)
    reasons: list[str] = []
    for key in sorted(set(left) & set(right), key=str):
        candidate, reference = left[key], right[key]
        if candidate.library_version != reference.library_version:
            reasons.append(f"library_version mismatch for paired library slice: {key!r}")
        for field, label in (("_machine", "machine metadata"), ("_effective_params", "effective_params")):
            candidate_value, reference_value = _metadata(candidate, field), _metadata(reference, field)
            if candidate_value is not None and reference_value is not None and candidate_value != reference_value:
                reasons.append(f"{label} mismatch for paired library slice: {key!r}")
    # A baseline may use a compact reference arm name. Those rows are paired
    # by _pairs, so do not call their differing library labels a mismatch.
    return reasons


def _rank_context_reasons(
    current: Sequence[BenchmarkSummaryV1], baseline: Sequence[BenchmarkSummaryV1], *, minimum_repetitions: int = 5
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
        current_by_library = {item.library: item for item in current_rows}
        baseline_by_library = {item.library: item for item in baseline_rows}
        for library in sorted(current_libraries):
            candidate, reference = current_by_library[library], baseline_by_library[library]
            if candidate.library_version != reference.library_version:
                reasons.append(f"library_version mismatch for rank-context library {library!r} in slice: {key!r}")
            for field, label in (("_machine", "machine metadata"), ("_effective_params", "effective_params")):
                candidate_value, reference_value = _metadata(candidate, field), _metadata(reference, field)
                if candidate_value is not None and reference_value is not None and candidate_value != reference_value:
                    reasons.append(f"{label} mismatch for rank-context library {library!r} in slice: {key!r}")
    return reasons


def evaluate_speed(
    current: Sequence[BenchmarkSummaryV1], baseline: Sequence[BenchmarkSummaryV1], *,
    target_scenarios: Sequence[str] | None = None, minimum_repetitions: int = 5,
) -> GateResult:
    if target_scenarios is not None:
        wanted = set(target_scenarios)
        current = [item for item in current if item.scenario in wanted]
        baseline = [item for item in baseline if item.scenario in wanted]
    compatibility = _compatibility_reasons(current, baseline)
    pairs, reasons = _pairs(current, baseline, minimum_repetitions=minimum_repetitions)
    reasons = compatibility + reasons
    if not pairs:
        return _insufficient("speed", reasons)
    if reasons:
        return _insufficient("speed", reasons)
    evidence: dict[str, object] = {}
    guardrail = False
    missed = False
    for candidate, reference in pairs:
        improvement = fit_time_improvement(candidate.fit_median_seconds, reference.fit_median_seconds)
        regression = relative_metric_regression(candidate.metric_name, candidate.metric_median, reference.metric_median)
        evidence[candidate.scenario] = {"fit_improvement": improvement, "metric_regression": regression}
        guardrail |= regression > 0.01
        # Tolerate the one-ulp representation of the exact 10% boundary.
        missed |= improvement < 0.10 - 1e-12
    if guardrail:
        return GateResult("speed", "reject", tuple(reasons) + ("metric regression exceeds the 1% speed guardrail",), evidence)
    if missed:
        return GateResult("speed", "defer", tuple(reasons) + ("one or more slices miss the 10% fit improvement threshold",), evidence)
    return GateResult("speed", "pass", tuple(reasons), evidence)


def evaluate_quality(
    current: Sequence[BenchmarkSummaryV1], baseline: Sequence[BenchmarkSummaryV1], *,
    target_scenarios: Sequence[str] | None = None, quality_first: bool = False,
    minimum_repetitions: int = 5,
) -> GateResult:
    if target_scenarios is not None:
        wanted = set(target_scenarios)
        current = [item for item in current if item.scenario in wanted]
        baseline = [item for item in baseline if item.scenario in wanted]
    compatibility = _compatibility_reasons(current, baseline)
    pairs, reasons = _pairs(current, baseline, minimum_repetitions=minimum_repetitions)
    reasons = compatibility + reasons
    if len(pairs) < 2:
        return _insufficient("quality", reasons + ["quality requires at least two comparable scenarios"])
    if reasons:
        return _insufficient("quality", reasons)
    evidence: dict[str, object] = {}
    meaningful = 0
    fit_regressions: list[float] = []
    for candidate, reference in pairs:
        improvement = relative_metric_improvement(candidate.metric_name, candidate.metric_median, reference.metric_median)
        absolute_improvement = (reference.metric_median - candidate.metric_median) if METRIC_DIRECTIONS[candidate.metric_name] == "minimize" else (candidate.metric_median - reference.metric_median)
        noise = candidate.metric_mad + reference.metric_mad
        floor = 0.005 * abs(reference.metric_median)
        fit_regression = candidate.fit_median_seconds / reference.fit_median_seconds - 1.0
        fit_regressions.append(fit_regression)
        clears_noise = absolute_improvement > noise and absolute_improvement > floor
        meaningful += int(clears_noise and improvement > 0.005)
        evidence[candidate.scenario] = {"metric_improvement": improvement, "absolute_improvement": absolute_improvement, "noise": noise, "relative_floor": floor, "fit_regression": fit_regression, "clears_noise": clears_noise}
    median_fit_regression = float(statistics.median(fit_regressions))
    evidence["median_fit_regression"] = median_fit_regression
    if not quality_first and median_fit_regression > 0.10:
        return GateResult("quality", "reject", tuple(reasons) + ("median fit-time regression exceeds the 10% quality guardrail",), evidence)
    if meaningful < 2:
        return GateResult("quality", "defer", tuple(reasons) + ("fewer than two scenarios clear noise and the 0.5% relative improvement floor",), evidence)
    return GateResult("quality", "pass", tuple(reasons), evidence)


def normalized_ranks(summaries: Sequence[BenchmarkSummaryV1]) -> dict[str, float]:
    """Return each library's median normalized rank over comparable slices."""

    slices: dict[tuple[object, ...], list[BenchmarkSummaryV1]] = {}
    for item in summaries:
        slices.setdefault(_slice_key(item), []).append(item)
    per_library: dict[str, list[float]] = {}
    for population in slices.values():
        if len({item.library for item in population}) < 2:
            continue
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
    current: Sequence[BenchmarkSummaryV1], baseline: Sequence[BenchmarkSummaryV1], *, minimum_repetitions: int = 5
) -> list[GateResult]:
    compatibility = _compatibility_reasons(current, baseline)
    pairs, reasons = _pairs(current, baseline, minimum_repetitions=minimum_repetitions)
    reasons = compatibility + reasons
    if reasons and pairs:
        return [_insufficient("catastrophic-regression", reasons)]
    if not pairs:
        return [_insufficient("catastrophic-regression", reasons)]
    results: list[GateResult] = []
    for candidate, reference in pairs:
        metric_regression = relative_metric_regression(candidate.metric_name, candidate.metric_median, reference.metric_median)
        fit_ratio = candidate.fit_median_seconds / reference.fit_median_seconds
        # The specification's strict boundaries are mathematical values; the
        # tiny tolerance avoids rejecting decimal inputs such as 1.05 due to
        # binary floating-point rounding.
        status = "reject" if metric_regression > 0.05 + 1e-12 or fit_ratio > 2.0 + 1e-12 else "pass"
        reason = (f"catastrophic regression on {candidate.scenario}",) if status == "reject" else ()
        results.append(GateResult("catastrophic-regression", status, reason, {candidate.scenario: {"metric_regression": metric_regression, "fit_ratio": fit_ratio}}))
    return results


def evaluate_default_policy(
    current: Sequence[BenchmarkSummaryV1], baseline: Sequence[BenchmarkSummaryV1], *,
    minimum_repetitions: int = 5,
) -> GateResult:
    compatibility = _compatibility_reasons(current, baseline) + _rank_context_reasons(
        current, baseline, minimum_repetitions=minimum_repetitions
    )
    if compatibility:
        return _insufficient("default-policy", compatibility)
    catastrophe = catastrophic_regressions(current, baseline, minimum_repetitions=minimum_repetitions)
    if any(result.status == "reject" for result in catastrophe):
        return GateResult("default-policy", "reject", ("catastrophic regression in a protected fixture",), {"catastrophic": [item.to_dict() for item in catastrophe]})
    if any(result.status == "insufficient-data" for result in catastrophe):
        return _insufficient("default-policy", [reason for item in catastrophe for reason in item.reasons])
    current_rank = normalized_ranks(current)
    baseline_rank = normalized_ranks(baseline)
    if "alloygbm" not in current_rank or "alloygbm" not in baseline_rank:
        return _insufficient("default-policy", ["no comparable competitor context for normalized rank"])
    evidence = {"candidate_median_normalized_rank": current_rank["alloygbm"], "baseline_median_normalized_rank": baseline_rank["alloygbm"], "candidate_ranks": current_rank, "baseline_ranks": baseline_rank}
    if current_rank["alloygbm"] < baseline_rank["alloygbm"]:
        return GateResult("default-policy", "pass", (), evidence)
    return GateResult("default-policy", "defer", ("candidate normalized rank is not strictly better than baseline",), evidence)


def evaluate_claim(
    claim: str | None, current: Sequence[BenchmarkSummaryV1], baseline: Sequence[BenchmarkSummaryV1], *,
    minimum_repetitions: int = 5, quality_first: bool = False,
) -> GateResult:
    if claim == "speed":
        return evaluate_speed(current, baseline, minimum_repetitions=minimum_repetitions)
    if claim == "quality":
        return evaluate_quality(current, baseline, minimum_repetitions=minimum_repetitions, quality_first=quality_first)
    if claim == "default-policy":
        return evaluate_default_policy(current, baseline, minimum_repetitions=minimum_repetitions)
    return _insufficient("comparison", ["a claim and baseline are required for gate evaluation"])


def evaluate_catastrophic_regression(
    current: Sequence[BenchmarkSummaryV1], baseline: Sequence[BenchmarkSummaryV1], *, minimum_repetitions: int = 5
) -> GateResult:
    """Collapse protected-fixture results into one gate result."""

    results = catastrophic_regressions(current, baseline, minimum_repetitions=minimum_repetitions)
    if any(item.status == "insufficient-data" for item in results):
        return _insufficient("catastrophic-regression", [reason for item in results for reason in item.reasons])
    if any(item.status == "reject" for item in results):
        return GateResult("catastrophic-regression", "reject", tuple(reason for item in results for reason in item.reasons), {"fixtures": [item.to_dict() for item in results]})
    return GateResult("catastrophic-regression", "pass", (), {"fixtures": [item.to_dict() for item in results]})


compute_normalized_ranks = normalized_ranks
