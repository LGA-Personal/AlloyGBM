"""Scenario workers for the deferred-architecture benchmark pack."""

from __future__ import annotations

import hashlib
import os
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    SCENARIO_CASES,
    CaseResult,
    current_rss_mb,
    peak_rss_mb,
    prediction_digest,
    rmse,
    rss_delta_mb,
)
from .fixtures import Fixture, make_fixture


def _artifact_stump_count(artifact: bytes) -> int | None:
    if len(artifact) < 16 or artifact[:4] != b"AGBM":
        return None
    section_count = struct.unpack_from("<I", artifact, 8)[0]
    for index in range(section_count):
        descriptor_offset = 16 + index * 20
        kind, payload_offset, payload_length = struct.unpack_from(
            "<IQQ", artifact, descriptor_offset
        )
        if kind != 1 or payload_length < 12:
            continue
        return int(struct.unpack_from("<I", artifact, payload_offset + 8)[0])
    return None


def _base_parameters(seed: int) -> dict[str, Any]:
    return {
        "learning_rate": 0.1,
        "n_estimators": 4,
        "max_depth": 6,
        "min_data_in_leaf": 1,
        "lambda_l2": 1.0,
        "training_policy": "manual",
        "deterministic": True,
        "continuous_binning_strategy": "quantile",
        "continuous_binning_max_bins": 64,
        "seed": seed,
    }


def _fit_regressor(
    fixture: Fixture,
    *,
    scenario: str,
    case: str,
    repetition: int,
    parameters: dict[str, Any],
) -> tuple[CaseResult, object]:
    from alloygbm import GBMRegressor

    before = current_rss_mb()
    estimator = GBMRegressor(**parameters)
    started = time.perf_counter()
    estimator.fit(fixture.X, fixture.y)
    fit_seconds = time.perf_counter() - started
    after = peak_rss_mb()
    predictions = np.asarray(estimator.predict(fixture.X_test), dtype=np.float32)
    timing = dict(getattr(estimator, "fit_timing_", None) or {})
    metrics: dict[str, Any] = {
        "fit_seconds": fit_seconds,
        "input_adaptation_seconds": float(
            timing.get("input_adaptation_seconds", 0.0)
        ),
        "native_bridge_prepare_seconds": float(
            timing.get("native_bridge_prepare_seconds", 0.0)
        ),
        "native_train_seconds": float(timing.get("native_train_seconds", fit_seconds)),
        "peak_rss_mb": after,
        "fit_peak_rss_mb": rss_delta_mb(before, after),
        "rmse": rmse(fixture.y_test, predictions),
        "prediction_digest": prediction_digest(predictions),
    }
    artifact = getattr(estimator, "artifact_bytes", None)
    if artifact is not None:
        artifact_bytes = bytes(artifact)
        metrics["artifact_bytes"] = len(artifact_bytes)
        metrics["artifact_digest"] = hashlib.sha256(artifact_bytes).hexdigest()
        stump_count = _artifact_stump_count(artifact_bytes)
        if stump_count is not None:
            metrics["stump_count"] = stump_count
    return (
        CaseResult(
            scenario=scenario,
            case=case,
            repetition=repetition,
            metrics=metrics,
            dimensions=fixture.dimensions,
            parameters=parameters,
        ),
        estimator,
    )


def _soa_case(
    case: str, profile: str, repetition: int, seed: int
) -> CaseResult:
    fixture = make_fixture("soa_histograms", case, profile, seed=seed)
    parameters = _base_parameters(seed)
    if profile == "full":
        parameters["n_estimators"] = {
            "standard_wide": 50,
            "standard_deep": 60,
            "dro_wide": 40,
            "linear_leaf": 24,
        }[case]
    if case == "standard_deep":
        parameters["max_depth"] = 10
    elif case == "dro_wide":
        parameters["leaf_solver"] = "dro"
    elif case == "linear_leaf":
        parameters["leaf_model"] = "linear"
    result, _ = _fit_regressor(
        fixture,
        scenario="soa_histograms",
        case=case,
        repetition=repetition,
        parameters=parameters,
    )
    return result


def _node_case(
    case: str, profile: str, repetition: int, seed: int
) -> CaseResult:
    expected_threads = "1" if case == "threads_1" else "8"
    actual_threads = os.environ.get("RAYON_NUM_THREADS")
    if actual_threads != expected_threads:
        raise RuntimeError(
            f"{case} requires RAYON_NUM_THREADS={expected_threads}, got {actual_threads!r}"
        )
    fixture = make_fixture("node_parallelism", case, profile, seed=seed)
    parameters = _base_parameters(seed)
    parameters.update(
        n_estimators=1,
        max_depth=8 if profile == "quick" else 12,
        min_data_in_leaf=1,
        lambda_l2=0.0,
    )
    result, _ = _fit_regressor(
        fixture,
        scenario="node_parallelism",
        case=case,
        repetition=repetition,
        parameters=parameters,
    )
    result.parameters["rayon_num_threads"] = int(expected_threads)
    return result


def _duplicate_bins_case(
    case: str, profile: str, repetition: int, seed: int
) -> CaseResult:
    fixture = make_fixture("duplicate_bins", case, profile, seed=seed)
    max_bins = 256 if case.endswith("u16") else 64
    parameters = _base_parameters(seed)
    parameters.update(
        n_estimators=1,
        max_depth=1,
        continuous_binning_max_bins=max_bins,
    )
    result, _ = _fit_regressor(
        fixture,
        scenario="duplicate_bins",
        case=case,
        repetition=repetition,
        parameters=parameters,
    )
    result.metrics["input_bytes"] = int(fixture.X.nbytes + fixture.y.nbytes)
    return result


def _predictor_artifact(tree_count: int, destination: Path, shape: str) -> bytes:
    repo_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--release",
            "-p",
            "alloygbm-engine",
            "--example",
            "sparse_predictor_fixture",
            "--",
            str(destination),
            str(tree_count),
            shape,
        ],
        cwd=repo_root,
        check=True,
    )
    return destination.read_bytes()


def _compact_case(
    case: str, profile: str, repetition: int, seed: int
) -> CaseResult:
    del seed
    fixture = make_fixture("compact_nodes", case, profile, seed=0)
    from alloygbm._alloygbm import NativePredictorHandle

    tree_count = 8 if profile == "full" else 1
    shape = "sparse" if case == "sparse_spines" else "balanced"
    with tempfile.TemporaryDirectory() as tmp:
        artifact = _predictor_artifact(
            tree_count, Path(tmp) / f"{shape}.agbm", shape
        )
    before = current_rss_mb()
    started = time.perf_counter()
    predictor = NativePredictorHandle(artifact)
    load_seconds = time.perf_counter() - started
    after = peak_rss_mb()
    X = np.ascontiguousarray(fixture.X_test, dtype=np.float32)
    predictor.predict_numpy_array(X[: min(128, len(X))])
    timings = []
    predictions = None
    for _ in range(5):
        started = time.perf_counter()
        predictions = np.asarray(predictor.predict_numpy_array(X), dtype=np.float32)
        timings.append(time.perf_counter() - started)
    assert predictions is not None
    metrics = {
        "fit_seconds": load_seconds,
        "load_seconds": load_seconds,
        "peak_rss_mb": after,
        "fit_peak_rss_mb": rss_delta_mb(before, after),
        "predict_seconds_per_row": float(np.median(timings)) / len(X),
        "artifact_bytes": len(artifact),
        "artifact_digest": hashlib.sha256(artifact).hexdigest(),
        "prediction_digest": prediction_digest(predictions),
        "rmse": rmse(fixture.y_test, predictions),
    }
    return CaseResult(
        "compact_nodes",
        case,
        repetition,
        metrics,
        fixture.dimensions,
        {
            "tree_count": tree_count,
            "shape": shape,
            "nodes_per_tree": 16 if shape == "sparse" else 7,
        },
    )


def _candidate_parameter(
    estimator_class: type, parameter: str, value: Any
) -> dict[str, Any]:
    available = estimator_class().get_params()
    if parameter not in available:
        raise RuntimeError(
            f"candidate mode requires estimator parameter {parameter!r}; "
            "the candidate implementation is not present"
        )
    return {parameter: value}


def _efb_case(
    case: str, profile: str, mode: str, repetition: int, seed: int
) -> CaseResult:
    from alloygbm import GBMRegressor

    fixture = make_fixture("efb", case, profile, seed=seed)
    parameters = _base_parameters(seed)
    parameters.update(n_estimators=4 if profile == "quick" else 24, max_depth=5)
    if mode == "candidate":
        parameters.update(
            _candidate_parameter(GBMRegressor, "feature_bundling", "exact")
        )
    result, estimator = _fit_regressor(
        fixture,
        scenario="efb",
        case=case,
        repetition=repetition,
        parameters=parameters,
    )
    diagnostics = getattr(estimator, "feature_bundling_diagnostics_", None)
    active = bool(diagnostics and diagnostics.get("active"))
    result.metrics.update(
        candidate_active=active if mode == "candidate" else False,
        original_feature_count=fixture.dimensions["features"],
        effective_feature_count=int(
            diagnostics.get("effective_feature_count", fixture.dimensions["features"])
            if diagnostics
            else fixture.dimensions["features"]
        ),
        conflict_rate=float(fixture.metadata.get("conflict_rate", 0.0)),
    )
    return result


def _rank_errors(X: np.ndarray, cuts_by_feature: object) -> tuple[float, float, float]:
    errors: list[float] = []
    if cuts_by_feature is None:
        raise RuntimeError("quantile cut metadata is missing")
    if len(cuts_by_feature) != X.shape[1]:
        raise RuntimeError(
            "quantile cut metadata feature count does not match the fixture"
        )
    for feature_index, raw_cuts in enumerate(cuts_by_feature):
        cuts = np.asarray(raw_cuts, dtype=np.float64)
        if not len(cuts):
            continue
        if not np.all(np.isfinite(cuts)) or not np.all(np.diff(cuts) > 0):
            return float("inf"), float("inf"), float("inf")
        values = np.sort(
            np.asarray(X[:, feature_index], dtype=np.float64)[
                np.isfinite(X[:, feature_index])
            ]
        )
        if len(np.unique(values)) < len(values) // 2:
            continue
        expected = np.arange(1, len(cuts) + 1, dtype=np.float64) / (len(cuts) + 1)
        left = np.searchsorted(values, cuts, side="left") / len(values)
        right = np.searchsorted(values, cuts, side="right") / len(values)
        errors.extend(np.maximum(left - expected, expected - right).clip(min=0.0))
    if not errors:
        raise RuntimeError("quantile cut metadata contains no scoreable cuts")
    array = np.asarray(errors)
    return float(array.mean()), float(np.quantile(array, 0.99)), float(array.max())


def _quantile_case(
    profile: str, mode: str, repetition: int, seed: int
) -> CaseResult:
    from alloygbm import GBMRegressor

    fixture = make_fixture("quantile_sketches", "large_skewed", profile, seed=seed)
    parameters = _base_parameters(seed)
    parameters.update(n_estimators=3 if profile == "quick" else 8, max_depth=3)
    if mode == "candidate":
        parameters.update(
            _candidate_parameter(
                GBMRegressor, "quantile_sketch_max_rows", 65_536
            )
        )
    result, estimator = _fit_regressor(
        fixture,
        scenario="quantile_sketches",
        case="large_skewed",
        repetition=repetition,
        parameters=parameters,
    )
    methods = getattr(estimator, "feature_quantile_cut_methods_", None)
    active = bool(methods and any(str(method) == "sketch" for method in methods))
    cuts = getattr(estimator, "_continuous_feature_quantile_cuts", None)
    mean_error, p99_error, max_error = _rank_errors(fixture.X, cuts)
    result.metrics.update(
        candidate_active=active if mode == "candidate" else False,
        mean_rank_error=mean_error,
        p99_rank_error=p99_error,
        max_rank_error=max_error,
    )
    return result


def run_case(
    *,
    scenario: str,
    case: str,
    profile: str,
    mode: str,
    repetition: int,
    seed: int,
) -> CaseResult:
    if scenario not in SCENARIO_CASES or case not in SCENARIO_CASES[scenario]:
        raise ValueError(f"unknown benchmark case {scenario}/{case}")
    if profile not in {"quick", "full"} or mode not in {"baseline", "candidate"}:
        raise ValueError(f"invalid profile/mode {profile}/{mode}")
    if scenario == "soa_histograms":
        return _soa_case(case, profile, repetition, seed)
    if scenario == "node_parallelism":
        return _node_case(case, profile, repetition, seed)
    if scenario == "duplicate_bins":
        return _duplicate_bins_case(case, profile, repetition, seed)
    if scenario == "compact_nodes":
        return _compact_case(case, profile, repetition, seed)
    if scenario == "efb":
        return _efb_case(case, profile, mode, repetition, seed)
    return _quantile_case(profile, mode, repetition, seed)
