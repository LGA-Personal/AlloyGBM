#!/usr/bin/env python3
"""Run auditable, paired-seed AlloyGBM accuracy-at-depth experiments.

The default matrix is the 15 numeric scenarios from the comparison harness and
the four standard models. ``synthetic_categorical`` is intentionally excluded:
the current numeric coercion drops every row in that fixture. Every arm only
changes AlloyGBM parameters; peers are freshly measured as controls. Five seeds
are required for an evidence-labelled run. Smaller matrices are plumbing smoke
runs and never produce a scientific verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from run_model_comparison import AVAILABLE_SCENARIOS, parse_alloy_param_overrides


DEFAULT_SCENARIOS: tuple[str, ...] = (
    "california_housing",
    "bike_sharing",
    "dense_numeric",
    "panel_time_series",
    "histogram_stress",
    "dow_jones_financial",
    "abalone_regression",
    "breast_cancer",
    "adult_income",
    "synthetic_classification",
    "wine_multiclass",
    "digits_multiclass",
    "synthetic_multiclass",
    "synthetic_ranking",
    "california_ranking",
)
DEFAULT_MODELS: tuple[str, ...] = ("alloygbm", "lightgbm", "xgboost", "catboost")
GUARD_SCENARIOS: tuple[str, ...] = ("histogram_stress", "panel_time_series")
KNOWN_UNSUPPORTED_SCENARIOS = {
    "synthetic_categorical": (
        "the current comparison runner numeric-coerces features and drops every row "
        "in this cat_* fixture (execution ruling 17)"
    )
}
METRIC_FOR_TASK = {
    "regression": "rmse",
    "classification": "log_loss_val",
    "multiclass_classification": "log_loss_val",
    "ranking": "ndcg_10",
}
LOWER_IS_BETTER_METRICS = frozenset({"rmse", "mae", "log_loss_val"})
VALID_MODELS = frozenset(
    {
        "alloygbm",
        "alloygbm_dro",
        "alloygbm_factor_neutral",
        "alloygbm_factor_neutral_dro",
        "alloygbm_linear",
        "alloygbm_morph",
        "alloygbm_morph_cosine",
        "alloygbm_morph_linear",
        "lightgbm",
        "xgboost",
        "catboost",
    }
)
EXPERIMENT_ENV_VARS = (
    "ALLOYGBM_EXPERIMENT_SPLIT_L2",
    "ALLOYGBM_EXPERIMENT_SPLIT_L1",
    "ALLOYGBM_EXPERIMENT_MIN_CHILD_HESS",
    "ALLOYGBM_EXPERIMENT_SPLIT_MIN_LEAF_MAGNITUDE",
    "ALLOYGBM_EXPERIMENT_FORCE_MANUAL_POLICY",
    "ALLOYGBM_EXPERIMENT_ENABLE_LEAF_REFINEMENT",
    "ALLOYGBM_EXPERIMENT_LINEAR_TAIL_RANK",
    "ALLOYGBM_EXPERIMENT_LINEAR_TAIL_CORE_SPAN_RATIO",
)
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
MANIFEST_NAME = "manifest.json"
RESULT_NAME = "model_comparison_latest.json"
MANIFEST_VERSION = 1
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class RunnerExitError(RuntimeError):
    """Comparison child failure that preserves its actual exit status."""

    def __init__(self, returncode: int, message: str) -> None:
        super().__init__(message)
        self.returncode = returncode


def normalise_gap(alloy: float, peer: float, lower_is_better: bool) -> float | None:
    """Return percent gap where negative always means AlloyGBM is better.

    A zero reference has no meaningful relative percentage, so the caller must
    use the separately reported absolute difference.
    """
    if not math.isfinite(alloy) or not math.isfinite(peer):
        raise ValueError("metric values must be finite")
    if peer == 0.0:
        return None
    gap = (alloy - peer) / abs(peer) * 100.0
    return gap if lower_is_better else -gap


def classify_verdict(alloy_gap_pct: float, band_pct: float) -> str:
    """Classify a relative gap against an inclusive, scenario-local spread."""
    if not math.isfinite(alloy_gap_pct) or not math.isfinite(band_pct):
        raise ValueError("gap and band must be finite")
    if band_pct < 0.0:
        raise ValueError("band must be non-negative")
    if abs(alloy_gap_pct) <= band_pct:
        return "tie"
    return "WIN" if alloy_gap_pct < 0.0 else "loss"


def parse_arm_specs(specs: list[str]) -> dict[str, dict[str, Any]]:
    """Parse repeatable ``NAME:KEY=VALUE[,KEY=VALUE]`` arms safely."""
    arms: dict[str, dict[str, Any]] = {}
    for spec in specs:
        name, separator, raw = spec.partition(":")
        if not separator or not SAFE_NAME.fullmatch(name) or name in {".", ".."}:
            raise ValueError(f"invalid or unsafe arm name in {spec!r}; use a safe arm name")
        if name in arms:
            raise ValueError(f"duplicate arm name {name!r}")
        overrides = [] if not raw.strip() else [part.strip() for part in raw.split(",")]
        if any(not override for override in overrides):
            raise ValueError(f"arm {name!r} contains an empty parameter override")
        parsed = parse_alloy_param_overrides(overrides)
        arms[name] = {
            "overrides": overrides,
            "parsed_overrides": parsed,
        }
    if not arms:
        raise ValueError("at least one --arm NAME:KEY=VALUE is required")
    return arms


def parse_seeds(spec: str, base_seed: int) -> list[int]:
    """Accept a seed count or explicit comma-separated seed identifiers."""
    raw = spec.strip()
    if not raw:
        raise ValueError("--seeds must be a positive count or comma-separated seed list")
    try:
        count = int(raw)
    except ValueError:
        parts = raw.split(",")
        if any(not part.strip() for part in parts):
            raise ValueError("--seeds has an empty seed")
        try:
            seeds = [int(part.strip()) for part in parts]
        except ValueError as exc:
            raise ValueError("--seeds must contain integers") from exc
    else:
        if count <= 0:
            raise ValueError("--seeds count must be positive")
        seeds = [base_seed + index for index in range(count)]
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds contains a duplicate seed")
    return seeds


def parse_depths(spec: str) -> list[int]:
    parts = [part.strip() for part in spec.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("--depths must be comma-separated positive integers")
    try:
        depths = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("--depths must be comma-separated positive integers") from exc
    if any(depth <= 0 for depth in depths):
        raise ValueError("--depths values must be positive")
    if len(set(depths)) != len(depths):
        raise ValueError("--depths contains a duplicate depth")
    return depths


def resolve_candidate_model(models: list[str], explicit: str | None) -> str:
    """Choose the AlloyGBM implementation whose accuracy is being measured."""
    alloy_models = [model for model in models if model.startswith("alloygbm")]
    if explicit is not None:
        if explicit not in models:
            raise ValueError(f"candidate model {explicit!r} must be included in --models")
        if not explicit.startswith("alloygbm"):
            raise ValueError("--candidate-model must name an AlloyGBM model")
        return explicit
    if len(alloy_models) != 1:
        raise ValueError(
            "select exactly one AlloyGBM model in --models or specify --candidate-model"
        )
    return alloy_models[0]


def evidence_class_for(
    seeds: list[int] | tuple[int, ...], scenarios: list[str] | tuple[str, ...]
) -> tuple[str, str | None]:
    """Evidence needs five seeds and both required guard scenarios."""
    reasons = []
    if len(seeds) != 5:
        reasons.append(f"seed count is {len(seeds)}, expected 5")
    missing_guards = [scenario for scenario in GUARD_SCENARIOS if scenario not in scenarios]
    if missing_guards:
        reasons.append("missing guard scenarios: " + ", ".join(missing_guards))
    return ("evidence", None) if not reasons else ("smoke", "; ".join(reasons))


def build_runner_command(
    *,
    repo_root: Path,
    target_dir: Path,
    overrides: list[str],
    depth: int,
    seed: int,
    args: dict[str, Any],
) -> list[str]:
    """Build the exact common-control command passed to the comparison runner."""
    command = [
        sys.executable,
        "-B",
        "benchmarks/run_model_comparison.py",
        "--max-depth",
        str(depth),
        "--threads",
        str(args["threads"]),
        "--seed",
        str(seed),
        "--rounds",
        str(args["rounds"]),
        "--learning-rate",
        str(args["learning_rate"]),
        "--alloy-continuous-binning-strategy",
        str(args["binning_strategy"]),
        "--alloy-continuous-binning-max-bins",
        str(args["bin_count"]),
        "--output-dir",
        str(target_dir),
        "--scenarios",
        *args["scenarios"],
        "--models",
        *args["models"],
    ]
    for override in overrides:
        command.extend(("--alloy-param", override))
    return command


def _metric_for_record(record: dict[str, Any]) -> str:
    task = record.get("task_type")
    if task not in METRIC_FOR_TASK:
        raise ValueError(f"unknown task_type {task!r} for {record.get('scenario')!r}")
    return METRIC_FOR_TASK[task]


def validate_records(
    payload: dict[str, Any],
    scenarios: list[str] | tuple[str, ...],
    models: list[str] | tuple[str, ...],
    seed: int,
    depth: int,
    rounds: int,
    learning_rate: float,
    expected_overrides: dict[str, Any] | None = None,
    expected_task_types: dict[str, str] | None = None,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Validate the exact one-profile result artifact and retain seed metrics."""
    unsupported = sorted(set(scenarios).intersection(KNOWN_UNSUPPORTED_SCENARIOS))
    if unsupported:
        detail = "; ".join(
            f"{name}: {KNOWN_UNSUPPORTED_SCENARIOS[name]}; numeric/finite filtering "
            "leaves no rows"
            for name in unsupported
        )
        raise ValueError(detail)
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError("comparison artifact must contain a records list")
    expected = {(scenario, model, seed) for scenario in scenarios for model in models}
    seen: dict[tuple[str, str, int], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("comparison artifact contains a non-object record")
        key = (record.get("scenario"), record.get("model"), record.get("seed"))
        if key not in expected:
            raise ValueError(f"unexpected result cell {key!r}")
        if key in seen:
            raise ValueError(f"duplicate result cell {key!r}")
        if record.get("status") != "PASS":
            raise ValueError(
                f"result cell {key!r} status {record.get('status')}: "
                f"{record.get('error', '')}"
            )
        if record.get("profile_name") != "single" or record.get("profile_index") != 1:
            raise ValueError(f"unrequested profile in result cell {key!r}")
        if record.get("run_index") != 1:
            raise ValueError(f"unrequested repeated run in result cell {key!r}")
        if record.get("seed") != seed:
            raise ValueError(f"seed mismatch in result cell {key!r}")
        if record.get("max_depth") != depth:
            raise ValueError(f"depth mismatch in result cell {key!r}")
        if record.get("rounds") != rounds:
            raise ValueError(f"rounds mismatch in result cell {key!r}")
        try:
            actual_lr = float(record.get("learning_rate"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"learning_rate missing in result cell {key!r}") from exc
        if not math.isfinite(actual_lr) or not math.isclose(
            actual_lr, learning_rate, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"learning_rate mismatch in result cell {key!r}")
        metric = _metric_for_record(record)
        expected_task = (
            expected_task_types.get(str(record.get("scenario")))
            if expected_task_types is not None
            else None
        )
        if expected_task is not None and record.get("task_type") != expected_task:
            raise ValueError(
                f"task_type mismatch in result cell {key!r}: "
                f"{record.get('task_type')!r} != {expected_task!r}"
            )
        if expected_task_types is not None and expected_task is None:
            raise ValueError(
                f"saved task_type identity missing for scenario {record.get('scenario')!r}"
            )
        value = record.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"non-finite or missing {metric} in result cell {key!r}")
        if (
            expected_overrides is not None
            and str(record.get("model", "")).startswith("alloygbm")
            and record.get("requested_alloy_param_overrides") != expected_overrides
        ):
            raise ValueError(f"requested AlloyGBM overrides mismatch in result cell {key!r}")
        retained = dict(record)
        retained["metric"] = metric
        retained["metric_value"] = float(value)
        seen[key] = retained
    missing = sorted(expected.difference(seen))
    if missing:
        raise ValueError(f"missing requested result cell(s): {missing!r}")
    return seen


def _metric_values(
    cells: dict[tuple[str, str, int], dict[str, Any]],
    scenario: str,
    model: str,
    seeds: list[int],
) -> tuple[str, dict[str, float]]:
    values: dict[str, float] = {}
    metric: str | None = None
    for seed in seeds:
        cell = cells.get((scenario, model, seed))
        if cell is None:
            raise ValueError(f"missing paired result cell {(scenario, model, seed)!r}")
        cell_metric = str(cell["metric"])
        if metric is not None and metric != cell_metric:
            raise ValueError(f"metric changed between seeds for {scenario}/{model}")
        metric = cell_metric
        value = float(cell.get("metric_value", cell.get("value")))
        if not math.isfinite(value):
            raise ValueError(f"non-finite metric for {scenario}/{model}/seed={seed}")
        values[str(seed)] = value
    assert metric is not None
    return metric, values


def _stats(metric: str, values: dict[str, float]) -> dict[str, Any]:
    items = list(values.values())
    median = float(statistics.median(items))
    minimum = min(items)
    maximum = max(items)
    spread = maximum - minimum
    spread_pct = spread / abs(median) * 100.0 if median != 0.0 else None
    return {
        "metric": metric,
        "values_by_seed": values,
        "n_seeds": len(items),
        "median": median,
        "minimum": minimum,
        "maximum": maximum,
        "absolute_spread": spread,
        "spread_pct": spread_pct,
    }


def _comparison(
    candidate_cells: dict[tuple[str, str, int], dict[str, Any]],
    reference_cells: dict[tuple[str, str, int], dict[str, Any]],
    scenario: str,
    candidate_model: str,
    reference_model: str,
    seeds: list[int],
) -> dict[str, Any]:
    candidate_metric, candidate_values = _metric_values(
        candidate_cells, scenario, candidate_model, seeds
    )
    reference_metric, reference_values = _metric_values(
        reference_cells, scenario, reference_model, seeds
    )
    if candidate_metric != reference_metric:
        raise ValueError(
            f"metric mismatch for {scenario}: {candidate_metric} vs {reference_metric}"
        )
    lower_is_better = candidate_metric in LOWER_IS_BETTER_METRICS
    candidate_stats = _stats(candidate_metric, candidate_values)
    reference_stats = _stats(reference_metric, reference_values)
    candidate_median = candidate_stats["median"]
    reference_median = reference_stats["median"]
    absolute_delta = candidate_median - reference_median
    relative_gap = normalise_gap(candidate_median, reference_median, lower_is_better)
    spread_values = [
        value
        for value in (candidate_stats["spread_pct"], reference_stats["spread_pct"])
        if value is not None
    ]
    band = max(spread_values) if len(spread_values) == 2 else None
    verdict = (
        classify_verdict(relative_gap, band)
        if relative_gap is not None and band is not None
        else None
    )
    paired = []
    for seed in seeds:
        candidate_value = candidate_values[str(seed)]
        reference_value = reference_values[str(seed)]
        paired.append(
            {
                "seed": seed,
                "candidate": candidate_value,
                "reference": reference_value,
                "delta": candidate_value - reference_value,
            }
        )
    return {
        "metric": candidate_metric,
        "lower_is_better": lower_is_better,
        "candidate": candidate_stats,
        "reference": reference_stats,
        "absolute_delta_median": absolute_delta,
        "absolute_delta_minimum": min(item["delta"] for item in paired),
        "absolute_delta_maximum": max(item["delta"] for item in paired),
        "relative_gap_pct": relative_gap,
        "band_pct": band,
        "verdict": verdict,
        "paired_absolute_deltas": paired,
    }


def analyse_cells(
    candidate: dict[tuple[str, str, int], dict[str, Any]],
    baseline: dict[tuple[str, str, int], dict[str, Any]],
    scenarios: list[str],
    models: list[str],
    seeds: list[int],
    baseline_arm: str,
    *,
    candidate_model: str,
) -> dict[str, Any]:
    """Summarise raw seed values, paired deltas, peer comparisons, and guards."""
    output: dict[str, Any] = {
        "baseline_arm": baseline_arm,
        "candidate_model": candidate_model,
        "scenarios": {},
        "guards": {},
    }
    if candidate_model not in models or not candidate_model.startswith("alloygbm"):
        raise ValueError("candidate_model must be a selected AlloyGBM model")
    peer_models = [
        model
        for model in models
        if model != candidate_model and not model.startswith("alloygbm")
    ]
    for scenario in scenarios:
        metric, _ = _metric_values(candidate, scenario, candidate_model, seeds)
        scenario_entry: dict[str, Any] = {
            "metric": metric,
            "models": {},
            "candidate_vs_baseline": _comparison(
                candidate, baseline, scenario, candidate_model, candidate_model, seeds
            ),
            "candidate_vs_peers": {},
        }
        for model in models:
            model_metric, values = _metric_values(candidate, scenario, model, seeds)
            scenario_entry["models"][model] = _stats(model_metric, values)
        for model in peer_models:
            scenario_entry["candidate_vs_peers"][model] = _comparison(
                candidate, candidate, scenario, candidate_model, model, seeds
            )
        output["scenarios"][scenario] = scenario_entry

    for guard in GUARD_SCENARIOS:
        scenario_entry = output["scenarios"].get(guard)
        if scenario_entry is None:
            output["guards"][guard] = {
                "status": "NOT_RUN",
                "peer_status": "NOT_RUN",
                "candidate_vs_baseline": None,
                "candidate_vs_peers": {},
            }
            continue
        baseline_verdict = scenario_entry["candidate_vs_baseline"]["verdict"]
        if baseline_verdict == "loss":
            status = "REGRESSION"
        elif baseline_verdict is None:
            status = "INCONCLUSIVE"
        else:
            status = "CLEAR"

        peer_verdicts = [
            comparison["verdict"]
            for comparison in scenario_entry["candidate_vs_peers"].values()
        ]
        if not peer_verdicts:
            peer_status = "NOT_RUN"
        elif any(verdict == "loss" for verdict in peer_verdicts):
            peer_status = "STANDING_DEFICIT"
        elif any(verdict is None for verdict in peer_verdicts):
            peer_status = "INCONCLUSIVE"
        else:
            peer_status = "CLEAR"
        output["guards"][guard] = {
            "status": status,
            "peer_status": peer_status,
            "candidate_vs_baseline": scenario_entry["candidate_vs_baseline"],
            "candidate_vs_peers": scenario_entry["candidate_vs_peers"],
        }
    return output


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo_root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def source_identity(repo_root: Path, scenarios: list[str]) -> dict[str, Any]:
    """Fingerprint runner, AlloyGBM, and selected data-generation source files."""
    relative_files: set[Path] = {
        Path("Cargo.toml"),
        Path("Cargo.lock"),
        Path("pyproject.toml"),
        Path("benchmarks/run_model_comparison.py"),
        Path("benchmarks/accuracy_depth_sweep.py"),
    }
    for base, suffixes in (
        (Path("crates"), {".rs", ".toml"}),
        (Path("bindings/python/src"), {".rs"}),
        (Path("bindings/python/alloygbm"), {".py"}),
    ):
        root = repo_root / base
        if root.exists():
            relative_files.update(
                path.relative_to(repo_root)
                for path in root.rglob("*")
                if path.is_file() and path.suffix in suffixes
            )
    for scenario in scenarios:
        relative_files.add(Path("benchmarks") / scenario / "manifest.yaml")
        prepare = Path("benchmarks") / scenario / "prepare.py"
        if (repo_root / prepare).exists():
            relative_files.add(prepare)

    file_hashes: dict[str, str] = {}
    for relative in sorted(relative_files, key=str):
        path = repo_root / relative
        if path.is_file():
            file_hashes[str(relative)] = _hash_file(path)
    aggregate = hashlib.sha256(
        json.dumps(file_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    status = _git_output(
        repo_root,
        "status",
        "--porcelain",
        "--",
        "Cargo.toml",
        "Cargo.lock",
        "pyproject.toml",
        "crates",
        "bindings/python/src",
        "bindings/python/alloygbm",
        "benchmarks/run_model_comparison.py",
        "benchmarks/accuracy_depth_sweep.py",
        *(f"benchmarks/{scenario}" for scenario in scenarios),
    )
    return {
        "git_sha": _git_output(repo_root, "rev-parse", "HEAD"),
        "source_fingerprint_sha256": aggregate,
        "source_file_sha256": file_hashes,
        "source_worktree_dirty": bool(status),
        "source_dirty_paths": [line[3:] for line in status.splitlines() if line],
        "scope": "runner, AlloyGBM Python/Rust source, and selected scenario preparation code",
    }


def runtime_identity() -> dict[str, Any]:
    """Hash the imported package and native extension used by the runner."""
    package = importlib.import_module("alloygbm")
    native = importlib.import_module("alloygbm._alloygbm")
    package_path = Path(package.__file__).resolve()
    native_path = Path(native.__file__).resolve()
    package_files = sorted(
        path for path in package_path.parent.rglob("*.py") if path.is_file()
    )
    package_hashes = {str(path.relative_to(package_path.parent)): _hash_file(path) for path in package_files}
    package_digest = hashlib.sha256(
        json.dumps(package_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.split()[0],
        "package_path": str(package_path),
        "package_sha256": package_digest,
        "native_module_path": str(native_path),
        "native_module_sha256": _hash_file(native_path),
        "alloygbm_version": str(getattr(package, "__version__", "unknown")),
    }


def selected_environment() -> dict[str, str]:
    """Record only known experiment knobs and thread-control variables."""
    return {
        name: os.environ[name]
        for name in (*EXPERIMENT_ENV_VARS, *THREAD_ENV_VARS)
        if name in os.environ
    }


def ensure_dataset(repo_root: Path, scenario: str, output_dir: Path) -> dict[str, Any]:
    """Prepare missing harness data and fingerprint its exact manifest and CSV."""
    manifest_path = repo_root / "benchmarks" / scenario / "manifest.yaml"
    if not manifest_path.is_file():
        raise ValueError(f"no dataset manifest for requested scenario {scenario!r}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    prepared_name = manifest["prepared"]["filename"]
    prepared_path = repo_root / "benchmarks" / "data" / scenario / "prepared" / prepared_name
    if not prepared_path.is_file():
        prep_script = repo_root / "benchmarks" / scenario / "prepare.py"
        if not prep_script.is_file():
            raise FileNotFoundError(f"missing dataset and preparation script for {scenario}")
        prep_log = output_dir / "preparation" / f"{scenario}.log"
        prep_log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-B", str(prep_script)]
        with prep_log.open("w", encoding="utf-8") as log:
            done = subprocess.run(
                command,
                cwd=repo_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if done.returncode != 0 or not prepared_path.is_file():
            raise RuntimeError(
                f"dataset preparation failed for {scenario} with exit {done.returncode}; "
                f"see {prep_log}"
            )
    return {
        "manifest_path": str(manifest_path.relative_to(repo_root)),
        "manifest_sha256": _hash_file(manifest_path),
        "prepared_path": str(prepared_path.relative_to(repo_root)),
        "prepared_sha256": _hash_file(prepared_path),
        "prepared_bytes": prepared_path.stat().st_size,
        "task_type": str(manifest.get("task_type", "regression")),
    }


def _manifest_config(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest.get("config", {})


def verify_resume_compatibility(saved: dict[str, Any], current: dict[str, Any]) -> None:
    """Reject resuming an experiment with changed config, code, runtime, or data."""
    mismatches: list[str] = []
    if saved.get("config") != current.get("config"):
        mismatches.append("config")
    saved_source = saved.get("source_state", {}).get("source_fingerprint_sha256")
    current_source = current.get("source_state", {}).get("source_fingerprint_sha256")
    if not saved_source or saved_source != current_source:
        mismatches.append("source")
    if saved.get("runtime") != current.get("runtime"):
        mismatches.append("runtime")
    if saved.get("dataset_identities") != current.get("dataset_identities"):
        mismatches.append("dataset")
    if saved.get("environment") != current.get("environment"):
        mismatches.append("environment")
    if saved.get("external_baseline") != current.get("external_baseline"):
        mismatches.append("external baseline")
    if mismatches:
        raise ValueError("resume mismatch: " + ", ".join(mismatches))


EXTERNAL_COMMON_CONFIG = (
    "depths",
    "seeds",
    "scenarios",
    "models",
    "candidate_model",
    "rounds",
    "learning_rate",
    "threads",
    "binning_strategy",
    "bin_count",
)


def validate_external_baseline(
    candidate: dict[str, Any], baseline: dict[str, Any], baseline_arm: str
) -> dict[str, Any]:
    """Validate a separate build's baseline while explicitly allowing treatment code."""
    if baseline_arm not in baseline.get("arms", {}):
        raise ValueError(f"external baseline has no arm {baseline_arm!r}")
    candidate_config = _manifest_config(candidate)
    baseline_config = _manifest_config(baseline)
    mismatches = [
        key
        for key in EXTERNAL_COMMON_CONFIG
        if candidate_config.get(key, "alloygbm" if key == "candidate_model" else None)
        != baseline_config.get(key, "alloygbm" if key == "candidate_model" else None)
    ]
    if candidate.get("dataset_identities") != baseline.get("dataset_identities"):
        mismatches.append("dataset identities")
    if mismatches:
        raise ValueError("external baseline mismatch: " + ", ".join(mismatches))
    candidate_source = candidate.get("source_state", {}).get("source_fingerprint_sha256")
    baseline_source = baseline.get("source_state", {}).get("source_fingerprint_sha256")
    candidate_native = candidate.get("runtime", {}).get("native_module_sha256")
    baseline_native = baseline.get("runtime", {}).get("native_module_sha256")
    candidate_package = candidate.get("runtime", {}).get("package_sha256")
    baseline_package = baseline.get("runtime", {}).get("package_sha256")
    if not all((candidate_source, baseline_source, candidate_native, baseline_native, candidate_package, baseline_package)):
        raise ValueError("external baseline lacks source or native runtime identity")
    return {
        "baseline_arm": baseline_arm,
        "source_changed": candidate_source != baseline_source,
        "candidate_source_fingerprint_sha256": candidate_source,
        "baseline_source_fingerprint_sha256": baseline_source,
        "native_module_changed": candidate_native != baseline_native,
        "candidate_native_module_sha256": candidate_native,
        "baseline_native_module_sha256": baseline_native,
        "package_changed": candidate_package != baseline_package,
        "candidate_package_sha256": candidate_package,
        "baseline_package_sha256": baseline_package,
        "explicit_source_treatment": True,
    }


def load_result_file(
    result_path: Path,
    *,
    expected_run_id: str | None,
    scenarios: list[str],
    models: list[str],
    seed: int,
    depth: int,
    rounds: int,
    learning_rate: float,
    binning_strategy: str | None = None,
    bin_count: int | None = None,
    threads: int | None = None,
    expected_overrides: dict[str, Any] | None = None,
    expected_task_types: dict[str, str] | None = None,
    expected_sha256: str | None = None,
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], str, str, dict[str, Any]]:
    if not result_path.is_file():
        raise ValueError(f"missing exact result artifact {result_path}")
    digest = _hash_file(result_path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"result artifact hash mismatch: {result_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    run_id = payload.get("run_id") if isinstance(payload, dict) else None
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError(f"result run_id mismatch for {result_path}")
    params = payload.get("params") if isinstance(payload, dict) else None
    if not isinstance(params, dict):
        raise ValueError(f"comparison artifact has no params provenance: {result_path}")
    if params.get("scenarios") != scenarios:
        raise ValueError(f"scenario config mismatch in {result_path}")
    if params.get("models_filter") != models:
        raise ValueError(f"model config mismatch in {result_path}")
    if expected_overrides is not None and params.get("requested_alloy_param_overrides") != expected_overrides:
        raise ValueError(f"requested AlloyGBM overrides mismatch in {result_path}")
    if params.get("profile_seeds") != [seed]:
        raise ValueError(f"seed config mismatch in {result_path}")
    profiles = params.get("profiles")
    expected_profile = {
        "name": "single",
        "learning_rate": learning_rate,
        "max_depth": depth,
        "rounds": rounds,
    }
    if profiles != [expected_profile]:
        raise ValueError(f"profile config mismatch in {result_path}")
    if (
        binning_strategy is not None
        and params.get("alloy_continuous_binning_strategy") != binning_strategy
    ):
        raise ValueError(f"binning strategy config mismatch in {result_path}")
    if (
        bin_count is not None
        and params.get("alloy_continuous_binning_max_bins") != bin_count
    ):
        raise ValueError(f"bin count config mismatch in {result_path}")
    if threads is not None:
        actual_threads = params.get("threads_per_library")
        if not isinstance(actual_threads, int) or actual_threads < 1:
            raise ValueError(f"thread budget missing in {result_path}")
        if threads > 0 and actual_threads != threads:
            raise ValueError(f"thread budget mismatch in {result_path}")
    cells = validate_records(
        payload,
        scenarios,
        models,
        seed,
        depth,
        rounds,
        learning_rate,
        expected_overrides=expected_overrides,
        expected_task_types=expected_task_types,
    )
    return cells, str(run_id), digest, params


def _unit_key(depth: int, arm: str, seed: int) -> str:
    return f"d{depth}/{arm}/s{seed}"


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_experiment_cells(
    manifest: dict[str, Any],
    experiment_root: Path,
    *,
    arms: list[str],
    depths: list[int],
) -> dict[tuple[str, int], dict[tuple[str, str, int], dict[str, Any]]]:
    config = manifest["config"]
    dataset_identities = manifest.get("dataset_identities", {})
    expected_task_types = {
        scenario: identity["task_type"]
        for scenario in config["scenarios"]
        if (identity := dataset_identities.get(scenario))
        and identity.get("task_type") is not None
    }
    if set(expected_task_types) != set(config["scenarios"]):
        raise ValueError("saved dataset identities are missing task_type for replay")
    loaded: dict[tuple[str, int], dict[tuple[str, str, int], dict[str, Any]]] = {}
    units = manifest.get("units", {})
    for depth in depths:
        for arm in arms:
            combined: dict[tuple[str, str, int], dict[str, Any]] = {}
            for seed in config["seeds"]:
                key = _unit_key(depth, arm, seed)
                unit = units.get(key)
                if not unit or unit.get("status") != "PASS":
                    raise ValueError(f"incomplete or failed experiment unit {key}")
                result_path = experiment_root / unit["result_path"]
                cells, run_id, digest, _ = load_result_file(
                    result_path,
                    expected_run_id=unit.get("result_run_id"),
                    scenarios=config["scenarios"],
                    models=config["models"],
                    seed=seed,
                    depth=depth,
                    rounds=config["rounds"],
                    learning_rate=config["learning_rate"],
                    binning_strategy=config["binning_strategy"],
                    bin_count=config["bin_count"],
                    threads=config["threads"],
                    expected_overrides=manifest["arms"][arm]["parsed_overrides"],
                    expected_task_types=expected_task_types,
                    expected_sha256=unit.get("result_sha256"),
                )
                unit["result_run_id"] = run_id
                unit["result_sha256"] = digest
                for cell_key, record in cells.items():
                    if cell_key in combined:
                        raise ValueError(f"duplicate seed cell while loading {key}: {cell_key!r}")
                    combined[cell_key] = record
            loaded[(arm, depth)] = combined
    return loaded


def _cells_for_arm_depth(
    loaded: dict[tuple[str, int], dict[tuple[str, str, int], dict[str, Any]]],
    arm: str,
    depth: int,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    return loaded[(arm, depth)]


def _render_analysis_report(
    manifest: dict[str, Any],
    results: dict[str, Any],
    treatment: dict[str, Any] | None,
) -> str:
    config = manifest["config"]
    evidence_class = manifest.get("evidence_class", "smoke")
    lines = [
        "# Accuracy-at-depth sweep",
        "",
        f"Evidence class: **{evidence_class}** ({len(config['seeds'])} seeds).",
        f"Candidate model: `{config['candidate_model']}`.",
        f"Scenarios: {', '.join(config['scenarios'])}.",
        f"Models: {', '.join(config['models'])}.",
        f"Depths: {', '.join(map(str, config['depths']))}; rounds={config['rounds']}; "
        f"learning_rate={config['learning_rate']}; threads={config['threads']}.",
        "",
        "The default suite excludes `synthetic_categorical` because numeric coercion "
        "drops every row in that fixture. Relative verdicts use each scenario's own "
        "observed seed range; zero reference medians have absolute differences and "
        "an undefined relative verdict.",
        "",
    ]
    if manifest.get("evidence_reason"):
        lines.insert(4, f"Smoke reason: {manifest['evidence_reason']}.")
    if treatment is not None:
        lines.extend(
            [
                "External baseline treatment identity:",
                f"- Source changed: {treatment['source_changed']}.",
                f"- Python package changed: {treatment['package_changed']}.",
                f"- Native module changed: {treatment['native_module_changed']}.",
                f"- Baseline arm: `{treatment['baseline_arm']}`.",
                "",
            ]
        )
    for depth_key, arm_results in results.items():
        lines.extend(
            [
                f"## {depth_key}",
                "",
                "| Scenario | Arm | Reference | Metric | Candidate median / spread | Reference median / spread | Absolute delta | Relative gap | Band | Verdict |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for arm_name, analysis in arm_results.items():
            for scenario, entry in analysis["scenarios"].items():
                comparisons = [(analysis["baseline_arm"], entry["candidate_vs_baseline"])]
                comparisons.extend(entry["candidate_vs_peers"].items())
                for reference_name, comparison in comparisons:
                    candidate_median = comparison["candidate"]["median"]
                    reference_median = comparison["reference"]["median"]
                    gap = comparison["relative_gap_pct"]
                    band = comparison["band_pct"]
                    gap_text = f"{gap:+.2f}%" if gap is not None else "undefined"
                    band_text = f"{band:.2f}%" if band is not None else "undefined"
                    verdict = comparison["verdict"] or "undefined"
                    candidate_spread = comparison["candidate"]["absolute_spread"]
                    candidate_spread_pct = comparison["candidate"]["spread_pct"]
                    reference_spread = comparison["reference"]["absolute_spread"]
                    reference_spread_pct = comparison["reference"]["spread_pct"]
                    candidate_pct_text = (
                        f"{candidate_spread_pct:.2f}%"
                        if candidate_spread_pct is not None
                        else "undefined"
                    )
                    reference_pct_text = (
                        f"{reference_spread_pct:.2f}%"
                        if reference_spread_pct is not None
                        else "undefined"
                    )
                    lines.append(
                        f"| {scenario} | {arm_name} | {reference_name} | {entry['metric']} "
                        f"| {candidate_median:.8g} / {candidate_spread:.8g} ({candidate_pct_text}) "
                        f"| {reference_median:.8g} / {reference_spread:.8g} ({reference_pct_text}) "
                        f"| {comparison['absolute_delta_median']:+.8g} | {gap_text} "
                        f"| {band_text} | {verdict} |"
                    )
        lines.append("")
    lines.extend(
        [
            "## Guard results",
            "",
            "Treatment status compares the candidate with the uncapped baseline: `REGRESSION` means the candidate is worse, `INCONCLUSIVE` means the relative comparison is undefined, and `CLEAR` means tie or improvement.",
            "Peer status reports current candidate-versus-peer standing: `STANDING_DEFICIT` means the candidate loses to at least one selected peer beyond the scenario band, `INCONCLUSIVE` means at least one peer comparison is undefined and none is a loss, and `CLEAR` means all peer comparisons are ties or wins. Peer status is not a measure of change from the uncapped baseline.",
            "",
        ]
    )
    for depth_key, arm_results in results.items():
        for arm_name, analysis in arm_results.items():
            for guard, entry in analysis["guards"].items():
                lines.append(
                    f"- {depth_key}, {arm_name}, {guard}: treatment **{entry['status']}**; "
                    f"current peer standing **{entry['peer_status']}**."
                )
    lines.append("")
    return "\n".join(lines)


def _write_analysis(
    output_dir: Path,
    manifest: dict[str, Any],
    loaded: dict[tuple[str, int], dict[tuple[str, str, int], dict[str, Any]]],
    arms_to_analyse: list[str],
    reference_arm: str,
    reference_loaded: dict[tuple[str, int], dict[tuple[str, str, int], dict[str, Any]]],
    treatment: dict[str, Any] | None,
) -> dict[str, Any]:
    config = manifest["config"]
    results: dict[str, Any] = {}
    for depth in config["depths"]:
        depth_key = f"d{depth}"
        results[depth_key] = {}
        for arm in arms_to_analyse:
            candidate = _cells_for_arm_depth(loaded, arm, depth)
            baseline = _cells_for_arm_depth(reference_loaded, reference_arm, depth)
            results[depth_key][arm] = analyse_cells(
                candidate,
                baseline,
                config["scenarios"],
                config["models"],
                config["seeds"],
                reference_arm,
                candidate_model=config["candidate_model"],
            )
    analysis = {
        "schema_version": MANIFEST_VERSION,
        "evidence_class": manifest.get("evidence_class", "smoke"),
        "evidence_reason": manifest.get("evidence_reason"),
        "candidate_model": config["candidate_model"],
        "source_state": manifest.get("source_state"),
        "runtime": manifest.get("runtime"),
        "dataset_identities": manifest.get("dataset_identities"),
        "config": config,
        "arms": manifest.get("arms"),
        "treatment": treatment,
        "results": results,
        "guard_scenarios": list(GUARD_SCENARIOS),
    }
    analysis_path = output_dir / "analysis.json"
    _save_json(analysis_path, analysis)
    report_path = output_dir / "report.md"
    report_path.write_text(_render_analysis_report(manifest, results, treatment), encoding="utf-8")
    print(_render_analysis_report(manifest, results, treatment))
    print(f"wrote {analysis_path}")
    print(f"wrote {report_path}")
    return analysis


def _validate_cli(args: argparse.Namespace) -> tuple[list[int], list[int], dict[str, Any]]:
    depths = parse_depths(args.depths)
    seeds = parse_seeds(args.seeds, args.base_seed)
    scenarios = list(args.scenarios or DEFAULT_SCENARIOS)
    models = list(args.models or DEFAULT_MODELS)
    unknown_scenarios = sorted(set(scenarios).difference(AVAILABLE_SCENARIOS))
    if unknown_scenarios:
        raise ValueError(f"unknown scenario(s): {unknown_scenarios}")
    unsupported = sorted(set(scenarios).intersection(KNOWN_UNSUPPORTED_SCENARIOS))
    if unsupported:
        detail = "; ".join(
            f"{name}: {KNOWN_UNSUPPORTED_SCENARIOS[name]}" for name in unsupported
        )
        raise ValueError(detail)
    if len(set(scenarios)) != len(scenarios):
        raise ValueError("--scenarios contains a duplicate scenario")
    if len(set(models)) != len(models):
        raise ValueError("--models contains a duplicate model")
    unknown_models = sorted(set(models).difference(VALID_MODELS))
    if unknown_models:
        raise ValueError(f"unknown model(s): {unknown_models}")
    candidate_model = resolve_candidate_model(models, args.candidate_model)
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be finite and positive")
    if args.threads < 0:
        raise ValueError("--threads must be non-negative")
    if not 2 <= args.bin_count <= 256:
        raise ValueError("--bin-count must be in [2, 256]")
    if args.binning_strategy not in {"linear", "rank", "quantile"}:
        raise ValueError("--binning-strategy must be linear, rank, or quantile")
    arms = parse_arm_specs(args.arm) if args.arm else {}
    if not args.analyze_only and not arms:
        raise ValueError("at least one --arm NAME:KEY=VALUE[,KEY=VALUE] is required")
    if not args.external_baseline and not args.analyze_only and args.baseline_arm not in arms:
        raise ValueError(f"--baseline-arm {args.baseline_arm!r} is not among --arm names")
    config = {
        "depths": depths,
        "seeds": seeds,
        "scenarios": scenarios,
        "models": models,
        "candidate_model": candidate_model,
        "rounds": args.rounds,
        "learning_rate": args.learning_rate,
        "threads": args.threads,
        "binning_strategy": args.binning_strategy,
        "bin_count": args.bin_count,
        "profile_name": "single",
        "default_scenario_exclusions": KNOWN_UNSUPPORTED_SCENARIOS,
    }
    return depths, seeds, {"config": config, "arms": arms}


def _initial_manifest(
    *,
    config: dict[str, Any],
    arms: dict[str, dict[str, Any]],
    source: dict[str, Any],
    runtime: dict[str, Any],
    datasets: dict[str, Any],
    environment: dict[str, str],
) -> dict[str, Any]:
    evidence_class, evidence_reason = evidence_class_for(
        config["seeds"], config["scenarios"]
    )
    return {
        "schema_version": MANIFEST_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "arms": arms,
        "source_state": source,
        "runtime": runtime,
        "dataset_identities": datasets,
        "environment": environment,
        "evidence_class": evidence_class,
        "evidence_reason": evidence_reason,
        "completed_unit_count": 0,
        "units": {},
        "exclusions": KNOWN_UNSUPPORTED_SCENARIOS,
    }


def _stream_child(command: list[str], repo_root: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(
            command,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert child.stdout is not None
        for line in child.stdout:
            log.write(line)
            log.flush()
            if line.startswith("[") or line.startswith("  error:"):
                print(line, end="", flush=True)
        return child.wait()


def run_sweep(
    *,
    output_dir: Path,
    repo_root: Path,
    config: dict[str, Any],
    arms: dict[str, dict[str, Any]],
    baseline_arm: str,
    resume: bool,
    external_baseline_path: Path | None,
    external_baseline_arm: str | None,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    saved: dict[str, Any] | None = None
    if manifest_path.exists():
        if not resume:
            raise ValueError(f"{manifest_path} already exists; pass --resume for a verified continuation")
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif any(output_dir.iterdir()):
        raise ValueError(
            f"output directory {output_dir} is non-empty and has no {MANIFEST_NAME}"
        )

    source = source_identity(repo_root, config["scenarios"])
    runtime = runtime_identity()
    environment = selected_environment()
    external_root: Path | None = None
    external_manifest: dict[str, Any] | None = None
    external_manifest_path: Path | None = None
    external_manifest_sha256: str | None = None
    treatment: dict[str, Any] | None = None
    external_arm: str | None = None
    external_reference: dict[str, Any] | None = None
    if external_baseline_path is not None:
        external_root = external_baseline_path.expanduser().resolve()
        external_manifest_path = external_root / MANIFEST_NAME
        if not external_manifest_path.is_file():
            raise ValueError(f"external baseline has no {MANIFEST_NAME}: {external_manifest_path}")
        external_manifest_sha256 = _hash_file(external_manifest_path)
        external_manifest = json.loads(external_manifest_path.read_text(encoding="utf-8"))
        external_arm = external_baseline_arm or "baseline"
        external_reference = {
            "path": str(external_root),
            "arm": external_arm,
            "manifest_sha256": external_manifest_sha256,
        }

    current = _initial_manifest(
        config=config,
        arms=arms,
        source=source,
        runtime=runtime,
        datasets={},
        environment=environment,
    )
    current["external_baseline"] = external_reference
    current["status"] = "INITIALIZING"
    if saved is not None:
        # Preparation may have stopped partway through. Compare stable identity
        # first, then check all saved dataset hashes once preparation completes.
        saved_static = dict(saved)
        current_static = dict(current)
        saved_static["dataset_identities"] = {}
        current_static["dataset_identities"] = {}
        if saved.get("external_baseline") is not None:
            saved_static["external_baseline"] = {
                key: saved["external_baseline"].get(key)
                for key in ("path", "arm", "manifest_sha256")
            }
        verify_resume_compatibility(saved_static, current_static)
        if saved.get("arms") != arms:
            raise ValueError("resume mismatch: arms")
        manifest = saved
        prior_status = saved.get("status", "READY")
    else:
        manifest = current
        _save_json(manifest_path, manifest)
        prior_status = None

    prior_datasets = (saved or {}).get("dataset_identities", {})
    manifest["status"] = "INITIALIZING"
    _save_json(manifest_path, manifest)
    datasets: dict[str, Any] = {}
    for scenario in config["scenarios"]:
        identity = ensure_dataset(repo_root, scenario, output_dir)
        if scenario in prior_datasets and prior_datasets[scenario] != identity:
            raise ValueError(f"resume mismatch: dataset identity changed for {scenario}")
        datasets[scenario] = identity
        manifest["dataset_identities"] = {**prior_datasets, **datasets}
        _save_json(manifest_path, manifest)

    current = _initial_manifest(
        config=config,
        arms=arms,
        source=source,
        runtime=runtime,
        datasets=datasets,
        environment=environment,
    )
    if external_manifest is not None and external_arm is not None:
        treatment = validate_external_baseline(current, external_manifest, external_arm)
        current["external_baseline"] = {
            **(external_reference or {}),
            "treatment_identity": treatment,
        }
    else:
        current["external_baseline"] = None

    if saved is not None:
        if prior_status != "INITIALIZING":
            verify_resume_compatibility(saved, current)
        elif (saved.get("external_baseline") or {}).get("treatment_identity") not in (
            None,
            (current.get("external_baseline") or {}).get("treatment_identity"),
        ):
            raise ValueError("resume mismatch: external baseline")
        manifest.update(
            {
                "dataset_identities": datasets,
                "evidence_class": current["evidence_class"],
                "evidence_reason": current["evidence_reason"],
                "external_baseline": current["external_baseline"],
                "status": "READY",
            }
        )
    else:
        manifest = current
        manifest["status"] = "READY"
    _save_json(manifest_path, manifest)

    for depth in config["depths"]:
        for arm_name, arm in arms.items():
            for seed in config["seeds"]:
                unit_key = _unit_key(depth, arm_name, seed)
                unit = manifest["units"].get(unit_key, {})
                target_dir = output_dir / "runs" / arm_name / f"d{depth}" / f"s{seed}"
                result_path = target_dir / RESULT_NAME
                if resume and unit.get("status") == "PASS":
                    try:
                        load_result_file(
                            result_path,
                            expected_run_id=unit.get("result_run_id"),
                            scenarios=config["scenarios"],
                            models=config["models"],
                            seed=seed,
                            depth=depth,
                            rounds=config["rounds"],
                            learning_rate=config["learning_rate"],
                            binning_strategy=config["binning_strategy"],
                            bin_count=config["bin_count"],
                            threads=config["threads"],
                            expected_overrides=arm["parsed_overrides"],
                            expected_task_types={
                                scenario: datasets[scenario]["task_type"]
                                for scenario in config["scenarios"]
                            },
                            expected_sha256=unit.get("result_sha256"),
                        )
                        continue
                    except (OSError, ValueError, json.JSONDecodeError):
                        print(f"[resume] rerunning unverifiable unit {unit_key}", flush=True)
                command_args = {
                    "scenarios": config["scenarios"],
                    "models": config["models"],
                    "threads": config["threads"],
                    "rounds": config["rounds"],
                    "learning_rate": config["learning_rate"],
                    "binning_strategy": config["binning_strategy"],
                    "bin_count": config["bin_count"],
                }
                command = build_runner_command(
                    repo_root=repo_root,
                    target_dir=target_dir,
                    overrides=arm["overrides"],
                    depth=depth,
                    seed=seed,
                    args=command_args,
                )
                log_path = target_dir / "runner.log"
                relative_result = result_path.relative_to(output_dir).as_posix()
                relative_log = log_path.relative_to(output_dir).as_posix()
                manifest["units"][unit_key] = {
                    "status": "RUNNING",
                    "command": command,
                    "returncode": None,
                    "log_path": relative_log,
                    "result_path": relative_result,
                    "requested_overrides": arm["parsed_overrides"],
                    "effective_parameters": None,
                }
                _save_json(manifest_path, manifest)
                print(f"[start] arm={arm_name} depth={depth} seed={seed}", flush=True)
                returncode = _stream_child(command, repo_root, log_path)
                manifest["units"][unit_key]["returncode"] = returncode
                if returncode != 0:
                    manifest["units"][unit_key]["status"] = "FAIL"
                    manifest["units"][unit_key]["error"] = "comparison runner returned non-zero"
                    _save_json(manifest_path, manifest)
                    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
                    raise RunnerExitError(
                        returncode,
                        f"arm={arm_name} depth={depth} seed={seed} failed with exit {returncode}; "
                        f"log={log_path}\n" + "\n".join(tail)
                    )
                try:
                    cells, run_id, result_hash, runner_params = load_result_file(
                        result_path,
                        expected_run_id=None,
                        scenarios=config["scenarios"],
                        models=config["models"],
                        seed=seed,
                        depth=depth,
                        rounds=config["rounds"],
                        learning_rate=config["learning_rate"],
                        binning_strategy=config["binning_strategy"],
                        bin_count=config["bin_count"],
                        threads=config["threads"],
                        expected_overrides=arm["parsed_overrides"],
                        expected_task_types={
                            scenario: datasets[scenario]["task_type"]
                            for scenario in config["scenarios"]
                        },
                    )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    manifest["units"][unit_key]["status"] = "FAIL"
                    manifest["units"][unit_key]["error"] = str(exc)
                    _save_json(manifest_path, manifest)
                    raise RuntimeError(
                        f"invalid comparison result for {unit_key}: {exc}; log={log_path}"
                    ) from exc
                effective = {
                    f"{scenario}/{model}": {
                        "requested_alloy_param_overrides": record.get("requested_alloy_param_overrides"),
                        "constructed_estimator_params": record.get("constructed_estimator_params"),
                        "resolved_training_policy": record.get("resolved_training_policy"),
                    }
                    for (scenario, model, _), record in cells.items()
                    if model.startswith("alloygbm")
                }
                manifest["units"][unit_key].update(
                    {
                        "status": "PASS",
                        "result_run_id": run_id,
                        "result_sha256": result_hash,
                        "runner_parameters": runner_params,
                        "effective_parameters": effective,
                        "cell_count": len(cells),
                    }
                )
                manifest["completed_unit_count"] = sum(
                    unit_record.get("status") == "PASS"
                    for unit_record in manifest["units"].values()
                )
                _save_json(manifest_path, manifest)
                print(
                    f"[unit-done] arm={arm_name} depth={depth} seed={seed} "
                    f"cells={len(cells)} result={relative_result}",
                    flush=True,
                )

    local_arm_names = list(arms)
    loaded = _load_experiment_cells(
        manifest,
        output_dir,
        arms=local_arm_names,
        depths=config["depths"],
    )
    reference_arm = baseline_arm
    reference_loaded = loaded
    if external_manifest is not None and external_root is not None:
        assert external_manifest_path is not None
        assert external_manifest_sha256 is not None
        if _hash_file(external_manifest_path) != external_manifest_sha256:
            raise ValueError("external baseline manifest changed while the sweep was running")
        reference_arm = current["external_baseline"]["arm"]
        reference_loaded = _load_experiment_cells(
            external_manifest,
            external_root,
            arms=[reference_arm],
            depths=config["depths"],
        )
        manifest["external_baseline"] = current["external_baseline"]
        _save_json(manifest_path, manifest)
    elif baseline_arm not in arms:
        raise ValueError(f"baseline arm {baseline_arm!r} is not present")
    manifest["analysis_completed_utc"] = datetime.now(timezone.utc).isoformat()
    _save_json(manifest_path, manifest)
    _write_analysis(
        output_dir,
        manifest,
        loaded,
        local_arm_names,
        reference_arm,
        reference_loaded,
        treatment,
    )
    return manifest


def analyse_saved_experiment(
    output_dir: Path,
    *,
    baseline_arm: str,
    external_baseline_path: Path | None,
    external_baseline_arm: str | None,
) -> dict[str, Any]:
    """Replay exact stored artifacts without consulting current code/runtime/data."""
    output_dir = output_dir.expanduser().resolve()
    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"analyze-only needs saved {MANIFEST_NAME} at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["config"]
    if "candidate_model" not in config:
        # Earlier v1 manifests used literal alloygbm as their analysis candidate.
        if "alloygbm" not in config.get("models", []):
            raise ValueError("saved experiment has no candidate_model identity")
        config["candidate_model"] = "alloygbm"
    manifest["evidence_class"], manifest["evidence_reason"] = evidence_class_for(
        config["seeds"], config["scenarios"]
    )
    arms = list(manifest["arms"])
    loaded = _load_experiment_cells(manifest, output_dir, arms=arms, depths=config["depths"])
    reference_arm = baseline_arm
    reference_loaded = loaded
    treatment = None
    if external_baseline_path is not None:
        external_root = external_baseline_path.expanduser().resolve()
        external_manifest_path = external_root / MANIFEST_NAME
        if not external_manifest_path.is_file():
            raise ValueError(f"external baseline has no {MANIFEST_NAME}: {external_manifest_path}")
        external_manifest = json.loads(external_manifest_path.read_text(encoding="utf-8"))
        reference_arm = external_baseline_arm or "baseline"
        treatment = validate_external_baseline(manifest, external_manifest, reference_arm)
        reference_loaded = _load_experiment_cells(
            external_manifest,
            external_root,
            arms=[reference_arm],
            depths=config["depths"],
        )
    elif reference_arm not in arms:
        raise ValueError(f"baseline arm {reference_arm!r} is not present in saved experiment")
    return _write_analysis(
        output_dir,
        manifest,
        loaded,
        arms,
        reference_arm,
        reference_loaded,
        treatment,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME:KEY=VALUE[,KEY=VALUE]",
        help="named AlloyGBM override set; repeat. `baseline:` is a valid empty arm.",
    )
    parser.add_argument("--depths", default="6,12", help="comma-separated depths (default: 6,12)")
    parser.add_argument(
        "--seeds",
        default="5",
        help="seed count (default 5) or explicit comma-separated seed identifiers",
    )
    parser.add_argument("--base-seed", type=int, default=20260919)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument(
        "--binning-strategy",
        "--alloy-continuous-binning-strategy",
        dest="binning_strategy",
        choices=("linear", "rank", "quantile"),
        default="linear",
    )
    parser.add_argument(
        "--bin-count",
        "--alloy-continuous-binning-max-bins",
        dest="bin_count",
        type=int,
        default=256,
    )
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--candidate-model",
        default=None,
        help="selected AlloyGBM model to compare (derive only when one AlloyGBM model is selected)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-arm", default="baseline")
    parser.add_argument("--external-baseline", type=Path)
    parser.add_argument("--external-baseline-arm", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        depths, seeds, parsed = _validate_cli(args)
        output_dir = args.output_dir.expanduser().resolve()
        repo_root = Path(__file__).resolve().parents[1]
        if args.analyze_only:
            analyse_saved_experiment(
                output_dir,
                baseline_arm=args.baseline_arm,
                external_baseline_path=args.external_baseline,
                external_baseline_arm=args.external_baseline_arm,
            )
            return 0
        config = parsed["config"]
        config["depths"] = depths
        config["seeds"] = seeds
        run_sweep(
            output_dir=output_dir,
            repo_root=repo_root,
            config=config,
            arms=parsed["arms"],
            baseline_arm=args.baseline_arm,
            resume=args.resume,
            external_baseline_path=args.external_baseline,
            external_baseline_arm=args.external_baseline_arm,
        )
    except RunnerExitError as exc:
        print(f"accuracy-depth-sweep: {exc}", file=sys.stderr)
        return exc.returncode if 0 < exc.returncode < 126 else 2
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"accuracy-depth-sweep: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
