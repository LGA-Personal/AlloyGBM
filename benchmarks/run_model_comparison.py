#!/usr/bin/env python3
"""Run cross-model benchmark comparisons for AlloyGBM, LightGBM, and XGBoost."""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

AVAILABLE_SCENARIOS = [
    "dense_numeric",
    "panel_time_series",
    "histogram_stress",
    "dow_jones_financial",
]


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    learning_rate: float
    max_depth: int
    rounds: int


DEFAULT_PROFILES = [
    BenchmarkProfile("shallow_high_lr", learning_rate=0.20, max_depth=4, rounds=200),
    BenchmarkProfile("mid_balanced", learning_rate=0.05, max_depth=6, rounds=1200),
    BenchmarkProfile("deep_low_lr", learning_rate=0.01, max_depth=8, rounds=5000),
]

ULTRA_PROFILE = BenchmarkProfile(
    "ultra_low_lr", learning_rate=0.005, max_depth=8, rounds=10000
)

REQUIRED_ALLOY_INIT_PARAMS = (
    "learning_rate",
    "max_depth",
    "n_estimators",
    "row_subsample",
    "col_subsample",
)


@dataclass
class BenchmarkRecord:
    scenario: str
    profile_name: str
    profile_index: int
    run_index: int
    seed: int
    learning_rate: float
    max_depth: int
    rounds: int
    model: str
    train_rows: int
    test_rows: int
    n_features: int
    fit_seconds: float
    predict_seconds: float
    rmse: float
    mae: float
    r2: float
    status: str
    error: str


def _verify_alloygbm_runtime_contract(
    gbm_regressor_cls: type, native_module: object
) -> tuple[str, ...]:
    signature = inspect.signature(gbm_regressor_cls.__init__)
    parameter_names = tuple(signature.parameters.keys())
    missing = sorted(
        required
        for required in REQUIRED_ALLOY_INIT_PARAMS
        if required not in signature.parameters
    )
    if missing:
        raise RuntimeError(
            "loaded alloygbm.GBMRegressor is not benchmark-compatible; "
            f"missing required __init__ parameters: {', '.join(missing)}"
        )
    if not hasattr(native_module, "train_regression_artifact"):
        raise RuntimeError(
            "loaded alloygbm native extension is not benchmark-compatible; "
            "missing train_regression_artifact"
        )
    return parameter_names


def _load_alloygbm_runtime() -> tuple[type, dict[str, object]]:
    try:
        import alloygbm
        import alloygbm._alloygbm as native_module
        from alloygbm import GBMRegressor
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "failed to import benchmark runtime package 'alloygbm'; "
            "install a current local alloygbm build before running benchmarks"
        ) from exc

    init_parameters = _verify_alloygbm_runtime_contract(GBMRegressor, native_module)
    runtime = {
        "module_path": str(Path(alloygbm.__file__).resolve()),
        "native_module_path": str(Path(native_module.__file__).resolve()),
        "init_parameters": list(init_parameters),
    }
    return GBMRegressor, runtime


def _prepare_dataset(
    repo_root: Path,
    scenario: str,
    force_prepare: bool,
    prepared_path: Path,
    manifest_kind: str,
) -> None:
    if prepared_path.exists() and not force_prepare:
        return

    scenario_script = repo_root / "benchmarks" / scenario / "prepare.py"
    command = [sys.executable, "-B", str(scenario_script)]
    if force_prepare and manifest_kind == "uci_download":
        command.append("--force-download")
    subprocess.run(command, cwd=repo_root, check=True)


def _load_manifest(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        return yaml.safe_load(manifest_file)


def _load_dataset(
    repo_root: Path, scenario: str, force_prepare: bool
) -> tuple[pd.DataFrame, str]:
    manifest_path = repo_root / "benchmarks" / scenario / "manifest.yaml"
    manifest = _load_manifest(manifest_path)
    target_column = str(manifest["prepared"]["target"])
    prepared_file = str(manifest["prepared"]["filename"])
    manifest_kind = str(manifest.get("kind", ""))
    prepared_path = repo_root / "benchmarks" / "data" / scenario / "prepared" / prepared_file

    _prepare_dataset(repo_root, scenario, force_prepare, prepared_path, manifest_kind)
    if not prepared_path.exists():
        raise FileNotFoundError(f"prepared dataset missing: {prepared_path}")

    frame = pd.read_csv(prepared_path)
    if target_column not in frame.columns:
        raise ValueError(f"target column '{target_column}' missing in {prepared_path}")
    return frame, target_column


def _split_by_timestamp(frame: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Split on unique timestamps so no timestamp appears in both train and test.
    ordered = frame.sort_values(["timestamp", "group_id"], kind="mergesort")
    unique_timestamps = sorted({str(value) for value in ordered["timestamp"]})
    if len(unique_timestamps) < 2:
        raise ValueError("need at least 2 unique timestamps for split")

    train_timestamp_count = max(1, int(len(unique_timestamps) * (1.0 - test_size)))
    if train_timestamp_count >= len(unique_timestamps):
        train_timestamp_count = len(unique_timestamps) - 1
    train_timestamps = set(unique_timestamps[:train_timestamp_count])

    train = ordered[ordered["timestamp"].astype(str).isin(train_timestamps)]
    test = ordered[~ordered["timestamp"].astype(str).isin(train_timestamps)]
    if train.empty or test.empty:
        raise ValueError("timestamp split produced empty train/test partition")
    return train, test


def _split_dataset(
    scenario: str,
    frame: pd.DataFrame,
    target_column: str,
    seed: int,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_equivalent_features: list[str] = []
    target_series = pd.to_numeric(frame[target_column], errors="coerce")
    for column in frame.columns:
        if column in {target_column, "group_id", "timestamp"}:
            continue
        feature_series = pd.to_numeric(frame[column], errors="coerce")
        comparable = feature_series.notna() & target_series.notna()
        if comparable.any() and (feature_series[comparable] == target_series[comparable]).all():
            target_equivalent_features.append(column)

    if target_equivalent_features:
        raise ValueError(
            f"{scenario}: target-equivalent features detected: "
            + ", ".join(sorted(target_equivalent_features))
        )

    feature_frame = frame.drop(columns=[target_column]).copy()
    if "group_id" in feature_frame.columns:
        feature_frame = feature_frame.drop(columns=["group_id"])

    if "timestamp" in feature_frame.columns:
        try:
            train, test = _split_by_timestamp(frame, test_size)
        except ValueError as exc:
            raise ValueError(f"{scenario}: {exc}") from exc

        x_train = train.drop(columns=[target_column, "group_id", "timestamp"], errors="ignore")
        x_test = test.drop(columns=[target_column, "group_id", "timestamp"], errors="ignore")
        y_train = train[target_column]
        y_test = test[target_column]
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            feature_frame,
            frame[target_column],
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )

    x_train = x_train.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x_test = x_test.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y_train = pd.to_numeric(y_train, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y_test = pd.to_numeric(y_test, errors="coerce").replace([np.inf, -np.inf], np.nan)

    train_mask = x_train.notna().all(axis=1) & y_train.notna()
    test_mask = x_test.notna().all(axis=1) & y_test.notna()
    x_train = x_train[train_mask]
    y_train = y_train[train_mask]
    x_test = x_test[test_mask]
    y_test = y_test[test_mask]

    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError(f"{scenario}: no rows left after numeric/finite filtering")

    return (
        x_train.to_numpy(dtype=float),
        x_test.to_numpy(dtype=float),
        y_train.to_numpy(dtype=float),
        y_test.to_numpy(dtype=float),
    )


def _run_model(
    model_name: str,
    factory: Callable[[], object],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    scenario: str,
    profile: BenchmarkProfile,
    profile_index: int,
    run_index: int,
    seed: int,
) -> BenchmarkRecord:
    try:
        model = factory()
        fit_start = time.perf_counter()
        if model_name == "alloygbm":
            model.fit(x_train.tolist(), y_train.tolist())
        else:
            model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - fit_start

        predict_start = time.perf_counter()
        if model_name == "alloygbm":
            predictions = np.array(model.predict(x_test.tolist()), dtype=float)
        else:
            predictions = np.array(model.predict(x_test), dtype=float)
        predict_seconds = time.perf_counter() - predict_start

        return BenchmarkRecord(
            scenario=scenario,
            profile_name=profile.name,
            profile_index=profile_index,
            run_index=run_index,
            seed=seed,
            learning_rate=profile.learning_rate,
            max_depth=profile.max_depth,
            rounds=profile.rounds,
            model=model_name,
            train_rows=int(len(x_train)),
            test_rows=int(len(x_test)),
            n_features=int(x_train.shape[1]),
            fit_seconds=float(fit_seconds),
            predict_seconds=float(predict_seconds),
            rmse=float(np.sqrt(mean_squared_error(y_test, predictions))),
            mae=float(mean_absolute_error(y_test, predictions)),
            r2=float(r2_score(y_test, predictions)),
            status="PASS",
            error="",
        )
    except Exception as exc:  # noqa: BLE001
        return BenchmarkRecord(
            scenario=scenario,
            profile_name=profile.name,
            profile_index=profile_index,
            run_index=run_index,
            seed=seed,
            learning_rate=profile.learning_rate,
            max_depth=profile.max_depth,
            rounds=profile.rounds,
            model=model_name,
            train_rows=0,
            test_rows=0,
            n_features=0,
            fit_seconds=0.0,
            predict_seconds=0.0,
            rmse=float("nan"),
            mae=float("nan"),
            r2=float("nan"),
            status="FAIL",
            error=f"{type(exc).__name__}: {exc}",
        )


def _model_factories(
    gbm_regressor_cls: type, seed: int, learning_rate: float, max_depth: int, rounds: int
) -> dict:
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    alloy_signature = inspect.signature(gbm_regressor_cls.__init__)
    alloy_params: dict[str, object] = {}
    if "learning_rate" in alloy_signature.parameters:
        alloy_params["learning_rate"] = learning_rate
    if "max_depth" in alloy_signature.parameters:
        alloy_params["max_depth"] = max_depth
    if "n_estimators" in alloy_signature.parameters:
        alloy_params["n_estimators"] = rounds
    if "rounds" in alloy_signature.parameters:
        alloy_params["rounds"] = rounds
    if "row_subsample" in alloy_signature.parameters:
        alloy_params["row_subsample"] = 0.8
    if "col_subsample" in alloy_signature.parameters:
        alloy_params["col_subsample"] = 0.8
    if "seed" in alloy_signature.parameters:
        alloy_params["seed"] = seed
    if "deterministic" in alloy_signature.parameters:
        alloy_params["deterministic"] = True

    return {
        "alloygbm": lambda: gbm_regressor_cls(**alloy_params),
        "lightgbm": lambda: LGBMRegressor(
            objective="regression",
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=rounds,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        ),
        "xgboost": lambda: XGBRegressor(
            objective="reg:squarederror",
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=rounds,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        ),
    }


def _parse_seed_list(seed_text: str) -> list[int]:
    raw_parts = [part.strip() for part in seed_text.split(",") if part.strip()]
    if not raw_parts:
        raise ValueError("profile seed list cannot be empty")

    seeds: list[int] = []
    seen: set[int] = set()
    for part in raw_parts:
        value = int(part)
        if value in seen:
            continue
        seeds.append(value)
        seen.add(value)
    return seeds


def _parse_profile_spec(spec: str) -> BenchmarkProfile:
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) != 4:
        raise ValueError(
            "profile must use 'name:learning_rate:max_depth:rounds' format"
        )

    name, learning_rate_raw, max_depth_raw, rounds_raw = parts
    learning_rate = float(learning_rate_raw)
    max_depth = int(max_depth_raw)
    rounds = int(rounds_raw)

    if not name:
        raise ValueError("profile name cannot be empty")
    if learning_rate <= 0.0:
        raise ValueError("profile learning_rate must be > 0")
    if max_depth <= 0:
        raise ValueError("profile max_depth must be > 0")
    if rounds <= 0:
        raise ValueError("profile rounds must be > 0")

    return BenchmarkProfile(
        name=name,
        learning_rate=learning_rate,
        max_depth=max_depth,
        rounds=rounds,
    )


def _resolve_profiles(
    args: argparse.Namespace,
) -> tuple[list[BenchmarkProfile], list[int], bool]:
    if args.profile and args.profile_grid != "none":
        raise ValueError("use either --profile-grid or --profile, not both")

    if args.profile:
        profiles = [_parse_profile_spec(spec) for spec in args.profile]
        seeds = _parse_seed_list(args.profile_seeds)
        return profiles, seeds, True

    if args.profile_grid == "default":
        profiles = list(DEFAULT_PROFILES)
        seeds = _parse_seed_list(args.profile_seeds)
        return profiles, seeds, True

    if args.profile_grid == "default_ultra":
        profiles = list(DEFAULT_PROFILES) + [ULTRA_PROFILE]
        seeds = _parse_seed_list(args.profile_seeds)
        return profiles, seeds, True

    profile = BenchmarkProfile(
        name="single",
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        rounds=args.rounds,
    )
    return [profile], [args.seed], False


def _summarize_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    passed = frame[frame["status"] == "PASS"].copy()
    if passed.empty:
        return pd.DataFrame()

    summary = (
        passed.groupby(
            [
                "scenario",
                "profile_name",
                "model",
                "learning_rate",
                "max_depth",
                "rounds",
            ],
            as_index=False,
        )
        .agg(
            runs=("fit_seconds", "count"),
            fit_seconds_median=("fit_seconds", "median"),
            predict_seconds_median=("predict_seconds", "median"),
            rmse_median=("rmse", "median"),
            mae_median=("mae", "median"),
            r2_median=("r2", "median"),
        )
        .sort_values(["scenario", "profile_name", "model"])
        .reset_index(drop=True)
    )
    return summary


def _best_rows_by_scenario(
    summary: pd.DataFrame, metric: str, ascending: bool
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    grouped = summary.groupby("scenario", as_index=False)
    index = grouped[metric].idxmin() if ascending else grouped[metric].idxmax()
    return summary.loc[index[metric]].sort_values("scenario")


def _render_results_markdown(
    run_id: str,
    params: dict,
    frame: pd.DataFrame,
    summary: pd.DataFrame,
) -> str:
    lines = [
        f"# Model Comparison ({run_id})",
        "",
        "## Params",
        f"- profile_mode: `{params['profile_mode']}`",
        f"- scenarios: `{', '.join(params['scenarios'])}`",
        f"- test_size: `{params['test_size']}`",
    ]

    if params["profile_mode"] == "single":
        lines.extend(
            [
                f"- seed: `{params['seed']}`",
                f"- learning_rate: `{params['learning_rate']}`",
                f"- max_depth: `{params['max_depth']}`",
                f"- rounds: `{params['rounds']}`",
            ]
        )
    else:
        lines.extend(
            [
                f"- profile_grid: `{params['profile_grid']}`",
                f"- profile_seeds: `{params['profile_seeds']}`",
                "",
                "### Profiles",
            ]
        )
        for profile in params["profiles"]:
            lines.append(
                "- "
                f"`{profile['name']}`: "
                f"lr={profile['learning_rate']}, "
                f"max_depth={profile['max_depth']}, "
                f"rounds={profile['rounds']}"
            )

    lines.extend(["", "## Raw Results", ""])

    if frame.empty:
        lines.append("No benchmark records were produced.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "| scenario | profile | model | seed | status | fit_seconds | predict_seconds | rmse | mae | r2 |",
            "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['scenario']} | {row['profile_name']} | {row['model']} | {int(row['seed'])} | "
            f"{row['status']} | {row['fit_seconds']:.6f} | {row['predict_seconds']:.6f} | "
            f"{row['rmse']:.6f} | {row['mae']:.6f} | {row['r2']:.6f} |"
        )

    lines.extend(["", "## Profile Median Summary", ""])
    if summary.empty:
        lines.append("No PASS records available for profile summary.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "| scenario | profile | model | runs | lr | depth | rounds | fit_seconds_median | predict_seconds_median | rmse_median | mae_median | r2_median |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['scenario']} | {row['profile_name']} | {row['model']} | {int(row['runs'])} | "
            f"{row['learning_rate']:.6f} | {int(row['max_depth'])} | {int(row['rounds'])} | "
            f"{row['fit_seconds_median']:.6f} | {row['predict_seconds_median']:.6f} | "
            f"{row['rmse_median']:.6f} | {row['mae_median']:.6f} | {row['r2_median']:.6f} |"
        )

    best_rmse = _best_rows_by_scenario(summary, metric="rmse_median", ascending=True)
    fastest_fit = _best_rows_by_scenario(
        summary, metric="fit_seconds_median", ascending=True
    )

    lines.extend(["", "## Best RMSE By Scenario", ""])
    if best_rmse.empty:
        lines.append("No best-RMSE rows available.")
    else:
        lines.extend(
            [
                "| scenario | profile | model | rmse_median |",
                "| --- | --- | --- | ---: |",
            ]
        )
        for _, row in best_rmse.iterrows():
            lines.append(
                f"| {row['scenario']} | {row['profile_name']} | {row['model']} | {row['rmse_median']:.6f} |"
            )

    lines.extend(["", "## Fastest Fit By Scenario", ""])
    if fastest_fit.empty:
        lines.append("No fastest-fit rows available.")
    else:
        lines.extend(
            [
                "| scenario | profile | model | fit_seconds_median |",
                "| --- | --- | --- | ---: |",
            ]
        )
        for _, row in fastest_fit.iterrows():
            lines.append(
                f"| {row['scenario']} | {row['profile_name']} | {row['model']} | {row['fit_seconds_median']:.6f} |"
            )

    return "\n".join(lines) + "\n"


def _write_outputs(
    output_dir: Path,
    run_id: str,
    records: list[BenchmarkRecord],
    params: dict,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame([asdict(record) for record in records])
    if not frame.empty:
        frame = frame.sort_values(
            ["scenario", "profile_index", "run_index", "model"]
        ).reset_index(drop=True)

    csv_timestamped = output_dir / f"model_comparison_{run_id}.csv"
    csv_latest = output_dir / "model_comparison_latest.csv"
    frame.to_csv(csv_timestamped, index=False)
    frame.to_csv(csv_latest, index=False)

    json_timestamped = output_dir / f"model_comparison_{run_id}.json"
    json_latest = output_dir / "model_comparison_latest.json"
    payload = {
        "run_id": run_id,
        "params": params,
        "records": frame.to_dict(orient="records"),
    }
    json_timestamped.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    json_latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = _summarize_profiles(frame)

    summary_csv_timestamped = output_dir / f"model_comparison_profile_summary_{run_id}.csv"
    summary_csv_latest = output_dir / "model_comparison_profile_summary_latest.csv"
    summary.to_csv(summary_csv_timestamped, index=False)
    summary.to_csv(summary_csv_latest, index=False)

    summary_json_timestamped = output_dir / f"model_comparison_profile_summary_{run_id}.json"
    summary_json_latest = output_dir / "model_comparison_profile_summary_latest.json"
    summary_payload = {
        "run_id": run_id,
        "params": params,
        "summary": summary.to_dict(orient="records"),
    }
    summary_json_timestamped.write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )
    summary_json_latest.write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )

    markdown_text = _render_results_markdown(run_id, params, frame, summary)
    markdown_timestamped = output_dir / f"model_comparison_{run_id}.md"
    markdown_latest = output_dir / "model_comparison_latest.md"
    markdown_timestamped.write_text(markdown_text, encoding="utf-8")
    markdown_latest.write_text(markdown_text, encoding="utf-8")

    return {
        "csv": csv_timestamped,
        "json": json_timestamped,
        "markdown": markdown_timestamped,
        "summary_csv": summary_csv_timestamped,
        "summary_json": summary_json_timestamped,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=AVAILABLE_SCENARIOS,
        choices=AVAILABLE_SCENARIOS,
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=120)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument(
        "--profile-grid",
        choices=["none", "default", "default_ultra"],
        default="none",
        help="named profile matrix; use none for single-profile compatibility mode",
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help="custom profile spec: name:learning_rate:max_depth:rounds (repeatable)",
    )
    parser.add_argument(
        "--profile-seeds",
        default="7,17,29",
        help="comma-separated seeds for profile-grid/custom-profile modes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks") / "results",
    )
    args = parser.parse_args(argv)

    try:
        profiles, seeds, matrix_mode = _resolve_profiles(args)
    except (TypeError, ValueError) as exc:
        print(f"invalid profile configuration: {exc}", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    try:
        gbm_regressor_cls, alloy_runtime = _load_alloygbm_runtime()
    except RuntimeError as exc:
        print(f"alloygbm runtime check failed: {exc}", file=sys.stderr)
        return 2
    print(
        "alloygbm runtime: "
        f"module={alloy_runtime['module_path']} "
        f"native={alloy_runtime['native_module_path']}"
    )

    datasets: dict[str, tuple[pd.DataFrame, str]] = {}
    dataset_errors: dict[str, str] = {}
    for scenario in args.scenarios:
        try:
            frame, target_column = _load_dataset(repo_root, scenario, args.force_prepare)
            datasets[scenario] = (frame, target_column)
        except Exception as exc:  # noqa: BLE001
            dataset_errors[scenario] = f"{type(exc).__name__}: {exc}"

    records: list[BenchmarkRecord] = []

    for profile_index, profile in enumerate(profiles, start=1):
        for run_index, seed in enumerate(seeds, start=1):
            factories = _model_factories(
                gbm_regressor_cls=gbm_regressor_cls,
                seed=seed,
                learning_rate=profile.learning_rate,
                max_depth=profile.max_depth,
                rounds=profile.rounds,
            )

            for scenario in args.scenarios:
                if scenario in dataset_errors:
                    error = dataset_errors[scenario]
                    for model_name in factories:
                        records.append(
                            BenchmarkRecord(
                                scenario=scenario,
                                profile_name=profile.name,
                                profile_index=profile_index,
                                run_index=run_index,
                                seed=seed,
                                learning_rate=profile.learning_rate,
                                max_depth=profile.max_depth,
                                rounds=profile.rounds,
                                model=model_name,
                                train_rows=0,
                                test_rows=0,
                                n_features=0,
                                fit_seconds=0.0,
                                predict_seconds=0.0,
                                rmse=float("nan"),
                                mae=float("nan"),
                                r2=float("nan"),
                                status="FAIL",
                                error=error,
                            )
                        )
                    continue

                frame, target_column = datasets[scenario]
                try:
                    x_train, x_test, y_train, y_test = _split_dataset(
                        scenario, frame, target_column, seed, args.test_size
                    )
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                    for model_name in factories:
                        records.append(
                            BenchmarkRecord(
                                scenario=scenario,
                                profile_name=profile.name,
                                profile_index=profile_index,
                                run_index=run_index,
                                seed=seed,
                                learning_rate=profile.learning_rate,
                                max_depth=profile.max_depth,
                                rounds=profile.rounds,
                                model=model_name,
                                train_rows=0,
                                test_rows=0,
                                n_features=0,
                                fit_seconds=0.0,
                                predict_seconds=0.0,
                                rmse=float("nan"),
                                mae=float("nan"),
                                r2=float("nan"),
                                status="FAIL",
                                error=error,
                            )
                        )
                    continue

                for model_name, factory in factories.items():
                    record = _run_model(
                        model_name=model_name,
                        factory=factory,
                        x_train=x_train,
                        y_train=y_train,
                        x_test=x_test,
                        y_test=y_test,
                        scenario=scenario,
                        profile=profile,
                        profile_index=profile_index,
                        run_index=run_index,
                        seed=seed,
                    )
                    records.append(record)
                    print(
                        f"[{scenario}][{profile.name}][seed={seed}] {model_name}: "
                        f"{record.status} fit={record.fit_seconds:.4f}s "
                        f"pred={record.predict_seconds:.4f}s rmse={record.rmse:.6f} "
                        f"mae={record.mae:.6f} r2={record.r2:.6f}"
                    )
                    if record.status != "PASS":
                        print(f"  error: {record.error}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    params = {
        "profile_mode": "matrix" if matrix_mode else "single",
        "profile_grid": args.profile_grid,
        "profiles": [asdict(profile) for profile in profiles],
        "profile_seeds": seeds,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "rounds": args.rounds,
        "test_size": args.test_size,
        "scenarios": args.scenarios,
        "alloygbm_runtime": alloy_runtime,
    }
    paths = _write_outputs(args.output_dir, run_id, records, params)
    print(f"wrote comparison csv: {paths['csv']}")
    print(f"wrote comparison json: {paths['json']}")
    print(f"wrote comparison markdown: {paths['markdown']}")
    print(f"wrote profile summary csv: {paths['summary_csv']}")
    print(f"wrote profile summary json: {paths['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
