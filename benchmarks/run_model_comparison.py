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


@dataclass
class BenchmarkRecord:
    scenario: str
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


def _prepare_dataset(
    repo_root: Path, scenario: str, force_prepare: bool, prepared_path: Path
) -> None:
    if prepared_path.exists() and not force_prepare:
        return

    scenario_script = repo_root / "benchmarks" / scenario / "prepare.py"
    command = [sys.executable, "-B", str(scenario_script)]
    if force_prepare and scenario in {"dense_numeric", "panel_time_series"}:
        command.append("--force-download")
    subprocess.run(command, cwd=repo_root, check=True)


def _load_manifest(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_dataset(
    repo_root: Path, scenario: str, force_prepare: bool
) -> tuple[pd.DataFrame, str]:
    manifest_path = repo_root / "benchmarks" / scenario / "manifest.yaml"
    manifest = _load_manifest(manifest_path)
    target_column = manifest["prepared"]["target"]
    prepared_file = manifest["prepared"]["filename"]
    prepared_path = repo_root / "benchmarks" / "data" / scenario / "prepared" / prepared_file

    _prepare_dataset(repo_root, scenario, force_prepare, prepared_path)
    if not prepared_path.exists():
        raise FileNotFoundError(f"prepared dataset missing: {prepared_path}")

    frame = pd.read_csv(prepared_path)
    if target_column not in frame.columns:
        raise ValueError(f"target column '{target_column}' missing in {prepared_path}")
    return frame, target_column


def _split_dataset(
    scenario: str, frame: pd.DataFrame, target_column: str, seed: int, test_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_frame = frame.drop(columns=[target_column]).copy()
    if "group_id" in feature_frame.columns:
        feature_frame = feature_frame.drop(columns=["group_id"])

    if "timestamp" in feature_frame.columns:
        ordered = frame.sort_values("timestamp")
        cutoff = max(1, int(len(ordered) * (1.0 - test_size)))
        train = ordered.iloc[:cutoff]
        test = ordered.iloc[cutoff:]
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
            model=model_name,
            train_rows=int(len(x_train)),
            test_rows=int(len(x_test)),
            n_features=int(x_train.shape[1]),
            fit_seconds=0.0,
            predict_seconds=0.0,
            rmse=float("nan"),
            mae=float("nan"),
            r2=float("nan"),
            status="FAIL",
            error=f"{type(exc).__name__}: {exc}",
        )


def _model_factories(seed: int, learning_rate: float, max_depth: int, rounds: int) -> dict:
    from alloygbm import GBMRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    alloy_signature = inspect.signature(GBMRegressor.__init__)
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
        "alloygbm": lambda: GBMRegressor(**alloy_params),
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


def _write_outputs(
    output_dir: Path,
    run_id: str,
    records: list[BenchmarkRecord],
    params: dict,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [asdict(record) for record in records]
    frame = pd.DataFrame(rows)

    csv_timestamped = output_dir / f"model_comparison_{run_id}.csv"
    csv_latest = output_dir / "model_comparison_latest.csv"
    frame.to_csv(csv_timestamped, index=False)
    frame.to_csv(csv_latest, index=False)

    json_timestamped = output_dir / f"model_comparison_{run_id}.json"
    json_latest = output_dir / "model_comparison_latest.json"
    json_payload = {
        "run_id": run_id,
        "params": params,
        "records": rows,
    }
    json_timestamped.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    json_latest.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    markdown_timestamped = output_dir / f"model_comparison_{run_id}.md"
    markdown_latest = output_dir / "model_comparison_latest.md"

    lines = [
        f"# Model Comparison ({run_id})",
        "",
        "## Params",
        f"- seed: `{params['seed']}`",
        f"- learning_rate: `{params['learning_rate']}`",
        f"- max_depth: `{params['max_depth']}`",
        f"- rounds: `{params['rounds']}`",
        "",
        "## Results",
        "",
        "| scenario | model | status | fit_seconds | predict_seconds | rmse | mae | r2 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            f"| {record.scenario} | {record.model} | {record.status} | "
            f"{record.fit_seconds:.6f} | {record.predict_seconds:.6f} | "
            f"{record.rmse:.6f} | {record.mae:.6f} | {record.r2:.6f} |"
        )

    markdown_text = "\n".join(lines) + "\n"
    markdown_timestamped.write_text(markdown_text, encoding="utf-8")
    markdown_latest.write_text(markdown_text, encoding="utf-8")

    return csv_timestamped, json_timestamped, markdown_timestamped


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["dense_numeric", "panel_time_series", "histogram_stress"],
        choices=["dense_numeric", "panel_time_series", "histogram_stress"],
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=120)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks") / "results",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    factories = _model_factories(args.seed, args.learning_rate, args.max_depth, args.rounds)
    records: list[BenchmarkRecord] = []

    for scenario in args.scenarios:
        try:
            frame, target_column = _load_dataset(repo_root, scenario, args.force_prepare)
            x_train, x_test, y_train, y_test = _split_dataset(
                scenario, frame, target_column, args.seed, args.test_size
            )
        except Exception as exc:  # noqa: BLE001
            for model_name in factories:
                records.append(
                    BenchmarkRecord(
                        scenario=scenario,
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
                )
                print(f"[{scenario}] {model_name}: FAIL")
                print(f"  error: {type(exc).__name__}: {exc}")
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
            )
            records.append(record)
            print(
                f"[{scenario}] {model_name}: {record.status} "
                f"fit={record.fit_seconds:.4f}s pred={record.predict_seconds:.4f}s "
                f"rmse={record.rmse:.6f} mae={record.mae:.6f} r2={record.r2:.6f}"
            )
            if record.status != "PASS":
                print(f"  error: {record.error}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    params = {
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "rounds": args.rounds,
        "test_size": args.test_size,
        "scenarios": args.scenarios,
    }
    csv_path, json_path, md_path = _write_outputs(args.output_dir, run_id, records, params)
    print(f"wrote comparison csv: {csv_path}")
    print(f"wrote comparison json: {json_path}")
    print(f"wrote comparison markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
