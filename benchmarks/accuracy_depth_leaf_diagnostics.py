#!/usr/bin/env python3
"""Inspect absolute terminal outputs in AlloyGBM classification artifacts.

This diagnostic follows the accuracy-depth comparison runner's dataset loader,
stratified split, classifier factories, and estimator settings. It is kept out
of the timed comparison path; parser and prediction-reconstruction checks run
on each diagnostic fit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from run_model_comparison import (
    _constructed_estimator_params,
    _classifier_factories,
    _load_alloygbm_classifier_runtime,
    _load_dataset,
    _multiclass_classifier_factories,
    _resolved_training_policy,
    _split_dataset,
    _load_optional_catboost_classifier,
)
from accuracy_depth_sweep import source_identity


TREE_NODE_STRIDE = 1 << 20
ARTIFACT_MAGIC = b"AGBM"
MODEL_FORMAT_V1 = 1
TREE_SECTION = 1
MULTICLASS_TREES_SECTION = 6
STUMP_SIZE = 32
DEFAULT_SCENARIOS = (
    "adult_income",
    "breast_cancer",
    "synthetic_classification",
    "digits_multiclass",
    "wine_multiclass",
    "synthetic_multiclass",
)
CAPS = (1.0, 2.0, 5.0)


@dataclass(frozen=True)
class Stump:
    tree_id: int
    node_id: int
    feature_index: int
    threshold_bin: int
    default_left: bool
    is_categorical: bool
    left_delta: float
    right_delta: float
    left_rows: int
    right_rows: int


@dataclass(frozen=True)
class ParsedArtifact:
    objective: str
    feature_count: int
    baselines: tuple[float, ...]
    # Scalar artifacts use class_id=None; multiclass artifacts keep classes apart.
    trees: dict[tuple[int | None, int], dict[int, Stump]]


@dataclass(frozen=True)
class ChildOutput:
    class_id: int | None
    tree_id: int
    parent_node_id: int
    child_node_id: int
    side: str
    value: float
    row_count: int
    terminal: bool


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_f32(data: bytes, offset: int) -> float:
    value = struct.unpack_from("<f", data, offset)[0]
    if not math.isfinite(value):
        raise ValueError(f"non-finite f32 at byte offset {offset}")
    return float(value)


def _section_payloads(artifact: bytes) -> dict[int, bytes]:
    if len(artifact) < 16 or artifact[:4] != ARTIFACT_MAGIC:
        raise ValueError("artifact has an invalid or truncated AGBM header")
    version, section_count, metadata_len = struct.unpack_from("<III", artifact, 4)
    if version != MODEL_FORMAT_V1:
        raise ValueError(f"unsupported artifact version {version}")
    descriptor_end = 16 + 20 * section_count
    data_start = descriptor_end + metadata_len
    if descriptor_end > len(artifact) or data_start > len(artifact):
        raise ValueError("artifact is truncated before section data")
    sections: dict[int, bytes] = {}
    for index in range(section_count):
        kind, offset, length = struct.unpack_from("<IQQ", artifact, 16 + 20 * index)
        end = offset + length
        if offset < data_start or end < offset or end > len(artifact):
            raise ValueError(f"section {kind} has an invalid byte range")
        if kind in sections:
            raise ValueError(f"duplicate artifact section kind {kind}")
        sections[kind] = artifact[offset:end]
    return sections


def _decode_stump(record: bytes) -> Stump:
    if len(record) != STUMP_SIZE:
        raise ValueError(f"stump record is {len(record)} bytes; expected {STUMP_SIZE}")
    node_id, feature, threshold, flags = struct.unpack_from("<IIHH", record, 0)
    if flags & ~3:
        raise ValueError(f"stump has unsupported flags {flags:#x}")
    gain = _read_f32(record, 12)
    del gain  # Gain is validated as finite but is not a leaf-output diagnostic.
    left_delta = _read_f32(record, 16)
    right_delta = _read_f32(record, 20)
    left_rows, right_rows = struct.unpack_from("<II", record, 24)
    tree_id, local_id = divmod(node_id, TREE_NODE_STRIDE)
    return Stump(
        tree_id=tree_id,
        node_id=local_id,
        feature_index=feature,
        threshold_bin=threshold,
        default_left=bool(flags & 1),
        is_categorical=bool(flags & 2),
        left_delta=left_delta,
        right_delta=right_delta,
        left_rows=left_rows,
        right_rows=right_rows,
    )


def parse_artifact(artifact: bytes) -> ParsedArtifact:
    """Parse scalar or multiclass tree records and validate parent-relative paths."""
    sections = _section_payloads(artifact)
    metadata_len = struct.unpack_from("<I", artifact, 12)[0]
    metadata_start = 16 + 20 * struct.unpack_from("<I", artifact, 8)[0]
    metadata = json.loads(artifact[metadata_start : metadata_start + metadata_len])
    objective = metadata.get("objective")
    if not isinstance(objective, str):
        raise ValueError("artifact metadata is missing objective")

    trees: dict[tuple[int | None, int], dict[int, Stump]] = {}
    if TREE_SECTION in sections:
        payload = sections[TREE_SECTION]
        if len(payload) < 16:
            raise ValueError("Trees payload is truncated")
        version, feature_count, stump_count = struct.unpack_from("<III", payload, 0)
        if version != MODEL_FORMAT_V1:
            raise ValueError(f"unsupported Trees payload version {version}")
        baseline = _read_f32(payload, 12)
        expected_len = 16 + stump_count * STUMP_SIZE
        if len(payload) != expected_len:
            raise ValueError(f"Trees payload length {len(payload)} != {expected_len}")
        for index in range(stump_count):
            stump = _decode_stump(payload[16 + index * STUMP_SIZE : 16 + (index + 1) * STUMP_SIZE])
            key = (None, stump.tree_id)
            nodes = trees.setdefault(key, {})
            if stump.node_id in nodes:
                raise ValueError(f"duplicate scalar tree/node {key}/{stump.node_id}")
            nodes[stump.node_id] = stump
        baselines = (baseline,)
    elif MULTICLASS_TREES_SECTION in sections:
        payload = sections[MULTICLASS_TREES_SECTION]
        if len(payload) < 12:
            raise ValueError("MultiClassTrees payload is truncated")
        version, class_count, feature_count = struct.unpack_from("<III", payload, 0)
        if version != MODEL_FORMAT_V1 or class_count < 2:
            raise ValueError("invalid MultiClassTrees header")
        cursor = 12
        header_end = cursor + 4 * class_count * 2
        if header_end > len(payload):
            raise ValueError("MultiClassTrees baseline/count arrays are truncated")
        baselines = tuple(_read_f32(payload, cursor + 4 * i) for i in range(class_count))
        cursor += 4 * class_count
        counts = struct.unpack_from(f"<{class_count}I", payload, cursor)
        cursor += 4 * class_count
        expected_len = cursor + STUMP_SIZE * sum(counts)
        if len(payload) != expected_len:
            raise ValueError(f"MultiClassTrees payload length {len(payload)} != {expected_len}")
        for class_id, count in enumerate(counts):
            for _ in range(count):
                stump = _decode_stump(payload[cursor : cursor + STUMP_SIZE])
                cursor += STUMP_SIZE
                key = (class_id, stump.tree_id)
                nodes = trees.setdefault(key, {})
                if stump.node_id in nodes:
                    raise ValueError(f"duplicate multiclass tree/node {key}/{stump.node_id}")
                nodes[stump.node_id] = stump
    else:
        raise ValueError("artifact has neither Trees nor MultiClassTrees section")

    if feature_count != len(metadata.get("feature_names", [])):
        raise ValueError("tree feature count differs from artifact metadata")
    if not trees:
        raise ValueError("classification artifact contains no split records")
    for key, nodes in trees.items():
        if 0 not in nodes:
            raise ValueError(f"tree {key} has no root split record")
        for node_id in nodes:
            if node_id > 0 and (node_id - 1) // 2 not in nodes:
                raise ValueError(f"tree {key} node {node_id} has no recorded parent")
    return ParsedArtifact(objective, feature_count, baselines, trees)


def _child_outputs(parsed: ParsedArtifact) -> list[ChildOutput]:
    children: list[ChildOutput] = []
    for (class_id, tree_id), nodes in sorted(parsed.trees.items(), key=lambda item: (item[0][0] or -1, item[0][1])):
        absolute: dict[int, np.float32] = {0: np.float32(0.0)}
        for local_id in sorted(nodes):
            if local_id not in absolute:
                raise ValueError(f"tree {tree_id} path to node {local_id} is incomplete")
            stump = nodes[local_id]
            for side, child_id, delta, rows in (
                ("left", 2 * local_id + 1, stump.left_delta, stump.left_rows),
                ("right", 2 * local_id + 2, stump.right_delta, stump.right_rows),
            ):
                if child_id in absolute:
                    raise ValueError(f"tree {tree_id} has duplicate child node {child_id}")
                value = np.float32(absolute[local_id] + np.float32(delta))
                absolute[child_id] = value
                children.append(
                    ChildOutput(
                        class_id=class_id,
                        tree_id=tree_id,
                        parent_node_id=local_id,
                        child_node_id=child_id,
                        side=side,
                        value=float(value),
                        row_count=rows,
                        terminal=child_id not in nodes,
                    )
                )
        # Only the tree's descendants are needed; root 0 is the parent output.
    return children


def _reconstructed_predictions(
    parsed: ParsedArtifact, quantized_rows: np.ndarray
) -> np.ndarray:
    if quantized_rows.ndim != 2 or quantized_rows.shape[1] != parsed.feature_count:
        raise ValueError("quantized prediction rows have the wrong shape")
    class_count = len(parsed.baselines)
    logits = np.tile(np.asarray(parsed.baselines, dtype=np.float32), (len(quantized_rows), 1))
    class_ids = range(class_count) if class_count > 1 else (None,)
    for class_id in class_ids:
        class_column = class_id if class_id is not None else 0
        tree_keys = sorted(key for key in parsed.trees if key[0] == class_id)
        for row_index, row in enumerate(quantized_rows):
            prediction = np.float32(parsed.baselines[class_column])
            for key in tree_keys:
                nodes = parsed.trees[key]
                local_id = 0
                contribution = np.float32(0.0)
                while local_id in nodes:
                    stump = nodes[local_id]
                    value = float(row[stump.feature_index])
                    if math.isnan(value):
                        go_left = stump.default_left
                    elif stump.is_categorical:
                        raise ValueError("prediction reconstruction needs categorical bitset state")
                    else:
                        go_left = value <= stump.threshold_bin
                    if go_left:
                        contribution = np.float32(contribution + np.float32(stump.left_delta))
                        local_id = 2 * local_id + 1
                    else:
                        contribution = np.float32(contribution + np.float32(stump.right_delta))
                        local_id = 2 * local_id + 2
                prediction = np.float32(prediction + contribution)
            logits[row_index, class_column] = prediction
    if class_count == 1:
        margin = logits[:, 0].astype(np.float64)
        prob = np.empty_like(margin)
        positive = margin >= 0
        prob[positive] = 1.0 / (1.0 + np.exp(-margin[positive]))
        exp_margin = np.exp(margin[~positive])
        prob[~positive] = exp_margin / (1.0 + exp_margin)
        return np.column_stack((1.0 - prob, prob))
    shifted = logits.astype(np.float64) - np.max(logits.astype(np.float64), axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)


def _diagnostic_prediction_rows(model: Any, rows: object) -> np.ndarray:
    """Mirror GBMRegressor's continuous-bin versus pre-binned prediction path."""
    numeric = model._validate_numeric_features(rows)
    if model._uses_continuous_binning:
        return np.asarray(model._quantize_rows_for_prediction(numeric), dtype=np.float32)
    return np.asarray(numeric, dtype=np.float32)


def _distribution(
    values: list[float],
    cap: float,
    row_counts: list[int] | None = None,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "max_abs": 0.0,
            "abs_q50": 0.0,
            "abs_q90": 0.0,
            "abs_q99": 0.0,
            "abs_gt_cap_count": 0,
            "abs_gt_cap_percent": 0.0,
            "abs_gt_cap_substantive_count": 0,
            "abs_gt_cap_substantive_percent": 0.0,
            "rows_in_gt_cap_terminal_leaves": 0,
            "rows_in_substantive_gt_cap_terminal_leaves": 0,
        }
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    gt = absolute > cap
    substantive_gt = absolute > cap + tolerance
    rows_exposed = 0
    rows_substantively_exposed = 0
    if row_counts is not None:
        rows_exposed = int(sum(count for count, exceeds in zip(row_counts, gt, strict=True) if exceeds))
        rows_substantively_exposed = int(
            sum(count for count, exceeds in zip(row_counts, substantive_gt, strict=True) if exceeds)
        )
    return {
        "count": int(absolute.size),
        "max_abs": float(absolute.max()),
        "abs_q50": float(np.quantile(absolute, 0.50)),
        "abs_q90": float(np.quantile(absolute, 0.90)),
        "abs_q99": float(np.quantile(absolute, 0.99)),
        "abs_gt_cap_count": int(gt.sum()),
        "abs_gt_cap_percent": float(gt.mean() * 100.0),
        "abs_gt_cap_substantive_count": int(substantive_gt.sum()),
        "abs_gt_cap_substantive_percent": float(substantive_gt.mean() * 100.0),
        "rows_in_gt_cap_terminal_leaves": rows_exposed,
        "rows_in_substantive_gt_cap_terminal_leaves": rows_substantively_exposed,
    }


def _saturation_counts(
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    task_type: str,
) -> dict[str, Any]:
    """Retain the requested 1e-6 probability saturation as secondary context."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    if probabilities.ndim != 2 or len(probabilities) != len(labels):
        raise ValueError("probability/label arrays are misaligned")
    per_label: dict[str, Any] = {}
    if task_type == "classification":
        positive_probability = probabilities[:, 1]
        low = positive_probability < 1e-6
        high = positive_probability > 1.0 - 1e-6
        criterion = "binary_positive_probability <1e-6 or >1-1e-6"
        for label in classes:
            mask = labels == label
            per_label[str(label)] = {
                "rows": int(mask.sum()),
                "low": int((low & mask).sum()),
                "high": int((high & mask).sum()),
            }
    else:
        class_indices = {label: index for index, label in enumerate(classes.tolist())}
        try:
            true_column = np.asarray([class_indices[label] for label in labels], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"unknown multiclass label in saturation check: {exc}") from exc
        true_probability = probabilities[np.arange(len(labels)), true_column]
        low = true_probability < 1e-6
        high = true_probability > 1.0 - 1e-6
        criterion = "true_class_probability <1e-6 or >1-1e-6"
        max_saturated = probabilities.max(axis=1) > 1.0 - 1e-6
        for label in classes:
            mask = labels == label
            per_label[str(label)] = {
                "rows": int(mask.sum()),
                "true_class_low": int((low & mask).sum()),
                "true_class_high": int((high & mask).sum()),
                "max_class_high": int((max_saturated & mask).sum()),
            }
    return {
        "rows": int(len(labels)),
        "criterion": criterion,
        "low_count": int(low.sum()),
        "high_count": int(high.sum()),
        "saturated_count": int((low | high).sum()),
        "max_class_probability_gt_1_minus_1e6_count": (
            int((probabilities.max(axis=1) > 1.0 - 1e-6).sum())
            if task_type == "multiclass_classification"
            else int(high.sum())
        ),
        "by_true_label": per_label,
    }


def summarize_artifact(parsed: ParsedArtifact, caps: tuple[float, ...] = CAPS) -> dict[str, Any]:
    children = _child_outputs(parsed)
    terminal = [child for child in children if child.terminal]
    accepted = children
    per_tree: list[dict[str, Any]] = []
    keys = sorted(parsed.trees, key=lambda key: (key[0] if key[0] is not None else -1, key[1]))
    for key in keys:
        group = [item for item in children if (item.class_id, item.tree_id) == key]
        terminal_group = [item for item in group if item.terminal]
        per_tree.append(
            {
                "class_id": key[0],
                "tree_id": key[1],
                "split_count": len(parsed.trees[key]),
                "terminal_leaf_count": len(terminal_group),
                "terminal_leaf_rows": int(sum(item.row_count for item in terminal_group)),
                "terminal_max_abs": float(max((abs(item.value) for item in terminal_group), default=0.0)),
                "accepted_child_count": len(group),
                "accepted_child_max_abs": float(max((abs(item.value) for item in group), default=0.0)),
                "terminal_gt_caps": {
                    f"{cap:g}": sum(abs(item.value) > cap for item in terminal_group)
                    for cap in caps
                },
                "terminal_substantive_gt_caps": {
                    f"{cap:g}": sum(abs(item.value) > cap + 1e-5 for item in terminal_group)
                    for cap in caps
                },
                "accepted_child_gt_caps": {
                    f"{cap:g}": sum(abs(item.value) > cap for item in group)
                    for cap in caps
                },
                "accepted_child_substantive_gt_caps": {
                    f"{cap:g}": sum(abs(item.value) > cap + 1e-5 for item in group)
                    for cap in caps
                },
            }
        )
    distributions = {}
    for cap in caps:
        tolerance = max(1e-5, cap * 1e-6)
        terminal_abs = np.abs(np.asarray([item.value for item in terminal], dtype=np.float64))
        accepted_abs = np.abs(np.asarray([item.value for item in accepted], dtype=np.float64))
        terminal_at_cap = np.isclose(terminal_abs, cap, rtol=0.0, atol=tolerance)
        accepted_at_cap = np.isclose(accepted_abs, cap, rtol=0.0, atol=tolerance)
        distributions[f"{cap:g}"] = {
            "terminal_outputs": _distribution(
                [item.value for item in terminal], cap, [item.row_count for item in terminal]
            ),
            "accepted_children": _distribution([item.value for item in accepted], cap),
            "terminal_outputs_at_cap_count": int(terminal_at_cap.sum()),
            "terminal_rows_at_cap": int(
                sum(item.row_count for item, at_cap in zip(terminal, terminal_at_cap, strict=True) if at_cap)
            ),
            "accepted_children_at_cap_count": int(accepted_at_cap.sum()),
        }
    return {
        "objective": parsed.objective,
        "feature_count": parsed.feature_count,
        "class_count": len(parsed.baselines),
        "baseline_logits": list(parsed.baselines),
        "tree_count": len(parsed.trees),
        "split_count": sum(len(nodes) for nodes in parsed.trees.values()),
        "distributions_by_cap": distributions,
        "per_tree": per_tree,
    }


def _model_factories(
    *,
    task_type: str,
    n_classes: int,
    seed: int,
    depth: int,
    rounds: int,
    learning_rate: float,
    threads: int,
    classifier_cls: type,
    catboost_cls: type | None,
) -> dict[str, Any]:
    common = dict(
        seed=seed,
        learning_rate=learning_rate,
        max_depth=depth,
        rounds=rounds,
        alloy_continuous_binning_strategy="linear",
        alloy_continuous_binning_max_bins=256,
        threads=threads,
        alloy_param_overrides=None,
    )
    if task_type == "classification":
        return _classifier_factories(classifier_cls, catboost_cls, **common)
    return _multiclass_classifier_factories(
        classifier_cls, catboost_cls, n_classes=n_classes, **common
    )


def _data_identity(repo_root: Path, scenario: str) -> dict[str, Any]:
    manifest_path = repo_root / "benchmarks" / scenario / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    csv_path = repo_root / "benchmarks" / "data" / scenario / "prepared" / manifest["prepared"]["filename"]
    return {
        "manifest_sha256": _sha256(manifest_path.read_bytes()),
        "prepared_sha256": _sha256(csv_path.read_bytes()),
        "prepared_path": str(csv_path.relative_to(repo_root)),
    }


def _fit_one(
    *,
    factory: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    scenario: str,
    seed: int,
    depth: int,
    label: str,
    check_rows: int,
) -> tuple[dict[str, Any], bytes, Any, float]:
    model = factory()
    model.fit(x_train, y_train)
    artifact = bytes(model.artifact_bytes)
    parsed = parse_artifact(artifact)
    if task_type == "classification" and len(parsed.baselines) != 1:
        raise ValueError(f"{scenario} produced {len(parsed.baselines)} baseline logits")
    if task_type == "multiclass_classification" and len(parsed.baselines) <= 1:
        raise ValueError(f"{scenario} did not produce a multiclass artifact")

    rows = min(check_rows, len(x_test))
    # The public pre-binned path retains raw integer bins and converts artifact
    # thresholds to bin+0.5. Integer input <= artifact threshold is equivalent.
    quantized = _diagnostic_prediction_rows(model, x_test[:rows])
    reconstructed = _reconstructed_predictions(parsed, quantized)
    train_probabilities = np.asarray(model.predict_proba(x_train), dtype=np.float64)
    test_probabilities = np.asarray(model.predict_proba(x_test), dtype=np.float64)
    actual = test_probabilities[:rows]
    if actual.shape != reconstructed.shape:
        raise ValueError(f"{scenario} prediction shape mismatch: {actual.shape} vs {reconstructed.shape}")
    max_error = float(np.max(np.abs(actual - reconstructed)))
    tolerance = 3e-5
    if not math.isfinite(max_error) or max_error > tolerance:
        raise ValueError(
            f"{scenario} artifact path reconstruction error {max_error:g} exceeds {tolerance:g}"
        )
    summary = summarize_artifact(parsed)
    summary.update(
        {
            "scenario": scenario,
            "task_type": task_type,
            "depth": depth,
            "seed": seed,
            "build_label": label,
            "n_jobs": int(model.n_jobs),
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "requested_estimator_parameters": {
                "learning_rate": model.learning_rate,
                "max_depth": model.max_depth,
                "n_estimators": model.n_estimators,
                "row_subsample": model.row_subsample,
                "col_subsample": model.col_subsample,
                "seed": model.seed,
                "n_jobs": model.n_jobs,
                "continuous_binning_strategy": model.continuous_binning_strategy,
                "continuous_binning_max_bins": model.continuous_binning_max_bins,
            },
            "constructed_estimator_parameters": _constructed_estimator_params(model),
            "resolved_training_policy": _resolved_training_policy(model, "alloygbm"),
            "train_label_saturation": _saturation_counts(
                train_probabilities, y_train, np.asarray(model.classes_), task_type
            ),
            "test_label_saturation": _saturation_counts(
                test_probabilities, y_test, np.asarray(model.classes_), task_type
            ),
            "prediction_rows_checked": rows,
            "prediction_max_abs_error": max_error,
            "artifact_bytes": len(artifact),
            "artifact_sha256": _sha256(artifact),
            "fit_timing_seconds": getattr(model, "fit_timing_", None),
        }
    )
    return summary, artifact, model, max_error


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    classifier_cls, runtime = _load_alloygbm_classifier_runtime()
    catboost_cls, catboost_info = _load_optional_catboost_classifier()
    package = importlib.import_module("alloygbm")
    native = importlib.import_module("alloygbm._alloygbm")
    native_path = Path(native.__file__).resolve()
    trainer_path = repo_root / "crates/engine/src/trainer/mod.rs"
    patch = __import__("subprocess").run(
        ["git", "diff", "--", "crates/engine/src/trainer/mod.rs"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.encode()

    results: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "build_label": args.build_label,
        "cap_value": args.cap_value,
        "config": {
            "scenarios": args.scenarios,
            "depths": args.depths,
            "seeds": args.seeds,
            "rounds": args.rounds,
            "learning_rate": args.learning_rate,
            "test_size": 0.2,
            "threads": 1,
            "row_subsample": 0.8,
            "col_subsample": 0.8,
            "binning_strategy": "linear",
            "bin_count": 256,
            "models": ["alloygbm"],
        },
        "runtime": {
            "python": sys.version.split()[0],
            "alloygbm_version": str(getattr(package, "__version__", "unknown")),
            "native_module_path": str(native_path),
            "native_module_sha256": _sha256(native_path.read_bytes()),
            "runtime_identity": runtime,
            "catboost_available": bool(catboost_info.get("available", catboost_cls is not None)),
        },
        "source": {
            "git_sha": __import__("subprocess").run(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, capture_output=True, check=True
            ).stdout.strip(),
            "diagnostic_script_sha256": _sha256(Path(__file__).read_bytes()),
            "benchmark_source_fingerprint": source_identity(repo_root, args.scenarios),
            "trainer_mod_sha256": _sha256(trainer_path.read_bytes()),
            "temporary_trainer_patch_sha256": _sha256(patch),
            "temporary_trainer_patch": patch.decode(),
        },
        "datasets": {},
        "fits": [],
        "determinism": [],
        "status": "RUNNING",
        "expected_fit_count": len(args.scenarios) * len(args.depths) * len(args.seeds),
        "completed_fit_count": 0,
    }
    loaded: dict[str, tuple[Any, str, str, str | None]] = {}
    for scenario in args.scenarios:
        loaded[scenario] = _load_dataset(repo_root, scenario, force_prepare=False)
        results["datasets"][scenario] = _data_identity(repo_root, scenario)

    output_path = output_dir / "diagnostics.json"
    if args.resume:
        if not output_path.is_file():
            raise ValueError(f"cannot resume: missing diagnostic checkpoint {output_path}")
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            saved.get("build_label") != args.build_label
            or saved.get("cap_value") != args.cap_value
            or saved.get("config") != results["config"]
        ):
            raise ValueError("cannot resume: build label or fit configuration changed")
        if saved.get("datasets") != results["datasets"]:
            raise ValueError("cannot resume: dataset identities changed")
        saved_source = saved.get("source", {})
        for key in (
            "trainer_mod_sha256",
            "temporary_trainer_patch_sha256",
            "benchmark_source_fingerprint",
        ):
            if saved_source.get(key) != results["source"].get(key):
                raise ValueError(f"cannot resume: source identity changed ({key})")
        if saved.get("runtime", {}).get("native_module_sha256") != results["runtime"]["native_module_sha256"]:
            raise ValueError("cannot resume: native module hash changed")
        results["source"]["resumed_from_diagnostic_script_sha256"] = saved_source.get(
            "diagnostic_script_sha256"
        )
        results["fits"] = saved.get("fits", [])
        results["completed_fit_count"] = len(results["fits"])
        results["determinism"] = saved.get("determinism", [])
        results["resume_checkpoint_sha256"] = _sha256(output_path.read_bytes())
    _write_checkpoint(output_path, results)

    completed_keys = {
        (fit["scenario"], int(fit["depth"]), int(fit["seed"]))
        for fit in results["fits"]
    }

    # One fit at a time. These fits are diagnostic only and are not used as
    # timed comparison records.
    deterministic_artifacts: dict[tuple[str, int, int], bytes] = {}
    for depth in args.depths:
        for seed in args.seeds:
            for scenario in args.scenarios:
                fit_key = (scenario, depth, seed)
                if fit_key in completed_keys:
                    continue
                frame, target_column, task_type, group_column = loaded[scenario]
                x_train, x_test, y_train, _y_test, _g_train, _g_test = _split_dataset(
                    scenario,
                    frame,
                    target_column,
                    seed,
                    0.2,
                    task_type=task_type,
                    group_column=group_column,
                )
                if task_type == "multiclass_classification":
                    n_classes = int(np.unique(y_train).size)
                else:
                    n_classes = 2
                factories = _model_factories(
                    task_type=task_type,
                    n_classes=n_classes,
                    seed=seed,
                    depth=depth,
                    rounds=args.rounds,
                    learning_rate=args.learning_rate,
                    threads=1,
                    classifier_cls=classifier_cls,
                    catboost_cls=catboost_cls,
                )
                summary, artifact, _model, _ = _fit_one(
                    factory=factories["alloygbm"],
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=_y_test,
                    task_type=task_type,
                    scenario=scenario,
                    seed=seed,
                    depth=depth,
                    label=args.build_label,
                    check_rows=args.prediction_rows,
                )
                summary["diagnostic_script_sha256"] = results["source"][
                    "diagnostic_script_sha256"
                ]
                results["fits"].append(summary)
                results["completed_fit_count"] += 1
                completed_keys.add(fit_key)
                _write_checkpoint(output_path, results)
                if (scenario, depth, seed) in {
                    (args.determinism_binary, args.determinism_depth, args.determinism_seed),
                    (args.determinism_multiclass, args.determinism_depth, args.determinism_seed),
                }:
                    deterministic_artifacts[(scenario, depth, seed)] = artifact
                print(
                    f"DIAG {args.build_label} {scenario} d{depth} s{seed}: "
                    f"trees={summary['tree_count']} splits={summary['split_count']} "
                    f"max_terminal_abs={summary['distributions_by_cap']['1']['terminal_outputs']['max_abs']:.8g} "
                    f"predict_err={summary['prediction_max_abs_error']:.3g}",
                    flush=True,
                )

    if args.check_determinism:
        checks = [
            (args.determinism_binary, "classification"),
            (args.determinism_multiclass, "multiclass_classification"),
        ]
        for scenario, expected_type in checks:
            frame, target_column, task_type, group_column = loaded[scenario]
            if task_type != expected_type:
                raise ValueError(f"determinism fixture {scenario} has task type {task_type}")
            x_train, x_test, y_train, _y_test, _g_train, _g_test = _split_dataset(
                scenario,
                frame,
                target_column,
                args.determinism_seed,
                0.2,
                task_type=task_type,
                group_column=group_column,
            )
            n_classes = int(np.unique(y_train).size) if task_type == "multiclass_classification" else 2
            factories4 = _model_factories(
                task_type=task_type,
                n_classes=n_classes,
                seed=args.determinism_seed,
                depth=args.determinism_depth,
                rounds=args.rounds,
                learning_rate=args.learning_rate,
                threads=4,
                classifier_cls=classifier_cls,
                catboost_cls=catboost_cls,
            )
            summary4, artifact4, _model4, _ = _fit_one(
                factory=factories4["alloygbm"],
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=_y_test,
                task_type=task_type,
                scenario=scenario,
                seed=args.determinism_seed,
                depth=args.determinism_depth,
                label=args.build_label + "-n_jobs4",
                check_rows=args.prediction_rows,
            )
            summary4["diagnostic_script_sha256"] = results["source"][
                "diagnostic_script_sha256"
            ]
            artifact1 = deterministic_artifacts.get(
                (scenario, args.determinism_depth, args.determinism_seed)
            )
            if artifact1 is None:
                raise ValueError(f"missing n_jobs=1 determinism artifact for {scenario}")
            identical = artifact1 == artifact4
            check = {
                "scenario": scenario,
                "depth": args.determinism_depth,
                "seed": args.determinism_seed,
                "n_jobs_1_sha256": _sha256(artifact1),
                "n_jobs_4_sha256": _sha256(artifact4),
                "byte_identical": identical,
                "n_jobs_4_prediction_max_abs_error": summary4["prediction_max_abs_error"],
            }
            results["determinism"].append(check)
            _write_checkpoint(output_path, results)
            print(
                f"DETERMINISM {args.build_label} {scenario} d{args.determinism_depth} "
                f"s{args.determinism_seed}: identical={identical} "
                f"n_jobs1={check['n_jobs_1_sha256']} n_jobs4={check['n_jobs_4_sha256']}",
                flush=True,
            )
            if not identical:
                raise ValueError(f"artifact differs across n_jobs for {scenario}")

    results["status"] = "COMPLETED"
    _write_checkpoint(output_path, results)
    print(f"wrote {output_path}", flush=True)
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-label", required=True)
    parser.add_argument("--cap-value", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--depths", nargs="+", type=int, default=[6, 12])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260919, 20260920, 20260921, 20260922, 20260923])
    parser.add_argument("--rounds", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--prediction-rows", type=int, default=128)
    parser.add_argument("--check-determinism", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--determinism-binary", default="synthetic_classification")
    parser.add_argument("--determinism-multiclass", default="synthetic_multiclass")
    parser.add_argument("--determinism-depth", type=int, default=12)
    parser.add_argument("--determinism-seed", type=int, default=20260919)
    return parser


if __name__ == "__main__":
    run_diagnostics(_parser().parse_args())
