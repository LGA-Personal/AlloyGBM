"""Tests for the MorphBoost scanner performance gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "benchmarks" / "morph_perf_gate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("morph_perf_gate_module", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_perf_gate_accepts_required_scanner_speedups() -> None:
    gate = _load_module().evaluate_perf_gate(
        {
            "best_split_morph_16": 100.0,
            "best_split_morph_64": 150.0,
            "best_split_morph_255": 400.0,
        },
        {
            "best_split_morph_16": 98.0,
            "best_split_morph_64": 90.0,
            "best_split_morph_255": 240.0,
        },
        end_to_end_improvement=0.0,
    )
    assert gate.passed


def test_perf_gate_accepts_end_to_end_fallback() -> None:
    gate = _load_module().evaluate_perf_gate(
        {
            "best_split_morph_16": 100.0,
            "best_split_morph_64": 150.0,
            "best_split_morph_255": 400.0,
        },
        {
            "best_split_morph_16": 99.0,
            "best_split_morph_64": 120.0,
            "best_split_morph_255": 300.0,
        },
        end_to_end_improvement=0.18,
    )
    assert gate.passed


def test_perf_gate_rejects_shape_regression() -> None:
    gate = _load_module().evaluate_perf_gate(
        {
            "best_split_morph_16": 100.0,
            "best_split_morph_64": 150.0,
            "best_split_morph_255": 400.0,
        },
        {
            "best_split_morph_16": 106.0,
            "best_split_morph_64": 90.0,
            "best_split_morph_255": 240.0,
        },
        end_to_end_improvement=0.20,
    )
    assert not gate.passed
    assert any("regression" in reason for reason in gate.reasons)


def test_parse_samples_uses_medians_and_rejects_malformed_rows() -> None:
    module = _load_module()
    text = "\n".join(
        [
            "best_split_morph_64: total_ms=1 iterations=1 ns_per_iter=100",
            "best_split_morph_64: total_ms=1 iterations=1 ns_per_iter=300",
            "best_split_morph_64: total_ms=1 iterations=1 ns_per_iter=200",
        ]
    )
    assert module.parse_benchmark_text(text)["best_split_morph_64"] == [100.0, 300.0, 200.0]
    assert module.aggregate_medians(module.parse_benchmark_text(text))["best_split_morph_64"] == 200.0
    with pytest.raises(ValueError, match="malformed"):
        module.parse_benchmark_text(
            "best_split_morph_64: total_ms=nope iterations=1 ns_per_iter=100"
        )


def test_validate_samples_requires_names_and_five_measurements() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="missing"):
        module.validate_samples({"best_split_morph_64": [1.0] * 5})
    with pytest.raises(ValueError, match="at least 5"):
        module.validate_samples(
            {
                "best_split_morph_16": [1.0] * 5,
                "best_split_morph_64": [1.0] * 4,
                "best_split_morph_255": [1.0] * 5,
            }
        )
