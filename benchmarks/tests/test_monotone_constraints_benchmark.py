"""Contract tests for the monotone-constraint acceptance benchmark."""

from __future__ import annotations

from contextlib import redirect_stderr
import importlib.util
from io import StringIO
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = REPO_ROOT / "benchmarks" / "monotone_constraints_benchmark.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "monotone_constraints_benchmark_module", BENCHMARK_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load benchmark from {BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark()


def _record(scenario, **overrides):
    record = {
        "scenario": scenario.name,
        "objective": scenario.objective,
        "direction": scenario.direction,
        "tree_growth": scenario.tree_growth,
        "seed": scenario.seed,
        "constrained_fit_seconds": 1.5,
        "unconstrained_fit_seconds": 1.0,
        "fit_time_ratio": 1.5,
        "constrained_loss": 0.8,
        "unconstrained_loss": 0.8,
        "constant_loss": 1.0,
        "checked_grid_pairs": 8 * 256,
        "worst_signed_monotone_margin": 0.0,
        "violation_count": 0,
        "grid_predictions_finite": True,
        "constrained_finite": True,
        "unconstrained_finite": True,
        "constrained_completed": True,
        "unconstrained_completed": True,
        "constrained_completed_rounds": 16,
        "unconstrained_completed_rounds": 16,
        "requested_rounds": 16,
    }
    record.update(overrides)
    return record


def _report(*, records=None, scenarios=None, rounds=None):
    canonical = BENCHMARK.quick_scenarios()
    scenarios = canonical if scenarios is None else scenarios
    records = (
        [_record(scenario) for scenario in canonical] if records is None else records
    )
    return {
        "quick": True,
        "rounds": BENCHMARK.QUICK_ROUNDS if rounds is None else rounds,
        "scenarios": [scenario.name for scenario in scenarios],
        "records": records,
    }


def test_quick_matrix_covers_objectives_directions_and_growth_modes() -> None:
    scenarios = BENCHMARK.quick_scenarios()

    assert len(scenarios) == 8
    assert {(scenario.n_rows, scenario.n_features) for scenario in scenarios} == {
        (128, 2)
    }
    assert {scenario.objective for scenario in scenarios} == {"regression", "binary"}
    assert {scenario.direction for scenario in scenarios} == {-1, 1}
    assert {scenario.tree_growth for scenario in scenarios} == {"level", "leaf"}
    assert {scenario.seed for scenario in scenarios} == {0}


def test_full_matrix_covers_each_required_dimension() -> None:
    scenarios = BENCHMARK.full_scenarios()

    assert len(scenarios) == 3 * 3 * 2 * 2 * 2 * 3
    assert {scenario.n_rows for scenario in scenarios} == {128, 4_096, 32_768}
    assert {scenario.n_features for scenario in scenarios} == {2, 16, 128}
    assert {scenario.objective for scenario in scenarios} == {"regression", "binary"}
    assert {scenario.direction for scenario in scenarios} == {-1, 1}
    assert {scenario.tree_growth for scenario in scenarios} == {"level", "leaf"}
    assert {scenario.seed for scenario in scenarios} == {0, 1, 2}


@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_fixture_is_deterministic_finite_float32_and_nontrivial(objective) -> None:
    scenario = BENCHMARK.Scenario(
        "target-contract", 128, 16, objective, 1, "level", 0
    )
    first = BENCHMARK.make_fixture(scenario)
    second = BENCHMARK.make_fixture(scenario)

    for first_array, second_array in zip(first.arrays(), second.arrays(), strict=True):
        np.testing.assert_array_equal(first_array, second_array)
        assert first_array.dtype == np.float32
        assert np.isfinite(first_array).all()

    components = BENCHMARK.target_components(first.X_train, direction=1)
    assert np.ptp(components["interaction"]) > 0.0
    assert np.ptp(components["nuisance"]) > 0.0
    assert not np.allclose(components["interaction"], 0.0)
    assert not np.allclose(components["nuisance"], 0.0)


def test_grid_validation_changes_only_feature_zero_for_each_context() -> None:
    class CapturingEstimator:
        def __init__(self) -> None:
            self.calls = []

        def predict(self, X):
            self.calls.append(X.copy())
            return X[:, 0]

    holdout = np.asarray(
        [[0.25, -0.5, 0.75], [-0.25, 0.5, -0.75]], dtype=np.float32
    )
    estimator = CapturingEstimator()

    check = BENCHMARK.validate_monotonicity(estimator, holdout, direction=1)

    assert check.checked_grid_pairs == 2 * 256
    assert check.violation_count == 0
    assert len(estimator.calls) == 2
    for context, grid in zip(holdout, estimator.calls, strict=True):
        assert grid.shape == (257, 3)
        assert np.all(np.diff(grid[:, 0]) > 0.0)
        np.testing.assert_array_equal(grid[:, 1:], np.tile(context[1:], (257, 1)))


def test_grid_validation_marks_mixed_finite_and_nan_predictions_non_finite() -> None:
    class MixedEstimator:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, X):
            self.calls += 1
            prediction = X[:, 0].astype(np.float64)
            if self.calls == 2:
                prediction[3] = np.nan
            return prediction

    holdout = np.asarray([[0.0, 0.5], [0.0, -0.5]], dtype=np.float32)
    check = BENCHMARK.validate_monotonicity(MixedEstimator(), holdout, direction=1)

    assert not check.grid_predictions_finite
    assert check.checked_grid_pairs == 2 * 256


@pytest.mark.parametrize(
    ("overrides", "expected_fragment"),
    [
        ({"constrained_finite": False}, "non-finite"),
        ({"constrained_completed": False}, "incomplete"),
        ({"constrained_completed_rounds": 15}, "incomplete"),
        ({"checked_grid_pairs": 0}, "grid pairs"),
        ({"grid_predictions_finite": False}, "grid predictions"),
        ({"violation_count": 1}, "monotone violations"),
        ({"constrained_loss": 1.251, "unconstrained_loss": 1.0}, "ratio"),
        (
            {"objective": "binary", "constrained_loss": 0.29, "unconstrained_loss": 0.2},
            "binary error degradation",
        ),
        ({"constrained_loss": 1.0, "constant_loss": 1.0}, "constant"),
    ],
)
def test_gate_rejects_invalid_record_conditions(overrides, expected_fragment) -> None:
    report = _report()
    report["records"][0].update(overrides)

    failures = BENCHMARK.evaluate_gate(report)

    assert any(expected_fragment in failure for failure in failures)


def test_gate_rejects_missing_records_and_ignores_timing_ratio() -> None:
    report = _report()
    report["records"].pop()
    failures = BENCHMARK.evaluate_gate(report)
    assert any("missing record" in failure for failure in failures)

    report = _report()
    report["records"][0]["fit_time_ratio"] = 10_000.0
    assert BENCHMARK.evaluate_gate(report) == []


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "duplicate"])
def test_gate_rejects_noncanonical_scenario_declarations(mutation) -> None:
    report = _report()
    if mutation == "missing":
        report["scenarios"].pop()
    elif mutation == "unexpected":
        report["scenarios"].append("unexpected-scenario")
    else:
        report["scenarios"].append(report["scenarios"][0])

    failures = BENCHMARK.evaluate_gate(report)

    assert any("scenario declarations" in failure for failure in failures)


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "duplicate"])
def test_gate_rejects_noncanonical_record_identities(mutation) -> None:
    report = _report()
    if mutation == "missing":
        report["records"].pop()
    elif mutation == "unexpected":
        extra = dict(report["records"][0])
        extra["scenario"] = "unexpected-scenario"
        report["records"].append(extra)
    else:
        report["records"].append(dict(report["records"][0]))

    failures = BENCHMARK.evaluate_gate(report)

    assert any("record identities" in failure for failure in failures)


def test_gate_rejects_empty_report_and_quick_round_mismatch() -> None:
    empty = {"quick": True, "rounds": BENCHMARK.QUICK_ROUNDS, "scenarios": [], "records": []}
    assert any("empty" in failure for failure in BENCHMARK.evaluate_gate(empty))

    report = _report(rounds=1)
    for record in report["records"]:
        record["requested_rounds"] = 1
        record["constrained_completed_rounds"] = 1
        record["unconstrained_completed_rounds"] = 1
    failures = BENCHMARK.evaluate_gate(report)
    assert any("requested rounds" in failure for failure in failures)
    assert any("incomplete rounds" in failure for failure in failures)


def test_quick_gate_passes_on_installed_implementation() -> None:
    report = BENCHMARK.run_benchmark(quick=True)

    assert BENCHMARK.evaluate_gate(report) == []
    for record in report["records"]:
        assert record["violation_count"] == 0
        assert record["constrained_finite"]
        assert record["unconstrained_finite"]
        assert record["constrained_completed"]
        assert record["unconstrained_completed"]


def test_cli_writes_rendered_markdown(tmp_path) -> None:
    output = tmp_path / "monotone.md"

    assert BENCHMARK.main(["--quick", "--gate", "--output", str(output)]) == 0

    rendered = output.read_text(encoding="utf-8")
    assert "# Monotone Constraint Acceptance Benchmark" in rendered
    assert "Acceptance Contract" in rendered
    assert "| Scenario |" in rendered


def test_cli_prints_gate_failures_to_stderr(monkeypatch) -> None:
    monkeypatch.setattr(BENCHMARK, "run_benchmark", lambda *, quick: {"records": []})
    stderr = StringIO()

    with redirect_stderr(stderr):
        assert BENCHMARK.main(["--quick", "--gate"]) == 1

    assert stderr.getvalue()


def test_ci_runs_monotone_contract_and_compact_gate() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert "python -m pytest benchmarks/tests/test_monotone_constraints_benchmark.py -q" in workflow
    assert "python benchmarks/monotone_constraints_benchmark.py --quick --gate" in workflow
