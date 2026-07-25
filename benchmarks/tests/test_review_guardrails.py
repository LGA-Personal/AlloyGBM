"""Contract tests for July-review benchmark evidence guardrails."""

from __future__ import annotations

from contextlib import redirect_stderr
import importlib.util
from io import StringIO
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_module(
    "review_guardrails_benchmark_module",
    REPO_ROOT / "benchmarks" / "review_guardrails.py",
)


def _rows_for_expected_matrix(expected):
    quantile_rows = []
    if "quantile" in expected.sections:
        quantile_rows = [
            BENCHMARK.QuantileSplitRow(
                seed,
                alpha,
                arm,
                0.0,
                1.0,
                0.9,
                1.0,
                8,
                8,
            )
            for seed in expected.seeds
            for alpha in expected.quantile_alphas
            for arm in expected.quantile_arms
        ]

    goss_rows = []
    if "goss" in expected.sections:
        for seed in expected.seeds:
            goss_rows.append(
                BENCHMARK.BoostingRow(
                    "goss",
                    seed,
                    "standard_full",
                    0.8,
                    2.0,
                    1.0,
                    expected.goss_rounds,
                    expected.goss_rounds,
                )
            )
            retained_fractions = sorted(
                {top_rate + other_rate for top_rate, other_rate in expected.goss_rates}
            )
            for retained in retained_fractions:
                goss_rows.append(
                    BENCHMARK.BoostingRow(
                        "goss",
                        seed,
                        f"uniform_{retained:.2f}",
                        1.0,
                        2.0,
                        1.0,
                        expected.goss_rounds,
                        expected.goss_rounds,
                        retained,
                    )
                )
            for top_rate, other_rate in expected.goss_rates:
                retained = top_rate + other_rate
                goss_rows.append(
                    BENCHMARK.BoostingRow(
                        "goss",
                        seed,
                        f"goss_{top_rate:.2f}_{other_rate:.2f}",
                        0.9,
                        2.0,
                        1.0,
                        expected.goss_rounds,
                        expected.goss_rounds,
                        retained,
                        f"uniform_{retained:.2f}",
                    )
                )

    dart_rows = []
    if "dart" in expected.sections:
        for seed in expected.seeds:
            for horizon in sorted({config.horizon for config in expected.dart_configs}):
                dart_rows.append(
                    BENCHMARK.BoostingRow(
                        "dart",
                        seed,
                        f"standard_{horizon}",
                        1.0,
                        2.0,
                        1.0,
                        horizon,
                        horizon,
                        dart_profile="standard_control",
                    )
                )
            for config in expected.dart_configs:
                dart_rows.append(
                    BENCHMARK.BoostingRow(
                        "dart",
                        seed,
                        f"dart_{config.horizon}_{config.drop_rate:.2f}_{config.max_drop}",
                        1.1,
                        2.0,
                        1.0,
                        config.horizon,
                        config.horizon,
                        None,
                        f"standard_{config.horizon}",
                        20.0,
                        dart_profile=config.profile,
                    )
                )
    return quantile_rows, goss_rows, dart_rows


def _expected_matrix(
    *,
    sections=("quantile", "goss", "dart"),
    seeds=(7,),
    quantile_alphas=(0.5,),
    goss_rates=((0.2, 0.1),),
    dart_configs=None,
):
    if dart_configs is None:
        dart_configs = (
            BENCHMARK.DartConfig(8, 0.10, 5, BENCHMARK.DART_PROFILE_DEFAULT_LIKE),
        )
    return BENCHMARK.make_expected_matrix(
        sections=sections,
        seeds=seeds,
        quick=True,
        quantile_alphas=quantile_alphas,
        goss_rates=goss_rates,
        goss_rounds=8,
        dart_configs=dart_configs,
    )


class ReviewGuardrailTests(unittest.TestCase):
    def test_ci_runs_review_guardrails_on_one_python_smoke_leg(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        step_blocks = [
            block.rstrip()
            for block in re.findall(r"(?ms)^      - name: .*?(?=^      - name:|\Z)", workflow)
        ]
        condition = "matrix.os == 'ubuntu-latest' && matrix.python-version == '3.13'"
        expected_steps = (
            (
                "Review benchmark contract tests",
                "python -m pytest benchmarks/tests/test_review_guardrails.py -q",
            ),
            (
                "Review benchmark quality gates",
                "python benchmarks/review_guardrails.py --quick --gate",
            ),
        )

        for name, command in expected_steps:
            self.assertIn(
                f"      - name: {name}\n        if: {condition}\n        run: {command}",
                step_blocks,
            )

    def test_quantile_fixture_and_weighted_quantile_are_deterministic(self) -> None:
        first = BENCHMARK.make_quantile_split_data(seed=17, n_train=96, n_test=48)
        second = BENCHMARK.make_quantile_split_data(seed=17, n_train=96, n_test=48)
        for left, right in zip(first, second, strict=True):
            np.testing.assert_array_equal(left, right)

        value = BENCHMARK.weighted_quantile(
            np.asarray([1.0, 5.0, 9.0]),
            np.asarray([1.0, 3.0, 1.0]),
            0.5,
        )
        self.assertEqual(value, 5.0)

    def test_quantile_primitives_match_hand_computed_boundaries(self) -> None:
        loss = BENCHMARK.pinball_loss(
            np.asarray([0.0, 2.0]),
            np.asarray([1.0, 1.0]),
            np.asarray([1.0, 3.0]),
            0.25,
        )
        self.assertEqual(loss, 0.375)

        exact_boundary = BENCHMARK.weighted_quantile(
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([1.0, 1.0, 2.0]),
            0.5,
        )
        self.assertEqual(exact_boundary, 2.0)

        zero_weight_ignored = BENCHMARK.weighted_quantile(
            np.asarray([-100.0, 5.0, 100.0]),
            np.asarray([0.0, 2.0, 0.0]),
            0.5,
        )
        self.assertEqual(zero_weight_ignored, 5.0)

        for invalid_alpha in (0.0, 1.0):
            with self.subTest(alpha=invalid_alpha):
                with self.assertRaises(ValueError):
                    BENCHMARK.weighted_quantile(
                        np.asarray([1.0, 2.0]),
                        np.asarray([1.0, 1.0]),
                        invalid_alpha,
                    )

    def test_proxy_pinball_gradient_matches_production_at_zero_residual(self) -> None:
        residual = np.asarray([-1.0, 0.0, 1.0])
        weights = np.asarray([2.0, 3.0, 4.0])

        gradient, hessian = BENCHMARK.proxy_pinball_grad_hess(
            residual,
            weights,
            alpha=0.2,
        )

        np.testing.assert_allclose(
            gradient,
            np.asarray([1.6, 2.4, -0.8]),
            rtol=0.0,
            atol=np.finfo(np.float64).eps * 2.0,
        )
        np.testing.assert_array_equal(hessian, weights)

    def test_expected_matrix_rejects_missing_quantile_seed_alpha_and_arm(self) -> None:
        expected = BENCHMARK.make_expected_matrix(
            sections=("quantile",),
            seeds=(7, 13),
            quick=True,
        )
        quantile_rows, _, _ = _rows_for_expected_matrix(expected)
        omissions = (
            [row for row in quantile_rows if row.seed != 13],
            [row for row in quantile_rows if not (row.seed == 7 and row.alpha == 0.9)],
            [
                row
                for row in quantile_rows
                if not (row.seed == 7 and row.alpha == 0.5 and row.arm == "smooth_0.10")
            ],
        )

        for rows in omissions:
            with self.subTest(row_count=len(rows)):
                gates = BENCHMARK.evaluate_gates(rows, [], [], expected=expected)
                self.assertFalse(
                    next(gate for gate in gates if gate.name == "quantile_contract").passed
                )

    def test_expected_matrix_rejects_missing_goss_rate(self) -> None:
        expected = BENCHMARK.make_expected_matrix(
            sections=("goss",),
            seeds=(7,),
            quick=True,
        )
        _, goss_rows, _ = _rows_for_expected_matrix(expected)
        goss_rows = [row for row in goss_rows if row.arm != "goss_0.30_0.10"]

        gates = BENCHMARK.evaluate_gates([], goss_rows, [], expected=expected)

        self.assertFalse(next(gate for gate in gates if gate.name == "goss_contract").passed)

    def test_expected_matrix_rejects_missing_or_reclassified_dart_stress(self) -> None:
        expected = BENCHMARK.make_expected_matrix(
            sections=("dart",),
            seeds=(7,),
            quick=True,
        )
        _, _, dart_rows = _rows_for_expected_matrix(expected)
        omitted_stress = [
            row for row in dart_rows if row.arm != "dart_16_0.20_5"
        ]
        reclassified_default = [
            (
                BENCHMARK.BoostingRow(
                    **{
                        **row.__dict__,
                        "dart_profile": "stress_profile",
                    }
                )
                if row.arm == "dart_8_0.10_5"
                else row
            )
            for row in dart_rows
        ]

        omitted_gates = BENCHMARK.evaluate_gates(
            [],
            [],
            omitted_stress,
            expected=expected,
        )
        reclassified_gates = BENCHMARK.evaluate_gates(
            [],
            [],
            reclassified_default,
            expected=expected,
        )

        self.assertFalse(
            next(gate for gate in omitted_gates if gate.name == "dart_contract").passed
        )
        self.assertFalse(
            next(gate for gate in reclassified_gates if gate.name == "dart_contract").passed
        )

    def test_expected_matrix_scopes_contracts_to_selected_section(self) -> None:
        expected = BENCHMARK.make_expected_matrix(
            sections=("quantile",),
            seeds=(7,),
            quick=True,
        )
        quantile_rows, _, _ = _rows_for_expected_matrix(expected)

        gates = BENCHMARK.evaluate_gates(
            quantile_rows,
            [],
            [],
            expected=expected,
        )

        self.assertEqual(
            [gate.name for gate in gates],
            ["quantile_contract", "quantile_quality"],
        )
        self.assertTrue(all(gate.passed for gate in gates))

    def test_smoothed_pinball_gradient_is_continuous_with_nonnegative_hessian(self) -> None:
        alpha = 0.2
        width = 0.5
        left_width = width * (1.0 - alpha)
        right_width = width * alpha
        residual = np.asarray(
            [
                -left_width - 1e-9,
                -left_width + 1e-9,
                0.0,
                right_width - 1e-9,
                right_width + 1e-9,
            ]
        )
        gradient, hessian = BENCHMARK.smoothed_pinball_grad_hess(
            residual, np.ones_like(residual), alpha=alpha, width=width
        )
        self.assertLess(np.max(np.abs(np.diff(gradient)[[0, 3]])), 1e-6)
        self.assertTrue(np.all(hessian >= 0.0))
        self.assertLess(abs(gradient[2]), 1e-12)

    def test_quantile_split_experiment_returns_valid_finite_children(self) -> None:
        rows = BENCHMARK.run_quantile_experiment(
            seeds=(7,), alphas=(0.1, 0.5, 0.9), n_train=160, n_test=96
        )
        self.assertEqual({row.arm for row in rows}, {"proxy", "smooth_0.05", "smooth_0.10"})
        self.assertTrue(
            all(np.isfinite(row.gain) and np.isfinite(row.pinball_loss) for row in rows)
        )
        self.assertTrue(all(row.left_count >= 8 and row.right_count >= 8 for row in rows))

    def test_boosting_fixture_and_dropout_pressure_are_deterministic(self) -> None:
        first = BENCHMARK.make_boosting_data(seed=19, n_train=128, n_test=64)
        second = BENCHMARK.make_boosting_data(seed=19, n_train=128, n_test=64)
        for left, right in zip(first, second, strict=True):
            np.testing.assert_array_equal(left, right)

        low_cap = BENCHMARK.configured_dropout_pressure(
            n_estimators=100, drop_rate=0.2, max_drop=5
        )
        high_cap = BENCHMARK.configured_dropout_pressure(
            n_estimators=100, drop_rate=0.2, max_drop=50
        )
        self.assertLess(0.0, low_cap)
        self.assertLess(low_cap, high_cap)

    def test_small_goss_and_dart_runs_are_finite_and_complete(self) -> None:
        goss = BENCHMARK.run_goss_experiment(
            seeds=(7,), n_train=192, n_test=96, n_estimators=8, rates=((0.2, 0.1),)
        )
        dart = BENCHMARK.run_dart_experiment(
            seeds=(7,),
            n_train=192,
            n_test=96,
            configs=(
                BENCHMARK.DartConfig(
                    8,
                    0.10,
                    5,
                    BENCHMARK.DART_PROFILE_DEFAULT_LIKE,
                ),
            ),
        )
        for row in [*goss, *dart]:
            self.assertTrue(np.isfinite(row.rmse))
            self.assertGreater(row.fit_seconds, 0.0)
            self.assertEqual(row.completed_rounds, row.requested_rounds)
        self.assertEqual(
            {row.dart_profile for row in dart},
            {"standard_control", "default_like"},
        )

    def test_quality_gates_reject_quality_and_completion_regressions(self) -> None:
        quantile_rows = [
            BENCHMARK.QuantileSplitRow(7, 0.5, "proxy", 0.0, 1.0, 1.0, 1.0, 8, 8),
            BENCHMARK.QuantileSplitRow(7, 0.5, "smooth_0.05", 0.0, 1.0, 1.11, 1.0, 8, 8),
            BENCHMARK.QuantileSplitRow(7, 0.5, "smooth_0.10", 0.0, 1.0, 1.0, 1.0, 8, 8),
        ]
        goss_rows = [
            BENCHMARK.BoostingRow("goss", 7, "standard_full", 1.0, 2.0, 1.0, 8, 8),
            BENCHMARK.BoostingRow("goss", 7, "uniform_0.30", 1.0, 2.0, 1.0, 8, 8, 0.3),
            BENCHMARK.BoostingRow(
                "goss", 7, "goss_0.20_0.10", 1.36, 2.0, 1.0, 8, 8, 0.3, "uniform_0.30"
            ),
        ]
        dart_rows = [
            BENCHMARK.BoostingRow(
                "dart", 7, "standard_8", 1.0, 2.0, 1.0, 8, 8, dart_profile="standard_control"
            ),
            BENCHMARK.BoostingRow(
                "dart", 7, "dart_8_0.10_5", 1.0, 2.0, 100.0, 7, 8, None, "standard_8", 20.0,
                dart_profile="default_like",
            ),
        ]

        gates = BENCHMARK.evaluate_gates(
            quantile_rows,
            goss_rows,
            dart_rows,
            expected=_expected_matrix(),
        )
        self.assertTrue(any(not gate.passed and gate.name == "quantile_quality" for gate in gates))
        self.assertTrue(any(not gate.passed and gate.name == "goss_quality" for gate in gates))
        self.assertTrue(any(not gate.passed and gate.name == "dart_completion" for gate in gates))

        timing_only_rows = [
            BENCHMARK.BoostingRow(
                "dart", 7, "standard_8", 1.0, 2.0, 0.01, 8, 8, dart_profile="standard_control"
            ),
            BENCHMARK.BoostingRow(
                "dart", 7, "dart_8_0.10_5", 1.1, 2.0, 99.0, 8, 8, None, "standard_8", 20.0,
                dart_profile="default_like",
            ),
        ]
        timing_gates = BENCHMARK.evaluate_gates(
            [],
            [],
            timing_only_rows,
            expected=_expected_matrix(sections=("dart",)),
        )
        self.assertTrue(all(gate.passed for gate in timing_gates))

    def test_dart_quality_gate_excludes_stress_profiles_only(self) -> None:
        quantile_rows = [
            BENCHMARK.QuantileSplitRow(7, 0.5, arm, 0.0, 1.0, 1.0, 1.2, 8, 8)
            for arm in ("proxy", "smooth_0.05", "smooth_0.10")
        ]
        goss_rows = [
            BENCHMARK.BoostingRow("goss", 7, "standard_full", 1.0, 2.0, 1.0, 8, 8),
            BENCHMARK.BoostingRow("goss", 7, "uniform_0.30", 1.0, 2.0, 1.0, 8, 8, 0.3),
            BENCHMARK.BoostingRow(
                "goss", 7, "goss_0.20_0.10", 1.0, 2.0, 1.0, 8, 8, 0.3, "uniform_0.30"
            ),
        ]
        standard = BENCHMARK.BoostingRow(
            "dart", 7, "standard_8", 1.0, 2.0, 1.0, 8, 8, dart_profile="standard_control"
        )
        stress = BENCHMARK.BoostingRow(
            "dart",
            7,
            "dart_8_0.20_5",
            1.60,
            2.0,
            1.0,
            8,
            8,
            None,
            "standard_8",
            20.0,
            dart_profile="stress_profile",
        )
        default_like = BENCHMARK.BoostingRow(
            "dart",
            7,
            "dart_8_0.10_5",
            1.60,
            2.0,
            1.0,
            8,
            8,
            None,
            "standard_8",
            20.0,
            dart_profile="default_like",
        )

        stress_only_gates = BENCHMARK.evaluate_gates(
            [],
            [],
            [standard, stress],
            expected=_expected_matrix(
                sections=("dart",),
                dart_configs=(
                    BENCHMARK.DartConfig(
                        8,
                        0.20,
                        5,
                        BENCHMARK.DART_PROFILE_STRESS,
                    ),
                ),
            ),
        )
        self.assertTrue(next(gate for gate in stress_only_gates if gate.name == "dart_contract").passed)
        self.assertTrue(next(gate for gate in stress_only_gates if gate.name == "dart_quality").passed)

        default_like_gates = BENCHMARK.evaluate_gates(
            [],
            [],
            [standard, default_like],
            expected=_expected_matrix(sections=("dart",)),
        )
        self.assertTrue(next(gate for gate in default_like_gates if gate.name == "dart_contract").passed)
        self.assertFalse(next(gate for gate in default_like_gates if gate.name == "dart_quality").passed)

    def test_quantile_quality_gate_uses_median_arm_losses_across_seeds(self) -> None:
        quantile_rows = [
            BENCHMARK.QuantileSplitRow(seed, 0.5, arm, 0.0, 1.0, loss, 1.0, 8, 8)
            for seed, loss in ((7, 1.11), (13, 0.89))
            for arm in ("proxy", "smooth_0.05", "smooth_0.10")
        ]
        goss_rows = [
            BENCHMARK.BoostingRow("goss", 7, "standard_full", 1.0, 2.0, 1.0, 8, 8),
            BENCHMARK.BoostingRow("goss", 7, "uniform_0.30", 1.0, 2.0, 1.0, 8, 8, 0.3),
            BENCHMARK.BoostingRow(
                "goss", 7, "goss_0.20_0.10", 1.0, 2.0, 1.0, 8, 8, 0.3, "uniform_0.30"
            ),
        ]
        dart_rows = [
            BENCHMARK.BoostingRow(
                "dart", 7, "standard_8", 1.0, 2.0, 1.0, 8, 8, dart_profile="standard_control"
            ),
            BENCHMARK.BoostingRow(
                "dart", 7, "dart_8_0.10_5", 1.0, 2.0, 1.0, 8, 8, None, "standard_8", 20.0,
                dart_profile="default_like",
            ),
        ]

        gates = BENCHMARK.evaluate_gates(
            quantile_rows,
            [],
            [],
            expected=_expected_matrix(
                sections=("quantile",),
                seeds=(7, 13),
            ),
        )
        self.assertTrue(next(gate for gate in gates if gate.name == "quantile_quality").passed)

    def test_control_contracts_reject_wrong_goss_and_dart_matches(self) -> None:
        expected_goss = BENCHMARK.make_expected_matrix(
            sections=("goss",),
            seeds=(7,),
            quick=True,
        )
        _, valid_goss_rows, _ = _rows_for_expected_matrix(expected_goss)

        for control in ("standard_full", "goss_0.20_0.10", "uniform_0.20"):
            goss_rows = [
                (
                    BENCHMARK.BoostingRow(
                        **{
                            **row.__dict__,
                            "matched_control": control,
                        }
                    )
                    if row.arm == "goss_0.20_0.10"
                    else row
                )
                for row in valid_goss_rows
            ]
            gates = BENCHMARK.evaluate_gates(
                [],
                goss_rows,
                [],
                expected=expected_goss,
            )
            self.assertFalse(next(gate for gate in gates if gate.name == "goss_contract").passed)

        expected_dart = BENCHMARK.make_expected_matrix(
            sections=("dart",),
            seeds=(7,),
            quick=True,
        )
        _, _, valid_dart_rows = _rows_for_expected_matrix(expected_dart)
        for control in ("goss_0.20_0.10", "standard_16"):
            dart_rows = [
                (
                    BENCHMARK.BoostingRow(
                        **{
                            **row.__dict__,
                            "matched_control": control,
                        }
                    )
                    if row.arm == "dart_8_0.10_5"
                    else row
                )
                for row in valid_dart_rows
            ]
            gates = BENCHMARK.evaluate_gates(
                [],
                [],
                dart_rows,
                expected=expected_dart,
            )
            self.assertFalse(next(gate for gate in gates if gate.name == "dart_contract").passed)

    def test_contract_gates_reject_duplicate_and_nonfinite_rows(self) -> None:
        quantile_rows = [
            BENCHMARK.QuantileSplitRow(7, 0.5, arm, 0.0, 1.0, 1.0, 1.2, 8, 8)
            for arm in ("proxy", "smooth_0.05", "smooth_0.10")
        ]
        valid_goss_rows = [
            BENCHMARK.BoostingRow("goss", 7, "standard_full", 1.0, 2.0, 1.0, 8, 8),
            BENCHMARK.BoostingRow("goss", 7, "uniform_0.30", 1.0, 2.0, 1.0, 8, 8, 0.3),
            BENCHMARK.BoostingRow(
                "goss", 7, "goss_0.20_0.10", 1.0, 2.0, 1.0, 8, 8, 0.3, "uniform_0.30"
            ),
        ]
        valid_dart_rows = [
            BENCHMARK.BoostingRow(
                "dart", 7, "standard_8", 1.0, 2.0, 1.0, 8, 8, dart_profile="standard_control"
            ),
            BENCHMARK.BoostingRow(
                "dart", 7, "dart_8_0.10_5", 1.0, 2.0, 1.0, 8, 8, None, "standard_8", 20.0,
                dart_profile="default_like",
            ),
        ]
        duplicate_quantile_rows = [*quantile_rows, quantile_rows[0]]
        nonfinite_goss_rows = [
            *valid_goss_rows[:-1],
            BENCHMARK.BoostingRow(
                "goss", 7, "goss_0.20_0.10", float("nan"), 2.0, 1.0, 8, 8, 0.3, "uniform_0.30"
            ),
        ]

        gates = BENCHMARK.evaluate_gates(
            duplicate_quantile_rows,
            nonfinite_goss_rows,
            valid_dart_rows,
            expected=_expected_matrix(),
        )
        self.assertFalse(next(gate for gate in gates if gate.name == "quantile_contract").passed)
        self.assertFalse(next(gate for gate in gates if gate.name == "goss_contract").passed)

    def test_cli_reports_failed_selected_gate_to_stderr_and_returns_nonzero(self) -> None:
        failed_goss_rows = [
            BENCHMARK.BoostingRow("goss", 7, "standard_full", 1.0, 2.0, 1.0, 8, 8),
            BENCHMARK.BoostingRow("goss", 7, "uniform_0.30", 1.0, 2.0, 1.0, 8, 8, 0.3),
            BENCHMARK.BoostingRow(
                "goss", 7, "goss_0.20_0.10", 1.5, 2.0, 1.0, 8, 8, 0.3, "uniform_0.30"
            ),
        ]
        stderr = StringIO()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "failed-gate.md"
            with patch.object(BENCHMARK, "run_benchmark", return_value=([], failed_goss_rows, [])):
                with redirect_stderr(stderr):
                    exit_code = BENCHMARK.main(["--section", "goss", "--gate", "--output", str(output)])

        self.assertEqual(exit_code, 1)
        self.assertIn("gate failed: goss_quality", stderr.getvalue())

    def test_cli_passes_exact_quick_and_full_expected_matrices_to_gates(self) -> None:
        captured = []

        def capture_expected(
            quantile_rows,
            goss_rows,
            dart_rows,
            *,
            expected,
        ):
            captured.append(expected)
            return []

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "matrix.md"
            with patch.object(BENCHMARK, "run_benchmark", return_value=([], [], [])):
                with patch.object(BENCHMARK, "evaluate_gates", side_effect=capture_expected):
                    self.assertEqual(
                        BENCHMARK.main(["--quick", "--gate", "--output", str(output)]),
                        0,
                    )
                    self.assertEqual(
                        BENCHMARK.main(["--gate", "--output", str(output)]),
                        0,
                    )

        quick, full = captured
        self.assertEqual(quick.sections, BENCHMARK.ALL_SECTIONS)
        self.assertEqual(quick.seeds, (7,))
        self.assertEqual(quick.goss_rounds, 8)
        self.assertEqual(quick.dart_configs, BENCHMARK.QUICK_DART_CONFIGS)
        self.assertEqual(full.sections, BENCHMARK.ALL_SECTIONS)
        self.assertEqual(full.seeds, BENCHMARK.DEFAULT_SEEDS)
        self.assertEqual(full.goss_rounds, 100)
        self.assertEqual(full.dart_configs, BENCHMARK.FULL_DART_CONFIGS)

    def test_report_renders_smoothed_pinball_tie_with_tolerance(self) -> None:
        quantile_rows = [
            BENCHMARK.QuantileSplitRow(
                seed,
                0.5,
                arm,
                0.0,
                1.0,
                loss,
                1.2,
                8,
                8,
            )
            for seed in (7, 13, 29)
            for arm, loss in (
                ("proxy", 1.0),
                ("smooth_0.05", 0.04010663203691039),
                ("smooth_0.10", 0.04010663203691039 + 5e-13),
            )
        ]

        report = BENCHMARK.render_report(
            quantile_rows=quantile_rows,
            goss_rows=[],
            dart_rows=[],
            seeds=(7, 13, 29),
            quick=False,
        )

        self.assertIn(
            "Smoothed-pinball median tie (absolute tolerance `1e-12`): "
            "`smooth_0.05=0.04010663203691039`, "
            "`smooth_0.10=0.04010663203741039`.",
            report,
        )
        self.assertNotIn("Best smoothed-pinball median arm", report)

    def test_report_renders_unique_smoothed_pinball_winner(self) -> None:
        quantile_rows = [
            BENCHMARK.QuantileSplitRow(
                7,
                0.5,
                arm,
                0.0,
                1.0,
                loss,
                1.2,
                8,
                8,
            )
            for arm, loss in (
                ("proxy", 1.0),
                ("smooth_0.05", 0.8),
                ("smooth_0.10", 0.9),
            )
        ]

        report = BENCHMARK.render_report(
            quantile_rows=quantile_rows,
            goss_rows=[],
            dart_rows=[],
            seeds=(7,),
            quick=True,
        )

        self.assertIn(
            "Best smoothed-pinball median arm: `smooth_0.05` "
            "(`0.80000000000000004` versus `0.90000000000000002`).",
            report,
        )
        self.assertNotIn("Smoothed-pinball median tie", report)

    def test_report_and_cli_render_requested_sections(self) -> None:
        quantile_rows = [
            BENCHMARK.QuantileSplitRow(7, 0.5, "proxy", 0.0, 1.0, 1.0, 1.2, 8, 8),
            BENCHMARK.QuantileSplitRow(7, 0.5, "smooth_0.05", 0.0, 1.0, 0.9, 1.2, 8, 8),
            BENCHMARK.QuantileSplitRow(7, 0.5, "smooth_0.10", 0.0, 1.0, 0.95, 1.2, 8, 8),
        ]
        goss_rows = [
            BENCHMARK.BoostingRow("goss", 7, "standard_full", 1.0, 2.0, 1.0, 8, 8),
            BENCHMARK.BoostingRow("goss", 7, "uniform_0.30", 1.0, 2.0, 0.9, 8, 8, 0.3),
            BENCHMARK.BoostingRow(
                "goss", 7, "goss_0.20_0.10", 1.1, 2.0, 1.1, 8, 8, 0.3, "uniform_0.30"
            ),
        ]
        dart_rows = [
            BENCHMARK.BoostingRow(
                "dart", 7, "standard_8", 1.0, 2.0, 1.0, 8, 8, dart_profile="standard_control"
            ),
            BENCHMARK.BoostingRow(
                "dart", 7, "dart_8_0.10_5", 1.1, 2.0, 1.1, 8, 8, None, "standard_8", 20.0,
                dart_profile="default_like",
            ),
        ]
        report = BENCHMARK.render_report(
            quantile_rows=quantile_rows,
            goss_rows=goss_rows,
            dart_rows=dart_rows,
            seeds=(7,),
            quick=True,
        )
        for text in (
            "Configuration",
            "Quantile fixture: 160 training rows, 96 held-out rows",
            "Boosting fixture: 256 training rows, 128 held-out rows",
            "Model settings: depth 4, learning rate 0.06, lambda_l2=1.0",
            "GOSS rates: (0.10, 0.10), (0.20, 0.10), (0.20, 0.20), (0.30, 0.10)",
            "DART configs: (8, 0.10, 5, default_like), (16, 0.20, 5, stress_profile)",
            "## Quantile Split Selection",
            "## GOSS Rate Sweep",
            "## DART Dropout Profile",
            "Timing is descriptive",
            "smooth_0.05",
            "uniform_0.30",
            "Standard time ratio",
            "dropout pressure",
            "default_like",
            "stress_profile",
            "quality is non-blocking",
            "Standard-time ratios use unrounded median fit times; displayed fit times are rounded.",
        ):
            self.assertIn(text, report)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "quantile.md"
            exit_code = BENCHMARK.main(
                [
                    "--quick",
                    "--section",
                    "quantile",
                    "--gate",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            selected_report = output.read_text(encoding="utf-8")
        self.assertIn("## Quantile Split Selection", selected_report)
        self.assertIn("| quantile_contract | pass |", selected_report)
        self.assertNotIn("| goss_contract |", selected_report)
        self.assertNotIn("| dart_contract |", selected_report)
        self.assertNotIn("## GOSS Rate Sweep", selected_report)
        self.assertNotIn("## DART Dropout Profile", selected_report)


if __name__ == "__main__":
    unittest.main()
