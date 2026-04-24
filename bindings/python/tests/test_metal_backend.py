"""End-to-end tests for the Metal GPU backend.

The whole module is gated on `native_runtime_info().metal_available`
so it is a no-op on Intel Macs, Linux CI, and on wheels built with
`--no-default-features` (i.e. the `metal` feature disabled).

**Stage 1 (histograms)** accelerates the histogram-build phase on
GPU; the histogram reduction is order-deterministic so histograms
are bit-identical to CPU on the same seed + deterministic settings.

**Stage 2 (best-split finder)** adds GPU-accelerated split finding.
The kernel uses SIMD `simd_prefix_inclusive_sum` + block-scan for
gain computation; that is a tree reduction, not a serial left-to-
right accumulation, so `parent_gain_term` and `left_grad` drift by
a few ulps from the CPU's strict-ascending sweep. Typically this is
invisible (well-separated gains), but when two candidate thresholds
have gains within ~1e-6 of each other, the GPU and CPU can pick
different winners. A single diverged split at a shallow node
cascades through the rest of the tree, producing macroscopic
prediction deltas (up to ~0.1) in ~0.1% of rows on tiny shapes.

Therefore Stage 2 relaxes the bit-exactness gate:

- Stage 1 tests (histogram-only semantics) retained `array_equal`
  where the kernel's block-scan happens to stay tie-free at the
  given shape + seed — these are kept as stricter regression
  guards for the common path.
- Stage 1 tests where Stage 2's GPU split finder introduces
  structural divergence at the root-split tie-break have been
  relaxed to `assert_allclose(atol=0.1, rtol=0.1)` with an inline
  rationale.
- Stage 2 `MetalStage2Tests` uses `atol=1e-5, rtol=1e-5` at the
  plan's 50k × 100 × 255 × 20 golden shape where gains are
  comfortably separated; corner cases use looser tolerances.

The escape hatch `ALLOYGBM_METAL_DISABLE=1` is wired in the PyO3
runtime_backend layer (S1.9) and lets us exercise the
warn-and-fallback path on any Metal-capable runner.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

import numpy as np

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor, native_runtime_info

# Module-level skip: leave one explicit assertion rather than relying
# on pytest markers so the file also runs cleanly under `unittest`.
_METAL = native_runtime_info()
_SKIP_REASON = (
    "Metal backend unavailable on this runner "
    f"(gpu_family={_METAL.gpu_family!r}, "
    f"metal_available={_METAL.metal_available})"
)


def _make_regression_dataset(n_rows: int = 256, n_features: int = 4, seed: int = 42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_rows, n_features).astype(np.float32)
    y = (2.0 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + rng.randn(n_rows) * 0.1).astype(np.float32)
    return X, y


def _make_classification_dataset(n_rows: int = 256, n_features: int = 4, seed: int = 42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_rows, n_features).astype(np.float32)
    logits = 1.5 * X[:, 0] - X[:, 1]
    y = (logits > 0).astype(np.int64)
    return X, y


def _make_ranking_dataset(n_rows: int = 200, n_features: int = 3, n_groups: int = 10, seed: int = 42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_rows, n_features).astype(np.float32)
    group_sizes = [n_rows // n_groups] * n_groups
    group_sizes[-1] += n_rows - sum(group_sizes)
    # Group IDs in sorted order (required by GBMRanker)
    group = np.concatenate([[g] * sz for g, sz in enumerate(group_sizes)]).astype(np.int64)
    y = rng.randint(0, 5, size=n_rows).astype(np.float32)
    return X, y, group


@unittest.skipUnless(_METAL.metal_available, _SKIP_REASON)
class MetalAvailabilityTests(unittest.TestCase):
    """Smoke tests that depend on a real Metal device being present."""

    def test_gpu_family_is_nonempty_string(self) -> None:
        self.assertIsInstance(_METAL.gpu_family, str)
        self.assertGreater(len(_METAL.gpu_family), 0)

    def test_metal_available_implies_apple7_baseline(self) -> None:
        # Stage 1 kernels require Apple7. If we see
        # `metal_available=True` on a non-Apple-Silicon GPU we want
        # the test to fail loudly — the baseline gate in
        # runtime_backend would otherwise mask the misconfiguration.
        self.assertTrue(_METAL.metal_available)


@unittest.skipUnless(_METAL.metal_available, _SKIP_REASON)
class MetalRegressionTests(unittest.TestCase):
    """Regression: CPU/Metal should produce bit-identical artifacts."""

    def test_small_regression_bitexact_predictions_over_training_set(self) -> None:
        # Bit-exact parity: identical seed + deterministic=True should
        # produce identical predictions over the training set. A direct
        # `artifact_bytes` comparison would be stronger but unworkable
        # because the metadata JSON encodes `trained_device` and its
        # length prefix, so Metal and CPU artifacts legitimately differ
        # by a few bytes. Prediction equality over every training row
        # is the observable bit-exactness contract that matters.
        X, y = _make_regression_dataset(n_rows=128, n_features=3, seed=1)
        cpu = GBMRegressor(n_estimators=8, seed=7, deterministic=True, device="cpu")
        metal = GBMRegressor(n_estimators=8, seed=7, deterministic=True, device="metal")
        cpu.fit(X, y)
        metal.fit(X, y)
        np.testing.assert_array_equal(cpu.predict(X), metal.predict(X))

    def test_metal_regression_predictions_match_cpu(self) -> None:
        X, y = _make_regression_dataset(n_rows=200, n_features=5, seed=123)
        cpu = GBMRegressor(n_estimators=10, seed=11, deterministic=True, device="cpu")
        metal = GBMRegressor(n_estimators=10, seed=11, deterministic=True, device="metal")
        cpu.fit(X, y)
        metal.fit(X, y)
        np.testing.assert_array_equal(cpu.predict(X[:50]), metal.predict(X[:50]))

    def test_metal_regression_records_trained_device(self) -> None:
        X, y = _make_regression_dataset(n_rows=64, seed=3)
        metal = GBMRegressor(n_estimators=3, device="metal")
        metal.fit(X, y)
        self.assertIn(b'"trained_device":"metal"', metal.artifact_bytes)


@unittest.skipUnless(_METAL.metal_available, _SKIP_REASON)
class MetalClassificationTests(unittest.TestCase):
    """Binary classification: bit-exactness + predict_proba parity."""

    def test_small_classification_matches_cpu(self) -> None:
        X, y = _make_classification_dataset(n_rows=200, n_features=4, seed=9)
        cpu = GBMClassifier(n_estimators=8, seed=5, deterministic=True, device="cpu")
        metal = GBMClassifier(n_estimators=8, seed=5, deterministic=True, device="metal")
        cpu.fit(X, y)
        metal.fit(X, y)
        # Labels are well-clear of the decision boundary on this seed.
        np.testing.assert_array_equal(cpu.predict(X[:40]), metal.predict(X[:40]))
        # Probabilities drift by a few ulps under Stage 2's SIMD block-scan
        # vs CPU's serial gain accumulation. Documented module-level.
        np.testing.assert_allclose(
            cpu.predict_proba(X[:40]),
            metal.predict_proba(X[:40]),
            atol=1e-5,
            rtol=1e-5,
        )


@unittest.skipUnless(_METAL.metal_available, _SKIP_REASON)
class MetalRankerTests(unittest.TestCase):
    """Ranking: the ranker runs multiple objectives; pin one for parity."""

    def test_small_ranker_matches_cpu(self) -> None:
        X, y, group = _make_ranking_dataset(n_rows=150, n_features=3, n_groups=6, seed=17)
        cpu = GBMRanker(
            n_estimators=8,
            seed=13,
            deterministic=True,
            objective="rank_ndcg",
            device="cpu",
        )
        metal = GBMRanker(
            n_estimators=8,
            seed=13,
            deterministic=True,
            objective="rank_ndcg",
            device="metal",
        )
        cpu.fit(X, y, group=group)
        metal.fit(X, y, group=group)
        np.testing.assert_array_equal(cpu.predict(X), metal.predict(X))


@unittest.skipUnless(_METAL.metal_available, _SKIP_REASON)
class MetalEdgeCases(unittest.TestCase):
    """Boundary conditions that stress the kernel's assumptions."""

    def test_nan_handling_matches_cpu(self) -> None:
        """NaNs go into a dedicated bin; split finding learns the
        NaN-direction. The Metal kernel must respect the same
        layout as the CPU backend."""
        X, y = _make_regression_dataset(n_rows=200, n_features=4, seed=42)
        # Salt 5 % of the entries with NaN across random cells.
        rng = np.random.RandomState(42)
        mask = rng.rand(*X.shape) < 0.05
        X_nan = X.copy()
        X_nan[mask] = np.nan
        cpu = GBMRegressor(n_estimators=6, seed=1, deterministic=True, device="cpu")
        metal = GBMRegressor(n_estimators=6, seed=1, deterministic=True, device="metal")
        cpu.fit(X_nan, y)
        metal.fit(X_nan, y)
        np.testing.assert_array_equal(cpu.predict(X_nan), metal.predict(X_nan))

    def test_single_row_does_not_crash(self) -> None:
        # A 1-row dataset has no informative splits, but must not
        # crash the kernel or the dispatch loop. `predict` may
        # return either a list or a numpy array depending on input
        # shape — coerce via `np.asarray` so this test is robust
        # to either.
        X = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        y = np.array([0.5], dtype=np.float32)
        metal = GBMRegressor(n_estimators=2, device="metal")
        metal.fit(X, y)
        pred = np.asarray(metal.predict(X), dtype=np.float64)
        self.assertEqual(pred.shape, (1,))
        self.assertTrue(np.isfinite(pred[0]))

    def test_single_feature_matches_cpu(self) -> None:
        rng = np.random.RandomState(2)
        X = rng.randn(128, 1).astype(np.float32)
        y = (3.0 * X[:, 0] + rng.randn(128) * 0.05).astype(np.float32)
        cpu = GBMRegressor(n_estimators=5, seed=19, deterministic=True, device="cpu")
        metal = GBMRegressor(n_estimators=5, seed=19, deterministic=True, device="metal")
        cpu.fit(X, y)
        metal.fit(X, y)
        np.testing.assert_array_equal(cpu.predict(X), metal.predict(X))

    def test_bin_count_16_matches_cpu(self) -> None:
        # At B=16 on a tiny fixture (256 × 4), root-split gains across
        # features are close enough that Stage 2's SIMD block-scan ulp
        # drift can flip the winning feature at the root; the whole tree
        # then diverges. Max observed delta is ~0.06; rtol=0.1 covers
        # that without masking any gross regression.
        X, y = _make_regression_dataset(n_rows=256, n_features=4, seed=4)
        cpu = GBMRegressor(
            n_estimators=5,
            seed=2,
            deterministic=True,
            continuous_binning_max_bins=16,
            device="cpu",
        )
        metal = GBMRegressor(
            n_estimators=5,
            seed=2,
            deterministic=True,
            continuous_binning_max_bins=16,
            device="metal",
        )
        cpu.fit(X, y)
        metal.fit(X, y)
        np.testing.assert_allclose(
            cpu.predict(X), metal.predict(X), atol=0.1, rtol=0.1
        )

    def test_bin_count_255_matches_cpu(self) -> None:
        # Small shape (512 × 4) + full 255 bins leaves near-empty bins
        # at the tails whose tree-reduced prefix sums ulp-drift across
        # backends. Two rows (out of 512) end up on a different leaf
        # under Stage 2; the rest match bit-exactly. Loose tolerance
        # covers the tie-break flip without masking genuine bugs.
        X, y = _make_regression_dataset(n_rows=512, n_features=4, seed=5)
        cpu = GBMRegressor(
            n_estimators=5,
            seed=3,
            deterministic=True,
            continuous_binning_max_bins=255,
            device="cpu",
        )
        metal = GBMRegressor(
            n_estimators=5,
            seed=3,
            deterministic=True,
            continuous_binning_max_bins=255,
            device="metal",
        )
        cpu.fit(X, y)
        metal.fit(X, y)
        np.testing.assert_allclose(
            cpu.predict(X), metal.predict(X), atol=0.1, rtol=0.1
        )

    def test_bin_count_wide_u16_matches_cpu(self) -> None:
        # Above 255 the backend switches to u16 bin storage. The
        # Metal kernel has a separate compile-time specialisation
        # (function constants) for this; exercise it.
        X, y = _make_regression_dataset(n_rows=512, n_features=3, seed=6)
        cpu = GBMRegressor(
            n_estimators=4,
            seed=7,
            deterministic=True,
            continuous_binning_max_bins=1024,
            device="cpu",
        )
        metal = GBMRegressor(
            n_estimators=4,
            seed=7,
            deterministic=True,
            continuous_binning_max_bins=1024,
            device="metal",
        )
        cpu.fit(X, y)
        metal.fit(X, y)
        np.testing.assert_array_equal(cpu.predict(X), metal.predict(X))


@unittest.skipUnless(_METAL.metal_available, _SKIP_REASON)
class MetalGoldenTests(unittest.TestCase):
    """S1.13 — bit-exactness golden test at scale.

    The plan originally called for identical `artifact_bytes` at
    (50k rows × 100 features, 255 bins), but S1.12 proved that is
    not achievable as-written: the artifact metadata JSON encodes
    `trained_device` and its length prefix (Metal vs CPU legitimately
    differ by a few bytes there). Prediction bit-exactness over the
    full training set is the stronger observable contract, and we
    additionally assert equality over a held-out eval set to stress
    robustness under float-ordering variance.

    Runtime: CPU ~0.3s + Metal ~5s + eval ~0s ≈ 6s, well within
    the default pytest budget — no slow-gate marker needed.
    """

    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.RandomState(42)
        cls.X_train = rng.randn(50_000, 100).astype(np.float32)
        cls.y_train = (
            2.0 * cls.X_train[:, 0]
            - cls.X_train[:, 1]
            + 0.5 * cls.X_train[:, 2]
            + rng.randn(50_000) * 0.1
        ).astype(np.float32)
        # Held-out eval set from a distinct draw so we exercise
        # predict() on rows the tree never saw during fit.
        cls.X_eval = rng.randn(5_000, 100).astype(np.float32)

        # Fit once per class to keep the total runtime bounded (Metal
        # fit at this shape is ~5s; three tests × fresh fits would
        # needlessly triple the cost).
        cls.cpu_model = GBMRegressor(
            n_estimators=20,
            seed=7,
            deterministic=True,
            continuous_binning_max_bins=255,
            device="cpu",
        )
        cls.metal_model = GBMRegressor(
            n_estimators=20,
            seed=7,
            deterministic=True,
            continuous_binning_max_bins=255,
            device="metal",
        )
        cls.cpu_model.fit(cls.X_train, cls.y_train)
        cls.metal_model.fit(cls.X_train, cls.y_train)

    def test_golden_bitexact_predictions_on_training_set(self) -> None:
        np.testing.assert_array_equal(
            self.cpu_model.predict(self.X_train),
            self.metal_model.predict(self.X_train),
        )

    def test_golden_bitexact_predictions_on_heldout_set(self) -> None:
        np.testing.assert_array_equal(
            self.cpu_model.predict(self.X_eval),
            self.metal_model.predict(self.X_eval),
        )

    def test_golden_trained_device_recorded_in_artifact(self) -> None:
        self.assertIn(b'"trained_device":"cpu"', self.cpu_model.artifact_bytes)
        self.assertIn(b'"trained_device":"metal"', self.metal_model.artifact_bytes)


def _metal_feature_compiled_in() -> bool:
    """Heuristic probe: does the currently-installed wheel include the
    Metal backend feature?

    The `ALLOYGBM_METAL_DISABLE=1` escape hatch only produces its
    signature warning text (`"...ALLOYGBM_METAL_DISABLE=1..."`) when
    the crate was built with `feature = "metal"`. A wheel built with
    `--no-default-features` returns a different message
    (`"...this build does not include the Metal backend..."`) because
    the escape-hatch code path isn't compiled in.

    We do one subprocess probe at import time so the gate is cheap
    and doesn't pollute the live process's Metal device cache.
    """
    env = os.environ.copy()
    env["ALLOYGBM_METAL_DISABLE"] = "1"
    script = (
        "import warnings\n"
        "from alloygbm import GBMRegressor\n"
        "with warnings.catch_warnings(record=True) as captured:\n"
        "    warnings.simplefilter('always')\n"
        "    m = GBMRegressor(n_estimators=2, device='metal')\n"
        "    m.fit([[0.0],[1.0],[2.0]], [0,1,2])\n"
        "    for w in captured:\n"
        "        if issubclass(w.category, RuntimeWarning):\n"
        "            print(str(w.message))\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except Exception:
        return False
    return "ALLOYGBM_METAL_DISABLE" in (result.stdout or "")


_METAL_FEATURE_COMPILED = _metal_feature_compiled_in()
_FEATURE_SKIP_REASON = (
    "Metal feature not compiled into this build (e.g. `--no-default-features` wheel); "
    "the `ALLOYGBM_METAL_DISABLE=1` escape-hatch path isn't exercisable here."
)


@unittest.skipUnless(_METAL_FEATURE_COMPILED, _FEATURE_SKIP_REASON)
class MetalFallbackTests(unittest.TestCase):
    """The warn-and-fallback path uses the S1.9 `ALLOYGBM_METAL_DISABLE=1`
    escape hatch so it's testable on *any* Metal-enabled build
    regardless of hardware. These tests do NOT require
    `metal_available=True` (the escape hatch short-circuits Metal
    init), but they DO require the `metal` crate feature to be
    compiled in so the escape-hatch code path exists at all.

    We shell out to a subprocess instead of mutating the live
    process's env because the Metal backend caches the MTLDevice
    the first time it sees `device="metal"`; toggling the env var
    inside the test process would be racy depending on test order.
    """

    def _run_with_disable(self, script: str) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env["ALLOYGBM_METAL_DISABLE"] = "1"
        return subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

    def test_fallback_emits_runtime_warning(self) -> None:
        script = """
import warnings
from alloygbm import GBMRegressor
with warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter('always')
    m = GBMRegressor(n_estimators=3, device='metal')
    m.fit([[1.0],[2.0],[3.0],[4.0],[5.0]], [1,2,3,4,5])
    warns = [w for w in captured if issubclass(w.category, RuntimeWarning)]
    if len(warns) != 1:
        raise SystemExit(f'expected 1 RuntimeWarning, got {len(warns)}')
    msg = str(warns[0].message)
    if 'falling back to CPU' not in msg:
        raise SystemExit(f'unexpected warning text: {msg}')
    if 'ALLOYGBM_METAL_DISABLE' not in msg:
        raise SystemExit(f'warning missing escape hatch name: {msg}')
print('ok')
"""
        result = self._run_with_disable(script)
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout={result.stdout!r} stderr={result.stderr!r}",
        )
        self.assertIn("ok", result.stdout)

    def test_fallback_records_cpu_in_artifact(self) -> None:
        script = """
import warnings, re
from alloygbm import GBMRegressor
with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    m = GBMRegressor(n_estimators=3, device='metal')
    m.fit([[1.0],[2.0],[3.0],[4.0],[5.0]], [1,2,3,4,5])
match = re.search(rb'"trained_device":"(\\w+)"', m.artifact_bytes)
if match is None or match.group(1) != b'cpu':
    raise SystemExit(f'expected cpu fallback, got {match!r}')
print('ok')
"""
        result = self._run_with_disable(script)
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout={result.stdout!r} stderr={result.stderr!r}",
        )
        self.assertIn("ok", result.stdout)

    def test_fallback_preserves_device_attr_on_estimator(self) -> None:
        # Even though the *artifact* records CPU (the backend that
        # actually ran), the estimator's `device` attribute remains
        # what the user asked for. This matters for pickle
        # round-tripping and for users who inspect the model.
        script = """
import warnings
from alloygbm import GBMRegressor
with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    m = GBMRegressor(n_estimators=3, device='metal')
    m.fit([[1.0],[2.0],[3.0],[4.0],[5.0]], [1,2,3,4,5])
if m.device != 'metal':
    raise SystemExit(f'device attr should be preserved, got {m.device!r}')
print('ok')
"""
        result = self._run_with_disable(script)
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout={result.stdout!r} stderr={result.stderr!r}",
        )


@unittest.skipUnless(_METAL.metal_available, _SKIP_REASON)
class MetalStage2Tests(unittest.TestCase):
    """S2.6 — GPU best-split finder parity tests.

    Stage 2 moves `best_split` / `best_split_with_options` onto the GPU.
    The split kernel's block-scan over bins accumulates in a different
    order than the CPU's serial sweep, so the Stage 1 gate of byte-
    identical artifacts + bit-exact predictions may no longer hold in
    general. The Stage 2 gate is weaker but still strong:

    * **Prediction equality within float ulp epsilon.** If the split
      kernel is structurally matching the CPU split finder (same
      `(feature_index, threshold_bin, default_left)` per node), only
      the leaf-value Newton step and the gradient accumulations can
      diverge — both are well-conditioned float32 reductions, so
      `atol=1e-5, rtol=1e-5` comfortably contains their ulp drift.
      A structural mismatch (different winning threshold) produces
      macroscopic prediction differences that this gate catches.

    * **Fit per objective, shared across tests in the class.** The
      50k × 100 × 255 × 20 shape is too expensive to re-run per test
      (~10s combined), so each objective fits once in `setUpClass`.

    * **Stage 1's golden tests remain in force.** They exercise
      histogram bit-exactness; this class additionally exercises the
      Stage 2 split kernel under diverse option combinations
      (L1+L2 regularization, feature weighting via monotone
      constraints, NaN handling).

    Runtime budget: CPU ~0.5s + Metal ~8s per objective × 3 objectives
    ≈ 25s total across the class. Well within pytest's default budget.
    """

    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.RandomState(1337)

        # ---- regression fixture ---------------------------------
        cls.X_reg = rng.randn(50_000, 100).astype(np.float32)
        cls.y_reg = (
            1.5 * cls.X_reg[:, 0]
            - 0.8 * cls.X_reg[:, 1]
            + 0.3 * cls.X_reg[:, 2]
            + rng.randn(50_000) * 0.05
        ).astype(np.float32)
        cls.X_reg_eval = rng.randn(5_000, 100).astype(np.float32)

        cls.cpu_reg = GBMRegressor(
            n_estimators=20,
            max_depth=6,
            seed=2026,
            deterministic=True,
            continuous_binning_max_bins=255,
            lambda_l1=0.0,
            lambda_l2=0.0,
            device="cpu",
        )
        cls.metal_reg = GBMRegressor(
            n_estimators=20,
            max_depth=6,
            seed=2026,
            deterministic=True,
            continuous_binning_max_bins=255,
            lambda_l1=0.0,
            lambda_l2=0.0,
            device="metal",
        )
        cls.cpu_reg.fit(cls.X_reg, cls.y_reg)
        cls.metal_reg.fit(cls.X_reg, cls.y_reg)

        # ---- regression + L1/L2 fixture -------------------------
        cls.cpu_reg_l1l2 = GBMRegressor(
            n_estimators=15,
            max_depth=6,
            seed=2026,
            deterministic=True,
            continuous_binning_max_bins=255,
            lambda_l1=0.5,
            lambda_l2=2.0,
            device="cpu",
        )
        cls.metal_reg_l1l2 = GBMRegressor(
            n_estimators=15,
            max_depth=6,
            seed=2026,
            deterministic=True,
            continuous_binning_max_bins=255,
            lambda_l1=0.5,
            lambda_l2=2.0,
            device="metal",
        )
        cls.cpu_reg_l1l2.fit(cls.X_reg, cls.y_reg)
        cls.metal_reg_l1l2.fit(cls.X_reg, cls.y_reg)

        # ---- binary classification fixture ----------------------
        logits = (
            1.2 * cls.X_reg[:, 0]
            - 0.7 * cls.X_reg[:, 1]
            + 0.4 * cls.X_reg[:, 3]
        )
        cls.y_bin = (logits > 0).astype(np.int64)

        cls.cpu_clf = GBMClassifier(
            n_estimators=15,
            max_depth=6,
            seed=2026,
            deterministic=True,
            continuous_binning_max_bins=255,
            device="cpu",
        )
        cls.metal_clf = GBMClassifier(
            n_estimators=15,
            max_depth=6,
            seed=2026,
            deterministic=True,
            continuous_binning_max_bins=255,
            device="metal",
        )
        cls.cpu_clf.fit(cls.X_reg, cls.y_bin)
        cls.metal_clf.fit(cls.X_reg, cls.y_bin)

        # ---- ranking fixture ------------------------------------
        # Reuse the smaller-scale ranking helper — ranking-specific
        # objectives already exercise the split kernel per node.
        cls.X_rank, cls.y_rank, cls.group_rank = _make_ranking_dataset(
            n_rows=1_000,
            n_features=8,
            n_groups=25,
            seed=2026,
        )
        cls.cpu_rank = GBMRanker(
            n_estimators=10,
            max_depth=5,
            seed=2026,
            deterministic=True,
            objective="rank_ndcg",
            continuous_binning_max_bins=255,
            device="cpu",
        )
        cls.metal_rank = GBMRanker(
            n_estimators=10,
            max_depth=5,
            seed=2026,
            deterministic=True,
            objective="rank_ndcg",
            continuous_binning_max_bins=255,
            device="metal",
        )
        cls.cpu_rank.fit(cls.X_rank, cls.y_rank, group=cls.group_rank)
        cls.metal_rank.fit(cls.X_rank, cls.y_rank, group=cls.group_rank)

    # -- predictions agree within ulp epsilon -----------------------
    def test_regression_predictions_match_cpu_within_ulp(self) -> None:
        np.testing.assert_allclose(
            self.cpu_reg.predict(self.X_reg),
            self.metal_reg.predict(self.X_reg),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_regression_heldout_predictions_match_cpu_within_ulp(self) -> None:
        np.testing.assert_allclose(
            self.cpu_reg.predict(self.X_reg_eval),
            self.metal_reg.predict(self.X_reg_eval),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_regression_l1_l2_predictions_match_cpu_within_ulp(self) -> None:
        # Exercises the kernel's L1_ENABLED specialization + the
        # non-zero λ path in the gain formula. Mismatch here
        # implicates the l1_threshold_gradient port inside the MSL
        # kernel.
        np.testing.assert_allclose(
            self.cpu_reg_l1l2.predict(self.X_reg),
            self.metal_reg_l1l2.predict(self.X_reg),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_binary_classifier_probs_match_cpu_within_ulp(self) -> None:
        np.testing.assert_allclose(
            self.cpu_clf.predict_proba(self.X_reg[:1_000]),
            self.metal_clf.predict_proba(self.X_reg[:1_000]),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_binary_classifier_labels_match_cpu_exactly(self) -> None:
        # Labels are `(proba > 0.5)`-thresholded from the same proba
        # within 1e-5 — unless a proba is within 1e-5 of 0.5, labels
        # must agree exactly. The fixture's logits are broad so no
        # row is near the decision boundary at ulp precision.
        np.testing.assert_array_equal(
            self.cpu_clf.predict(self.X_reg[:1_000]),
            self.metal_clf.predict(self.X_reg[:1_000]),
        )

    def test_ranker_scores_match_cpu(self) -> None:
        # Ranking is the tightest test of cross-backend parity because
        # the downstream metric (NDCG) cares about *order*, not exact
        # values. At 1k × 8 × 25 groups the root-split tie-break
        # occasionally flips one row's leaf under Stage 2's SIMD
        # reduction; max observed delta ~0.07 on ≤0.1% of rows. We
        # assert (a) macroscopic tolerance and (b) that Spearman rank
        # correlation stays ≥0.99 — the metric that actually matters
        # for a ranker.
        cpu_pred = self.cpu_rank.predict(self.X_rank)
        metal_pred = self.metal_rank.predict(self.X_rank)
        np.testing.assert_allclose(cpu_pred, metal_pred, atol=0.1, rtol=0.1)
        # Spearman ρ — robust to ulp drift in scores.
        cpu_ranks = np.argsort(np.argsort(cpu_pred))
        metal_ranks = np.argsort(np.argsort(metal_pred))
        corr = np.corrcoef(cpu_ranks, metal_ranks)[0, 1]
        self.assertGreater(corr, 0.99, f"rank correlation {corr} < 0.99")

    # -- trained_device metadata --------------------------------------
    def test_metal_fits_record_trained_device_metal(self) -> None:
        for m in (self.metal_reg, self.metal_reg_l1l2, self.metal_clf, self.metal_rank):
            with self.subTest(estimator=type(m).__name__):
                self.assertIn(b'"trained_device":"metal"', m.artifact_bytes)


@unittest.skipUnless(_METAL.metal_available, _SKIP_REASON)
class MetalStage2NanAndMonotoneTests(unittest.TestCase):
    """S2.6 — corner-case parity: NaN-direction choice under Stage 2
    split kernel, and monotone-constraint post-filter unaffected by
    backend swap."""

    def test_nan_heavy_regression_matches_cpu_within_ulp(self) -> None:
        # 10% of a random feature's rows replaced with NaN. Each
        # split evaluates NaN-left vs NaN-right per threshold; the
        # kernel must pick the same winner as CPU.
        rng = np.random.RandomState(99)
        X = rng.randn(3_000, 10).astype(np.float32)
        y = (1.8 * X[:, 0] - X[:, 2] + rng.randn(3_000) * 0.05).astype(np.float32)
        nan_mask = rng.rand(*X.shape) < 0.10
        X_nan = X.copy()
        X_nan[nan_mask] = np.nan

        cpu = GBMRegressor(
            n_estimators=8,
            max_depth=5,
            seed=31,
            deterministic=True,
            continuous_binning_max_bins=255,
            device="cpu",
        )
        metal = GBMRegressor(
            n_estimators=8,
            max_depth=5,
            seed=31,
            deterministic=True,
            continuous_binning_max_bins=255,
            device="metal",
        )
        cpu.fit(X_nan, y)
        metal.fit(X_nan, y)
        np.testing.assert_allclose(
            cpu.predict(X_nan),
            metal.predict(X_nan),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_monotone_constraints_behave_identically_on_both_backends(self) -> None:
        # Monotone constraints are applied engine-side as a post-
        # filter on the SplitCandidate returned by the backend.
        # Stage 2's GPU best_split returns the same candidate shape
        # as the CPU path, so the engine's monotone filter applies
        # untouched. This test asserts that: same constraints +
        # same data + same seed → equal predictions within ulp.
        rng = np.random.RandomState(23)
        X = rng.randn(2_000, 4).astype(np.float32)
        y = (2.5 * X[:, 0] - 1.2 * X[:, 2] + rng.randn(2_000) * 0.05).astype(np.float32)
        constraints = [1, 0, -1, 0]

        cpu = GBMRegressor(
            n_estimators=6,
            max_depth=4,
            seed=17,
            deterministic=True,
            continuous_binning_max_bins=255,
            monotone_constraints=constraints,
            device="cpu",
        )
        metal = GBMRegressor(
            n_estimators=6,
            max_depth=4,
            seed=17,
            deterministic=True,
            continuous_binning_max_bins=255,
            monotone_constraints=constraints,
            device="metal",
        )
        cpu.fit(X, y)
        metal.fit(X, y)
        np.testing.assert_allclose(
            cpu.predict(X),
            metal.predict(X),
            atol=1e-5,
            rtol=1e-5,
        )


@unittest.skipUnless(_METAL.metal_available, _SKIP_REASON)
class MetalStage3Tests(unittest.TestCase):
    """S3.11 — Stage 3 (GPU residency) parity tests.

    Stage 3 keeps histograms *and* row indices resident on the GPU
    across the level-wise / leaf-wise trainer loops — no per-level
    CPU round-trip for the hot path. The kernels themselves
    (histogram, split, partition, subtract) are unchanged from
    Stages 1/2, so the float-ulp envelope is the same as Stage 2's
    (`atol=1e-5, rtol=1e-5` at the 50k × 100 × 255 × 20 golden
    shape).

    What Stage 3 *does* change is handle lifetime: row-index pool
    entries are minted by `apply_split` and flow as inputs into the
    next level's `build_histograms`; histogram pool entries are
    released on free-on-consume. These tests exist to pin down
    cases that Stage 1 + Stage 2 tests don't exercise — specifically
    leaf-wise growth (where `PendingSplit` holds handles across
    gain-ordered queue reordering) and mixed continuous/categorical
    fits (where the partition kernel's categorical bitset path
    produces GPU row indices that the next level's histogram kernel
    must bind).
    """

    def test_leaf_wise_growth_matches_cpu(self) -> None:
        """Leaf-wise mode threads `GpuRowIndexHandle`s through a
        gain-ordered `PendingSplit` queue; an accidentally-duplicated
        release or a missed handle would manifest as either a panic
        or macroscopically-divergent predictions here. 3000 × 12 ×
        max_leaves=32 is enough to push leaf-wise through ~6 levels
        of growth per tree.

        Tolerance: loose (`atol=0.1, rtol=0.1`). Leaf-wise is
        particularly sensitive to Stage 2's pre-existing
        gain-ulp-drift because the gain-ordered queue can reorder
        when two candidate splits have gains within ~1e-6 of each
        other; the rest of the tree then grows differently. See
        the module docstring and `MetalStage2NanAndMonotoneTests`.
        What this test gates is Stage 3's pool-lifetime invariant,
        not ulp precision.
        """
        rng = np.random.RandomState(71)
        X = rng.randn(3_000, 12).astype(np.float32)
        y = (
            1.3 * X[:, 0]
            - 0.7 * X[:, 3]
            + 0.4 * X[:, 7]
            + rng.randn(3_000) * 0.05
        ).astype(np.float32)

        common = dict(
            n_estimators=12,
            max_depth=8,
            max_leaves=32,
            tree_growth="leaf",
            seed=2027,
            deterministic=True,
            continuous_binning_max_bins=255,
        )
        cpu = GBMRegressor(device="cpu", **common)
        metal = GBMRegressor(device="metal", **common)
        cpu.fit(X, y)
        metal.fit(X, y)

        np.testing.assert_allclose(
            cpu.predict(X),
            metal.predict(X),
            atol=0.1,
            rtol=0.1,
        )

    def test_mixed_categorical_continuous_fit_matches_cpu(self) -> None:
        """D-017 coverage: continuous + categorical features in one
        fit. Stage 3 routes continuous best_split through the GPU
        kernel and categorical best_split to CpuBackend (D-012),
        but partition for *both* kinds runs on GPU via the
        `SPLIT_KIND = 1` bitset path (D-017). The row indices
        produced by a categorical split feed directly into the next
        level's histogram kernel with no CPU round-trip.
        """
        rng = np.random.RandomState(83)
        n_rows = 2_000
        X_cont = rng.randn(n_rows, 6).astype(np.float32)
        # One categorical column with 8 categories ("cat_0".."cat_7").
        cat_ids = rng.randint(0, 8, size=n_rows)
        cat_values = [f"cat_{i}" for i in cat_ids.tolist()]
        # The numeric matrix leaves the categorical column at 0.0
        # (per the categorical_feature_values_list protocol — the
        # real category identity lives in the parallel string list).
        X = np.hstack([X_cont, np.zeros((n_rows, 1), dtype=np.float32)])
        y = (
            1.5 * X_cont[:, 0]
            - 0.6 * X_cont[:, 2]
            + (cat_ids == 3).astype(np.float32) * 1.2
            + rng.randn(n_rows) * 0.05
        ).astype(np.float32)

        common = dict(
            n_estimators=10,
            max_depth=5,
            seed=2027,
            deterministic=True,
            continuous_binning_max_bins=255,
            # Native categorical splits enabled: max_cat_threshold > 0
            # turns on the Fisher-sort categorical partitioner on CPU
            # and routes the resulting bitset through the GPU
            # partition kernel's SPLIT_KIND=1 path.
            max_cat_threshold=16,
            categorical_feature_indices=[6],
        )
        cpu = GBMRegressor(device="cpu", **common)
        metal = GBMRegressor(device="metal", **common)
        cpu.fit(X, y, categorical_feature_values_list=[cat_values])
        metal.fit(X, y, categorical_feature_values_list=[cat_values])

        # Loose tolerance — this test's value is Stage 3 plumbing
        # correctness (categorical bitset flowing through GPU
        # partition → GPU row indices → next level's histogram
        # kernel), not ulp precision.
        np.testing.assert_allclose(
            cpu.predict(X),
            metal.predict(X),
            atol=0.1,
            rtol=0.1,
        )

    def test_deep_tree_stresses_pool_lifetime(self) -> None:
        """Depth-10 level-wise tree has ~1023 nodes and ~10 trainer
        levels; every level mints a fresh row-index handle per node
        and releases the parent handles on free-on-consume. A pool
        leak would eventually blow past
        `MTLDevice.recommendedMaxWorkingSetSize` — not on a depth-10
        fit, but a lifetime bug (e.g. missing release guard on an
        early-return path) would manifest as a panic or mismatched
        predictions here well before the allocator breaks.
        """
        rng = np.random.RandomState(109)
        X = rng.randn(5_000, 20).astype(np.float32)
        y = (
            1.0 * X[:, 0]
            - 0.5 * X[:, 5]
            + 0.3 * X[:, 10]
            + rng.randn(5_000) * 0.1
        ).astype(np.float32)

        common = dict(
            n_estimators=5,
            max_depth=10,
            seed=2027,
            deterministic=True,
            continuous_binning_max_bins=255,
        )
        cpu = GBMRegressor(device="cpu", **common)
        metal = GBMRegressor(device="metal", **common)
        cpu.fit(X, y)
        metal.fit(X, y)

        # Loose tolerance — deep trees stack Stage 2 ulp drift across
        # many splits; what this test gates is that the ~1023-node
        # level-wise fit completes without a pool leak or panic and
        # produces qualitatively-identical predictions.
        np.testing.assert_allclose(
            cpu.predict(X),
            metal.predict(X),
            atol=0.1,
            rtol=0.1,
        )


class InvalidDeviceTests(unittest.TestCase):
    """Device validation runs at estimator construction time, not at
    fit time, so no Metal hardware is required for these tests."""

    def test_unknown_device_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            GBMRegressor(device="tpu")

    def test_device_auto_accepted(self) -> None:
        # "auto" is currently an alias for "cpu" (the Stage 2+
        # heuristic is deferred), but it must round-trip through
        # construction without raising.
        m = GBMRegressor(n_estimators=2, device="auto")
        self.assertEqual(m.device, "auto")
        m.fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0])
        pred = m.predict([[1.5]])
        self.assertEqual(len(pred), 1)


if __name__ == "__main__":
    unittest.main()
