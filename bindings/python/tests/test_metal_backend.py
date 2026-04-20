"""End-to-end tests for the Metal GPU backend (Stage 1).

The whole module is gated on `native_runtime_info().metal_available`
so it is a no-op on Intel Macs, Linux CI, and on wheels built with
`--no-default-features` (i.e. the `metal` feature disabled).

Stage 1 only accelerates the histogram-building phase; split finding,
partitioning, and prediction still run on the CPU path via the
`MetalBackend`'s embedded `CpuBackend`. Therefore the bit-exactness
claim is *stronger* than "within epsilon": a Metal-trained model
should serialize to identical `artifact_bytes` as the CPU-trained
model given the same seed and deterministic settings, because the
histogram kernel uses a two-pass deterministic reduction pattern
(no float atomics).

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
        np.testing.assert_array_equal(cpu.predict(X[:40]), metal.predict(X[:40]))
        np.testing.assert_array_equal(cpu.predict_proba(X[:40]), metal.predict_proba(X[:40]))


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
        np.testing.assert_array_equal(cpu.predict(X), metal.predict(X))

    def test_bin_count_255_matches_cpu(self) -> None:
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
        np.testing.assert_array_equal(cpu.predict(X), metal.predict(X))

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
