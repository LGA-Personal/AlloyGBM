import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SCALE = _load_module("numerai_scale_module", REPO_ROOT / "benchmarks" / "numerai_scale.py")

ERAS = [f"{i:04d}" for i in range(1, 41)]


class PlanChunksTests(unittest.TestCase):
    def test_chunks_are_chronological_disjoint_and_embargoed(self) -> None:
        plan = SCALE.plan_chunks(ERAS, train_eras=10, valid_eras=5, test_eras=8, gap_eras=4)
        self.assertEqual(plan.test[-1], ERAS[-1])
        self.assertEqual((len(plan.train), len(plan.valid), len(plan.test)), (10, 5, 8))
        pos = {e: i for i, e in enumerate(ERAS)}
        self.assertEqual(pos[plan.valid[0]] - pos[plan.train[-1]], 5)
        self.assertEqual(pos[plan.test[0]] - pos[plan.valid[-1]], 5)
        self.assertFalse(set(plan.train) & set(plan.valid) | set(plan.valid) & set(plan.test))
        self.assertEqual([pos[e] for e in plan.train], sorted(pos[e] for e in plan.train))

    def test_end_offset_moves_the_window_back(self) -> None:
        plan = SCALE.plan_chunks(ERAS, train_eras=5, valid_eras=5, test_eras=5, gap_eras=0,
                                 end_offset=3)
        self.assertEqual(plan.test[-1], ERAS[-4])

    def test_too_few_eras_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "plan needs 41 eras"):
            SCALE.plan_chunks(ERAS, train_eras=30, valid_eras=5, test_eras=6, gap_eras=0)

    def test_era_sort_is_numeric(self) -> None:
        self.assertEqual(sorted(["0010", "0002", "era3"], key=SCALE.era_sort_key),
                         ["0002", "era3", "0010"])


class ParquetChunkTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow not installed")
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        rng = np.random.default_rng(0)
        self.features = ["feature_a", "feature_b"]

        def write(path: Path, eras: list[str], nan_eras: set[str]) -> None:
            era_col, target, fa, fb = [], [], [], []
            for era in eras:
                for _ in range(3):
                    era_col.append(era)
                    target.append(None if era in nan_eras else float(rng.integers(0, 5)) / 4)
                    fa.append(int(era))
                    fb.append(int(rng.integers(0, 5)))
            pq.write_table(pa.table({
                "era": era_col, "target": target,
                "feature_a": pa.array(fa, pa.int8()), "feature_b": pa.array(fb, pa.int8()),
                "feature_unused": pa.array(fb, pa.int8()),
            }), path)

        self.sources = {"train": root / "train.parquet", "validation": root / "validation.parquet"}
        write(self.sources["train"], ERAS[:20], set())
        # The newest validation eras are live rows with no target.
        write(self.sources["validation"], ERAS[20:] + ["0041", "0042"], {"0041", "0042"})

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_timeline_spans_both_files_and_skips_unlabelled_eras(self) -> None:
        timeline = SCALE.labelled_era_timeline(self.sources, "target")
        self.assertEqual([e.era for e in timeline], ERAS)
        self.assertEqual({e.source for e in timeline[:20]}, {"train"})
        self.assertEqual({e.source for e in timeline[20:]}, {"validation"})

    def test_load_chunk_reads_only_requested_eras_in_order(self) -> None:
        timeline = SCALE.labelled_era_timeline(self.sources, "target")
        eras = ERAS[17:23]  # straddles the two files
        chunk = SCALE.load_chunk(eras, timeline, self.sources, self.features, "target")
        self.assertEqual(chunk.X.shape, (18, 2))
        self.assertEqual(chunk.X.dtype, np.float32)
        self.assertTrue(chunk.X.flags["C_CONTIGUOUS"])
        self.assertEqual(sorted(set(chunk.eras)), eras)
        np.testing.assert_array_equal(chunk.X[:, 0], np.repeat(np.arange(18, 24), 3))
        self.assertTrue(np.isfinite(chunk.y).all())


class LibraryParamsTests(unittest.TestCase):
    def test_deep_profile_maps_onto_every_library(self) -> None:
        profile = SCALE.PROFILES["deep"]
        alloy, unmapped = SCALE.library_params(profile, "alloygbm", seed=85, threads=4)
        self.assertEqual(unmapped, [])
        self.assertEqual(alloy["tree_growth"], "leaf")
        self.assertEqual(alloy["max_leaves"], 1024)
        self.assertEqual(alloy["min_data_in_leaf"], 10_000)
        self.assertEqual(alloy["training_policy"], "manual")
        lgb, _ = SCALE.library_params(profile, "lightgbm", seed=85, threads=4)
        self.assertEqual((lgb["num_leaves"], lgb["max_depth"], lgb["min_child_samples"]),
                         (1024, 12, 10_000))
        xgb, _ = SCALE.library_params(profile, "xgboost", seed=85, threads=4)
        self.assertEqual(xgb["min_child_weight"], 10_000.0)
        cat, cat_unmapped = SCALE.library_params(profile, "catboost", seed=85, threads=4)
        self.assertEqual(cat["min_data_in_leaf"], 10_000)
        self.assertEqual(cat_unmapped, ["l1", "min_split_gain"])
        _, sym_unmapped = SCALE.library_params(profile, "catboost", seed=85, threads=4,
                                               catboost_grow_policy="SymmetricTree")
        self.assertIn("min_data_in_leaf", sym_unmapped)


if __name__ == "__main__":
    unittest.main()
