import json
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "docs" / "benchmarks" / "data" / "2026-09-06-bin-budget"


class TestBinBudgetEvidence(unittest.TestCase):
    def setUp(self):
        self.before_path = DATA_DIR / "before.json"
        self.after_path = DATA_DIR / "after.json"
        self.assertTrue(self.before_path.exists(), f"Missing {self.before_path}")
        self.assertTrue(self.after_path.exists(), f"Missing {self.after_path}")

        with open(self.before_path) as f:
            self.before = json.load(f)
        with open(self.after_path) as f:
            self.after = json.load(f)

    def test_parameters_enforce_quantile_strategy(self):
        for run_name, data in [("before", self.before), ("after", self.after)]:
            params = data.get("params", {})
            self.assertEqual(
                params.get("alloy_continuous_binning_strategy"),
                "quantile",
                f"{run_name}.json was not run with --alloy-continuous-binning-strategy quantile",
            )
            self.assertEqual(
                params.get("alloy_continuous_binning_max_bins"),
                256,
                f"{run_name}.json was not run with --alloy-continuous-binning-max-bins 256",
            )

    def test_peer_models_are_stable_controls(self):
        before_records = {(r["scenario"], r["model"]): r for r in self.before["records"]}
        after_records = {(r["scenario"], r["model"]): r for r in self.after["records"]}

        peer_models = {"lightgbm", "xgboost", "catboost"}
        for key, b_rec in before_records.items():
            scenario, model = key
            if model in peer_models and key in after_records:
                a_rec = after_records[key]
                for metric in ("rmse", "mae", "accuracy", "log_loss_val", "ndcg_10"):
                    b_val = b_rec.get(metric)
                    a_val = a_rec.get(metric)
                    if b_val is not None and a_val is not None:
                        if not (b_val != b_val and a_val != a_val):
                            self.assertAlmostEqual(
                                b_val,
                                a_val,
                                places=5,
                                msg=f"Peer control drift for {model} on {scenario} ({metric})",
                            )

    def test_alloygbm_demonstrates_non_null_metric_movement(self):
        before_records = {(r["scenario"], r["model"]): r for r in self.before["records"]}
        after_records = {(r["scenario"], r["model"]): r for r in self.after["records"]}

        metric_deltas = []
        for key, b_rec in before_records.items():
            scenario, model = key
            if model == "alloygbm" and key in after_records:
                a_rec = after_records[key]
                for metric in ("rmse", "mae", "accuracy", "log_loss_val", "ndcg_10"):
                    b_val = b_rec.get(metric)
                    a_val = a_rec.get(metric)
                    if (
                        b_val is not None
                        and a_val is not None
                        and not (b_val != b_val or a_val != a_val)
                    ):
                        delta = abs(a_val - b_val)
                        metric_deltas.append(delta)

        self.assertTrue(len(metric_deltas) > 0, "No valid AlloyGBM metrics found to compare")
        has_movement = any(d > 1e-9 for d in metric_deltas)
        self.assertTrue(
            has_movement,
            "AlloyGBM metrics did not move between before and after runs. Quantile binning cuts were not exercised.",
        )


if __name__ == "__main__":
    unittest.main()
