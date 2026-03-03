import csv
import importlib.util
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


PANEL_PREPARE = _load_module(
    "panel_prepare_module", REPO_ROOT / "benchmarks" / "panel_time_series" / "prepare.py"
)
RUNNER = _load_module(
    "benchmark_runner_module", REPO_ROOT / "benchmarks" / "run_model_comparison.py"
)
DOW_PREPARE = _load_module(
    "dow_prepare_module", REPO_ROOT / "benchmarks" / "dow_jones_financial" / "prepare.py"
)


class TemporalLeakageTests(unittest.TestCase):
    def test_panel_prepare_uses_future_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_zip = tmp / "AirQualityUCI.zip"
            prepared_csv = tmp / "prepared.csv"

            rows = [
                {
                    "Date": "01/01/2004",
                    "Time": "00.00.00",
                    "CO(GT)": "1.0",
                    "PT08.S1(CO)": "10",
                    "C6H6(GT)": "20",
                    "PT08.S2(NMHC)": "30",
                    "NOx(GT)": "40",
                    "PT08.S3(NOx)": "50",
                    "NO2(GT)": "60",
                    "PT08.S4(NO2)": "70",
                    "PT08.S5(O3)": "80",
                    "T": "10",
                    "RH": "50",
                    "AH": "1.0",
                },
                {
                    "Date": "01/01/2004",
                    "Time": "01.00.00",
                    "CO(GT)": "2.0",
                    "PT08.S1(CO)": "11",
                    "C6H6(GT)": "21",
                    "PT08.S2(NMHC)": "31",
                    "NOx(GT)": "41",
                    "PT08.S3(NOx)": "51",
                    "NO2(GT)": "61",
                    "PT08.S4(NO2)": "71",
                    "PT08.S5(O3)": "81",
                    "T": "11",
                    "RH": "51",
                    "AH": "1.1",
                },
                {
                    "Date": "01/01/2004",
                    "Time": "02.00.00",
                    "CO(GT)": "3.0",
                    "PT08.S1(CO)": "12",
                    "C6H6(GT)": "22",
                    "PT08.S2(NMHC)": "32",
                    "NOx(GT)": "42",
                    "PT08.S3(NOx)": "52",
                    "NO2(GT)": "62",
                    "PT08.S4(NO2)": "72",
                    "PT08.S5(O3)": "82",
                    "T": "12",
                    "RH": "52",
                    "AH": "1.2",
                },
                {
                    "Date": "01/01/2004",
                    "Time": "03.00.00",
                    "CO(GT)": "4.0",
                    "PT08.S1(CO)": "13",
                    "C6H6(GT)": "23",
                    "PT08.S2(NMHC)": "33",
                    "NOx(GT)": "43",
                    "PT08.S3(NOx)": "53",
                    "NO2(GT)": "63",
                    "PT08.S4(NO2)": "73",
                    "PT08.S5(O3)": "83",
                    "T": "13",
                    "RH": "53",
                    "AH": "1.3",
                },
            ]

            fieldnames = list(rows[0].keys())
            with tempfile.TemporaryDirectory() as csv_tmp:
                csv_path = Path(csv_tmp) / "AirQualityUCI.csv"
                with csv_path.open("w", encoding="latin1", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
                    writer.writeheader()
                    writer.writerows(rows)

                with zipfile.ZipFile(raw_zip, "w") as archive:
                    archive.write(csv_path, arcname="AirQualityUCI.csv")

            kept_rows, dropped_no_future = PANEL_PREPARE._prepare_rows(
                raw_zip, prepared_csv, max_rows=0
            )
            self.assertEqual(kept_rows, 3)
            self.assertEqual(dropped_no_future, 1)

            frame = pd.read_csv(prepared_csv)
            self.assertEqual(frame["co_gt"].tolist(), [1.0, 2.0, 3.0])
            self.assertEqual(frame["target_co_gt"].tolist(), [2.0, 3.0, 4.0])
            self.assertFalse((frame["co_gt"] == frame["target_co_gt"]).all())

    def test_timestamp_split_has_no_overlap(self) -> None:
        frame = pd.DataFrame(
            {
                "group_id": ["A", "B", "A", "B", "A", "B", "A", "B"],
                "timestamp": [
                    "2024-01-01T00:00:00",
                    "2024-01-01T00:00:00",
                    "2024-01-02T00:00:00",
                    "2024-01-02T00:00:00",
                    "2024-01-03T00:00:00",
                    "2024-01-03T00:00:00",
                    "2024-01-04T00:00:00",
                    "2024-01-04T00:00:00",
                ],
                "f1": [1, 2, 3, 4, 5, 6, 7, 8],
                "target": [10, 20, 30, 40, 50, 60, 70, 80],
            }
        )

        train, test = RUNNER._split_by_timestamp(frame, test_size=0.25)
        train_timestamps = set(train["timestamp"].astype(str))
        test_timestamps = set(test["timestamp"].astype(str))
        self.assertTrue(train_timestamps.isdisjoint(test_timestamps))
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)

    def test_split_dataset_rejects_target_equivalent_feature(self) -> None:
        frame = pd.DataFrame(
            {
                "group_id": ["A", "A", "A", "A"],
                "timestamp": [
                    "2024-01-01T00:00:00",
                    "2024-01-02T00:00:00",
                    "2024-01-03T00:00:00",
                    "2024-01-04T00:00:00",
                ],
                "leaky_feature": [1.0, 2.0, 3.0, 4.0],
                "target": [1.0, 2.0, 3.0, 4.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "target-equivalent features"):
            RUNNER._split_dataset(
                scenario="test_scenario",
                frame=frame,
                target_column="target",
                seed=7,
                test_size=0.25,
            )

    def test_dow_manifest_output_fields_exclude_future_features(self) -> None:
        self.assertIn(
            "target_percent_change_next_weeks_price", DOW_PREPARE.OUTPUT_FIELDS
        )
        forbidden_prefixes = ("next_weeks_", "percent_change_next_weeks_price")
        leaked_feature_fields = [
            name
            for name in DOW_PREPARE.OUTPUT_FIELDS
            if name != "target_percent_change_next_weeks_price"
            and any(name.startswith(prefix) for prefix in forbidden_prefixes)
        ]
        self.assertEqual(leaked_feature_fields, [])


if __name__ == "__main__":
    unittest.main()
