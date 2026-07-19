from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from import_ercot_generation_reports import import_generation_reports  # noqa: E402


class ManualGenerationImportTests(unittest.TestCase):
    def test_imports_manual_wind_csv_with_inferred_publish_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "wind_report_20250101_095500.csv"
            input_path.write_text(
                "\n".join(
                    [
                        "Interval Start,Interval End,ACTUAL SYSTEM WIDE,"
                        "STWPF SYSTEM WIDE,WGRPP SYSTEM WIDE",
                        "2025-01-01 00:00,2025-01-01 01:00,"
                        "1000,1200,1300",
                    ]
                ),
                encoding="utf-8",
            )

            results = import_generation_reports(
                report_key="wind-system",
                inputs=[str(input_path)],
                raw_root=root / "raw",
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["status"], "imported")
            data_path = Path(results[0]["data_path"])
            metadata_path = Path(results[0]["metadata_path"])
            self.assertTrue(data_path.exists())
            self.assertTrue(metadata_path.exists())

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["source"], "ercot_manual")
            self.assertEqual(metadata["report_type_id"], 13028)
            self.assertEqual(metadata["row_count"], 1)

            with gzip.open(data_path, "rt", encoding="utf-8") as handle:
                frame = pd.read_csv(handle)
            self.assertIn("interval_start_utc", frame.columns)
            self.assertIn("publish_time_utc", frame.columns)
            self.assertIn("gen_system_wide", frame.columns)
            self.assertEqual(
                frame["interval_start_utc"].iloc[0],
                "2025-01-01 06:00:00+00:00",
            )
            self.assertEqual(
                frame["publish_time_utc"].iloc[0],
                "2025-01-01 15:55:00+00:00",
            )


if __name__ == "__main__":
    unittest.main()
