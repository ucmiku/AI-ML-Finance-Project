from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from build_clean import build_clean_database  # noqa: E402
from ingest_raw import initialize_database  # noqa: E402


class CleanDatabaseTests(unittest.TestCase):
    def _insert_file(
        self,
        connection: sqlite3.Connection,
        *,
        source: str,
        dataset: str,
        collected_at: str,
    ) -> int:
        cursor = connection.execute(
            """
            INSERT INTO raw_files(
                source, dataset, file_path, file_name, file_format, sha256,
                file_size_bytes, source_row_count, collected_at_utc,
                request_json, metadata_json, imported_at_utc
            ) VALUES (?, ?, ?, ?, 'json', ?, 1, 1, ?, '{}', '{}', ?)
            """,
            (
                source,
                dataset,
                f"/{dataset}-{collected_at}.json.gz",
                f"{dataset}.json.gz",
                f"sha-{dataset}-{collected_at}",
                collected_at,
                collected_at,
            ),
        )
        return int(cursor.lastrowid)

    def _insert_record(
        self,
        connection: sqlite3.Connection,
        *,
        file_id: int,
        record_number: int,
        record: dict,
        interval: str | None = None,
        publish: str | None = None,
        observation: str | None = None,
        location: str | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO raw_records(
                file_id, record_number, record_json, interval_start_utc,
                publish_time_utc, observation_date, location
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                record_number,
                json.dumps(record),
                interval,
                publish,
                observation,
                location,
            ),
        )

    def test_builds_typed_utc_tables_and_deduplicates_weather(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "raw.sqlite"
            analytics_path = root / "analytics.sqlite"
            connection = sqlite3.connect(raw_path)
            initialize_database(connection)

            old_weather = self._insert_file(
                connection,
                source="openmeteo",
                dataset="previous-runs-hybrid",
                collected_at="2025-01-01T00:00:00Z",
            )
            new_weather = self._insert_file(
                connection,
                source="openmeteo",
                dataset="previous-runs-hybrid",
                collected_at="2025-01-02T00:00:00Z",
            )
            ignored_weather = self._insert_file(
                connection,
                source="openmeteo",
                dataset="previous-runs-hybrid",
                collected_at="2025-01-02T00:00:01Z",
            )
            weather = {
                "delivery_date_local": "2025-01-01",
                "forecast_run_time_utc": "2024-12-30T00:00:00Z",
                "decision_cutoff_utc": "2024-12-31T15:55:00Z",
                "forecast_model": "openmeteo_previous_runs_default_model",
                "availability_assumption": (
                    "openmeteo_previous_day1_local_hours_00_08_else_day2_before_pre_dam_cutoff"
                ),
                "temperature_2m": 10.0,
                "relative_humidity_2m": 50,
                "wind_speed_10m": 3.0,
                "wind_gusts_10m": 5.0,
                "cloud_cover": 20,
                "shortwave_radiation": 100.0,
                "precipitation": 0.0,
            }
            weather_06 = {
                **weather,
                "forecast_run_time_utc": "2024-12-31T06:00:00Z",
            }
            self._insert_record(
                connection,
                file_id=old_weather,
                record_number=1,
                record=weather,
                interval="2025-01-01T00:00",
                location="Dallas",
            )
            self._insert_record(
                connection,
                file_id=new_weather,
                record_number=1,
                record={**weather, "temperature_2m": 11.0},
                interval="2025-01-01T00:00",
                location="Dallas",
            )
            self._insert_record(
                connection,
                file_id=new_weather,
                record_number=2,
                record=weather_06,
                interval="2025-01-01T06:00",
                location="Dallas",
            )
            self._insert_record(
                connection,
                file_id=ignored_weather,
                record_number=1,
                record=weather,
                interval="2025-01-01T00:00",
                location="Austin",
            )
            for file_index, (location, temperature) in enumerate((
                ("Fort_Worth", 10.0),
                ("Denton", 10.0),
                ("McKinney", 10.0),
                ("Arlington", 10.0),
                ("Wichita_Falls", 8.0),
            ), start=2):
                location_file = self._insert_file(
                    connection,
                    source="openmeteo",
                    dataset="previous-runs-hybrid",
                    collected_at=f"2025-01-02T00:00:{file_index:02d}Z",
                )
                self._insert_record(
                    connection,
                    file_id=location_file,
                    record_number=1,
                    record={**weather, "temperature_2m": temperature},
                    interval="2025-01-01T00:00",
                    location=location,
                )
                self._insert_record(
                    connection,
                    file_id=location_file,
                    record_number=2,
                    record=weather_06,
                    interval="2025-01-01T06:00",
                    location=location,
                )

            da_file = self._insert_file(
                connection,
                source="gridstatusio",
                dataset="ercot_spp_day_ahead_hourly",
                collected_at="2025-01-01T00:00:00Z",
            )
            self._insert_record(
                connection,
                file_id=da_file,
                record_number=1,
                record={
                    "interval_end_utc": "2025-01-01 07:00:00+00:00",
                    "location_type": "Trading Hub",
                    "market": "DAY_AHEAD_HOURLY",
                    "spp": 20.0,
                },
                interval="2025-01-01 06:00:00+00:00",
                location="HB_NORTH",
            )

            rt_file = self._insert_file(
                connection,
                source="gridstatusio",
                dataset="ercot_spp_real_time_15_min",
                collected_at="2025-01-01T00:00:00Z",
            )
            for index, price in enumerate((22.0, 24.0, 26.0, 28.0)):
                minute = index * 15
                end_minute = minute + 15
                self._insert_record(
                    connection,
                    file_id=rt_file,
                    record_number=index + 1,
                    record={
                        "interval_end_utc": (
                            f"2025-01-01 06:{end_minute:02d}:00+00:00"
                            if end_minute < 60
                            else "2025-01-01 07:00:00+00:00"
                        ),
                        "location_type": "Trading Hub",
                        "market": "REAL_TIME_15_MIN",
                        "spp": price,
                    },
                    interval=f"2025-01-01 06:{minute:02d}:00+00:00",
                    location="HB_NORTH",
                )

            gas_file = self._insert_file(
                connection,
                source="fred",
                dataset="DHHNGSP",
                collected_at="2025-01-02T00:00:00Z",
            )
            self._insert_record(
                connection,
                file_id=gas_file,
                record_number=1,
                record={
                    "value": "3.25",
                    "realtime_start": "2025-01-02",
                    "realtime_end": "2025-01-02",
                },
                observation="2025-01-01",
            )

            load_file = self._insert_file(
                connection,
                source="gridstatusio",
                dataset="ercot_seven_day_load_forecast",
                collected_at="2024-12-31T16:00:00Z",
            )
            self._insert_record(
                connection,
                file_id=load_file,
                record_number=1,
                record={
                    "interval_end_utc": "2025-01-01 06:05:00+00:00",
                    "load_forecast": 50000.0,
                },
                interval="2025-01-01 06:00:00+00:00",
                publish="2024-12-31 15:55:00+00:00",
            )
            connection.commit()
            connection.close()

            summary = build_clean_database(raw_path, analytics_path)
            self.assertEqual(summary["integrity_check"], "ok")

            clean = sqlite3.connect(analytics_path)
            try:
                weather_row = clean.execute(
                    """
                    SELECT city, target_hour_utc, temperature_2m_c
                    FROM clean_weather_hourly
                    WHERE city = 'Dallas'
                      AND target_hour_utc = '2025-01-01T00:00:00Z'
                    """
                ).fetchone()
                self.assertEqual(
                    weather_row,
                    ("Dallas", "2025-01-01T00:00:00Z", 11.0),
                )
                price_row = clean.execute(
                    """
                    SELECT da_price_usd_per_mwh, rt_price_usd_per_mwh,
                           spread_usd_per_mwh, is_label_complete
                    FROM clean_price_hourly
                    """
                ).fetchone()
                self.assertEqual(price_row, (20.0, 25.0, 5.0, 1))
                time_row = clean.execute(
                    """
                    SELECT decision_time_utc, delivery_time_local,
                           ercot_local_hour, decision_date_local
                    FROM feature_time_hourly
                    WHERE delivery_hour_utc = '2025-01-01T06:00:00Z'
                    """
                ).fetchone()
                self.assertEqual(
                    time_row,
                    (
                        "2024-12-31T15:55:00Z",
                        "2025-01-01T00:00:00-06:00",
                        0,
                        "2024-12-31",
                    ),
                )
                feature_row = clean.execute(
                    """
                    SELECT dfw_city_count, temperature_dfw_mean_c,
                           temperature_wichita_c,
                           temperature_wichita_minus_dfw_c,
                           north_temperature_min_c
                    FROM feature_weather_hourly
                    WHERE target_hour_utc = '2025-01-01T00:00:00Z'
                    """
                ).fetchone()
                self.assertEqual(feature_row[0], 5)
                self.assertAlmostEqual(feature_row[1], 10.2)
                self.assertAlmostEqual(feature_row[2], 8.0)
                self.assertAlmostEqual(feature_row[3], -2.2)
                self.assertAlmostEqual(feature_row[4], 8.0)
                self.assertEqual(
                    clean.execute(
                        "SELECT COUNT(*) FROM clean_load_forecast"
                    ).fetchone()[0],
                    1,
                )
                self.assertEqual(
                    clean.execute(
                        "SELECT COUNT(*) FROM clean_gas_daily"
                    ).fetchone()[0],
                    1,
                )
                self.assertIsNone(
                    clean.execute(
                        """
                        SELECT gas_price_usd_per_mmbtu
                        FROM feature_gas_da_daily
                        WHERE decision_date_local = '2024-12-31'
                        """
                    ).fetchone()[0]
                )
                self.assertEqual(
                    clean.execute(
                        """
                        SELECT split_name FROM model_split_assignments
                        WHERE delivery_hour_utc = '2025-01-01T06:00:00Z'
                        """
                    ).fetchone()[0],
                    "test",
                )
                self.assertEqual(
                    clean.execute(
                        "SELECT COUNT(*) FROM quality_check_results"
                    ).fetchone()[0],
                    14,
                )
                self.assertIsNone(
                    clean.execute(
                        """
                        SELECT name FROM sqlite_master
                        WHERE name LIKE '%wind%' OR name LIKE '%solar%'
                        """
                    ).fetchone()
                )
            finally:
                clean.close()


if __name__ == "__main__":
    unittest.main()
