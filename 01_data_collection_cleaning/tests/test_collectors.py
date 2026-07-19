from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from collectors.common import iter_date_chunks  # noqa: E402
from collectors.fred_collector import collect_fred_series  # noqa: E402
from collectors.gridstatus_collector import (  # noqa: E402
    collect_ercot_asof_forecasts,
    collect_ercot_forecasts,
    collect_ercot_prices,
)
from collectors.openmeteo_collector import (  # noqa: E402
    DEFAULT_TEXAS_LOCATIONS,
    WeatherLocation,
    collect_openmeteo_weather,
    parse_location,
)


class FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.content = json.dumps(payload).encode("utf-8")

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class FakeSession:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, *, params: dict, timeout: int) -> FakeResponse:
        self.calls.append((url, params))
        return FakeResponse(self.payload)


class FakeGridStatusClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def get_dataset(self, **kwargs) -> pd.DataFrame:
        self.calls.append(kwargs)
        requested_location = kwargs["filter_value"]
        return pd.DataFrame(
            {
                "interval_start_utc": ["2025-01-01T06:00:00+00:00"],
                "location": [requested_location],
                "market": [kwargs["dataset"]],
                "spp": [25.0],
            }
        )


class CollectorTests(unittest.TestCase):
    def test_date_chunks_cover_inclusive_range(self) -> None:
        chunks = list(
            iter_date_chunks(date(2025, 1, 1), date(2025, 1, 5), 2)
        )
        self.assertEqual(
            chunks,
            [
                (date(2025, 1, 1), date(2025, 1, 3)),
                (date(2025, 1, 3), date(2025, 1, 5)),
                (date(2025, 1, 5), date(2025, 1, 6)),
            ],
        )

    def test_parse_location(self) -> None:
        location = parse_location("Dallas,32.7767,-96.7970")
        self.assertEqual(location.name, "Dallas")
        self.assertAlmostEqual(location.latitude, 32.7767)

    def test_default_weather_locations_cover_north_texas(self) -> None:
        self.assertEqual(
            [location.name for location in DEFAULT_TEXAS_LOCATIONS],
            [
                "Dallas",
                "Fort_Worth",
                "Denton",
                "McKinney",
                "Arlington",
                "Wichita_Falls",
            ],
        )

    def test_fred_saves_raw_response_without_key_in_metadata(self) -> None:
        payload = {"observations": [{"date": "2025-01-01", "value": "3.00"}]}
        session = FakeSession(payload)
        with tempfile.TemporaryDirectory() as temp_dir:
            result = collect_fred_series(
                date(2025, 1, 1),
                date(2025, 1, 2),
                api_key="secret",
                raw_root=Path(temp_dir),
                session=session,
            )[0]
            metadata = json.loads(Path(result["metadata_path"]).read_text())
            self.assertNotIn("api_key", metadata["request_params"])
            with gzip.open(result["data_path"], "rt", encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), payload)

    def test_openmeteo_saves_hourly_response(self) -> None:
        payload = {
            "hourly": {
                "time": ["2025-01-01T00:00", "2025-01-01T01:00"],
                "temperature_2m": [10.0, 11.0],
            }
        }
        session = FakeSession(payload)
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_root = Path(temp_dir)
            result = collect_openmeteo_weather(
                date(2025, 1, 1),
                date(2025, 1, 1),
                locations=(WeatherLocation("Dallas", 32.7767, -96.7970),),
                raw_root=raw_root,
                session=session,
            )[0]
            self.assertEqual(result["row_count"], 2)
            self.assertTrue(Path(result["data_path"]).exists())

            resumed = collect_openmeteo_weather(
                date(2025, 1, 1),
                date(2025, 1, 1),
                locations=(WeatherLocation("Dallas", 32.7767, -96.7970),),
                raw_root=raw_root,
                session=session,
            )[0]
            self.assertEqual(resumed["status"], "skipped_existing")
            self.assertEqual(len(session.calls), 1)

    def test_gridstatusio_saves_filtered_north_hub_response(self) -> None:
        client = FakeGridStatusClient()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = collect_ercot_prices(
                date(2025, 1, 1),
                date(2025, 1, 1),
                markets=("DAY_AHEAD_HOURLY",),
                raw_root=Path(temp_dir),
                client=client,
            )[0]
            self.assertEqual(result["row_count"], 1)
            frame = pd.read_csv(result["data_path"])
            self.assertEqual(set(frame["location"]), {"HB_NORTH"})
            self.assertEqual(
                client.calls[0]["dataset"], "ercot_spp_day_ahead_hourly"
            )
            self.assertEqual(client.calls[0]["filter_column"], "location")
            self.assertEqual(client.calls[0]["filter_value"], "HB_NORTH")
            metadata = json.loads(Path(result["metadata_path"]).read_text())
            self.assertEqual(metadata["source"], "gridstatusio")
            self.assertNotIn("api_key", metadata["request"])

    def test_gridstatusio_saves_ercot_forecast_response(self) -> None:
        class FakeForecastClient:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def get_dataset(self, **kwargs) -> pd.DataFrame:
                self.calls.append(kwargs)
                return pd.DataFrame(
                    {
                        "interval_start_utc": ["2025-01-01T06:00:00+00:00"],
                        "interval_end_utc": ["2025-01-01T07:00:00+00:00"],
                        "publish_time_utc": ["2024-12-31T17:55:00+00:00"],
                        "stwpf_system_wide": [10000.0],
                    }
                )

        client = FakeForecastClient()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = collect_ercot_forecasts(
                date(2025, 1, 1),
                date(2025, 1, 1),
                forecasts=("WIND_PRODUCTION_FORECAST",),
                raw_root=Path(temp_dir),
                client=client,
            )[0]
            self.assertEqual(result["dataset"], "ercot_wind_production_forecast")
            self.assertEqual(result["row_count"], 1)
            frame = pd.read_csv(result["data_path"])
            self.assertIn("publish_time_utc", frame.columns)
            self.assertEqual(
                client.calls[0]["dataset"],
                "ercot_wind_actual_and_forecast_hourly",
            )
            metadata = json.loads(Path(result["metadata_path"]).read_text())
            self.assertEqual(metadata["forecast"], "WIND_PRODUCTION_FORECAST")
            self.assertNotIn("api_key", metadata["request"])

    def test_gridstatusio_saves_asof_forecast_response(self) -> None:
        class FakeAsofForecastClient:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def get_dataset(self, **kwargs) -> pd.DataFrame:
                self.calls.append(kwargs)
                return pd.DataFrame(
                    {
                        "interval_start_utc": [
                            "2025-01-01T06:00:00+00:00",
                            "2025-01-01T06:00:00+00:00",
                            "2025-01-01T06:00:00+00:00",
                        ],
                        "publish_time_utc": [
                            "2024-12-31T14:55:00+00:00",
                            "2024-12-31T15:55:00+00:00",
                            "2024-12-31T17:00:00+00:00",
                        ],
                        "stwpf_system_wide": [9000.0, 10000.0, 11000.0],
                    }
                )

        client = FakeAsofForecastClient()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = collect_ercot_asof_forecasts(
                date(2025, 1, 1),
                date(2025, 1, 1),
                forecasts=("WIND_PRODUCTION_FORECAST",),
                asof_days_before=1,
                asof_hour_local=10,
                raw_root=Path(temp_dir),
                client=client,
            )[0]
            self.assertEqual(
                result["dataset"],
                "ercot_wind_da",
            )
            self.assertIn("publish_time_end", client.calls[0])
            self.assertNotIn("publish_time", client.calls[0])
            frame = pd.read_csv(result["data_path"])
            self.assertEqual(len(frame), 1)
            self.assertEqual(float(frame["stwpf_system_wide"].iloc[0]), 10000.0)
            metadata = json.loads(Path(result["metadata_path"]).read_text())
            self.assertEqual(metadata["asof_rule"]["days_before"], 1)
            self.assertEqual(metadata["asof_rule"]["hour_local"], 10)
            self.assertEqual(metadata["candidate_row_count"], 3)


if __name__ == "__main__":
    unittest.main()
