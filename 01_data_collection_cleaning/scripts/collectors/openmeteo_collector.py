from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests

from .common import (
    DEFAULT_RAW_ROOT,
    build_http_session,
    filename_timestamp,
    find_completed_file,
    iso_utc,
    iter_date_chunks,
    partitioned_directory,
    safe_slug,
    utc_now,
    validate_date_range,
    write_gzip_bytes,
    write_metadata,
)


OPENMETEO_ENDPOINTS = {
    "forecast": "https://api.open-meteo.com/v1/forecast",
    "historical-forecast": (
        "https://historical-forecast-api.open-meteo.com/v1/forecast"
    ),
}

DEFAULT_HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "shortwave_radiation",
    "precipitation",
)


@dataclass(frozen=True)
class WeatherLocation:
    name: str
    latitude: float
    longitude: float


DEFAULT_TEXAS_LOCATIONS = (
    WeatherLocation("Dallas", 32.7767, -96.7970),
    WeatherLocation("Fort_Worth", 32.7555, -97.3308),
    WeatherLocation("Denton", 33.2148, -97.1331),
    WeatherLocation("McKinney", 33.1972, -96.6397),
    WeatherLocation("Arlington", 32.7357, -97.1081),
    WeatherLocation("Wichita_Falls", 33.9137, -98.4934),
)


def parse_location(value: str) -> WeatherLocation:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError(
            "Location must use NAME,LATITUDE,LONGITUDE, for example "
            "Dallas,32.7767,-96.7970"
        )
    try:
        latitude = float(parts[1])
        longitude = float(parts[2])
    except ValueError as exc:
        raise ValueError(f"Invalid coordinates in location {value!r}") from exc
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError(f"Coordinates out of range in location {value!r}")
    return WeatherLocation(parts[0], latitude, longitude)


def collect_openmeteo_weather(
    start: date,
    end: date,
    *,
    locations: tuple[WeatherLocation, ...] = DEFAULT_TEXAS_LOCATIONS,
    mode: str = "historical-forecast",
    hourly_variables: tuple[str, ...] = DEFAULT_HOURLY_VARIABLES,
    chunk_days: int = 31,
    api_key: str | None = None,
    raw_root: Path = DEFAULT_RAW_ROOT,
    session: requests.Session | None = None,
    skip_existing: bool = True,
) -> list[dict[str, Any]]:
    """Download hourly Open-Meteo responses for one or more locations."""
    validate_date_range(start, end)
    if mode not in OPENMETEO_ENDPOINTS:
        raise ValueError(f"Unsupported Open-Meteo mode {mode!r}")
    if not locations:
        raise ValueError("At least one weather location is required")

    endpoint = OPENMETEO_ENDPOINTS[mode]
    api_key = api_key or os.getenv("OPENMETEO_API_KEY")
    http = session or build_http_session()
    results: list[dict[str, Any]] = []

    for location in locations:
        for chunk_start, chunk_end_exclusive in iter_date_chunks(
            start, end, chunk_days
        ):
            chunk_end = chunk_end_exclusive - timedelta(days=1)
            dataset = f"{mode}_{safe_slug(location.name)}"
            directory = partitioned_directory(
                raw_root, "openmeteo", dataset, chunk_start
            )
            file_prefix = (
                f"{dataset}_{chunk_start.isoformat()}_{chunk_end.isoformat()}_"
            )
            if skip_existing:
                completed = find_completed_file(
                    directory, file_prefix=file_prefix, extension=".json.gz"
                )
                if completed:
                    data_path, metadata = completed
                    results.append(
                        {
                            "source": "openmeteo",
                            "dataset": dataset,
                            "data_path": str(data_path),
                            "metadata_path": str(data_path) + ".metadata.json",
                            "row_count": metadata.get("row_count"),
                            "status": "skipped_existing",
                        }
                    )
                    continue

            collected_at = utc_now()
            params: dict[str, Any] = {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "start_date": chunk_start.isoformat(),
                "end_date": chunk_end.isoformat(),
                "hourly": ",".join(hourly_variables),
                "timezone": "UTC",
                "temperature_unit": "celsius",
                "wind_speed_unit": "ms",
                "precipitation_unit": "mm",
            }
            if api_key:
                params["apikey"] = api_key

            response = http.get(endpoint, params=params, timeout=90)
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                raise RuntimeError(
                    f"Open-Meteo error for {location.name}: "
                    f"{payload.get('reason', json.dumps(payload)[:500])}"
                )
            hourly = payload.get("hourly")
            if not isinstance(hourly, dict) or "time" not in hourly:
                raise RuntimeError(
                    f"Unexpected Open-Meteo response for {location.name}"
                )

            file_name = (
                f"{file_prefix}"
                f"{filename_timestamp(collected_at)}.json.gz"
            )
            data_path = directory / file_name
            write_gzip_bytes(data_path, response.content)

            public_params = {
                key: value for key, value in params.items() if key != "apikey"
            }
            metadata_path = write_metadata(
                data_path,
                {
                    "source": "openmeteo",
                    "dataset": dataset,
                    "mode": mode,
                    "endpoint": endpoint,
                    "request_params": public_params,
                    "location": {
                        "name": location.name,
                        "latitude": location.latitude,
                        "longitude": location.longitude,
                    },
                    "collected_at_utc": iso_utc(collected_at),
                    "row_count": len(hourly["time"]),
                    "hourly_variables": list(hourly_variables),
                },
            )
            results.append(
                {
                    "source": "openmeteo",
                    "dataset": dataset,
                    "data_path": str(data_path),
                    "metadata_path": str(metadata_path),
                    "row_count": len(hourly["time"]),
                    "status": "downloaded",
                }
            )
    return results
