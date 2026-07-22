from __future__ import annotations

import json
import logging
import os
import time as time_module
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

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
    "single-run": "https://single-runs-api.open-meteo.com/v1/forecast",
    "previous-runs": "https://previous-runs-api.open-meteo.com/v1/forecast",
}

ERCOT_TIMEZONE = ZoneInfo("America/Chicago")
ECMWF_SINGLE_RUN_ARCHIVE_START = date(2024, 3, 14)
SINGLE_RUN_AVAILABILITY_ASSUMPTION = (
    "ecmwf_ifs_single_run_initialized_before_pre_dam_cutoff"
)
PREVIOUS_RUNS_DAY2_ASSUMPTION = (
    "openmeteo_previous_day2_fixed_48h_lead_before_pre_dam_cutoff"
)
PREVIOUS_RUNS_HYBRID_ASSUMPTION = (
    "openmeteo_previous_day1_local_hours_00_08_else_day2_before_pre_dam_cutoff"
)
LOGGER = logging.getLogger(__name__)

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


def _previous_runs_location_records(
    payload: dict[str, Any],
    location: WeatherLocation,
    *,
    start_date: date,
    end_date: date,
    variable_names: tuple[str, ...],
    lead_mode: str,
) -> dict[str, Any]:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
        raise RuntimeError(
            f"Unexpected Previous Runs response for {location.name}"
        )

    output: dict[str, list[Any]] = {
        "time": [],
        "delivery_date_local": [],
        "forecast_run_time_utc": [],
        "decision_cutoff_utc": [],
        "forecast_lead_hours": [],
        **{variable: [] for variable in variable_names},
    }
    for index, value in enumerate(hourly["time"]):
        target = datetime.fromisoformat(str(value)).replace(tzinfo=timezone.utc)
        target_local = target.astimezone(ERCOT_TIMEZONE)
        delivery_date = target_local.date()
        if not start_date <= delivery_date <= end_date:
            continue
        lead_hours = (
            24
            if lead_mode == "hybrid" and target_local.hour <= 8
            else 48
        )
        output["time"].append(str(value))
        output["delivery_date_local"].append(delivery_date.isoformat())
        output["forecast_run_time_utc"].append(
            iso_utc(target - timedelta(hours=lead_hours))
        )
        output["decision_cutoff_utc"].append(
            iso_utc(_decision_cutoff_utc(delivery_date))
        )
        output["forecast_lead_hours"].append(lead_hours)
        for variable in variable_names:
            source_variable = (
                f"{variable}_previous_day{lead_hours // 24}"
            )
            values = hourly.get(source_variable)
            output[variable].append(
                values[index] if isinstance(values, list) and index < len(values)
                else None
            )

    return {
        "location": location.name,
        "requested_latitude": location.latitude,
        "requested_longitude": location.longitude,
        "response_latitude": payload.get("latitude"),
        "response_longitude": payload.get("longitude"),
        "forecast_model": "openmeteo_previous_runs_default_model",
        "forecast_lead_rule": (
            "previous_day1_local_hours_00_08_else_previous_day2"
            if lead_mode == "hybrid"
            else "previous_day2_fixed_48_hours"
        ),
        "availability_assumption": (
            PREVIOUS_RUNS_HYBRID_ASSUMPTION
            if lead_mode == "hybrid"
            else PREVIOUS_RUNS_DAY2_ASSUMPTION
        ),
        "hourly_units": payload.get("hourly_units", {}),
        "hourly": output,
    }


def collect_openmeteo_previous_runs_weather(
    start: date,
    end: date,
    *,
    locations: tuple[WeatherLocation, ...] = DEFAULT_TEXAS_LOCATIONS,
    hourly_variables: tuple[str, ...] = DEFAULT_HOURLY_VARIABLES,
    chunk_days: int = 31,
    api_key: str | None = None,
    raw_root: Path = DEFAULT_RAW_ROOT,
    session: requests.Session | None = None,
    skip_existing: bool = True,
    request_delay_seconds: float = 1.0,
    lead_mode: str = "day2",
) -> list[dict[str, Any]]:
    """Collect leakage-safe Previous Runs forecasts in date blocks."""
    validate_date_range(start, end)
    if not locations:
        raise ValueError("At least one weather location is required")
    if chunk_days < 1:
        raise ValueError("chunk_days must be at least 1")
    if request_delay_seconds < 0:
        raise ValueError("request_delay_seconds must not be negative")
    if lead_mode not in {"day2", "hybrid"}:
        raise ValueError("lead_mode must be 'day2' or 'hybrid'")

    api_key = api_key or os.getenv("OPENMETEO_API_KEY")
    http = session or build_http_session()
    endpoint = OPENMETEO_ENDPOINTS["previous-runs"]
    dataset = (
        "previous-runs-hybrid"
        if lead_mode == "hybrid"
        else "previous-runs-day2"
    )
    availability_assumption = (
        PREVIOUS_RUNS_HYBRID_ASSUMPTION
        if lead_mode == "hybrid"
        else PREVIOUS_RUNS_DAY2_ASSUMPTION
    )
    results: list[dict[str, Any]] = []
    query_start = start - timedelta(days=1)
    query_end = end + timedelta(days=1)

    for chunk_start, chunk_end_exclusive in iter_date_chunks(
        query_start, query_end, chunk_days
    ):
        chunk_end = chunk_end_exclusive - timedelta(days=1)
        directory = partitioned_directory(
            raw_root, "openmeteo", dataset, chunk_start
        )
        file_prefix = (
            f"{dataset}_{start.isoformat()}_{end.isoformat()}_"
            f"query_{chunk_start.isoformat()}_{chunk_end.isoformat()}_"
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

        params: dict[str, Any] = {
            "latitude": ",".join(str(item.latitude) for item in locations),
            "longitude": ",".join(str(item.longitude) for item in locations),
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "hourly": ",".join(
                f"{variable}_previous_day{day}"
                for variable in hourly_variables
                for day in ((1, 2) if lead_mode == "hybrid" else (2,))
            ),
            "timezone": "UTC",
            "temperature_unit": "celsius",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
        }
        if api_key:
            params["apikey"] = api_key
        if request_delay_seconds:
            time_module.sleep(request_delay_seconds)
        try:
            response = http.get(endpoint, params=params, timeout=120)
        except requests.exceptions.RetryError as exc:
            LOGGER.warning(
                "Open-Meteo Previous Runs rate limit reached after %d files",
                len(results),
            )
            results.append(
                {
                    "source": "openmeteo",
                    "dataset": dataset,
                    "status": "rate_limited",
                    "reason": str(exc),
                }
            )
            return results
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(
                "Open-Meteo Previous Runs error: "
                f"{payload.get('reason', json.dumps(payload)[:500])}"
            )
        payloads = payload if isinstance(payload, list) else [payload]
        if len(payloads) != len(locations):
            raise RuntimeError(
                f"Previous Runs returned {len(payloads)} locations; "
                f"expected {len(locations)}"
            )

        filtered_locations = [
            _previous_runs_location_records(
                item,
                location,
                start_date=start,
                end_date=end,
                variable_names=hourly_variables,
                lead_mode=lead_mode,
            )
            for item, location in zip(payloads, locations)
        ]
        output_payload = {
            "source": "openmeteo",
            "dataset": dataset,
            "delivery_date_start_local": start.isoformat(),
            "delivery_date_end_local": end.isoformat(),
            "query_start_utc": chunk_start.isoformat(),
            "query_end_utc": chunk_end.isoformat(),
            "forecast_model": "openmeteo_previous_runs_default_model",
            "forecast_lead_hours": [24, 48] if lead_mode == "hybrid" else [48],
            "availability_assumption": availability_assumption,
            "weather_values_complete": all(
                value is not None
                for item in filtered_locations
                for variable in hourly_variables
                for value in item["hourly"][variable]
            ),
            "locations": filtered_locations,
        }
        response_bytes = json.dumps(
            output_payload, ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")
        collected_at = utc_now()
        file_name = f"{file_prefix}{filename_timestamp(collected_at)}.json.gz"
        data_path = directory / file_name
        write_gzip_bytes(data_path, response_bytes)
        row_count = sum(
            len(item["hourly"]["time"]) for item in filtered_locations
        )
        public_params = {
            key: value for key, value in params.items() if key != "apikey"
        }
        metadata_path = write_metadata(
            data_path,
            {
                "source": "openmeteo",
                "dataset": dataset,
                "mode": f"previous-runs-{lead_mode}",
                "endpoint": endpoint,
                "request_params": public_params,
                "delivery_date_start_local": start.isoformat(),
                "delivery_date_end_local": end.isoformat(),
                "query_start_utc": chunk_start.isoformat(),
                "query_end_utc": chunk_end.isoformat(),
                "forecast_model": "openmeteo_previous_runs_default_model",
                "forecast_lead_hours": (
                    [24, 48] if lead_mode == "hybrid" else [48]
                ),
                "availability_assumption": availability_assumption,
                "weather_values_complete": output_payload[
                    "weather_values_complete"
                ],
                "locations": [item.name for item in locations],
                "collected_at_utc": iso_utc(collected_at),
                "row_count": row_count,
                "hourly_variables": list(hourly_variables),
            },
        )
        results.append(
            {
                "source": "openmeteo",
                "dataset": dataset,
                "data_path": str(data_path),
                "metadata_path": str(metadata_path),
                "row_count": row_count,
                "status": "downloaded",
            }
        )

    return results


def _delivery_day_bounds_utc(delivery_date: date) -> tuple[datetime, datetime]:
    start_local = datetime.combine(
        delivery_date, time.min, tzinfo=ERCOT_TIMEZONE
    )
    end_local = datetime.combine(
        delivery_date + timedelta(days=1), time.min, tzinfo=ERCOT_TIMEZONE
    )
    return start_local.astimezone(timezone.utc), end_local.astimezone(
        timezone.utc
    )


def _decision_cutoff_utc(delivery_date: date) -> datetime:
    cutoff_local = datetime.combine(
        delivery_date - timedelta(days=1),
        time(hour=9, minute=55),
        tzinfo=ERCOT_TIMEZONE,
    )
    return cutoff_local.astimezone(timezone.utc)


def _filter_single_run_location(
    payload: dict[str, Any],
    location: WeatherLocation,
    *,
    delivery_date: date,
    run_time_utc: datetime,
    model: str,
) -> dict[str, Any]:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
        raise RuntimeError(
            f"Unexpected Single Runs response for {location.name}"
        )

    start_utc, end_utc = _delivery_day_bounds_utc(delivery_date)
    selected_indices: list[int] = []
    for index, value in enumerate(hourly["time"]):
        target = datetime.fromisoformat(str(value)).replace(tzinfo=timezone.utc)
        if start_utc <= target < end_utc:
            selected_indices.append(index)

    expected_hours = int((end_utc - start_utc).total_seconds() // 3600)
    if len(selected_indices) != expected_hours:
        raise RuntimeError(
            f"Single Runs response for {location.name} delivery date "
            f"{delivery_date} has {len(selected_indices)} target hours; "
            f"expected {expected_hours}"
        )

    filtered_hourly: dict[str, list[Any]] = {}
    for key, values in hourly.items():
        if not isinstance(values, list):
            continue
        filtered_hourly[key] = [
            values[index] if index < len(values) else None
            for index in selected_indices
        ]

    return {
        "location": location.name,
        "requested_latitude": location.latitude,
        "requested_longitude": location.longitude,
        "response_latitude": payload.get("latitude"),
        "response_longitude": payload.get("longitude"),
        "delivery_date_local": delivery_date.isoformat(),
        "forecast_run_time_utc": iso_utc(run_time_utc),
        "decision_cutoff_utc": iso_utc(_decision_cutoff_utc(delivery_date)),
        "forecast_model": model,
        "forecast_lead_rule": "one_fixed_run_per_ercot_delivery_date",
        "availability_assumption": SINGLE_RUN_AVAILABILITY_ASSUMPTION,
        "hourly_units": payload.get("hourly_units", {}),
        "hourly": filtered_hourly,
    }


def collect_openmeteo_single_run_weather(
    start: date,
    end: date,
    *,
    locations: tuple[WeatherLocation, ...] = DEFAULT_TEXAS_LOCATIONS,
    hourly_variables: tuple[str, ...] = DEFAULT_HOURLY_VARIABLES,
    model: str = "ecmwf_ifs",
    run_hour_utc: int = 0,
    api_key: str | None = None,
    raw_root: Path = DEFAULT_RAW_ROOT,
    session: requests.Session | None = None,
    skip_existing: bool = True,
    request_delay_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    """Collect one fixed ECMWF run for every ERCOT local delivery date."""
    validate_date_range(start, end)
    if not locations:
        raise ValueError("At least one weather location is required")
    if run_hour_utc not in {0, 12}:
        raise ValueError("Historical ECMWF single runs support 00Z or 12Z")
    if request_delay_seconds < 0:
        raise ValueError("request_delay_seconds must not be negative")

    api_key = api_key or os.getenv("OPENMETEO_API_KEY")
    http = session or build_http_session()
    dataset = "single-run-ecmwf"
    endpoint = OPENMETEO_ENDPOINTS["single-run"]
    results: list[dict[str, Any]] = []

    earliest_delivery_date = ECMWF_SINGLE_RUN_ARCHIVE_START + timedelta(days=1)
    effective_start = max(start, earliest_delivery_date)
    if start < effective_start:
        results.append(
            {
                "source": "openmeteo",
                "dataset": dataset,
                "status": "unavailable_before_archive",
                "start_date": start.isoformat(),
                "end_date": min(end, effective_start - timedelta(days=1)).isoformat(),
                "reason": (
                    "ECMWF Single Runs archive begins with the "
                    f"{ECMWF_SINGLE_RUN_ARCHIVE_START.isoformat()} run"
                ),
            }
        )
    if effective_start > end:
        return results

    delivery_date = effective_start
    processed_days = 0
    total_days = (end - effective_start).days + 1
    while delivery_date <= end:
        run_date = delivery_date - timedelta(days=1)
        preferred_run_time_utc = datetime.combine(
            run_date, time(hour=run_hour_utc), tzinfo=timezone.utc
        )
        cutoff_utc = _decision_cutoff_utc(delivery_date)
        if preferred_run_time_utc >= cutoff_utc:
            raise RuntimeError(
                f"Run {iso_utc(preferred_run_time_utc)} is not before cutoff "
                f"{iso_utc(cutoff_utc)}"
            )

        directory = partitioned_directory(
            raw_root, "openmeteo", dataset, delivery_date
        )
        resume_prefix = f"{dataset}_delivery_{delivery_date.isoformat()}_run_"
        if skip_existing:
            completed = find_completed_file(
                directory, file_prefix=resume_prefix, extension=".json.gz"
            )
            if completed:
                data_path, metadata = completed
                results.append(
                    {
                        "source": "openmeteo",
                        "dataset": dataset,
                        "delivery_date_local": delivery_date.isoformat(),
                        "forecast_run_time_utc": metadata.get(
                            "forecast_run_time_utc"
                        ),
                        "data_path": str(data_path),
                        "metadata_path": str(data_path) + ".metadata.json",
                        "row_count": metadata.get("row_count"),
                        "status": "skipped_existing",
                    }
                )
                processed_days += 1
                if processed_days % 25 == 0 or processed_days == total_days:
                    LOGGER.info(
                        "Open-Meteo Single Runs progress: %d/%d delivery days",
                        processed_days,
                        total_days,
                    )
                delivery_date += timedelta(days=1)
                continue

        run_candidates = [preferred_run_time_utc]
        if run_hour_utc == 0:
            for days_back in range(2, 8):
                for fallback_hour in (0, 12):
                    run_candidates.append(
                        datetime.combine(
                            delivery_date - timedelta(days=days_back),
                            time(hour=fallback_hour),
                            tzinfo=timezone.utc,
                        )
                    )

        collected_at = utc_now()
        params: dict[str, Any] = {}
        run_time_utc: datetime | None = None
        filtered_locations: list[dict[str, Any]] | None = None
        unavailable_reasons: list[str] = []
        for candidate_run in run_candidates:
            if candidate_run >= cutoff_utc:
                continue
            params = {
                "latitude": ",".join(str(item.latitude) for item in locations),
                "longitude": ",".join(str(item.longitude) for item in locations),
                "run": candidate_run.strftime("%Y-%m-%dT%H:%M"),
                "models": model,
                "forecast_days": 10,
                "hourly": ",".join(hourly_variables),
                "timezone": "UTC",
                "temperature_unit": "celsius",
                "wind_speed_unit": "ms",
                "precipitation_unit": "mm",
            }
            if api_key:
                params["apikey"] = api_key

            if request_delay_seconds:
                time_module.sleep(request_delay_seconds)
            response = http.get(endpoint, params=params, timeout=90)
            candidate_payload = response.json()
            if (
                getattr(response, "status_code", 200) == 400
                and isinstance(candidate_payload, dict)
                and candidate_payload.get("error")
                and "run is not available"
                in str(candidate_payload.get("reason", "")).lower()
            ):
                unavailable_reasons.append(
                    str(candidate_payload.get("reason", "run unavailable"))
                )
                continue
            response.raise_for_status()
            if isinstance(candidate_payload, dict) and candidate_payload.get(
                "error"
            ):
                raise RuntimeError(
                    "Open-Meteo Single Runs error for delivery date "
                    f"{delivery_date}: "
                    f"{candidate_payload.get('reason', json.dumps(candidate_payload)[:500])}"
                )
            candidate_payloads = (
                candidate_payload
                if isinstance(candidate_payload, list)
                else [candidate_payload]
            )
            if len(candidate_payloads) != len(locations):
                raise RuntimeError(
                    f"Single Runs returned {len(candidate_payloads)} locations; "
                    f"expected {len(locations)}"
                )
            candidate_locations = [
                _filter_single_run_location(
                    item,
                    location,
                    delivery_date=delivery_date,
                    run_time_utc=candidate_run,
                    model=model,
                )
                for item, location in zip(candidate_payloads, locations)
            ]
            missing_values = sum(
                value is None
                for item in candidate_locations
                for variable in hourly_variables
                for value in item["hourly"].get(variable, [])
            )
            missing_variables = [
                variable
                for variable in hourly_variables
                if any(
                    variable not in item["hourly"]
                    or len(item["hourly"][variable])
                    != len(item["hourly"]["time"])
                    for item in candidate_locations
                )
            ]
            if missing_values or missing_variables:
                unavailable_reasons.append(
                    f"{iso_utc(candidate_run)} incomplete: "
                    f"{missing_values} null values, missing variables "
                    f"{','.join(missing_variables) or 'none'}"
                )
                continue

            run_time_utc = candidate_run
            filtered_locations = candidate_locations
            break

        if run_time_utc is None or filtered_locations is None:
            results.append(
                {
                    "source": "openmeteo",
                    "dataset": dataset,
                    "delivery_date_local": delivery_date.isoformat(),
                    "status": "unavailable_no_complete_run",
                    "reason": "; ".join(unavailable_reasons),
                }
            )
            processed_days += 1
            delivery_date += timedelta(days=1)
            continue

        output_payload = {
            "source": "openmeteo",
            "dataset": dataset,
            "delivery_date_local": delivery_date.isoformat(),
            "forecast_run_time_utc": iso_utc(run_time_utc),
            "decision_cutoff_utc": iso_utc(cutoff_utc),
            "forecast_model": model,
            "availability_assumption": SINGLE_RUN_AVAILABILITY_ASSUMPTION,
            "preferred_forecast_run_time_utc": iso_utc(
                preferred_run_time_utc
            ),
            "used_fallback_run": run_time_utc != preferred_run_time_utc,
            "weather_values_complete": True,
            "locations": filtered_locations,
        }
        response_bytes = json.dumps(
            output_payload, ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")
        run_slug = run_time_utc.strftime("%Y%m%dT%H%MZ")
        file_prefix = f"{resume_prefix}{run_slug}_"
        file_name = f"{file_prefix}{filename_timestamp(collected_at)}.json.gz"
        data_path = directory / file_name
        write_gzip_bytes(data_path, response_bytes)

        row_count = sum(
            len(item["hourly"]["time"]) for item in filtered_locations
        )
        public_params = {
            key: value for key, value in params.items() if key != "apikey"
        }
        metadata_path = write_metadata(
            data_path,
            {
                "source": "openmeteo",
                "dataset": dataset,
                "mode": "single-run",
                "endpoint": endpoint,
                "request_params": public_params,
                "delivery_date_local": delivery_date.isoformat(),
                "forecast_run_time_utc": iso_utc(run_time_utc),
                "preferred_forecast_run_time_utc": iso_utc(
                    preferred_run_time_utc
                ),
                "used_fallback_run": run_time_utc != preferred_run_time_utc,
                "weather_values_complete": True,
                "decision_cutoff_utc": iso_utc(cutoff_utc),
                "forecast_model": model,
                "availability_assumption": SINGLE_RUN_AVAILABILITY_ASSUMPTION,
                "locations": [item.name for item in locations],
                "collected_at_utc": iso_utc(collected_at),
                "row_count": row_count,
                "hourly_variables": list(hourly_variables),
            },
        )
        results.append(
            {
                "source": "openmeteo",
                "dataset": dataset,
                "delivery_date_local": delivery_date.isoformat(),
                "forecast_run_time_utc": iso_utc(run_time_utc),
                "data_path": str(data_path),
                "metadata_path": str(metadata_path),
                "row_count": row_count,
                "status": "downloaded",
            }
        )
        processed_days += 1
        if processed_days % 25 == 0 or processed_days == total_days:
            LOGGER.info(
                "Open-Meteo Single Runs progress: %d/%d delivery days",
                processed_days,
                total_days,
            )
        delivery_date += timedelta(days=1)

    return results
