from __future__ import annotations

import importlib.metadata
import gzip
import io
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .common import (
    DEFAULT_RAW_ROOT,
    filename_timestamp,
    find_completed_file,
    iso_utc,
    iter_date_chunks,
    partitioned_directory,
    utc_now,
    validate_date_range,
    write_metadata,
)


MARKET_DATASETS = {
    "DAY_AHEAD_HOURLY": "ercot_spp_day_ahead_hourly",
    "REAL_TIME_15_MIN": "ercot_spp_real_time_15_min",
}
DEFAULT_MARKETS = tuple(MARKET_DATASETS)

FORECAST_DATASETS = {
    "SEVEN_DAY_LOAD_FORECAST": {
        "api_dataset": "ercot_load_forecast",
        "dataset": "ercot_seven_day_load_forecast",
        "rows_per_day_limit": 500,
    },
    "WIND_PRODUCTION_FORECAST": {
        "api_dataset": "ercot_wind_actual_and_forecast_hourly",
        "dataset": "ercot_wind_production_forecast",
        "rows_per_day_limit": 8000,
    },
    "SOLAR_PRODUCTION_FORECAST": {
        "api_dataset": "ercot_solar_actual_and_forecast_hourly",
        "dataset": "ercot_solar_production_forecast",
        "rows_per_day_limit": 7000,
    },
}
DEFAULT_FORECASTS = tuple(FORECAST_DATASETS)
ERCOT_LOCAL_TIMEZONE = ZoneInfo("America/Chicago")


def _load_gridstatusio_client(api_key: str | None = None) -> Any:
    api_key = api_key or os.getenv("GRIDSTATUS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GRIDSTATUS_API_KEY is not set. Add it to .env or the process "
            "environment."
        )
    try:
        from gridstatusio import GridStatusClient
    except ImportError as exc:
        raise RuntimeError(
            "gridstatusio is not installed. Run: "
            "pip install -r requirements-data.txt"
        ) from exc
    return GridStatusClient(api_key=api_key, return_format="pandas")


def _dataset_for_market(market_name: str) -> str:
    try:
        return MARKET_DATASETS[market_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown market {market_name!r}. Available values: "
            f"{', '.join(MARKET_DATASETS)}"
        ) from exc


def _forecast_config(forecast_name: str) -> dict[str, Any]:
    try:
        return FORECAST_DATASETS[forecast_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown forecast {forecast_name!r}. Available values: "
            f"{', '.join(FORECAST_DATASETS)}"
        ) from exc


def _find_column(frame: pd.DataFrame, expected: str) -> str | None:
    normalized = {
        str(column).strip().lower().replace("_", " "): str(column)
        for column in frame.columns
    }
    return normalized.get(expected.strip().lower().replace("_", " "))


def _write_gridstatusio_frame(
    frame: pd.DataFrame,
    *,
    data_path: Path,
) -> None:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = data_path.with_name(data_path.name + ".part")
    try:
        buffer = io.StringIO()
        frame.to_csv(buffer, index=False)
        payload = gzip.compress(buffer.getvalue().encode("utf-8"))
        temp_path.write_bytes(payload)
        temp_path.replace(data_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _gridstatusio_library_version() -> str:
    try:
        return importlib.metadata.version("gridstatusio")
    except importlib.metadata.PackageNotFoundError:
        return "injected-test-double"


def collect_ercot_prices(
    start: date,
    end: date,
    *,
    markets: tuple[str, ...] = DEFAULT_MARKETS,
    location: str = "HB_NORTH",
    location_type: str = "Trading Hub",
    chunk_days: int = 31,
    raw_root: Path = DEFAULT_RAW_ROOT,
    api_key: str | None = None,
    client: Any | None = None,
    skip_existing: bool = True,
) -> list[dict[str, Any]]:
    """Download ERCOT SPP data from GridStatus.io and save raw CSV files."""
    validate_date_range(start, end)
    if client is None:
        client = _load_gridstatusio_client(api_key)

    results: list[dict[str, Any]] = []
    for market_name in markets:
        api_dataset = _dataset_for_market(market_name)
        for chunk_start, chunk_end_exclusive in iter_date_chunks(
            start, end, chunk_days
        ):
            dataset = f"ercot_spp_{market_name.lower()}"
            directory = partitioned_directory(
                raw_root, "gridstatusio", dataset, chunk_start
            )
            file_prefix = (
                f"{dataset}_{chunk_start.isoformat()}_"
                f"{chunk_end_exclusive.isoformat()}_"
            )
            if skip_existing:
                completed = find_completed_file(
                    directory, file_prefix=file_prefix, extension=".csv.gz"
                )
                if completed:
                    data_path, metadata = completed
                    results.append(
                        {
                            "source": "gridstatusio",
                            "dataset": dataset,
                            "data_path": str(data_path),
                            "metadata_path": str(data_path) + ".metadata.json",
                            "row_count": metadata.get("row_count"),
                            "status": "skipped_existing",
                        }
                    )
                    continue

            collected_at = utc_now()
            chunk_length_days = (chunk_end_exclusive - chunk_start).days
            row_limit = max(200, chunk_length_days * 101)
            try:
                frame = client.get_dataset(
                    dataset=api_dataset,
                    start=chunk_start.isoformat(),
                    end=chunk_end_exclusive.isoformat(),
                    filter_column="location",
                    filter_value=location,
                    timezone="US/Central",
                    limit=row_limit,
                    verbose=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"GridStatus.io request failed for {api_dataset}, "
                    f"{chunk_start} through {chunk_end_exclusive}: {exc}"
                ) from exc
            if not isinstance(frame, pd.DataFrame):
                raise RuntimeError(
                    f"GridStatus.io returned {type(frame).__name__}, "
                    "expected DataFrame"
                )

            location_column = _find_column(frame, "location")
            if location_column is None:
                raise RuntimeError(
                    "GridStatus.io response has no location column: "
                    f"{list(frame.columns)}"
                )
            matching_rows = int(
                frame[location_column]
                .astype(str)
                .str.upper()
                .eq(location.upper())
                .sum()
            )
            if frame.empty:
                raise RuntimeError(
                    f"GridStatus.io returned no {location} rows for {api_dataset}, "
                    f"{chunk_start} through {chunk_end_exclusive}"
                )
            if matching_rows != len(frame):
                raise RuntimeError(
                    f"GridStatus.io returned {len(frame) - matching_rows} rows "
                    f"outside requested location {location!r}"
                )
            if len(frame) >= row_limit:
                raise RuntimeError(
                    f"GridStatus.io returned {len(frame)} rows, reaching the safety "
                    f"limit of {row_limit}. Reduce --chunk-days to prevent truncation."
                )

            file_name = f"{file_prefix}{filename_timestamp(collected_at)}.csv.gz"
            data_path = directory / file_name
            _write_gridstatusio_frame(frame, data_path=data_path)

            metadata_path = write_metadata(
                data_path,
                {
                    "source": "gridstatusio",
                    "dataset": dataset,
                    "api_dataset": api_dataset,
                    "library_version": _gridstatusio_library_version(),
                    "request": {
                        "start": chunk_start.isoformat(),
                        "end_exclusive": chunk_end_exclusive.isoformat(),
                        "filter_column": "location",
                        "filter_value": location,
                        "timezone": "US/Central",
                        "limit": row_limit,
                    },
                    "market": market_name,
                    "requested_location": location,
                    "requested_location_type": location_type,
                    "matching_location_rows": matching_rows,
                    "collected_at_utc": iso_utc(collected_at),
                    "row_count": len(frame),
                    "columns": [str(column) for column in frame.columns],
                    "status": "downloaded",
                },
            )
            results.append(
                {
                    "source": "gridstatusio",
                    "dataset": dataset,
                    "data_path": str(data_path),
                    "metadata_path": str(metadata_path),
                    "row_count": len(frame),
                    "status": "downloaded",
                }
            )
    return results


def collect_ercot_forecasts(
    start: date,
    end: date,
    *,
    forecasts: tuple[str, ...] = DEFAULT_FORECASTS,
    chunk_days: int = 7,
    raw_root: Path = DEFAULT_RAW_ROOT,
    api_key: str | None = None,
    client: Any | None = None,
    skip_existing: bool = True,
) -> list[dict[str, Any]]:
    """Download ERCOT forecast datasets from GridStatus.io and save raw CSV files."""
    validate_date_range(start, end)
    if client is None:
        client = _load_gridstatusio_client(api_key)

    results: list[dict[str, Any]] = []
    for forecast_name in forecasts:
        config = _forecast_config(forecast_name)
        api_dataset = str(config["api_dataset"])
        dataset = str(config["dataset"])
        rows_per_day_limit = int(config["rows_per_day_limit"])

        for chunk_start, chunk_end_exclusive in iter_date_chunks(
            start, end, chunk_days
        ):
            directory = partitioned_directory(
                raw_root, "gridstatusio", dataset, chunk_start
            )
            file_prefix = (
                f"{dataset}_{chunk_start.isoformat()}_"
                f"{chunk_end_exclusive.isoformat()}_"
            )
            if skip_existing:
                completed = find_completed_file(
                    directory, file_prefix=file_prefix, extension=".csv.gz"
                )
                if completed:
                    data_path, metadata = completed
                    results.append(
                        {
                            "source": "gridstatusio",
                            "dataset": dataset,
                            "data_path": str(data_path),
                            "metadata_path": str(data_path) + ".metadata.json",
                            "row_count": metadata.get("row_count"),
                            "status": "skipped_existing",
                        }
                    )
                    continue

            collected_at = utc_now()
            chunk_length_days = (chunk_end_exclusive - chunk_start).days
            row_limit = max(1000, chunk_length_days * rows_per_day_limit)
            try:
                frame = client.get_dataset(
                    dataset=api_dataset,
                    start=chunk_start.isoformat(),
                    end=chunk_end_exclusive.isoformat(),
                    timezone="US/Central",
                    limit=row_limit,
                    filter_value="",
                    verbose=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"GridStatus.io request failed for {api_dataset}, "
                    f"{chunk_start} through {chunk_end_exclusive}: {exc}"
                ) from exc
            if not isinstance(frame, pd.DataFrame):
                raise RuntimeError(
                    f"GridStatus.io returned {type(frame).__name__}, "
                    "expected DataFrame"
                )
            if frame.empty:
                raise RuntimeError(
                    f"GridStatus.io returned no rows for {api_dataset}, "
                    f"{chunk_start} through {chunk_end_exclusive}"
                )
            if len(frame) >= row_limit:
                raise RuntimeError(
                    f"GridStatus.io returned {len(frame)} rows, reaching the safety "
                    f"limit of {row_limit}. Reduce --chunk-days to prevent truncation."
                )

            file_name = f"{file_prefix}{filename_timestamp(collected_at)}.csv.gz"
            data_path = directory / file_name
            _write_gridstatusio_frame(frame, data_path=data_path)

            metadata_path = write_metadata(
                data_path,
                {
                    "source": "gridstatusio",
                    "dataset": dataset,
                    "api_dataset": api_dataset,
                    "library_version": _gridstatusio_library_version(),
                    "request": {
                        "start": chunk_start.isoformat(),
                        "end_exclusive": chunk_end_exclusive.isoformat(),
                        "timezone": "US/Central",
                        "limit": row_limit,
                    },
                    "forecast": forecast_name,
                    "collected_at_utc": iso_utc(collected_at),
                    "row_count": len(frame),
                    "columns": [str(column) for column in frame.columns],
                    "status": "downloaded",
                },
            )
            results.append(
                {
                    "source": "gridstatusio",
                    "dataset": dataset,
                    "data_path": str(data_path),
                    "metadata_path": str(metadata_path),
                    "row_count": len(frame),
                    "status": "downloaded",
                }
            )
    return results


def _local_asof_datetime(
    delivery_date: date,
    *,
    days_before: int,
    hour: int,
) -> datetime:
    if days_before < 0:
        raise ValueError("days_before must be non-negative")
    if hour < 0 or hour > 23:
        raise ValueError("hour must be between 0 and 23")
    asof_date = delivery_date - timedelta(days=days_before)
    return datetime.combine(
        asof_date,
        time(hour=hour),
        tzinfo=ERCOT_LOCAL_TIMEZONE,
    )


def _asof_dataset_name(dataset: str, *, days_before: int, hour: int) -> str:
    short_names = {
        "ercot_seven_day_load_forecast": "ercot_load_da",
        "ercot_wind_production_forecast": "ercot_wind_da",
        "ercot_solar_production_forecast": "ercot_solar_da",
    }
    return short_names.get(dataset, f"{dataset}_da")


def _latest_before_asof(frame: pd.DataFrame, asof_time: datetime) -> pd.DataFrame:
    interval_column = _find_column(frame, "interval_start_utc")
    publish_column = _find_column(frame, "publish_time_utc")
    if interval_column is None or publish_column is None:
        raise RuntimeError(
            "GridStatus.io forecast response must include interval_start_utc "
            f"and publish_time_utc columns: {list(frame.columns)}"
        )

    selected = frame.copy()
    selected[publish_column] = pd.to_datetime(
        selected[publish_column],
        utc=True,
        format="ISO8601",
    )
    asof_utc = pd.Timestamp(asof_time).tz_convert("UTC")
    selected = selected[selected[publish_column] <= asof_utc]
    if selected.empty:
        return selected

    selected = selected.sort_values([interval_column, publish_column])
    selected = selected.drop_duplicates(subset=[interval_column], keep="last")
    return selected.reset_index(drop=True)


def collect_ercot_asof_forecasts(
    start: date,
    end: date,
    *,
    forecasts: tuple[str, ...] = DEFAULT_FORECASTS,
    asof_days_before: int = 1,
    asof_hour_local: int = 10,
    raw_root: Path = DEFAULT_RAW_ROOT,
    api_key: str | None = None,
    client: Any | None = None,
    skip_existing: bool = True,
) -> list[dict[str, Any]]:
    """Download one as-of forecast vintage per ERCOT delivery day.

    GridStatus.io does not allow publish_time="latest" together with
    publish_time_end. This collector therefore requests forecast rows published
    before the local as-of timestamp, then keeps the latest publish_time_utc per
    interval_start_utc locally.
    """
    validate_date_range(start, end)
    if client is None:
        client = _load_gridstatusio_client(api_key)

    results: list[dict[str, Any]] = []
    current = start
    while current <= end:
        delivery_end = current + timedelta(days=1)
        asof_time = _local_asof_datetime(
            current,
            days_before=asof_days_before,
            hour=asof_hour_local,
        )
        for forecast_name in forecasts:
            config = _forecast_config(forecast_name)
            api_dataset = str(config["api_dataset"])
            base_dataset = str(config["dataset"])
            rows_per_day_limit = int(config["rows_per_day_limit"])
            dataset = _asof_dataset_name(
                base_dataset,
                days_before=asof_days_before,
                hour=asof_hour_local,
            )
            row_limit = max(1000, rows_per_day_limit)
            directory = partitioned_directory(
                raw_root, "gridstatusio", dataset, current
            )
            file_prefix = f"{dataset}_{current.isoformat()}_{delivery_end.isoformat()}_"
            if skip_existing:
                completed = find_completed_file(
                    directory, file_prefix=file_prefix, extension=".csv.gz"
                )
                if completed:
                    data_path, metadata = completed
                    results.append(
                        {
                            "source": "gridstatusio",
                            "dataset": dataset,
                            "data_path": str(data_path),
                            "metadata_path": str(data_path) + ".metadata.json",
                            "row_count": metadata.get("row_count"),
                            "status": "skipped_existing",
                        }
                    )
                    continue

            collected_at = utc_now()
            try:
                frame = client.get_dataset(
                    dataset=api_dataset,
                    start=current.isoformat(),
                    end=delivery_end.isoformat(),
                    publish_time_end=asof_time.isoformat(),
                    timezone="US/Central",
                    limit=row_limit,
                    filter_value="",
                    verbose=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"GridStatus.io request failed for {api_dataset}, "
                    f"delivery date {current}, as of {asof_time.isoformat()}: {exc}"
                ) from exc
            if not isinstance(frame, pd.DataFrame):
                raise RuntimeError(
                    f"GridStatus.io returned {type(frame).__name__}, "
                    "expected DataFrame"
                )
            if frame.empty:
                raise RuntimeError(
                    f"GridStatus.io returned no rows for {api_dataset}, "
                    f"delivery date {current}, as of {asof_time.isoformat()}"
                )
            original_row_count = len(frame)
            frame = _latest_before_asof(frame, asof_time)
            if frame.empty:
                raise RuntimeError(
                    f"GridStatus.io returned no rows with publish_time_utc on or "
                    f"before {asof_time.isoformat()} for {api_dataset}, "
                    f"delivery date {current}"
                )
            if len(frame) >= row_limit:
                raise RuntimeError(
                    f"GridStatus.io returned {len(frame)} rows, reaching the safety "
                    f"limit of {row_limit}."
                )

            file_name = f"{file_prefix}{filename_timestamp(collected_at)}.csv.gz"
            data_path = directory / file_name
            _write_gridstatusio_frame(frame, data_path=data_path)

            metadata_path = write_metadata(
                data_path,
                {
                    "source": "gridstatusio",
                    "dataset": dataset,
                    "base_dataset": base_dataset,
                    "api_dataset": api_dataset,
                    "library_version": _gridstatusio_library_version(),
                    "request": {
                        "start": current.isoformat(),
                        "end_exclusive": delivery_end.isoformat(),
                        "publish_time_end": asof_time.isoformat(),
                        "timezone": "US/Central",
                        "limit": row_limit,
                    },
                    "forecast": forecast_name,
                    "asof_selection": "local_latest_publish_time_per_interval",
                    "candidate_row_count": original_row_count,
                    "asof_rule": {
                        "delivery_date": current.isoformat(),
                        "days_before": asof_days_before,
                        "hour_local": asof_hour_local,
                        "timezone": "America/Chicago",
                    },
                    "collected_at_utc": iso_utc(collected_at),
                    "row_count": len(frame),
                    "columns": [str(column) for column in frame.columns],
                    "status": "downloaded",
                },
            )
            results.append(
                {
                    "source": "gridstatusio",
                    "dataset": dataset,
                    "data_path": str(data_path),
                    "metadata_path": str(metadata_path),
                    "row_count": len(frame),
                    "status": "downloaded",
                }
            )
        current = delivery_end
    return results
