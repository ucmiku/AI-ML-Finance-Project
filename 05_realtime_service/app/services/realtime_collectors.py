from __future__ import annotations

import asyncio
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from app.services.realtime_store import (
    insert_collection_run,
    replace_price_actuals,
    replace_ercot_rows,
    replace_gas_rows,
    replace_weather_rows,
)


ERCOT_TZ = ZoneInfo("America/Chicago")
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
HOURLY_WEATHER = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "shortwave_radiation",
    "precipitation",
)
WEATHER_LOCATIONS = (
    ("Dallas", 32.7767, -96.7970),
    ("Fort_Worth", 32.7555, -97.3308),
    ("Denton", 33.2148, -97.1331),
    ("McKinney", 33.1972, -96.6397),
    ("Arlington", 32.7357, -97.1081),
    ("Wichita_Falls", 33.9137, -98.4934),
)
ERCOT_FORECAST_DATASETS = {
    "load_forecast": ("ercot_load_forecast", 5000),
    "load_forecast_by_weather_zone": (
        "ercot_load_forecast_by_weather_zone",
        5000,
    ),
    "wind_forecast": ("ercot_wind_actual_and_forecast_hourly", 10000),
    "solar_forecast": ("ercot_solar_actual_and_forecast_hourly", 10000),
}
PRICE_DATASETS = {
    "day_ahead": "ercot_spp_day_ahead_hourly",
    "real_time": "ercot_spp_real_time_15_min",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def tomorrow_ercot_date(now: datetime | None = None) -> date:
    now = now or utc_now()
    return now.astimezone(ERCOT_TZ).date() + timedelta(days=1)


def _local_date_from_utc(value: str) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/Chicago").date().isoformat()


def _timestamp_to_utc_text(value: Any) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return _timestamp_to_utc_text(value)
    if isinstance(value, datetime):
        return iso_utc(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _find_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    normalized = {
        str(col).strip().lower().replace("_", " "): str(col) for col in frame.columns
    }
    for name in names:
        match = normalized.get(name.strip().lower().replace("_", " "))
        if match:
            return match
    return None


def _find_price_column(frame: pd.DataFrame) -> str:
    column = _find_column(
        frame,
        (
            "spp",
            "settlement_point_price",
            "settlement point price",
            "price",
            "lmp",
        ),
    )
    if column is None:
        numeric_columns = [
            str(col)
            for col in frame.select_dtypes(include=["number"]).columns
            if str(col).lower() not in {"interval", "hour"}
        ]
        if len(numeric_columns) == 1:
            return numeric_columns[0]
        raise RuntimeError(f"Could not identify price column: {list(frame.columns)}")
    return column


async def _fetch_weather_location(
    client: httpx.AsyncClient,
    location_name: str,
    latitude: float,
    longitude: float,
    delivery_date: date,
    collected_at_utc: str,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        # ERCOT local delivery day spans two UTC dates during CDT/CST.
        "start_date": delivery_date.isoformat(),
        "end_date": (delivery_date + timedelta(days=1)).isoformat(),
        "hourly": ",".join(HOURLY_WEATHER),
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }
    api_key = os.getenv("OPENMETEO_API_KEY")
    if api_key:
        params["apikey"] = api_key

    response = await client.get(OPENMETEO_URL, params=params)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    rows: list[dict[str, Any]] = []
    for index, hour_text in enumerate(times):
        delivery_hour_utc = (
            pd.Timestamp(hour_text)
            .tz_localize("UTC")
            .isoformat()
            .replace("+00:00", "Z")
        )
        row = {
            "delivery_date_local": _local_date_from_utc(delivery_hour_utc),
            "delivery_hour_utc": delivery_hour_utc,
            "location": location_name,
            "collected_at_utc": collected_at_utc,
            "raw": {"location": location_name},
        }
        for variable in HOURLY_WEATHER:
            values = hourly.get(variable, [])
            row[variable] = values[index] if index < len(values) else None
            row["raw"][variable] = row[variable]
        rows.append(row)
    return [row for row in rows if row["delivery_date_local"] == delivery_date.isoformat()]


async def collect_weather(delivery_date: date, collected_at_utc: str) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=60) as client:
        tasks = [
            _fetch_weather_location(
                client, name, lat, lon, delivery_date, collected_at_utc
            )
            for name, lat, lon in WEATHER_LOCATIONS
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    rows: list[dict[str, Any]] = []
    for result in results:
        if isinstance(result, Exception):
            raise result
        rows.extend(result)
    replace_weather_rows(rows)
    return rows


async def _fetch_gas(collected_at_utc: str) -> list[dict[str, Any]]:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY is not set.")
    end = utc_now().date()
    start = end - timedelta(days=14)
    params = {
        "series_id": "DHHNGSP",
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start.isoformat(),
        "observation_end": end.isoformat(),
        "sort_order": "asc",
    }
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(FRED_URL, params=params)
    response.raise_for_status()
    observations = response.json().get("observations", [])
    rows: list[dict[str, Any]] = []
    for obs in observations:
        value_text = obs.get("value")
        try:
            value = float(value_text)
        except (TypeError, ValueError):
            value = None
        rows.append(
            {
                "series_id": "DHHNGSP",
                "observation_date": obs.get("date"),
                "collected_at_utc": collected_at_utc,
                "value": value,
                "raw": obs,
            }
        )
    replace_gas_rows(rows)
    return rows


def _collect_ercot_sync(delivery_date: date, collected_at_utc: str) -> list[dict[str, Any]]:
    api_key = os.getenv("GRIDSTATUS_API_KEY")
    if not api_key:
        raise RuntimeError("GRIDSTATUS_API_KEY is not set.")
    from gridstatusio import GridStatusClient

    client = GridStatusClient(api_key=api_key, return_format="pandas")
    rows: list[dict[str, Any]] = []
    # Include the previous local delivery day so model ramp features have a
    # six-hour warmup window before the target day starts.
    start = (delivery_date - timedelta(days=1)).isoformat()
    end = (delivery_date + timedelta(days=1)).isoformat()

    for label, (dataset, limit) in ERCOT_FORECAST_DATASETS.items():
        request_kwargs: dict[str, Any] = {}
        if label == "load_forecast_by_weather_zone":
            request_kwargs["publish_time"] = "latest_report"
        frame = client.get_dataset(
            dataset=dataset,
            start=start,
            end=end,
            timezone="US/Central",
            limit=limit,
            filter_value="",
            verbose=False,
            **request_kwargs,
        )
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue

        interval_col = _find_column(
            frame,
            (
                "interval_start_utc",
                "interval start utc",
                "interval_start",
                "interval start",
            ),
        )
        publish_col = _find_column(
            frame,
            (
                "publish_time_utc",
                "publish time utc",
                "publish_time",
                "publish time",
            ),
        )
        if interval_col is None:
            continue
        if publish_col is not None:
            frame = frame.sort_values([interval_col, publish_col])
            frame = frame.drop_duplicates(subset=[interval_col], keep="last")
        else:
            frame = frame.drop_duplicates(subset=[interval_col], keep="last")

        for record in frame.to_dict(orient="records"):
            delivery_hour_utc = _timestamp_to_utc_text(record[interval_col])
            rows.append(
                {
                    "dataset": label,
                    "delivery_date_local": _local_date_from_utc(delivery_hour_utc),
                    "delivery_hour_utc": delivery_hour_utc,
                    "collected_at_utc": collected_at_utc,
                    "publish_time_utc": (
                        _timestamp_to_utc_text(record[publish_col])
                        if publish_col is not None and pd.notna(record[publish_col])
                        else None
                    ),
                    "raw": {
                        key: _jsonable(value)
                        for key, value in record.items()
                    },
                }
            )
    replace_ercot_rows(rows)
    return rows


async def collect_ercot(delivery_date: date, collected_at_utc: str) -> list[dict[str, Any]]:
    return await asyncio.to_thread(_collect_ercot_sync, delivery_date, collected_at_utc)


def collect_price_actuals(
    start: date,
    end: date,
    *,
    location: str = "HB_NORTH",
    collected_at_utc: str | None = None,
) -> list[dict[str, Any]]:
    api_key = os.getenv("GRIDSTATUS_API_KEY")
    if not api_key:
        raise RuntimeError("GRIDSTATUS_API_KEY is not set.")
    from gridstatusio import GridStatusClient

    collected_at_utc = collected_at_utc or iso_utc(utc_now())
    client = GridStatusClient(api_key=api_key, return_format="pandas")
    frames: dict[str, pd.DataFrame] = {}
    for name, dataset in PRICE_DATASETS.items():
        limit = max(1000, (end - start).days * (120 if name == "real_time" else 30))
        frame = client.get_dataset(
            dataset=dataset,
            start=start.isoformat(),
            end=end.isoformat(),
            filter_column="location",
            filter_value=location,
            timezone="US/Central",
            limit=limit,
            verbose=False,
        )
        if not isinstance(frame, pd.DataFrame):
            raise RuntimeError(f"GridStatus.io returned {type(frame).__name__}")
        if frame.empty:
            frames[name] = frame
            continue
        interval_col = _find_column(
            frame,
            (
                "interval_start_utc",
                "interval start utc",
                "interval_start",
                "interval start",
            ),
        )
        if interval_col is None:
            raise RuntimeError(f"Could not identify interval column: {list(frame.columns)}")
        price_col = _find_price_column(frame)
        normalized = frame[[interval_col, price_col]].copy()
        normalized["delivery_hour_utc"] = normalized[interval_col].map(
            lambda value: pd.Timestamp(value)
            .tz_convert("UTC")
            .floor("h")
            .isoformat()
            .replace("+00:00", "Z")
        )
        normalized["price"] = pd.to_numeric(normalized[price_col], errors="coerce")
        frames[name] = normalized[["delivery_hour_utc", "price"]]

    da = frames.get("day_ahead", pd.DataFrame(columns=["delivery_hour_utc", "price"]))
    rt_raw = frames.get("real_time", pd.DataFrame(columns=["delivery_hour_utc", "price"]))
    da_hourly = da.groupby("delivery_hour_utc", as_index=False)["price"].mean()
    rt_hourly = (
        rt_raw.groupby("delivery_hour_utc", as_index=False)
        .agg(rt_price=("price", "mean"), rt_interval_count=("price", "count"))
    )
    merged = da_hourly.rename(columns={"price": "da_price"}).merge(
        rt_hourly, on="delivery_hour_utc", how="outer"
    )
    rows: list[dict[str, Any]] = []
    for record in merged.to_dict(orient="records"):
        hour = str(record["delivery_hour_utc"])
        da_price = _jsonable(record.get("da_price"))
        rt_price = _jsonable(record.get("rt_price"))
        spread = (
            float(rt_price) - float(da_price)
            if da_price is not None and rt_price is not None
            else None
        )
        row = {
            "delivery_date_local": _local_date_from_utc(hour),
            "delivery_hour_utc": hour,
            "location": location,
            "collected_at_utc": collected_at_utc,
            "da_price_usd_per_mwh": da_price,
            "rt_price_usd_per_mwh": rt_price,
            "spread_usd_per_mwh": spread,
            "rt_interval_count": int(record.get("rt_interval_count") or 0),
        }
        row["raw"] = row.copy()
        rows.append(row)
    replace_price_actuals(rows)
    return rows


async def collect_realtime_snapshot() -> dict[str, Any]:
    delivery_date = tomorrow_ercot_date()
    collected_at = iso_utc(utc_now())
    status = "success"
    messages: list[str] = []
    counts: dict[str, int] = {}

    jobs = {
        "weather": collect_weather(delivery_date, collected_at),
        "gas": _fetch_gas(collected_at),
        "ercot": collect_ercot(delivery_date, collected_at),
    }
    results = await asyncio.gather(*jobs.values(), return_exceptions=True)
    for name, result in zip(jobs.keys(), results):
        if isinstance(result, Exception):
            counts[name] = 0
            status = "partial_failure"
            messages.append(f"{name}: {result}")
        else:
            counts[name] = len(result)

    insert_collection_run(
        collected_at_utc=collected_at,
        delivery_date_local=delivery_date.isoformat(),
        status=status,
        message="; ".join(messages) if messages else None,
    )
    return {
        "delivery_date_local": delivery_date.isoformat(),
        "collected_at_utc": collected_at,
        "status": status,
        "counts": counts,
        "messages": messages,
    }
