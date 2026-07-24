from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from app.core.config import DEFAULT_SQLITE_PATH, get_database_uri
from app.services.realtime_query_service import (
    get_realtime_as_market_forecast,
    get_realtime_as_market_status,
)


TABLE_NAME = "model_wide_hourly_2024_2026"

SELECT_COLUMNS = [
    "delivery_hour_utc",
    "delivery_date_local",
    "delivery_time_local",
    "ercot_local_hour",
    "is_dst",
    "decision_time_utc",
    "gas_price_usd_per_mmbtu",
    "temperature_dfw_mean_c",
    "wind_speed_dfw_mean_ms",
    "cloud_cover_dfw_mean_pct",
    "load_system_total_mw",
    "wind_stwpf_system_wide_mw",
    "solar_pvgrpp_system_mw",
    "spread_usd_per_mwh",
    "rt_above_da",
    "split_name",
]


def _sqlite_path_from_uri(db_uri: str) -> Path:
    if not db_uri.startswith("sqlite:///"):
        raise ValueError("Only sqlite:/// database URIs are supported by this service.")
    return Path(db_uri.removeprefix("sqlite:///"))


def _connect() -> sqlite3.Connection:
    db_uri = get_database_uri()
    path = _sqlite_path_from_uri(db_uri)
    if not path.exists():
        path = DEFAULT_SQLITE_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def get_day_ahead_forecast(delivery_date: str) -> dict[str, Any]:
    realtime = get_realtime_as_market_forecast(delivery_date)
    if realtime is not None:
        if not any(row.get("predicted_spread") is not None for row in realtime["hours"]):
            from app.services.model_service import run_prediction

            prediction = run_prediction(delivery_date)
            if prediction.get("row_count", 0) > 0:
                refreshed = get_realtime_as_market_forecast(delivery_date)
                if refreshed is not None:
                    return refreshed
        return realtime

    query = f"""
        SELECT {", ".join(SELECT_COLUMNS)}
        FROM {TABLE_NAME}
        WHERE delivery_date_local = ?
        ORDER BY delivery_hour_utc
    """
    with _connect() as conn:
        rows = [dict(row) for row in conn.execute(query, (delivery_date,)).fetchall()]
    for row in rows:
        row["hour"] = row.get("ercot_local_hour")
    if rows:
        from app.services.model_service import run_offline_prediction

        prediction = run_offline_prediction(delivery_date)
        prediction_rows = {
            row["delivery_hour_utc"]: row
            for row in prediction.get("predictions", [])
        }
        for row in rows:
            pred = prediction_rows.get(row["delivery_hour_utc"])
            if pred:
                row.update(
                    {
                        "predicted_spread": pred.get("predicted_spread"),
                        "confidence": pred.get("confidence"),
                        "prediction_signal": pred.get("signal"),
                        "prediction_confidence": pred.get("confidence"),
                        "feature_missing_count": pred.get("feature_missing_count"),
                        "p_negative": pred.get("p_negative"),
                        "p_neutral": pred.get("p_neutral"),
                        "p_positive": pred.get("p_positive"),
                        "model_name": pred.get("model_name"),
                        "model_version": pred.get("model_version"),
                        "predicted_at_utc": pred.get("predicted_at_utc"),
                    },
                )

    return {
        "delivery_date": delivery_date,
        "table": TABLE_NAME,
        "row_count": len(rows),
        "hours": rows,
    }


def get_data_status(delivery_date: str) -> dict[str, Any]:
    realtime = get_realtime_as_market_status(delivery_date)
    if realtime is not None:
        return realtime

    query = f"""
        SELECT
            COUNT(*) AS row_count,
            MIN(delivery_hour_utc) AS first_delivery_hour_utc,
            MAX(delivery_hour_utc) AS last_delivery_hour_utc,
            MIN(has_all_three_forecasts) AS has_all_three_forecasts,
            MIN(load_pre_dam_valid) AS load_pre_dam_valid,
            MIN(wind_pre_dam_valid) AS wind_pre_dam_valid,
            MIN(solar_pre_dam_valid) AS solar_pre_dam_valid,
            MIN(all_issue_times_pre_dam_valid) AS all_issue_times_pre_dam_valid
        FROM {TABLE_NAME}
        WHERE delivery_date_local = ?
    """
    with _connect() as conn:
        row = conn.execute(query, (delivery_date,)).fetchone()

    row_count = int(row["row_count"] or 0)
    expected_hours = 24
    return {
        "delivery_date": delivery_date,
        "table": TABLE_NAME,
        "row_count": row_count,
        "expected_hours": expected_hours,
        "missing_hours": max(expected_hours - row_count, 0),
        "complete_day": row_count == expected_hours,
        "first_delivery_hour_utc": row["first_delivery_hour_utc"],
        "last_delivery_hour_utc": row["last_delivery_hour_utc"],
        "has_all_three_forecasts": bool(row["has_all_three_forecasts"] or 0),
        "load_pre_dam_valid": bool(row["load_pre_dam_valid"] or 0),
        "wind_pre_dam_valid": bool(row["wind_pre_dam_valid"] or 0),
        "solar_pre_dam_valid": bool(row["solar_pre_dam_valid"] or 0),
        "all_issue_times_pre_dam_valid": bool(
            row["all_issue_times_pre_dam_valid"] or 0
        ),
    }
