from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from app.services.realtime_store import connect, init_realtime_tables


ERCOT_TZ = ZoneInfo("America/Chicago")
DFW_LOCATIONS = {"Dallas", "Fort_Worth", "Denton", "McKinney", "Arlington"}
WEATHER_ZONE_LOAD_MAPPING = {
    "load_coast_mw": "coast",
    "load_east_mw": "east",
    "load_far_west_mw": "far_west",
    "load_north_mw": "north",
    "load_north_central_mw": "north_central",
    "load_south_central_mw": "south_central",
    "load_southern_mw": "southern",
    "load_west_mw": "west",
    "load_system_total_mw": "system_total",
}


def _row_to_dict(row: Any) -> dict[str, Any]:
    data = dict(row)
    if "raw_json" in data:
        try:
            data["raw"] = json.loads(data.pop("raw_json"))
        except json.JSONDecodeError:
            data["raw"] = data.pop("raw_json")
    return data


def _local_time_parts(delivery_hour_utc: str) -> tuple[str, int, int]:
    dt_utc = datetime.fromisoformat(delivery_hour_utc.replace("Z", "+00:00"))
    dt_local = dt_utc.astimezone(ERCOT_TZ)
    return dt_local.isoformat(), dt_local.hour, int(dt_local.dst().total_seconds() > 0)


def _floor_utc_hour(value: str) -> str:
    dt_utc = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt_utc.replace(minute=0, second=0, microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _latest_gas_observation(conn: Any) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT *
        FROM online_gas_observations
        WHERE value IS NOT NULL
        ORDER BY observation_date DESC
        LIMIT 1
        """
    ).fetchone()
    return _row_to_dict(row) if row else None


def _weather_hourly_features(conn: Any, delivery_date: str) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT *
        FROM online_weather_forecast_hourly
        WHERE delivery_date_local = ?
        """,
        (delivery_date,),
    ).fetchall()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        data = dict(row)
        grouped.setdefault(data["delivery_hour_utc"], []).append(data)

    features: dict[str, dict[str, Any]] = {}
    for hour, hour_rows in grouped.items():
        dfw = [row for row in hour_rows if row["location"] in DFW_LOCATIONS]

        def mean_value(items: list[dict[str, Any]], column: str) -> float | None:
            values = [item[column] for item in items if item.get(column) is not None]
            if not values:
                return None
            return sum(values) / len(values)

        features[hour] = {
            "temperature_dfw_mean_c": mean_value(dfw, "temperature_2m"),
            "wind_speed_dfw_mean_ms": mean_value(dfw, "wind_speed_10m"),
            "cloud_cover_dfw_mean_pct": mean_value(dfw, "cloud_cover"),
        }
    return features


def _ercot_hourly_features(conn: Any, delivery_date: str) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT dataset, delivery_hour_utc, raw_json
        FROM online_ercot_forecast_rows
        WHERE delivery_date_local = ?
        """,
        (delivery_date,),
    ).fetchall()
    features: dict[str, dict[str, Any]] = {}
    load_values: dict[str, list[float]] = {}

    for row in rows:
        data = dict(row)
        raw = json.loads(data["raw_json"])
        hour = _floor_utc_hour(data["delivery_hour_utc"])
        features.setdefault(hour, {})

        if data["dataset"] == "load_forecast":
            value = raw.get("load_forecast")
            if value is not None:
                load_values.setdefault(hour, []).append(float(value))
        elif data["dataset"] == "load_forecast_by_weather_zone":
            for target, source in WEATHER_ZONE_LOAD_MAPPING.items():
                value = raw.get(source)
                if value is not None:
                    features[hour][target] = float(value)
        elif data["dataset"] == "wind_forecast":
            value = raw.get("stwpf_system_wide")
            if value is not None:
                features[hour]["wind_stwpf_system_wide_mw"] = float(value)
        elif data["dataset"] == "solar_forecast":
            value = raw.get("pvgrpp_system_wide")
            if value is not None:
                features[hour]["solar_pvgrpp_system_mw"] = float(value)

    for hour, values in load_values.items():
        if values and features.setdefault(hour, {}).get("load_system_total_mw") is None:
            features.setdefault(hour, {})["load_system_total_mw"] = sum(values) / len(values)
    return features


def get_realtime_as_market_forecast(delivery_date: str) -> dict[str, Any] | None:
    init_realtime_tables()
    with connect() as conn:
        weather = _weather_hourly_features(conn, delivery_date)
        ercot = _ercot_hourly_features(conn, delivery_date)
        gas = _latest_gas_observation(conn)
        latest_collected_at_utc = conn.execute(
            """
            SELECT MAX(collected_at_utc)
            FROM (
                SELECT collected_at_utc FROM online_weather_forecast_hourly
                WHERE delivery_date_local = ?
                UNION ALL
                SELECT collected_at_utc FROM online_ercot_forecast_rows
                WHERE delivery_date_local = ?
                UNION ALL
                SELECT collected_at_utc FROM online_gas_observations
            )
            """,
            (delivery_date, delivery_date),
        ).fetchone()[0]
        prediction_rows = {
            row["delivery_hour_utc"]: dict(row)
            for row in conn.execute(
                """
                SELECT delivery_hour_utc, predicted_at_utc, model_name,
                       model_version, model_fold, predicted_spread,
                       p_negative, p_neutral, p_positive, predicted_class,
                       confidence, signal, numeric_signal, feature_missing_count
                FROM online_model_predictions
                WHERE delivery_date_local = ?
                ORDER BY delivery_hour_utc
                """,
                (delivery_date,),
            ).fetchall()
        }
        zscore_rows = {
            row["delivery_hour_utc"]: dict(row)
            for row in conn.execute(
                """
                SELECT delivery_hour_utc, gas_price_z30,
                       load_system_z30_same_hour, net_load_z30_same_hour
                FROM online_zscore_features
                WHERE delivery_date_local = ?
                  AND location = 'HB_NORTH'
                ORDER BY delivery_hour_utc
                """,
                (delivery_date,),
            ).fetchall()
        }

    hours = sorted(set(weather) | set(ercot))
    if len(hours) < 24:
        return None

    rows = []
    for hour in hours:
        delivery_time_local, local_hour, is_dst = _local_time_parts(hour)
        row = {
            "delivery_hour_utc": hour,
            "delivery_date_local": delivery_date,
            "delivery_time_local": delivery_time_local,
            "ercot_local_hour": local_hour,
            "hour": local_hour,
            "is_dst": is_dst,
            "decision_time_utc": latest_collected_at_utc or "",
            "gas_price_usd_per_mmbtu": gas["value"] if gas else None,
            "temperature_dfw_mean_c": None,
            "wind_speed_dfw_mean_ms": None,
            "cloud_cover_dfw_mean_pct": None,
            "load_system_total_mw": None,
            "wind_stwpf_system_wide_mw": None,
            "solar_pvgrpp_system_mw": None,
            "spread_usd_per_mwh": None,
            "rt_above_da": None,
            "split_name": "realtime",
        }
        row.update(weather.get(hour, {}))
        row.update(ercot.get(hour, {}))
        zscore = zscore_rows.get(hour)
        if zscore:
            row.update(
                {
                    "gas_price_z30": zscore["gas_price_z30"],
                    "load_system_z30_same_hour": zscore[
                        "load_system_z30_same_hour"
                    ],
                    "net_load_z30_same_hour": zscore[
                        "net_load_z30_same_hour"
                    ],
                }
            )
        prediction = prediction_rows.get(hour)
        if prediction:
            row.update(
                {
                    "predicted_spread": prediction["predicted_spread"],
                    "confidence": prediction["confidence"],
                    "prediction_signal": prediction["signal"],
                    "prediction_confidence": prediction["confidence"],
                    "feature_missing_count": prediction["feature_missing_count"],
                    "p_negative": prediction["p_negative"],
                    "p_neutral": prediction["p_neutral"],
                    "p_positive": prediction["p_positive"],
                    "model_name": prediction["model_name"],
                    "model_version": prediction["model_version"],
                    "predicted_at_utc": prediction["predicted_at_utc"],
                }
            )
        rows.append(row)

    return {
        "delivery_date": delivery_date,
        "table": "online_realtime_features",
        "row_count": len(rows),
        "hours": rows,
    }


def get_realtime_as_market_status(delivery_date: str) -> dict[str, Any] | None:
    forecast = get_realtime_as_market_forecast(delivery_date)
    if forecast is None:
        return None
    hours = forecast["hours"]
    row_count = len(hours)
    return {
        "delivery_date": delivery_date,
        "table": forecast["table"],
        "row_count": row_count,
        "expected_hours": 24,
        "missing_hours": max(24 - row_count, 0),
        "complete_day": row_count == 24,
        "first_delivery_hour_utc": hours[0]["delivery_hour_utc"] if hours else None,
        "last_delivery_hour_utc": hours[-1]["delivery_hour_utc"] if hours else None,
        "has_all_three_forecasts": all(
            row.get("load_system_total_mw") is not None
            and row.get("wind_stwpf_system_wide_mw") is not None
            and row.get("solar_pvgrpp_system_mw") is not None
            for row in hours
        ),
        "load_pre_dam_valid": any(row.get("load_system_total_mw") is not None for row in hours),
        "wind_pre_dam_valid": any(
            row.get("wind_stwpf_system_wide_mw") is not None for row in hours
        ),
        "solar_pre_dam_valid": any(
            row.get("solar_pvgrpp_system_mw") is not None for row in hours
        ),
        "all_issue_times_pre_dam_valid": True,
    }


def get_realtime_status() -> dict[str, Any]:
    init_realtime_tables()
    with connect() as conn:
        latest_run = conn.execute(
            """
            SELECT run_id, collected_at_utc, delivery_date_local, status, message
            FROM online_collection_runs
            ORDER BY run_id DESC
            LIMIT 1
            """
        ).fetchone()
        row_counts = {
            "weather": conn.execute(
                "SELECT COUNT(*) FROM online_weather_forecast_hourly"
            ).fetchone()[0],
            "gas": conn.execute(
                "SELECT COUNT(*) FROM online_gas_observations"
            ).fetchone()[0],
            "ercot": conn.execute(
                "SELECT COUNT(*) FROM online_ercot_forecast_rows"
            ).fetchone()[0],
        }
    return {
        "latest_run": dict(latest_run) if latest_run else None,
        "row_counts": row_counts,
    }


def get_realtime_day_ahead(delivery_date: str) -> dict[str, Any]:
    init_realtime_tables()
    with connect() as conn:
        weather_rows = [
            _row_to_dict(row)
            for row in conn.execute(
                """
                SELECT *
                FROM online_weather_forecast_hourly
                WHERE delivery_date_local = ?
                ORDER BY delivery_hour_utc, location
                """,
                (delivery_date,),
            ).fetchall()
        ]
        ercot_rows = [
            _row_to_dict(row)
            for row in conn.execute(
                """
                SELECT *
                FROM online_ercot_forecast_rows
                WHERE delivery_date_local = ?
                ORDER BY delivery_hour_utc, dataset
                """,
                (delivery_date,),
            ).fetchall()
        ]
        gas_observation = conn.execute(
            """
            SELECT *
            FROM online_gas_observations
            WHERE value IS NOT NULL
            ORDER BY observation_date DESC
            LIMIT 1
            """
        ).fetchone()
        latest_collected_at_utc = conn.execute(
            """
            SELECT MAX(collected_at_utc)
            FROM (
                SELECT collected_at_utc FROM online_weather_forecast_hourly
                WHERE delivery_date_local = ?
                UNION ALL
                SELECT collected_at_utc FROM online_ercot_forecast_rows
                WHERE delivery_date_local = ?
                UNION ALL
                SELECT collected_at_utc FROM online_gas_observations
            )
            """,
            (delivery_date, delivery_date),
        ).fetchone()[0]
    return {
        "delivery_date": delivery_date,
        "latest_collected_at_utc": latest_collected_at_utc,
        "weather_rows": weather_rows,
        "gas_observation": _row_to_dict(gas_observation) if gas_observation else None,
        "ercot_rows": ercot_rows,
    }
