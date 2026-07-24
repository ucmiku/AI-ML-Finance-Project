from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from app.core.config import DEFAULT_SQLITE_PATH, get_model_dir, get_model_fold
from app.services.realtime_store import (
    connect,
    init_realtime_tables,
    replace_model_predictions,
    replace_zscore_features,
)


ERCOT_TZ = ZoneInfo("America/Chicago")
CLASS_COLUMNS = ["p_c1", "p_c2", "p_c3", "p_c4", "p_c5"]
DFW_LOCATIONS = {"Dallas", "Fort_Worth", "Denton", "McKinney", "Arlington"}
ZSCORE_MIN_OBSERVATIONS = 15
ZSCORE_WINDOW_OBSERVATIONS = 30
LOCATION = "HB_NORTH"
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


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _floor_utc_hour(value: str) -> str:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt.replace(minute=0, second=0, microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _local_datetime(delivery_hour_utc: str) -> datetime:
    return datetime.fromisoformat(delivery_hour_utc.replace("Z", "+00:00")).astimezone(
        ERCOT_TZ
    )


def _delivery_utc_bounds(delivery_date: str) -> tuple[str, str]:
    start_local = datetime.fromisoformat(delivery_date).replace(tzinfo=ERCOT_TZ)
    end_local = start_local + pd.Timedelta(days=1)
    return (
        start_local.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        end_local.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


def _utc_text_minus_hours(value: str, hours: int) -> str:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00")) - pd.Timedelta(
        hours=hours
    )
    return dt.isoformat().replace("+00:00", "Z")


def _offline_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DEFAULT_SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float | None]) -> float | None:
    real = [float(value) for value in values if value is not None]
    return sum(real) / len(real) if real else None


def _max(values: list[float | None]) -> float | None:
    real = [float(value) for value in values if value is not None]
    return max(real) if real else None


def _min(values: list[float | None]) -> float | None:
    real = [float(value) for value in values if value is not None]
    return min(real) if real else None


def _zscore(current: float | None, history: list[float]) -> float | None:
    if current is None or len(history) < ZSCORE_MIN_OBSERVATIONS:
        return None
    series = pd.Series(history[-ZSCORE_WINDOW_OBSERVATIONS:], dtype="float64").dropna()
    if len(series) < ZSCORE_MIN_OBSERVATIONS:
        return None
    std = float(series.std(ddof=1))
    if std == 0 or math.isnan(std):
        return None
    return (float(current) - float(series.mean())) / std


@lru_cache(maxsize=1)
def load_model_bundle() -> dict[str, Any]:
    model_dir = get_model_dir()
    fold = get_model_fold()
    schema_path = model_dir / "feature_schema.json"
    metadata_path = model_dir / "model_metadata.json"
    thresholds_path = model_dir / "thresholds.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema not found: {schema_path}")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    thresholds = (
        json.loads(thresholds_path.read_text(encoding="utf-8"))
        if thresholds_path.exists()
        else {"probability_threshold": 0.6}
    )

    reg_path = model_dir / "b2a_xgboost_regressor" / f"{fold}_pipeline.joblib"
    clf_path = model_dir / "b2b_xgboost_classifier" / f"{fold}_pipeline.joblib"
    if not reg_path.exists():
        raise FileNotFoundError(f"Regressor model not found: {reg_path}")
    if not clf_path.exists():
        raise FileNotFoundError(f"Classifier model not found: {clf_path}")

    return {
        "model_dir": str(model_dir),
        "fold": fold,
        "schema": schema,
        "metadata": metadata,
        "thresholds": thresholds,
        "b2a_features": list(schema["b2a_xgboost_regressor"]["feature_order"]),
        "b2b_features": list(schema["b2b_xgboost_classifier"]["feature_order"]),
        "regressor": joblib.load(reg_path),
        "classifier": joblib.load(clf_path),
    }


def get_model_info() -> dict[str, Any]:
    try:
        bundle = load_model_bundle()
        metadata = bundle["metadata"]
        return {
            "model_loaded": True,
            "model_name": metadata.get("model_name", "C1_XGBoost_Prediction_Agent"),
            "model_version": metadata.get("model_version", "v3"),
            "model_fold": bundle["fold"],
            "model_dir": bundle["model_dir"],
            "selected_agent": metadata.get("selected_agent"),
            "target": "spread_usd_per_mwh",
            "prediction_unit": "USD/MWh",
            "location": "HB_NORTH",
            "feature_schema_version": bundle["schema"].get("schema_version"),
            "b2a_feature_count": len(bundle["b2a_features"]),
            "b2b_feature_count": len(bundle["b2b_features"]),
            "signal_probability_threshold": bundle["thresholds"].get(
                "probability_threshold", 0.6
            ),
            "realtime_enabled": True,
            "realtime_refresh_interval_seconds": 900,
            "prediction_horizon": "next_day_hourly",
            "production_note": metadata.get("production_note"),
        }
    except Exception as exc:
        return {
            "model_loaded": False,
            "model_name": "C1_XGBoost_Prediction_Agent",
            "model_version": "not_loaded",
            "error": str(exc),
            "realtime_enabled": True,
            "realtime_refresh_interval_seconds": 900,
        }


def _latest_gas_price(conn: Any) -> float | None:
    row = conn.execute(
        """
        SELECT value
        FROM online_gas_observations
        WHERE value IS NOT NULL
        ORDER BY observation_date DESC
        LIMIT 1
        """
    ).fetchone()
    return _safe_float(row["value"]) if row else None


def _historical_same_hour_values(
    delivery_hour_utc: str,
    ercot_local_hour: int,
) -> dict[str, list[float]]:
    query = """
        SELECT
            gas_price_usd_per_mmbtu,
            load_system_total_mw,
            net_load_st_forecast_system_mw
        FROM model_wide_hourly_2024_2026
        WHERE location = ?
          AND ercot_local_hour = ?
          AND delivery_hour_utc < ?
        ORDER BY delivery_hour_utc DESC
        LIMIT ?
    """
    with _offline_connect() as conn:
        rows = conn.execute(
            query,
            (
                LOCATION,
                ercot_local_hour,
                delivery_hour_utc,
                ZSCORE_WINDOW_OBSERVATIONS,
            ),
        ).fetchall()

    oldest_to_newest = list(reversed(rows))
    return {
        "gas_price": [
            value
            for value in (
                _safe_float(row["gas_price_usd_per_mmbtu"])
                for row in oldest_to_newest
            )
            if value is not None
        ],
        "load_system_total_mw": [
            value
            for value in (
                _safe_float(row["load_system_total_mw"]) for row in oldest_to_newest
            )
            if value is not None
        ],
        "net_load_st_forecast_system_mw": [
            value
            for value in (
                _safe_float(row["net_load_st_forecast_system_mw"])
                for row in oldest_to_newest
            )
            if value is not None
        ],
    }


def _historical_spread_values(delivery_hour_utc: str) -> list[float]:
    values_by_hour: dict[str, float] = {}
    offline_query = """
        SELECT delivery_hour_utc, spread_usd_per_mwh
        FROM model_wide_hourly_2024_2026
        WHERE location = ?
          AND delivery_hour_utc < ?
          AND spread_usd_per_mwh IS NOT NULL
        ORDER BY delivery_hour_utc DESC
        LIMIT 240
    """
    with _offline_connect() as offline:
        for row in offline.execute(offline_query, (LOCATION, delivery_hour_utc)):
            value = _safe_float(row["spread_usd_per_mwh"])
            if value is not None:
                values_by_hour[row["delivery_hour_utc"]] = value

    init_realtime_tables()
    realtime_query = """
        SELECT delivery_hour_utc, spread_usd_per_mwh
        FROM online_price_actuals
        WHERE location = ?
          AND delivery_hour_utc < ?
          AND spread_usd_per_mwh IS NOT NULL
        ORDER BY delivery_hour_utc DESC
        LIMIT 240
    """
    with connect() as realtime:
        for row in realtime.execute(realtime_query, (LOCATION, delivery_hour_utc)):
            value = _safe_float(row["spread_usd_per_mwh"])
            if value is not None:
                values_by_hour[row["delivery_hour_utc"]] = value

    return [values_by_hour[hour] for hour in sorted(values_by_hour)]


def _mad(series: pd.Series) -> float | None:
    if series.empty:
        return None
    median = float(series.median())
    return float((series - median).abs().median())


def _spread_asof_features(delivery_hour_utc: str) -> dict[str, Any]:
    history = _historical_spread_values(delivery_hour_utc)

    def lag(hours: int) -> float | None:
        return history[-hours] if len(history) >= hours else None

    roll24 = pd.Series(history[-24:], dtype="float64")
    roll72 = pd.Series(history[-72:], dtype="float64")
    out: dict[str, Any] = {
        "spread_asof_lag24": lag(24),
        "spread_asof_lag48": lag(48),
        "spread_asof_lag168": lag(168),
        "lag24_available": int(len(history) >= 24),
        "lag48_available": int(len(history) >= 48),
        "lag168_available": int(len(history) >= 168),
        "rolling24_available": int(len(roll24) >= 24),
        "rolling72_available": int(len(roll72) >= 72),
    }
    if len(roll24) >= 24:
        out.update(
            {
                "spread_asof_roll_mean24": float(roll24.mean()),
                "spread_asof_roll_median24": float(roll24.median()),
                "spread_asof_roll_std24": float(roll24.std(ddof=1)),
                "spread_asof_roll_mad24": _mad(roll24),
                "spread_spike_count_24h_gt20": int((roll24.abs() > 20).sum()),
            }
        )
    if len(roll72) >= 72:
        out.update(
            {
                "spread_asof_roll_mean72": float(roll72.mean()),
                "spread_asof_roll_std72": float(roll72.std(ddof=1)),
                "spread_spike_count_72h_gt20": int((roll72.abs() > 20).sum()),
            }
        )
    return out


def _calculate_zscore_features(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    calculated_at = _utc_now_text()
    rows: list[dict[str, Any]] = []
    out: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        hour = str(row["delivery_hour_utc"])
        local_hour = int(row["ercot_local_hour"])
        history = _historical_same_hour_values(hour, local_hour)
        current_gas = _safe_float(row.get("gas_price"))
        current_load = _safe_float(row.get("load_system_total_mw"))
        current_net_load = _safe_float(row.get("net_load_st_forecast_system_mw"))
        features = {
            "gas_price_z30": _zscore(current_gas, history["gas_price"]),
            "load_system_z30_same_hour": _zscore(
                current_load, history["load_system_total_mw"]
            ),
            "net_load_z30_same_hour": _zscore(
                current_net_load, history["net_load_st_forecast_system_mw"]
            ),
        }
        out[hour] = features
        zscore_row = {
            "delivery_date_local": row["delivery_date_local"],
            "delivery_hour_utc": hour,
            "location": LOCATION,
            "calculated_at_utc": calculated_at,
            "ercot_local_hour": local_hour,
            "history_window_observations": min(
                len(history["gas_price"]),
                len(history["load_system_total_mw"]),
                len(history["net_load_st_forecast_system_mw"]),
            ),
            "min_required_observations": ZSCORE_MIN_OBSERVATIONS,
            **features,
        }
        zscore_row["raw"] = {
            **zscore_row,
            "rule": (
                "group by location and ERCOT local hour; shift(1) by requiring "
                "historical delivery_hour_utc < current delivery_hour_utc; use "
                "latest 30 same-hour observations with at least 15 observations"
            ),
            "history_counts": {
                "gas_price": len(history["gas_price"]),
                "load_system_total_mw": len(history["load_system_total_mw"]),
                "net_load_st_forecast_system_mw": len(
                    history["net_load_st_forecast_system_mw"]
                ),
            },
        }
        rows.append(zscore_row)
    replace_zscore_features(rows)
    return out


def _load_weather_features(conn: Any, delivery_date: str) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT *
        FROM online_weather_forecast_hourly
        WHERE delivery_date_local = ?
        """,
        (delivery_date,),
    ).fetchall()
    by_hour: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        data = dict(row)
        by_hour.setdefault(data["delivery_hour_utc"], []).append(data)

    out: dict[str, dict[str, Any]] = {}
    for hour, items in by_hour.items():
        dfw = [item for item in items if item["location"] in DFW_LOCATIONS]
        wichita = next(
            (item for item in items if item["location"] == "Wichita_Falls"), None
        )
        dfw_temp_mean = _mean([_safe_float(item["temperature_2m"]) for item in dfw])
        dfw_humidity_mean = _mean(
            [_safe_float(item["relative_humidity_2m"]) for item in dfw]
        )
        dfw_wind_mean = _mean([_safe_float(item["wind_speed_10m"]) for item in dfw])
        dfw_gust_mean = _mean([_safe_float(item["wind_gusts_10m"]) for item in dfw])
        dfw_cloud_mean = _mean([_safe_float(item["cloud_cover"]) for item in dfw])
        dfw_radiation_mean = _mean(
            [_safe_float(item["shortwave_radiation"]) for item in dfw]
        )
        dfw_precip_mean = _mean([_safe_float(item["precipitation"]) for item in dfw])
        w_temp = _safe_float(wichita["temperature_2m"]) if wichita else None
        w_humidity = _safe_float(wichita["relative_humidity_2m"]) if wichita else None
        w_wind = _safe_float(wichita["wind_speed_10m"]) if wichita else None
        w_gust = _safe_float(wichita["wind_gusts_10m"]) if wichita else None
        w_cloud = _safe_float(wichita["cloud_cover"]) if wichita else None
        w_radiation = _safe_float(wichita["shortwave_radiation"]) if wichita else None
        w_precip = _safe_float(wichita["precipitation"]) if wichita else None
        all_temps = [_safe_float(item["temperature_2m"]) for item in items]
        all_gusts = [_safe_float(item["wind_gusts_10m"]) for item in items]
        all_precips = [_safe_float(item["precipitation"]) for item in items]

        out[hour] = {
            "dfw_city_count": len(dfw),
            "temperature_dfw_mean_c": dfw_temp_mean,
            "temperature_dfw_min_c": _min([_safe_float(item["temperature_2m"]) for item in dfw]),
            "temperature_dfw_max_c": _max([_safe_float(item["temperature_2m"]) for item in dfw]),
            "temperature_wichita_c": w_temp,
            "temperature_wichita_minus_dfw_c": (
                w_temp - dfw_temp_mean if w_temp is not None and dfw_temp_mean is not None else None
            ),
            "humidity_dfw_mean_pct": dfw_humidity_mean,
            "humidity_wichita_pct": w_humidity,
            "humidity_wichita_minus_dfw_pct": (
                w_humidity - dfw_humidity_mean
                if w_humidity is not None and dfw_humidity_mean is not None
                else None
            ),
            "wind_speed_dfw_mean_ms": dfw_wind_mean,
            "wind_speed_dfw_max_ms": _max([_safe_float(item["wind_speed_10m"]) for item in dfw]),
            "wind_speed_wichita_ms": w_wind,
            "wind_speed_wichita_minus_dfw_ms": (
                w_wind - dfw_wind_mean if w_wind is not None and dfw_wind_mean is not None else None
            ),
            "wind_gust_dfw_mean_ms": dfw_gust_mean,
            "wind_gust_dfw_max_ms": _max([_safe_float(item["wind_gusts_10m"]) for item in dfw]),
            "wind_gust_wichita_ms": w_gust,
            "wind_gust_wichita_minus_dfw_ms": (
                w_gust - dfw_gust_mean if w_gust is not None and dfw_gust_mean is not None else None
            ),
            "cloud_cover_dfw_mean_pct": dfw_cloud_mean,
            "cloud_cover_wichita_pct": w_cloud,
            "cloud_cover_wichita_minus_dfw_pct": (
                w_cloud - dfw_cloud_mean if w_cloud is not None and dfw_cloud_mean is not None else None
            ),
            "radiation_dfw_mean_wm2": dfw_radiation_mean,
            "radiation_wichita_wm2": w_radiation,
            "radiation_wichita_minus_dfw_wm2": (
                w_radiation - dfw_radiation_mean
                if w_radiation is not None and dfw_radiation_mean is not None
                else None
            ),
            "precipitation_dfw_mean_mm": dfw_precip_mean,
            "precipitation_dfw_max_mm": _max([_safe_float(item["precipitation"]) for item in dfw]),
            "precipitation_wichita_mm": w_precip,
            "precipitation_wichita_minus_dfw_mm": (
                w_precip - dfw_precip_mean if w_precip is not None and dfw_precip_mean is not None else None
            ),
            "north_temperature_min_c": _min(all_temps),
            "north_temperature_max_c": _max(all_temps),
            "north_wind_gust_max_ms": _max(all_gusts),
            "north_precipitation_max_mm": _max(all_precips),
            "freezing_city_count": sum(
                1 for value in all_temps if value is not None and value <= 0
            ),
            "extreme_heat_city_count": sum(
                1 for value in all_temps if value is not None and value >= 35
            ),
            "high_wind_city_count": sum(
                1 for value in all_gusts if value is not None and value >= 15
            ),
            "rainy_city_count": sum(
                1 for value in all_precips if value is not None and value > 0
            ),
        }
    return out


def _load_ercot_features(conn: Any, delivery_date: str) -> dict[str, dict[str, Any]]:
    start_utc, end_utc = _delivery_utc_bounds(delivery_date)
    warmup_start_utc = _utc_text_minus_hours(start_utc, 6)
    rows = conn.execute(
        """
        SELECT dataset, delivery_hour_utc, raw_json
        FROM online_ercot_forecast_rows
        WHERE delivery_hour_utc >= ?
          AND delivery_hour_utc < ?
        """,
        (warmup_start_utc, end_utc),
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    load_values: dict[str, list[float]] = {}
    for row in rows:
        data = dict(row)
        raw = json.loads(data["raw_json"])
        hour = _floor_utc_hour(data["delivery_hour_utc"])
        out.setdefault(hour, {})
        if data["dataset"] == "load_forecast":
            value = _safe_float(raw.get("load_forecast"))
            if value is not None:
                load_values.setdefault(hour, []).append(value)
        elif data["dataset"] == "load_forecast_by_weather_zone":
            for target, source in WEATHER_ZONE_LOAD_MAPPING.items():
                value = _safe_float(raw.get(source))
                if value is not None:
                    out[hour][target] = value
            if out[hour].get("load_system_total_mw") is not None:
                out[hour]["load_hb_north_proxy_mw"] = out[hour][
                    "load_system_total_mw"
                ]
        elif data["dataset"] == "wind_forecast":
            mapping = {
                "wind_stwpf_system_wide_mw": "stwpf_system_wide",
                "wind_wgrpp_system_wide_mw": "wgrpp_system_wide",
                "wind_stwpf_lz_north_mw": "stwpf_lz_north",
                "wind_wgrpp_lz_north_mw": "wgrpp_lz_north",
                "wind_stwpf_lz_south_houston_mw": "stwpf_lz_south_houston",
                "wind_wgrpp_lz_south_houston_mw": "wgrpp_lz_south_houston",
                "wind_stwpf_lz_west_mw": "stwpf_lz_west",
                "wind_wgrpp_lz_west_mw": "wgrpp_lz_west",
            }
            for target, source in mapping.items():
                out[hour][target] = _safe_float(raw.get(source))
        elif data["dataset"] == "solar_forecast":
            out[hour]["solar_stppf_system_mw"] = _safe_float(raw.get("stppf_system_wide"))
            out[hour]["solar_pvgrpp_system_mw"] = _safe_float(raw.get("pvgrpp_system_wide"))

    for hour, values in load_values.items():
        if values and out.setdefault(hour, {}).get("load_system_total_mw") is None:
            out.setdefault(hour, {})["load_system_total_mw"] = sum(values) / len(values)
            out[hour]["load_hb_north_proxy_mw"] = out[hour]["load_system_total_mw"]
    for hour, values in out.items():
        load = values.get("load_system_total_mw")
        wind = values.get("wind_stwpf_system_wide_mw")
        solar = values.get("solar_stppf_system_mw")
        wind_potential = values.get("wind_wgrpp_system_wide_mw")
        solar_potential = values.get("solar_pvgrpp_system_mw")
        if wind is not None and solar is not None:
            values["renewable_st_forecast_system_mw"] = wind + solar
        if load is not None and wind is not None and solar is not None:
            values["net_load_st_forecast_system_mw"] = load - wind - solar
            values["renewable_st_share_of_load"] = (
                values["renewable_st_forecast_system_mw"] / load if load else None
            )
        if wind_potential is not None and solar_potential is not None:
            values["renewable_potential_system_mw"] = wind_potential + solar_potential
        if load is not None and wind_potential is not None and solar_potential is not None:
            values["net_load_potential_system_mw"] = load - wind_potential - solar_potential
        north_load = values.get("load_hb_north_proxy_mw")
        north_wind = values.get("wind_stwpf_lz_north_mw")
        if north_load and north_wind is not None:
            values["wind_north_share_of_north_load"] = north_wind / north_load
        for region in ("system", "north", "south_houston", "west"):
            st_key = (
                "wind_stwpf_system_wide_mw"
                if region == "system"
                else f"wind_stwpf_lz_{region}_mw"
            )
            wg_key = (
                "wind_wgrpp_system_wide_mw"
                if region == "system"
                else f"wind_wgrpp_lz_{region}_mw"
            )
            st = values.get(st_key)
            wg = values.get(wg_key)
            if st is not None and wg is not None:
                values[f"wind_gap_{region}_mw"] = wg - st
                if region == "system" and wg:
                    values["wind_gap_system_pct"] = (wg - st) / wg
        pv = values.get("solar_pvgrpp_system_mw")
        st = values.get("solar_stppf_system_mw")
        if pv is not None and st is not None:
            values["solar_gap_system_mw"] = pv - st
            values["solar_gap_system_pct"] = (pv - st) / pv if pv else 0.0
    return out


def _apply_ramps(frame: pd.DataFrame) -> pd.DataFrame:
    ramp_sources = {
        "load_system_total_mw": "load_ramp",
        "wind_stwpf_system_wide_mw": "wind_ramp",
        "solar_stppf_system_mw": "solar_ramp",
        "renewable_st_forecast_system_mw": "renewable_ramp",
        "net_load_st_forecast_system_mw": "net_load_ramp",
    }
    frame = frame.sort_values("delivery_hour_utc").reset_index(drop=True)
    for source, prefix in ramp_sources.items():
        if source in frame.columns:
            for lag in (1, 3, 6):
                frame[f"{prefix}_{lag}h_mw"] = frame[source] - frame[source].shift(lag)
    if "net_load_ramp_3h_mw" in frame.columns:
        frame["net_load_ramp_3h_x_peak_hour"] = (
            frame["net_load_ramp_3h_mw"] * frame.get("is_peak_hour", 0)
        )
        frame["renewable_ramp_3h_x_net_load"] = frame.get(
            "renewable_ramp_3h_mw", np.nan
        ) * frame.get("net_load_st_forecast_system_mw", np.nan)
    return frame


def _apply_weather_durations(frame: pd.DataFrame) -> pd.DataFrame:
    duration_specs = {
        "freezing_hour_flag": "freezing_duration_h",
        "extreme_heat_hour_flag": "extreme_heat_duration_h",
        "high_wind_hour_flag": "high_wind_duration_h",
    }
    frame = frame.sort_values("delivery_hour_utc").reset_index(drop=True)
    for flag_column, duration_column in duration_specs.items():
        duration = 0
        values: list[int] = []
        for flag in frame.get(flag_column, pd.Series([0] * len(frame))).fillna(0):
            duration = duration + 1 if int(flag) == 1 else 0
            values.append(duration)
        frame[duration_column] = values
    return frame


def build_realtime_feature_frame(delivery_date: str) -> pd.DataFrame:
    bundle = load_model_bundle()
    features = sorted(set(bundle["b2a_features"]) | set(bundle["b2b_features"]))
    init_realtime_tables()
    with connect() as conn:
        weather = _load_weather_features(conn, delivery_date)
        ercot = _load_ercot_features(conn, delivery_date)
        gas_price = _latest_gas_price(conn)

    start_utc, end_utc = _delivery_utc_bounds(delivery_date)
    warmup_start_utc = _utc_text_minus_hours(start_utc, 6)
    hours = [
        hour
        for hour in sorted(set(weather) | set(ercot))
        if warmup_start_utc <= hour < end_utc
    ]
    rows: list[dict[str, Any]] = []
    for hour in hours:
        local = _local_datetime(hour)
        row: dict[str, Any] = {feature: np.nan for feature in features}
        row["delivery_hour_utc"] = hour
        row["delivery_date_local"] = local.date().isoformat()
        row["ercot_local_hour"] = local.hour
        row["ercot_local_day_of_week"] = local.weekday()
        row["ercot_local_month"] = local.month
        row["is_weekend"] = int(local.weekday() >= 5)
        row["is_dst"] = int(local.dst().total_seconds() > 0)
        row["is_peak_hour"] = int(7 <= local.hour <= 22)
        row["is_evening_peak"] = int(17 <= local.hour <= 20)
        row["hour_sin"] = math.sin(2 * math.pi * local.hour / 24)
        row["hour_cos"] = math.cos(2 * math.pi * local.hour / 24)
        row["dow_sin"] = math.sin(2 * math.pi * local.weekday() / 7)
        row["dow_cos"] = math.cos(2 * math.pi * local.weekday() / 7)
        row["month_sin"] = math.sin(2 * math.pi * local.month / 12)
        row["month_cos"] = math.cos(2 * math.pi * local.month / 12)
        row["gas_price"] = gas_price
        row["gas_is_forward_filled"] = 1
        row["weather_forecast_lead_hours"] = 24
        row["source_payload_present"] = 1
        row["required_raw_inputs_available"] = 1
        row["dataset_ready_for_training"] = 1
        row["causal_feature_warmup_complete"] = 0
        row.update(_spread_asof_features(hour))
        row["causal_feature_warmup_complete"] = int(
            row.get("lag168_available", 0) == 1
            and row.get("rolling72_available", 0) == 1
        )
        row.update(weather.get(hour, {}))
        row.update(ercot.get(hour, {}))
        row["freezing_hour_flag"] = int((row.get("freezing_city_count") or 0) > 0)
        row["extreme_heat_hour_flag"] = int((row.get("extreme_heat_city_count") or 0) > 0)
        row["high_wind_hour_flag"] = int((row.get("high_wind_city_count") or 0) > 0)
        row["rainy_hour_flag"] = int((row.get("rainy_city_count") or 0) > 0)
        row["fixed_extreme_weather_flag"] = int(
            row["freezing_hour_flag"]
            or row["extreme_heat_hour_flag"]
            or row["high_wind_hour_flag"]
        )
        row["fixed_compound_extreme_count"] = (
            row["freezing_hour_flag"] + row["extreme_heat_hour_flag"] + row["high_wind_hour_flag"]
        )
        row["fixed_extreme_weather_coverage_24h"] = row["fixed_extreme_weather_flag"]
        row["freezing_x_gas_price"] = (
            row["freezing_hour_flag"] * gas_price if gas_price is not None else np.nan
        )
        load = row.get("load_system_total_mw")
        row["extreme_heat_x_load_fixed"] = (
            row["extreme_heat_hour_flag"] * load if load is not None else np.nan
        )
        wind_gap = row.get("wind_gap_system_mw")
        row["high_wind_x_wind_gap"] = (
            row["high_wind_hour_flag"] * wind_gap if wind_gap is not None else np.nan
        )
        solar_gap = row.get("solar_gap_system_mw")
        row["solar_gap_x_peak_hour"] = (
            solar_gap * row["is_peak_hour"] if solar_gap is not None else np.nan
        )
        net_load = row.get("net_load_st_forecast_system_mw")
        row["wind_gap_x_net_load"] = (
            wind_gap * net_load
            if wind_gap is not None and net_load is not None
            else np.nan
        )
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = _apply_ramps(frame)
    frame = frame[
        (frame["delivery_hour_utc"] >= start_utc)
        & (frame["delivery_hour_utc"] < end_utc)
    ].reset_index(drop=True)
    frame["delivery_date_local"] = delivery_date
    frame = _apply_weather_durations(frame)
    zscore_features = _calculate_zscore_features(frame)
    for column in (
        "gas_price_z30",
        "load_system_z30_same_hour",
        "net_load_z30_same_hour",
    ):
        frame[column] = frame["delivery_hour_utc"].map(
            lambda hour: zscore_features.get(str(hour), {}).get(column)
        )
    return frame


def _predict_proba_5(model: Any, matrix: Any) -> np.ndarray:
    raw = model.predict_proba(matrix)
    probabilities = np.zeros((len(matrix), 5), dtype=float)
    classes = getattr(model, "classes_", np.arange(5)).astype(int)
    probabilities[:, classes] = raw
    return probabilities


def _transform_with_compatible_imputer(imputer: Any, frame: pd.DataFrame) -> Any:
    # Packaged artifacts were pickled with scikit-learn 1.6.1. Newer sklearn
    # expects this private attribute when transforming old SimpleImputer objects.
    if not hasattr(imputer, "_fill_dtype") and hasattr(imputer, "_fit_dtype"):
        imputer._fill_dtype = imputer._fit_dtype
    return imputer.transform(frame)


def _offline_feature_frame(delivery_date: str) -> pd.DataFrame:
    bundle = load_model_bundle()
    features = sorted(set(bundle["b2a_features"]) | set(bundle["b2b_features"]))
    with _offline_connect() as conn:
        available_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(model_wide_hourly_2024_2026)")
        }
    selected_features = [feature for feature in features if feature in available_columns]
    base_columns = [
        "delivery_hour_utc",
        "delivery_date_local",
        "delivery_time_local",
        "ercot_local_hour",
        "ercot_local_day_of_week",
        "ercot_local_month",
        "is_weekend",
        "is_dst",
        "gas_price_usd_per_mmbtu",
        "freezing_city_count",
        "extreme_heat_city_count",
        "high_wind_city_count",
        "rainy_city_count",
        "load_system_total_mw",
        "wind_stwpf_system_wide_mw",
        "wind_wgrpp_system_wide_mw",
        "wind_stwpf_lz_north_mw",
        "wind_wgrpp_lz_north_mw",
        "wind_stwpf_lz_south_houston_mw",
        "wind_wgrpp_lz_south_houston_mw",
        "wind_stwpf_lz_west_mw",
        "wind_wgrpp_lz_west_mw",
        "solar_pvgrpp_system_mw",
        "solar_stppf_system_mw",
        "renewable_st_forecast_system_mw",
        "net_load_st_forecast_system_mw",
    ]
    columns = [
        column
        for column in dict.fromkeys([*base_columns, *selected_features])
        if column in available_columns
    ]
    start_utc, end_utc = _delivery_utc_bounds(delivery_date)
    warmup_start_utc = _utc_text_minus_hours(start_utc, 6)
    query = f"""
        SELECT {", ".join(columns)}
        FROM model_wide_hourly_2024_2026
        WHERE location = ?
          AND delivery_hour_utc >= ?
          AND delivery_hour_utc < ?
        ORDER BY delivery_hour_utc
    """
    with _offline_connect() as conn:
        rows = [
            dict(row)
            for row in conn.execute(query, (LOCATION, warmup_start_utc, end_utc))
        ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    enriched_rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        hour = str(record["delivery_hour_utc"])
        local = _local_datetime(hour)
        record["delivery_date_local"] = local.date().isoformat()
        record["delivery_time_local"] = local.isoformat()
        record["ercot_local_hour"] = local.hour
        record["ercot_local_day_of_week"] = local.weekday()
        record["ercot_local_month"] = local.month
        record["is_weekend"] = int(local.weekday() >= 5)
        record["is_dst"] = int(local.dst().total_seconds() > 0)
        record["is_peak_hour"] = int(7 <= local.hour <= 22)
        record["is_evening_peak"] = int(17 <= local.hour <= 20)
        record["hour_sin"] = math.sin(2 * math.pi * local.hour / 24)
        record["hour_cos"] = math.cos(2 * math.pi * local.hour / 24)
        record["dow_sin"] = math.sin(2 * math.pi * local.weekday() / 7)
        record["dow_cos"] = math.cos(2 * math.pi * local.weekday() / 7)
        record["month_sin"] = math.sin(2 * math.pi * local.month / 12)
        record["month_cos"] = math.cos(2 * math.pi * local.month / 12)
        record["gas_price"] = _safe_float(record.get("gas_price_usd_per_mmbtu"))
        record["source_payload_present"] = 1
        record["required_raw_inputs_available"] = 1
        record["dataset_ready_for_training"] = 1

        record.update(_spread_asof_features(hour))
        record["causal_feature_warmup_complete"] = int(
            record.get("lag168_available", 0) == 1
            and record.get("rolling72_available", 0) == 1
        )

        record["freezing_hour_flag"] = int(
            (record.get("freezing_city_count") or 0) > 0
        )
        record["extreme_heat_hour_flag"] = int(
            (record.get("extreme_heat_city_count") or 0) > 0
        )
        record["high_wind_hour_flag"] = int(
            (record.get("high_wind_city_count") or 0) > 0
        )
        record["rainy_hour_flag"] = int((record.get("rainy_city_count") or 0) > 0)
        record["fixed_extreme_weather_flag"] = int(
            record["freezing_hour_flag"]
            or record["extreme_heat_hour_flag"]
            or record["high_wind_hour_flag"]
        )
        record["fixed_compound_extreme_count"] = (
            record["freezing_hour_flag"]
            + record["extreme_heat_hour_flag"]
            + record["high_wind_hour_flag"]
        )
        record["fixed_extreme_weather_coverage_24h"] = record[
            "fixed_extreme_weather_flag"
        ]
        gas_price = record.get("gas_price")
        record["freezing_x_gas_price"] = (
            record["freezing_hour_flag"] * gas_price
            if gas_price is not None
            else np.nan
        )

        for region in ("system", "north", "south_houston", "west"):
            st_key = (
                "wind_stwpf_system_wide_mw"
                if region == "system"
                else f"wind_stwpf_lz_{region}_mw"
            )
            wg_key = (
                "wind_wgrpp_system_wide_mw"
                if region == "system"
                else f"wind_wgrpp_lz_{region}_mw"
            )
            st = _safe_float(record.get(st_key))
            wg = _safe_float(record.get(wg_key))
            if st is not None and wg is not None:
                record[f"wind_gap_{region}_mw"] = wg - st
                if region == "system" and wg:
                    record["wind_gap_system_pct"] = (wg - st) / wg

        pv = _safe_float(record.get("solar_pvgrpp_system_mw"))
        st_solar = _safe_float(record.get("solar_stppf_system_mw"))
        if pv is not None and st_solar is not None:
            record["solar_gap_system_mw"] = pv - st_solar
            record["solar_gap_system_pct"] = (pv - st_solar) / pv if pv else 0.0

        load = _safe_float(record.get("load_system_total_mw"))
        wind_gap = _safe_float(record.get("wind_gap_system_mw"))
        solar_gap = _safe_float(record.get("solar_gap_system_mw"))
        net_load = _safe_float(record.get("net_load_st_forecast_system_mw"))
        record["extreme_heat_x_load_fixed"] = (
            record["extreme_heat_hour_flag"] * load if load is not None else np.nan
        )
        record["high_wind_x_wind_gap"] = (
            record["high_wind_hour_flag"] * wind_gap
            if wind_gap is not None
            else np.nan
        )
        record["solar_gap_x_peak_hour"] = (
            solar_gap * record["is_peak_hour"] if solar_gap is not None else np.nan
        )
        record["wind_gap_x_net_load"] = (
            wind_gap * net_load
            if wind_gap is not None and net_load is not None
            else np.nan
        )
        enriched_rows.append(record)

    frame = pd.DataFrame(enriched_rows)
    frame = _apply_ramps(frame)
    frame = frame[
        (frame["delivery_hour_utc"] >= start_utc)
        & (frame["delivery_hour_utc"] < end_utc)
    ].reset_index(drop=True)
    if frame.empty:
        return frame
    frame["delivery_date_local"] = delivery_date
    frame = _apply_weather_durations(frame)
    zscore_features = _calculate_zscore_features(frame)
    for column in (
        "gas_price_z30",
        "load_system_z30_same_hour",
        "net_load_z30_same_hour",
    ):
        frame[column] = frame["delivery_hour_utc"].map(
            lambda hour: zscore_features.get(str(hour), {}).get(column)
        )

    for feature in features:
        if feature not in frame.columns:
            frame[feature] = np.nan
    return frame


def _predict_from_feature_frame(
    frame: pd.DataFrame,
    delivery_date: str,
    *,
    persist: bool,
) -> dict[str, Any]:
    bundle = load_model_bundle()
    if len(frame) < 24:
        return {
            "delivery_date": delivery_date,
            "row_count": 0,
            "predictions": [],
            "status": "incomplete_feature_rows",
            "message": f"Expected 24 hourly feature rows, found {len(frame)}.",
        }

    reg = bundle["regressor"]
    clf = bundle["classifier"]
    b2a_features = bundle["b2a_features"]
    b2b_features = bundle["b2b_features"]

    if isinstance(reg, dict):
        predicted_spread = reg["model"].predict(
            _transform_with_compatible_imputer(reg["imputer"], frame[b2a_features])
        )
    else:
        predicted_spread = reg.predict(frame[b2a_features])

    if isinstance(clf, dict):
        probabilities = _predict_proba_5(
            clf["model"],
            _transform_with_compatible_imputer(clf["imputer"], frame[b2b_features]),
        )
    else:
        probabilities = _predict_proba_5(clf, frame[b2b_features])

    threshold = float(bundle["thresholds"].get("probability_threshold", 0.6))
    predicted_at = _utc_now_text()
    metadata = bundle["metadata"]
    model_name = metadata.get("model_name", "C1_XGBoost_Prediction_Agent")
    model_version = metadata.get("model_version", "v3")
    rows: list[dict[str, Any]] = []
    for idx, (_, feature_row) in enumerate(frame.iterrows()):
        local = _local_datetime(str(feature_row["delivery_hour_utc"]))
        p_c1, p_c2, p_c3, p_c4, p_c5 = [float(x) for x in probabilities[idx]]
        p_negative = p_c1 + p_c2
        p_positive = p_c4 + p_c5
        signal = "NO_TRADE"
        numeric_signal = 0
        if p_positive >= threshold and p_positive > p_negative:
            signal = "DEC"
            numeric_signal = 1
        elif p_negative >= threshold and p_negative > p_positive:
            signal = "INC"
            numeric_signal = -1
        prediction = {
            "delivery_date_local": delivery_date,
            "delivery_hour_utc": feature_row["delivery_hour_utc"],
            "delivery_time_local": local.isoformat(),
            "ercot_local_hour": local.hour,
            "hour": local.hour,
            "predicted_at_utc": predicted_at,
            "model_name": model_name,
            "model_version": model_version,
            "model_fold": bundle["fold"],
            "predicted_spread": float(predicted_spread[idx]),
            "p_c1": p_c1,
            "p_c2": p_c2,
            "p_c3": p_c3,
            "p_c4": p_c4,
            "p_c5": p_c5,
            "p_negative": p_negative,
            "p_neutral": p_c3,
            "p_positive": p_positive,
            "predicted_class": int(np.argmax(probabilities[idx]) + 1),
            "confidence": float(probabilities[idx].max()),
            "signal": signal,
            "prediction_signal": signal,
            "prediction_confidence": float(probabilities[idx].max()),
            "numeric_signal": numeric_signal,
            "feature_missing_count": int(frame[b2b_features].iloc[idx].isna().sum()),
        }
        prediction["raw"] = prediction.copy()
        rows.append(prediction)
    if persist:
        replace_model_predictions(rows)
    return {
        "delivery_date": delivery_date,
        "row_count": len(rows),
        "model_name": model_name,
        "model_version": model_version,
        "model_fold": bundle["fold"],
        "predicted_at_utc": predicted_at,
        "predictions": rows,
        "status": "success",
    }


def run_prediction(delivery_date: str) -> dict[str, Any]:
    frame = build_realtime_feature_frame(delivery_date)
    if frame.empty:
        return {
            "delivery_date": delivery_date,
            "row_count": 0,
            "predictions": [],
            "status": "no_realtime_features",
        }
    return _predict_from_feature_frame(frame, delivery_date, persist=True)


def run_offline_prediction(delivery_date: str) -> dict[str, Any]:
    frame = _offline_feature_frame(delivery_date)
    if frame.empty:
        return {
            "delivery_date": delivery_date,
            "row_count": 0,
            "predictions": [],
            "status": "no_offline_features",
        }
    return _predict_from_feature_frame(frame, delivery_date, persist=False)


def get_predictions(delivery_date: str) -> dict[str, Any]:
    init_realtime_tables()
    with connect() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT *
                FROM online_model_predictions
                WHERE delivery_date_local = ?
                ORDER BY delivery_hour_utc
                """,
                (delivery_date,),
            ).fetchall()
        ]
    if not rows:
        prediction = run_prediction(delivery_date)
        if prediction.get("row_count", 0) == 0:
            prediction = run_offline_prediction(delivery_date)
        if prediction.get("row_count", 0) == 0:
            return {
                "delivery_date": delivery_date,
                "row_count": 0,
                "predictions": [],
                "status": prediction.get("status", "no_predictions"),
                "message": prediction.get("message"),
            }
        rows = prediction["predictions"]
    for row in rows:
        row.pop("raw_json", None)
        if "hour" not in row or row["hour"] is None:
            local = _local_datetime(str(row["delivery_hour_utc"]))
            row["delivery_time_local"] = local.isoformat()
            row["ercot_local_hour"] = local.hour
            row["hour"] = local.hour
        row.setdefault("prediction_signal", row.get("signal"))
        row.setdefault("prediction_confidence", row.get("confidence"))
    return {
        "delivery_date": delivery_date,
        "row_count": len(rows),
        "predictions": rows,
    }
