from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from app.core.config import DEFAULT_REALTIME_SQLITE_PATH, get_realtime_database_uri


def sqlite_path_from_uri(db_uri: str | None = None) -> Path:
    db_uri = db_uri or get_realtime_database_uri()
    if not db_uri.startswith("sqlite:///"):
        raise ValueError("Realtime SQLite store only supports sqlite:/// URIs.")
    return Path(db_uri.removeprefix("sqlite:///"))


def connect() -> sqlite3.Connection:
    path = sqlite_path_from_uri()
    if not path.exists():
        path = DEFAULT_REALTIME_SQLITE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_realtime_tables() -> None:
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS online_collection_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                collected_at_utc TEXT NOT NULL,
                delivery_date_local TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT
            );

            CREATE TABLE IF NOT EXISTS online_weather_forecast_hourly (
                delivery_date_local TEXT NOT NULL,
                delivery_hour_utc TEXT NOT NULL,
                location TEXT NOT NULL,
                collected_at_utc TEXT NOT NULL,
                temperature_2m REAL,
                relative_humidity_2m REAL,
                wind_speed_10m REAL,
                wind_gusts_10m REAL,
                cloud_cover REAL,
                shortwave_radiation REAL,
                precipitation REAL,
                raw_json TEXT NOT NULL,
                PRIMARY KEY (delivery_hour_utc, location)
            );

            CREATE TABLE IF NOT EXISTS online_gas_observations (
                series_id TEXT NOT NULL,
                observation_date TEXT NOT NULL,
                collected_at_utc TEXT NOT NULL,
                value REAL,
                raw_json TEXT NOT NULL,
                PRIMARY KEY (series_id, observation_date)
            );

            CREATE TABLE IF NOT EXISTS online_ercot_forecast_rows (
                dataset TEXT NOT NULL,
                delivery_date_local TEXT NOT NULL,
                delivery_hour_utc TEXT NOT NULL,
                collected_at_utc TEXT NOT NULL,
                publish_time_utc TEXT,
                raw_json TEXT NOT NULL,
                PRIMARY KEY (dataset, delivery_hour_utc)
            );

            CREATE TABLE IF NOT EXISTS online_model_predictions (
                delivery_date_local TEXT NOT NULL,
                delivery_hour_utc TEXT NOT NULL,
                predicted_at_utc TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                model_fold TEXT NOT NULL,
                predicted_spread REAL,
                p_c1 REAL,
                p_c2 REAL,
                p_c3 REAL,
                p_c4 REAL,
                p_c5 REAL,
                p_negative REAL,
                p_neutral REAL,
                p_positive REAL,
                predicted_class INTEGER,
                confidence REAL,
                signal TEXT NOT NULL,
                numeric_signal INTEGER NOT NULL,
                feature_missing_count INTEGER NOT NULL,
                raw_json TEXT NOT NULL,
                PRIMARY KEY (delivery_hour_utc, model_name, model_version, model_fold)
            );

            CREATE TABLE IF NOT EXISTS online_price_actuals (
                delivery_date_local TEXT NOT NULL,
                delivery_hour_utc TEXT NOT NULL,
                location TEXT NOT NULL,
                collected_at_utc TEXT NOT NULL,
                da_price_usd_per_mwh REAL,
                rt_price_usd_per_mwh REAL,
                spread_usd_per_mwh REAL,
                rt_interval_count INTEGER NOT NULL,
                raw_json TEXT NOT NULL,
                PRIMARY KEY (delivery_hour_utc, location)
            );

            CREATE TABLE IF NOT EXISTS online_zscore_features (
                delivery_date_local TEXT NOT NULL,
                delivery_hour_utc TEXT NOT NULL,
                location TEXT NOT NULL,
                calculated_at_utc TEXT NOT NULL,
                ercot_local_hour INTEGER NOT NULL,
                history_window_observations INTEGER NOT NULL,
                min_required_observations INTEGER NOT NULL,
                gas_price_z30 REAL,
                load_system_z30_same_hour REAL,
                net_load_z30_same_hour REAL,
                raw_json TEXT NOT NULL,
                PRIMARY KEY (delivery_hour_utc, location)
            );

            CREATE INDEX IF NOT EXISTS idx_online_weather_delivery_date
                ON online_weather_forecast_hourly(delivery_date_local);
            CREATE INDEX IF NOT EXISTS idx_online_ercot_delivery_date
                ON online_ercot_forecast_rows(delivery_date_local);
            CREATE INDEX IF NOT EXISTS idx_online_predictions_delivery_date
                ON online_model_predictions(delivery_date_local);
            CREATE INDEX IF NOT EXISTS idx_online_price_actuals_delivery_date
                ON online_price_actuals(delivery_date_local);
            CREATE INDEX IF NOT EXISTS idx_online_zscore_delivery_date
                ON online_zscore_features(delivery_date_local);
            """
        )


def insert_collection_run(
    *,
    collected_at_utc: str,
    delivery_date_local: str,
    status: str,
    message: str | None,
) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO online_collection_runs
                (collected_at_utc, delivery_date_local, status, message)
            VALUES (?, ?, ?, ?)
            """,
            (collected_at_utc, delivery_date_local, status, message),
        )


def replace_weather_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with connect() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO online_weather_forecast_hourly (
                delivery_date_local,
                delivery_hour_utc,
                location,
                collected_at_utc,
                temperature_2m,
                relative_humidity_2m,
                wind_speed_10m,
                wind_gusts_10m,
                cloud_cover,
                shortwave_radiation,
                precipitation,
                raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["delivery_date_local"],
                    row["delivery_hour_utc"],
                    row["location"],
                    row["collected_at_utc"],
                    row.get("temperature_2m"),
                    row.get("relative_humidity_2m"),
                    row.get("wind_speed_10m"),
                    row.get("wind_gusts_10m"),
                    row.get("cloud_cover"),
                    row.get("shortwave_radiation"),
                    row.get("precipitation"),
                    json.dumps(row.get("raw", row), ensure_ascii=False),
                )
                for row in rows
            ],
        )


def replace_gas_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with connect() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO online_gas_observations (
                series_id,
                observation_date,
                collected_at_utc,
                value,
                raw_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    row["series_id"],
                    row["observation_date"],
                    row["collected_at_utc"],
                    row.get("value"),
                    json.dumps(row.get("raw", row), ensure_ascii=False),
                )
                for row in rows
            ],
        )


def replace_ercot_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with connect() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO online_ercot_forecast_rows (
                dataset,
                delivery_date_local,
                delivery_hour_utc,
                collected_at_utc,
                publish_time_utc,
                raw_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["dataset"],
                    row["delivery_date_local"],
                    row["delivery_hour_utc"],
                    row["collected_at_utc"],
                    row.get("publish_time_utc"),
                    json.dumps(row.get("raw", row), ensure_ascii=False),
                )
                for row in rows
            ],
        )


def replace_model_predictions(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with connect() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO online_model_predictions (
                delivery_date_local,
                delivery_hour_utc,
                predicted_at_utc,
                model_name,
                model_version,
                model_fold,
                predicted_spread,
                p_c1,
                p_c2,
                p_c3,
                p_c4,
                p_c5,
                p_negative,
                p_neutral,
                p_positive,
                predicted_class,
                confidence,
                signal,
                numeric_signal,
                feature_missing_count,
                raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["delivery_date_local"],
                    row["delivery_hour_utc"],
                    row["predicted_at_utc"],
                    row["model_name"],
                    row["model_version"],
                    row["model_fold"],
                    row.get("predicted_spread"),
                    row.get("p_c1"),
                    row.get("p_c2"),
                    row.get("p_c3"),
                    row.get("p_c4"),
                    row.get("p_c5"),
                    row.get("p_negative"),
                    row.get("p_neutral"),
                    row.get("p_positive"),
                    row.get("predicted_class"),
                    row.get("confidence"),
                    row["signal"],
                    row["numeric_signal"],
                    row["feature_missing_count"],
                    json.dumps(row.get("raw", row), ensure_ascii=False),
                )
                for row in rows
            ],
        )


def replace_price_actuals(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with connect() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO online_price_actuals (
                delivery_date_local,
                delivery_hour_utc,
                location,
                collected_at_utc,
                da_price_usd_per_mwh,
                rt_price_usd_per_mwh,
                spread_usd_per_mwh,
                rt_interval_count,
                raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["delivery_date_local"],
                    row["delivery_hour_utc"],
                    row.get("location", "HB_NORTH"),
                    row["collected_at_utc"],
                    row.get("da_price_usd_per_mwh"),
                    row.get("rt_price_usd_per_mwh"),
                    row.get("spread_usd_per_mwh"),
                    row.get("rt_interval_count", 0),
                    json.dumps(row.get("raw", row), ensure_ascii=False),
                )
                for row in rows
            ],
        )


def replace_zscore_features(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with connect() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO online_zscore_features (
                delivery_date_local,
                delivery_hour_utc,
                location,
                calculated_at_utc,
                ercot_local_hour,
                history_window_observations,
                min_required_observations,
                gas_price_z30,
                load_system_z30_same_hour,
                net_load_z30_same_hour,
                raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["delivery_date_local"],
                    row["delivery_hour_utc"],
                    row.get("location", "HB_NORTH"),
                    row["calculated_at_utc"],
                    row["ercot_local_hour"],
                    row["history_window_observations"],
                    row["min_required_observations"],
                    row.get("gas_price_z30"),
                    row.get("load_system_z30_same_hour"),
                    row.get("net_load_z30_same_hour"),
                    json.dumps(row.get("raw", row), ensure_ascii=False),
                )
                for row in rows
            ],
        )
