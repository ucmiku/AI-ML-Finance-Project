from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from collectors.common import DATA_WORKSPACE


DEFAULT_RAW_DATABASE = DATA_WORKSPACE / "interim" / "ercot_data.sqlite"
DEFAULT_ANALYTICS_DATABASE = (
    DATA_WORKSPACE / "interim" / "ercot_analytics.sqlite"
)
SCHEMA_VERSION = 6
ERCOT_TIMEZONE = ZoneInfo("America/Chicago")

WEATHER_LOCATIONS = (
    "Dallas",
    "Fort Worth",
    "Denton",
    "McKinney",
    "Arlington",
    "Wichita Falls",
)

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE clean_build_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE clean_da_price_hourly (
    delivery_hour_utc TEXT NOT NULL,
    interval_end_utc TEXT NOT NULL,
    location TEXT NOT NULL,
    location_type TEXT,
    market TEXT NOT NULL,
    da_price_usd_per_mwh REAL NOT NULL,
    source_file_id INTEGER NOT NULL,
    source_record_id INTEGER NOT NULL,
    PRIMARY KEY (delivery_hour_utc, location),
    CHECK (delivery_hour_utc GLOB '????-??-??T??:??:??Z')
) WITHOUT ROWID;

CREATE TABLE clean_rt_price_15min (
    interval_start_utc TEXT NOT NULL,
    interval_end_utc TEXT NOT NULL,
    location TEXT NOT NULL,
    location_type TEXT,
    market TEXT NOT NULL,
    rt_price_usd_per_mwh REAL NOT NULL,
    source_file_id INTEGER NOT NULL,
    source_record_id INTEGER NOT NULL,
    PRIMARY KEY (interval_start_utc, location),
    CHECK (interval_start_utc GLOB '????-??-??T??:??:??Z')
) WITHOUT ROWID;

CREATE TABLE clean_rt_price_hourly (
    delivery_hour_utc TEXT NOT NULL,
    location TEXT NOT NULL,
    rt_price_usd_per_mwh REAL NOT NULL,
    rt_price_min_usd_per_mwh REAL NOT NULL,
    rt_price_max_usd_per_mwh REAL NOT NULL,
    interval_count INTEGER NOT NULL,
    is_complete_hour INTEGER NOT NULL,
    PRIMARY KEY (delivery_hour_utc, location),
    CHECK (is_complete_hour IN (0, 1))
) WITHOUT ROWID;

CREATE TABLE clean_price_hourly (
    delivery_hour_utc TEXT NOT NULL,
    location TEXT NOT NULL,
    da_price_usd_per_mwh REAL,
    rt_price_usd_per_mwh REAL,
    spread_usd_per_mwh REAL,
    spread_sign INTEGER,
    rt_above_da INTEGER,
    rt_interval_count INTEGER,
    is_label_complete INTEGER NOT NULL,
    PRIMARY KEY (delivery_hour_utc, location),
    CHECK (spread_sign IS NULL OR spread_sign IN (-1, 0, 1)),
    CHECK (rt_above_da IS NULL OR rt_above_da IN (0, 1)),
    CHECK (is_label_complete IN (0, 1))
) WITHOUT ROWID;

CREATE TABLE clean_weather_hourly (
    city TEXT NOT NULL,
    target_hour_utc TEXT NOT NULL,
    delivery_date_local TEXT NOT NULL,
    forecast_run_time_utc TEXT NOT NULL,
    decision_cutoff_utc TEXT NOT NULL,
    forecast_model TEXT NOT NULL,
    forecast_lead_hours REAL NOT NULL,
    temperature_2m_c REAL,
    relative_humidity_2m_pct REAL,
    wind_speed_10m_ms REAL,
    wind_gusts_10m_ms REAL,
    cloud_cover_pct REAL,
    shortwave_radiation_wm2 REAL,
    precipitation_mm REAL,
    availability_assumption TEXT NOT NULL,
    source_dataset TEXT NOT NULL,
    source_file_id INTEGER NOT NULL,
    source_record_id INTEGER NOT NULL,
    PRIMARY KEY (city, target_hour_utc),
    CHECK (target_hour_utc GLOB '????-??-??T??:??:??Z'),
    CHECK (forecast_run_time_utc GLOB '????-??-??T??:??:??Z'),
    CHECK (decision_cutoff_utc GLOB '????-??-??T??:??:??Z'),
    CHECK (forecast_run_time_utc < decision_cutoff_utc),
    CHECK (forecast_lead_hours > 0)
) WITHOUT ROWID;

CREATE TABLE clean_gas_daily (
    observation_date TEXT NOT NULL,
    vintage_start_date TEXT NOT NULL,
    vintage_end_date TEXT NOT NULL,
    henry_hub_usd_per_mmbtu REAL,
    is_missing INTEGER NOT NULL,
    source_file_id INTEGER NOT NULL,
    source_record_id INTEGER NOT NULL,
    PRIMARY KEY (observation_date, vintage_start_date),
    CHECK (is_missing IN (0, 1))
) WITHOUT ROWID;

CREATE TABLE clean_load_forecast (
    target_time_utc TEXT NOT NULL,
    interval_end_utc TEXT NOT NULL,
    publish_time_utc TEXT NOT NULL,
    load_forecast_mw REAL NOT NULL,
    source_file_id INTEGER NOT NULL,
    source_record_id INTEGER NOT NULL,
    PRIMARY KEY (target_time_utc, publish_time_utc),
    CHECK (target_time_utc GLOB '????-??-??T??:??:??Z'),
    CHECK (publish_time_utc GLOB '????-??-??T??:??:??Z')
) WITHOUT ROWID;

CREATE TABLE feature_weather_hourly (
    target_hour_utc TEXT PRIMARY KEY,
    delivery_date_local TEXT NOT NULL,
    forecast_run_time_utc TEXT NOT NULL,
    decision_cutoff_utc TEXT NOT NULL,
    forecast_model TEXT NOT NULL,
    forecast_lead_hours REAL NOT NULL,
    dfw_city_count INTEGER NOT NULL,
    temperature_dfw_mean_c REAL NOT NULL,
    temperature_dfw_min_c REAL NOT NULL,
    temperature_dfw_max_c REAL NOT NULL,
    temperature_wichita_c REAL NOT NULL,
    temperature_wichita_minus_dfw_c REAL NOT NULL,
    humidity_dfw_mean_pct REAL NOT NULL,
    humidity_wichita_pct REAL NOT NULL,
    humidity_wichita_minus_dfw_pct REAL NOT NULL,
    wind_speed_dfw_mean_ms REAL NOT NULL,
    wind_speed_dfw_max_ms REAL NOT NULL,
    wind_speed_wichita_ms REAL NOT NULL,
    wind_speed_wichita_minus_dfw_ms REAL NOT NULL,
    wind_gust_dfw_mean_ms REAL NOT NULL,
    wind_gust_dfw_max_ms REAL NOT NULL,
    wind_gust_wichita_ms REAL NOT NULL,
    wind_gust_wichita_minus_dfw_ms REAL NOT NULL,
    cloud_cover_dfw_mean_pct REAL NOT NULL,
    cloud_cover_wichita_pct REAL NOT NULL,
    cloud_cover_wichita_minus_dfw_pct REAL NOT NULL,
    radiation_dfw_mean_wm2 REAL NOT NULL,
    radiation_wichita_wm2 REAL NOT NULL,
    radiation_wichita_minus_dfw_wm2 REAL NOT NULL,
    precipitation_dfw_mean_mm REAL NOT NULL,
    precipitation_dfw_max_mm REAL NOT NULL,
    precipitation_wichita_mm REAL NOT NULL,
    precipitation_wichita_minus_dfw_mm REAL NOT NULL,
    north_temperature_min_c REAL NOT NULL,
    north_temperature_max_c REAL NOT NULL,
    north_wind_gust_max_ms REAL NOT NULL,
    north_precipitation_max_mm REAL NOT NULL,
    freezing_city_count INTEGER NOT NULL,
    extreme_heat_city_count INTEGER NOT NULL,
    high_wind_city_count INTEGER NOT NULL,
    rainy_city_count INTEGER NOT NULL,
    availability_assumption TEXT NOT NULL,
    CHECK (dfw_city_count = 5),
    CHECK (target_hour_utc GLOB '????-??-??T??:??:??Z'),
    CHECK (forecast_run_time_utc GLOB '????-??-??T??:??:??Z'),
    CHECK (decision_cutoff_utc GLOB '????-??-??T??:??:??Z'),
    CHECK (forecast_run_time_utc < decision_cutoff_utc)
) WITHOUT ROWID;

CREATE TABLE quality_check_results (
    check_name TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    observed_value TEXT NOT NULL,
    expected_value TEXT NOT NULL,
    details TEXT NOT NULL,
    CHECK (status IN ('PASS', 'WARN'))
) WITHOUT ROWID;

CREATE TABLE feature_time_hourly (
    delivery_hour_utc TEXT PRIMARY KEY,
    decision_time_utc TEXT NOT NULL,
    delivery_date_local TEXT NOT NULL,
    delivery_time_local TEXT NOT NULL,
    decision_date_local TEXT NOT NULL,
    ercot_local_hour INTEGER NOT NULL,
    ercot_local_day_of_week INTEGER NOT NULL,
    ercot_local_month INTEGER NOT NULL,
    is_weekend INTEGER NOT NULL,
    is_dst INTEGER NOT NULL,
    CHECK (delivery_hour_utc GLOB '????-??-??T??:??:??Z'),
    CHECK (decision_time_utc GLOB '????-??-??T??:??:??Z'),
    CHECK (is_weekend IN (0, 1)),
    CHECK (is_dst IN (0, 1))
) WITHOUT ROWID;

CREATE TABLE feature_load_da_hourly (
    delivery_hour_utc TEXT PRIMARY KEY,
    decision_time_utc TEXT NOT NULL,
    load_forecast_mw REAL NOT NULL,
    interval_count INTEGER NOT NULL,
    is_complete_hour INTEGER NOT NULL,
    latest_publish_time_utc TEXT NOT NULL,
    CHECK (delivery_hour_utc GLOB '????-??-??T??:??:??Z'),
    CHECK (decision_time_utc GLOB '????-??-??T??:??:??Z'),
    CHECK (latest_publish_time_utc GLOB '????-??-??T??:??:??Z'),
    CHECK (is_complete_hour IN (0, 1))
) WITHOUT ROWID;

CREATE TABLE feature_gas_da_daily (
    decision_date_local TEXT PRIMARY KEY,
    gas_observation_date TEXT,
    gas_price_usd_per_mmbtu REAL,
    is_forward_filled INTEGER NOT NULL,
    gas_availability_assumption TEXT NOT NULL,
    CHECK (is_forward_filled IN (0, 1))
) WITHOUT ROWID;

CREATE TABLE model_split_assignments (
    delivery_hour_utc TEXT PRIMARY KEY,
    split_name TEXT NOT NULL,
    CHECK (split_name IN ('train', 'validation', 'test'))
) WITHOUT ROWID;

CREATE INDEX idx_clean_weather_target
    ON clean_weather_hourly(target_hour_utc, city);
CREATE INDEX idx_clean_gas_vintage
    ON clean_gas_daily(vintage_start_date, observation_date);
CREATE INDEX idx_clean_load_publish
    ON clean_load_forecast(publish_time_utc, target_time_utc);
CREATE INDEX idx_clean_price_complete
    ON clean_price_hourly(is_label_complete, delivery_hour_utc);

CREATE VIEW vw_complete_price_labels AS
SELECT *
FROM clean_price_hourly
WHERE is_label_complete = 1;

CREATE VIEW vw_model_price_weather_hourly AS
SELECT
    p.delivery_hour_utc,
    p.location,
    p.da_price_usd_per_mwh,
    p.rt_price_usd_per_mwh,
    p.spread_usd_per_mwh,
    p.spread_sign,
    p.rt_above_da,
    w.*
FROM vw_complete_price_labels AS p
JOIN feature_weather_hourly AS w
  ON w.target_hour_utc = p.delivery_hour_utc;

CREATE VIEW vw_model_dataset_hourly AS
SELECT
    p.delivery_hour_utc,
    p.location,
    s.split_name,
    t.decision_time_utc,
    t.delivery_date_local,
    t.delivery_time_local,
    t.decision_date_local,
    t.ercot_local_hour,
    t.ercot_local_day_of_week,
    t.ercot_local_month,
    t.is_weekend,
    t.is_dst,
    p.da_price_usd_per_mwh,
    p.rt_price_usd_per_mwh,
    p.spread_usd_per_mwh,
    p.spread_sign,
    p.rt_above_da,
    w.*,
    l.load_forecast_mw,
    l.interval_count AS load_interval_count,
    l.is_complete_hour AS load_is_complete_hour,
    l.latest_publish_time_utc AS load_latest_publish_time_utc,
    g.gas_observation_date,
    g.gas_price_usd_per_mmbtu,
    g.is_forward_filled AS gas_is_forward_filled,
    g.gas_availability_assumption
FROM vw_complete_price_labels AS p
JOIN model_split_assignments AS s
  ON s.delivery_hour_utc = p.delivery_hour_utc
JOIN feature_time_hourly AS t
  ON t.delivery_hour_utc = p.delivery_hour_utc
JOIN feature_weather_hourly AS w
  ON w.target_hour_utc = p.delivery_hour_utc
LEFT JOIN feature_load_da_hourly AS l
  ON l.delivery_hour_utc = p.delivery_hour_utc
LEFT JOIN feature_gas_da_daily AS g
  ON g.decision_date_local = t.decision_date_local;
"""


PRICE_INSERT_SQL = {
    "clean_da_price_hourly": """
        WITH ranked AS (
            SELECT
                strftime('%Y-%m-%dT%H:%M:%SZ', r.interval_start_utc)
                    AS interval_start_utc,
                strftime(
                    '%Y-%m-%dT%H:%M:%SZ',
                    json_extract(r.record_json, '$.interval_end_utc')
                ) AS interval_end_utc,
                r.location,
                json_extract(r.record_json, '$.location_type')
                    AS location_type,
                json_extract(r.record_json, '$.market') AS market,
                CAST(json_extract(r.record_json, '$.spp') AS REAL) AS price,
                f.file_id AS source_file_id,
                r.record_id AS source_record_id,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        strftime(
                            '%Y-%m-%dT%H:%M:%SZ', r.interval_start_utc
                        ),
                        r.location
                    ORDER BY
                        COALESCE(f.collected_at_utc, '') DESC,
                        f.imported_at_utc DESC,
                        r.record_id DESC
                ) AS duplicate_rank
            FROM raw.raw_records AS r
            JOIN raw.raw_files AS f ON f.file_id = r.file_id
            WHERE f.dataset = 'ercot_spp_day_ahead_hourly'
              AND r.interval_start_utc IS NOT NULL
              AND r.location IS NOT NULL
              AND json_extract(r.record_json, '$.spp') IS NOT NULL
        )
        INSERT INTO clean_da_price_hourly
        SELECT
            interval_start_utc,
            interval_end_utc,
            location,
            location_type,
            market,
            price,
            source_file_id,
            source_record_id
        FROM ranked
        WHERE duplicate_rank = 1
          AND interval_start_utc IS NOT NULL
          AND interval_end_utc IS NOT NULL
    """,
    "clean_rt_price_15min": """
        WITH ranked AS (
            SELECT
                strftime('%Y-%m-%dT%H:%M:%SZ', r.interval_start_utc)
                    AS interval_start_utc,
                strftime(
                    '%Y-%m-%dT%H:%M:%SZ',
                    json_extract(r.record_json, '$.interval_end_utc')
                ) AS interval_end_utc,
                r.location,
                json_extract(r.record_json, '$.location_type')
                    AS location_type,
                json_extract(r.record_json, '$.market') AS market,
                CAST(json_extract(r.record_json, '$.spp') AS REAL) AS price,
                f.file_id AS source_file_id,
                r.record_id AS source_record_id,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        strftime(
                            '%Y-%m-%dT%H:%M:%SZ', r.interval_start_utc
                        ),
                        r.location
                    ORDER BY
                        COALESCE(f.collected_at_utc, '') DESC,
                        f.imported_at_utc DESC,
                        r.record_id DESC
                ) AS duplicate_rank
            FROM raw.raw_records AS r
            JOIN raw.raw_files AS f ON f.file_id = r.file_id
            WHERE f.dataset = 'ercot_spp_real_time_15_min'
              AND r.interval_start_utc IS NOT NULL
              AND r.location IS NOT NULL
              AND json_extract(r.record_json, '$.spp') IS NOT NULL
        )
        INSERT INTO clean_rt_price_15min
        SELECT
            interval_start_utc,
            interval_end_utc,
            location,
            location_type,
            market,
            price,
            source_file_id,
            source_record_id
        FROM ranked
        WHERE duplicate_rank = 1
          AND interval_start_utc IS NOT NULL
          AND interval_end_utc IS NOT NULL
    """,
}


WEATHER_INSERT_SQL = """
WITH ranked AS (
    SELECT
        CASE r.location
            WHEN 'Fort_Worth' THEN 'Fort Worth'
            WHEN 'Wichita_Falls' THEN 'Wichita Falls'
            ELSE r.location
        END AS city,
        strftime('%Y-%m-%dT%H:%M:%SZ', r.interval_start_utc)
            AS target_hour_utc,
        json_extract(r.record_json, '$.delivery_date_local')
            AS delivery_date_local,
        json_extract(r.record_json, '$.forecast_run_time_utc')
            AS forecast_run_time_utc,
        json_extract(r.record_json, '$.decision_cutoff_utc')
            AS decision_cutoff_utc,
        json_extract(r.record_json, '$.forecast_model')
            AS forecast_model,
        ROUND(
            (julianday(strftime('%Y-%m-%dT%H:%M:%SZ', r.interval_start_utc))
             - julianday(json_extract(r.record_json, '$.forecast_run_time_utc')))
            * 24.0,
            3
        ) AS forecast_lead_hours,
        CAST(json_extract(r.record_json, '$.temperature_2m') AS REAL)
            AS temperature_2m_c,
        CAST(json_extract(r.record_json, '$.relative_humidity_2m') AS REAL)
            AS relative_humidity_2m_pct,
        CAST(json_extract(r.record_json, '$.wind_speed_10m') AS REAL)
            AS wind_speed_10m_ms,
        CAST(json_extract(r.record_json, '$.wind_gusts_10m') AS REAL)
            AS wind_gusts_10m_ms,
        CAST(json_extract(r.record_json, '$.cloud_cover') AS REAL)
            AS cloud_cover_pct,
        CAST(json_extract(r.record_json, '$.shortwave_radiation') AS REAL)
            AS shortwave_radiation_wm2,
        CAST(json_extract(r.record_json, '$.precipitation') AS REAL)
            AS precipitation_mm,
        COALESCE(
            json_extract(r.record_json, '$.availability_assumption'),
            'single_run_initialization_before_pre_dam_cutoff'
        ) AS availability_assumption,
        f.dataset AS source_dataset,
        f.file_id AS source_file_id,
        r.record_id AS source_record_id,
        ROW_NUMBER() OVER (
            PARTITION BY
                CASE r.location
                    WHEN 'Fort_Worth' THEN 'Fort Worth'
                    WHEN 'Wichita_Falls' THEN 'Wichita Falls'
                    ELSE r.location
                END,
                strftime('%Y-%m-%dT%H:%M:%SZ', r.interval_start_utc)
            ORDER BY
                COALESCE(json_extract(r.record_json, '$.forecast_run_time_utc'), '') DESC,
                COALESCE(f.collected_at_utc, '') DESC,
                f.imported_at_utc DESC,
                r.record_id DESC
        ) AS duplicate_rank
    FROM raw.raw_records AS r
    JOIN raw.raw_files AS f ON f.file_id = r.file_id
    WHERE f.source = 'openmeteo'
      AND f.dataset = 'previous-runs-hybrid'
      AND r.location IN (
          'Dallas', 'Fort_Worth', 'Denton', 'McKinney',
          'Arlington', 'Wichita_Falls',
          'Fort Worth', 'Wichita Falls'
      )
      AND r.interval_start_utc IS NOT NULL
      AND json_extract(r.record_json, '$.forecast_run_time_utc') IS NOT NULL
      AND json_extract(r.record_json, '$.decision_cutoff_utc') IS NOT NULL
      AND json_extract(r.record_json, '$.temperature_2m') IS NOT NULL
      AND json_extract(r.record_json, '$.relative_humidity_2m') IS NOT NULL
      AND json_extract(r.record_json, '$.wind_speed_10m') IS NOT NULL
      AND json_extract(r.record_json, '$.wind_gusts_10m') IS NOT NULL
      AND json_extract(r.record_json, '$.cloud_cover') IS NOT NULL
      AND json_extract(r.record_json, '$.shortwave_radiation') IS NOT NULL
      AND json_extract(r.record_json, '$.precipitation') IS NOT NULL
)
INSERT INTO clean_weather_hourly
SELECT
    city,
    target_hour_utc,
    delivery_date_local,
    forecast_run_time_utc,
    decision_cutoff_utc,
    forecast_model,
    forecast_lead_hours,
    temperature_2m_c,
    relative_humidity_2m_pct,
    wind_speed_10m_ms,
    wind_gusts_10m_ms,
    cloud_cover_pct,
    shortwave_radiation_wm2,
    precipitation_mm,
    availability_assumption,
    source_dataset,
    source_file_id,
    source_record_id
FROM ranked
WHERE duplicate_rank = 1
  AND target_hour_utc IS NOT NULL
  AND delivery_date_local IS NOT NULL
  AND forecast_run_time_utc IS NOT NULL
  AND decision_cutoff_utc IS NOT NULL
  AND forecast_model IS NOT NULL
  AND forecast_lead_hours > 0
  AND forecast_run_time_utc < decision_cutoff_utc
  AND temperature_2m_c IS NOT NULL
  AND relative_humidity_2m_pct IS NOT NULL
  AND wind_speed_10m_ms IS NOT NULL
  AND wind_gusts_10m_ms IS NOT NULL
  AND cloud_cover_pct IS NOT NULL
  AND shortwave_radiation_wm2 IS NOT NULL
  AND precipitation_mm IS NOT NULL
"""


WEATHER_FEATURE_INSERT_SQL = """
WITH dfw AS (
    SELECT
        target_hour_utc,
        MIN(delivery_date_local) AS delivery_date_local,
        MIN(forecast_run_time_utc) AS forecast_run_time_utc,
        MIN(decision_cutoff_utc) AS decision_cutoff_utc,
        MIN(forecast_model) AS forecast_model,
        MIN(forecast_lead_hours) AS forecast_lead_hours,
        MIN(availability_assumption) AS availability_assumption,
        COUNT(*) AS city_count,
        AVG(temperature_2m_c) AS temperature_mean,
        MIN(temperature_2m_c) AS temperature_min,
        MAX(temperature_2m_c) AS temperature_max,
        AVG(relative_humidity_2m_pct) AS humidity_mean,
        AVG(wind_speed_10m_ms) AS wind_speed_mean,
        MAX(wind_speed_10m_ms) AS wind_speed_max,
        AVG(wind_gusts_10m_ms) AS wind_gust_mean,
        MAX(wind_gusts_10m_ms) AS wind_gust_max,
        AVG(cloud_cover_pct) AS cloud_cover_mean,
        AVG(shortwave_radiation_wm2) AS radiation_mean,
        AVG(precipitation_mm) AS precipitation_mean,
        MAX(precipitation_mm) AS precipitation_max
    FROM clean_weather_hourly
    WHERE city <> 'Wichita Falls'
    GROUP BY target_hour_utc
    HAVING COUNT(*) = 5
),
wichita AS (
    SELECT *
    FROM clean_weather_hourly
    WHERE city = 'Wichita Falls'
),
north_extremes AS (
    SELECT
        target_hour_utc,
        MIN(temperature_2m_c) AS temperature_min,
        MAX(temperature_2m_c) AS temperature_max,
        MAX(wind_gusts_10m_ms) AS wind_gust_max,
        MAX(precipitation_mm) AS precipitation_max,
        SUM(CASE WHEN temperature_2m_c <= 0 THEN 1 ELSE 0 END)
            AS freezing_city_count,
        SUM(CASE WHEN temperature_2m_c >= 35 THEN 1 ELSE 0 END)
            AS extreme_heat_city_count,
        SUM(CASE WHEN wind_gusts_10m_ms >= 15 THEN 1 ELSE 0 END)
            AS high_wind_city_count,
        SUM(CASE WHEN precipitation_mm > 0.1 THEN 1 ELSE 0 END)
            AS rainy_city_count
    FROM clean_weather_hourly
    GROUP BY target_hour_utc
    HAVING COUNT(*) = 6
)
INSERT INTO feature_weather_hourly
SELECT
    dfw.target_hour_utc,
    dfw.delivery_date_local,
    dfw.forecast_run_time_utc,
    dfw.decision_cutoff_utc,
    dfw.forecast_model,
    dfw.forecast_lead_hours,
    dfw.city_count,
    dfw.temperature_mean,
    dfw.temperature_min,
    dfw.temperature_max,
    wichita.temperature_2m_c,
    wichita.temperature_2m_c - dfw.temperature_mean,
    dfw.humidity_mean,
    wichita.relative_humidity_2m_pct,
    wichita.relative_humidity_2m_pct - dfw.humidity_mean,
    dfw.wind_speed_mean,
    dfw.wind_speed_max,
    wichita.wind_speed_10m_ms,
    wichita.wind_speed_10m_ms - dfw.wind_speed_mean,
    dfw.wind_gust_mean,
    dfw.wind_gust_max,
    wichita.wind_gusts_10m_ms,
    wichita.wind_gusts_10m_ms - dfw.wind_gust_mean,
    dfw.cloud_cover_mean,
    wichita.cloud_cover_pct,
    wichita.cloud_cover_pct - dfw.cloud_cover_mean,
    dfw.radiation_mean,
    wichita.shortwave_radiation_wm2,
    wichita.shortwave_radiation_wm2 - dfw.radiation_mean,
    dfw.precipitation_mean,
    dfw.precipitation_max,
    wichita.precipitation_mm,
    wichita.precipitation_mm - dfw.precipitation_mean,
    north_extremes.temperature_min,
    north_extremes.temperature_max,
    north_extremes.wind_gust_max,
    north_extremes.precipitation_max,
    north_extremes.freezing_city_count,
    north_extremes.extreme_heat_city_count,
    north_extremes.high_wind_city_count,
    north_extremes.rainy_city_count,
    dfw.availability_assumption
FROM dfw
JOIN wichita ON wichita.target_hour_utc = dfw.target_hour_utc
JOIN north_extremes
  ON north_extremes.target_hour_utc = dfw.target_hour_utc
"""


GAS_INSERT_SQL = """
WITH ranked AS (
    SELECT
        r.observation_date,
        json_extract(r.record_json, '$.realtime_start')
            AS vintage_start_date,
        json_extract(r.record_json, '$.realtime_end')
            AS vintage_end_date,
        CASE
            WHEN json_extract(r.record_json, '$.value') IS NULL
              OR TRIM(CAST(json_extract(r.record_json, '$.value') AS TEXT))
                 IN ('', '.') THEN NULL
            ELSE CAST(json_extract(r.record_json, '$.value') AS REAL)
        END AS price,
        CASE
            WHEN json_extract(r.record_json, '$.value') IS NULL
              OR TRIM(CAST(json_extract(r.record_json, '$.value') AS TEXT))
                 IN ('', '.') THEN 1
            ELSE 0
        END AS is_missing,
        f.file_id AS source_file_id,
        r.record_id AS source_record_id,
        ROW_NUMBER() OVER (
            PARTITION BY
                r.observation_date,
                json_extract(r.record_json, '$.realtime_start')
            ORDER BY
                COALESCE(f.collected_at_utc, '') DESC,
                f.imported_at_utc DESC,
                r.record_id DESC
        ) AS duplicate_rank
    FROM raw.raw_records AS r
    JOIN raw.raw_files AS f ON f.file_id = r.file_id
    WHERE f.source = 'fred'
      AND f.dataset = 'DHHNGSP'
      AND r.observation_date IS NOT NULL
)
INSERT INTO clean_gas_daily
SELECT
    observation_date,
    vintage_start_date,
    vintage_end_date,
    price,
    is_missing,
    source_file_id,
    source_record_id
FROM ranked
WHERE duplicate_rank = 1
  AND vintage_start_date IS NOT NULL
  AND vintage_end_date IS NOT NULL
"""


LOAD_INSERT_SQL = """
WITH ranked AS (
    SELECT
        strftime('%Y-%m-%dT%H:%M:%SZ', r.interval_start_utc)
            AS target_time_utc,
        strftime(
            '%Y-%m-%dT%H:%M:%SZ',
            json_extract(r.record_json, '$.interval_end_utc')
        ) AS interval_end_utc,
        strftime('%Y-%m-%dT%H:%M:%SZ', r.publish_time_utc)
            AS publish_time_utc,
        CAST(json_extract(r.record_json, '$.load_forecast') AS REAL)
            AS load_forecast_mw,
        f.file_id AS source_file_id,
        r.record_id AS source_record_id,
        ROW_NUMBER() OVER (
            PARTITION BY
                strftime('%Y-%m-%dT%H:%M:%SZ', r.interval_start_utc),
                strftime('%Y-%m-%dT%H:%M:%SZ', r.publish_time_utc)
            ORDER BY
                COALESCE(f.collected_at_utc, '') DESC,
                f.imported_at_utc DESC,
                r.record_id DESC
        ) AS duplicate_rank
    FROM raw.raw_records AS r
    JOIN raw.raw_files AS f ON f.file_id = r.file_id
    WHERE f.dataset = 'ercot_seven_day_load_forecast'
      AND r.interval_start_utc IS NOT NULL
      AND r.publish_time_utc IS NOT NULL
      AND json_extract(r.record_json, '$.load_forecast') IS NOT NULL
)
INSERT INTO clean_load_forecast
SELECT
    target_time_utc,
    interval_end_utc,
    publish_time_utc,
    load_forecast_mw,
    source_file_id,
    source_record_id
FROM ranked
WHERE duplicate_rank = 1
  AND target_time_utc IS NOT NULL
  AND interval_end_utc IS NOT NULL
  AND publish_time_utc IS NOT NULL
"""


LOAD_FEATURE_INSERT_SQL = """
WITH candidates AS (
    SELECT
        target_time_utc,
        publish_time_utc,
        load_forecast_mw,
        decision_cutoff_utc(target_time_utc) AS decision_time_utc,
        ROW_NUMBER() OVER (
            PARTITION BY target_time_utc
            ORDER BY publish_time_utc DESC, source_record_id DESC
        ) AS publish_rank
    FROM clean_load_forecast
    WHERE publish_time_utc <= decision_cutoff_utc(target_time_utc)
),
selected AS (
    SELECT *
    FROM candidates
    WHERE publish_rank = 1
)
INSERT INTO feature_load_da_hourly
SELECT
    strftime('%Y-%m-%dT%H:00:00Z', target_time_utc),
    MIN(decision_time_utc),
    AVG(load_forecast_mw),
    COUNT(*),
    CASE WHEN COUNT(*) = 12 THEN 1 ELSE 0 END,
    MAX(publish_time_utc)
FROM selected
GROUP BY strftime('%Y-%m-%dT%H:00:00Z', target_time_utc)
"""


DERIVED_PRICE_SQL = """
INSERT INTO clean_rt_price_hourly
SELECT
    strftime('%Y-%m-%dT%H:00:00Z', interval_start_utc),
    location,
    AVG(rt_price_usd_per_mwh),
    MIN(rt_price_usd_per_mwh),
    MAX(rt_price_usd_per_mwh),
    COUNT(*),
    CASE WHEN COUNT(*) = 4 THEN 1 ELSE 0 END
FROM clean_rt_price_15min
GROUP BY strftime('%Y-%m-%dT%H:00:00Z', interval_start_utc), location;

WITH all_keys AS (
    SELECT delivery_hour_utc, location FROM clean_da_price_hourly
    UNION
    SELECT delivery_hour_utc, location FROM clean_rt_price_hourly
)
INSERT INTO clean_price_hourly
SELECT
    k.delivery_hour_utc,
    k.location,
    da.da_price_usd_per_mwh,
    rt.rt_price_usd_per_mwh,
    CASE
        WHEN da.da_price_usd_per_mwh IS NOT NULL
         AND rt.rt_price_usd_per_mwh IS NOT NULL
        THEN rt.rt_price_usd_per_mwh - da.da_price_usd_per_mwh
    END,
    CASE
        WHEN da.da_price_usd_per_mwh IS NULL
          OR rt.rt_price_usd_per_mwh IS NULL THEN NULL
        WHEN rt.rt_price_usd_per_mwh > da.da_price_usd_per_mwh THEN 1
        WHEN rt.rt_price_usd_per_mwh < da.da_price_usd_per_mwh THEN -1
        ELSE 0
    END,
    CASE
        WHEN da.da_price_usd_per_mwh IS NULL
          OR rt.rt_price_usd_per_mwh IS NULL THEN NULL
        WHEN rt.rt_price_usd_per_mwh > da.da_price_usd_per_mwh THEN 1
        ELSE 0
    END,
    rt.interval_count,
    CASE
        WHEN da.da_price_usd_per_mwh IS NOT NULL
         AND rt.rt_price_usd_per_mwh IS NOT NULL
         AND rt.is_complete_hour = 1 THEN 1
        ELSE 0
    END
FROM all_keys AS k
LEFT JOIN clean_da_price_hourly AS da
  ON da.delivery_hour_utc = k.delivery_hour_utc
 AND da.location = k.location
LEFT JOIN clean_rt_price_hourly AS rt
  ON rt.delivery_hour_utc = k.delivery_hour_utc
 AND rt.location = k.location;
"""


TABLES = (
    "clean_da_price_hourly",
    "clean_rt_price_15min",
    "clean_rt_price_hourly",
    "clean_price_hourly",
    "clean_weather_hourly",
    "clean_gas_daily",
    "clean_load_forecast",
    "feature_weather_hourly",
    "feature_time_hourly",
    "feature_load_da_hourly",
    "feature_gas_da_daily",
    "model_split_assignments",
    "quality_check_results",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def _decision_cutoff_utc(target_time_utc: str) -> str:
    delivery_local = _parse_utc(target_time_utc).astimezone(ERCOT_TIMEZONE)
    cutoff_local = datetime.combine(
        delivery_local.date() - timedelta(days=1),
        time(9, 55),
        tzinfo=ERCOT_TIMEZONE,
    )
    return cutoff_local.astimezone(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _register_sql_functions(connection: sqlite3.Connection) -> None:
    connection.create_function(
        "decision_cutoff_utc", 1, _decision_cutoff_utc
    )


def _build_time_features(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        SELECT DISTINCT delivery_hour_utc
        FROM clean_price_hourly
        ORDER BY delivery_hour_utc
        """
    )
    output: list[tuple[Any, ...]] = []
    for (delivery_hour_utc,) in rows:
        delivery_utc = _parse_utc(delivery_hour_utc)
        delivery_local = delivery_utc.astimezone(ERCOT_TIMEZONE)
        decision_time_utc = _decision_cutoff_utc(delivery_hour_utc)
        output.append(
            (
                delivery_hour_utc,
                decision_time_utc,
                delivery_local.date().isoformat(),
                delivery_local.isoformat(),
                (delivery_local.date() - timedelta(days=1)).isoformat(),
                delivery_local.hour,
                delivery_local.weekday(),
                delivery_local.month,
                int(delivery_local.weekday() >= 5),
                int(bool(delivery_local.dst())),
            )
        )
    connection.executemany(
        """
        INSERT INTO feature_time_hourly(
            delivery_hour_utc, decision_time_utc, delivery_date_local,
            delivery_time_local, decision_date_local, ercot_local_hour,
            ercot_local_day_of_week, ercot_local_month, is_weekend, is_dst
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        output,
    )


def _next_business_day(value: date) -> date:
    candidate = value + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def _build_gas_features(connection: sqlite3.Connection) -> None:
    snapshot_date = connection.execute(
        "SELECT MAX(vintage_start_date) FROM clean_gas_daily"
    ).fetchone()[0]
    gas_rows = connection.execute(
        """
        SELECT observation_date, henry_hub_usd_per_mmbtu, is_missing
        FROM clean_gas_daily
        WHERE vintage_start_date = ?
        ORDER BY observation_date
        """,
        (snapshot_date,),
    ).fetchall()
    observations = [
        (
            date.fromisoformat(observation_date),
            price,
            is_missing,
            _next_business_day(date.fromisoformat(observation_date)),
        )
        for observation_date, price, is_missing in gas_rows
    ]
    decision_dates = [
        row[0]
        for row in connection.execute(
            """
            SELECT DISTINCT decision_date_local
            FROM feature_time_hourly
            ORDER BY decision_date_local
            """
        )
    ]

    output: list[tuple[Any, ...]] = []
    current_price: float | None = None
    current_observation: date | None = None
    observation_index = 0
    for decision_date_text in decision_dates:
        decision_date = date.fromisoformat(decision_date_text)
        while (
            observation_index < len(observations)
            and observations[observation_index][3] <= decision_date
        ):
            observation_date, price, is_missing, _ = observations[
                observation_index
            ]
            if not is_missing and price is not None:
                current_price = float(price)
                current_observation = observation_date
            observation_index += 1

        output.append(
            (
                decision_date_text,
                current_observation.isoformat()
                if current_observation is not None
                else None,
                current_price,
                int(
                    current_observation is not None
                    and current_observation < decision_date
                ),
                "observation_date_plus_one_business_day_forward_fill",
            )
        )
    connection.executemany(
        """
        INSERT INTO feature_gas_da_daily(
            decision_date_local, gas_observation_date,
            gas_price_usd_per_mmbtu, is_forward_filled,
            gas_availability_assumption
        ) VALUES (?, ?, ?, ?, ?)
        """,
        output,
    )


def _build_split_assignments(connection: sqlite3.Connection) -> None:
    keys = [
        row[0]
        for row in connection.execute(
            """
            SELECT p.delivery_hour_utc
            FROM vw_complete_price_labels AS p
            JOIN feature_weather_hourly AS w
              ON w.target_hour_utc = p.delivery_hour_utc
            ORDER BY p.delivery_hour_utc
            """
        )
    ]
    if not keys:
        raise RuntimeError("No complete price-weather rows available for splits")
    train_end = int(len(keys) * 0.70)
    validation_end = int(len(keys) * 0.85)
    assignments = []
    for index, key in enumerate(keys):
        split = (
            "train"
            if index < train_end
            else "validation"
            if index < validation_end
            else "test"
        )
        assignments.append((key, split))
    connection.executemany(
        "INSERT INTO model_split_assignments(delivery_hour_utc, split_name) "
        "VALUES (?, ?)",
        assignments,
    )


def _attach_raw(connection: sqlite3.Connection, raw_database: Path) -> None:
    connection.execute(
        "ATTACH DATABASE ? AS raw", (str(raw_database.resolve()),)
    )
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM raw.sqlite_master WHERE type = 'table'"
        )
    }
    missing = {"raw_files", "raw_records"} - tables
    if missing:
        raise RuntimeError(
            "Raw database is missing required tables: "
            + ", ".join(sorted(missing))
        )


def _write_build_info(
    connection: sqlite3.Connection, raw_database: Path
) -> None:
    values = {
        "schema_version": str(SCHEMA_VERSION),
        "built_at_utc": _utc_now(),
        "raw_database": str(raw_database.resolve()),
        "canonical_timezone": "UTC",
        "weather_locations": json.dumps(
            WEATHER_LOCATIONS, ensure_ascii=True
        ),
        "weather_availability_assumption": (
            "openmeteo_previous_day1_local_hours_00_08_else_day2_before_pre_dam_cutoff"
        ),
        "weather_forecast_source": "openmeteo_previous_runs_hybrid",
        "weather_run_selection_rule": (
            "previous_day1_for_local_hours_00_08_else_previous_day2"
        ),
        "weather_extreme_thresholds": json.dumps(
            {
                "freezing_temperature_c_lte": 0,
                "extreme_heat_temperature_c_gte": 35,
                "high_wind_gust_ms_gte": 15,
                "rainy_precipitation_mm_gt": 0.1,
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        "gas_vintage_semantics": (
            "collection_snapshot_not_original_publication_time"
        ),
        "model_delivery_scope": (
            "price_weather_load_gas_with_documented_asof_assumptions"
        ),
        "decision_cutoff_rule": (
            "delivery_local_date_minus_one_day_09:55_America/Chicago"
        ),
        "time_split_rule": "chronological_70_15_15_train_validation_test",
        "gas_fill_rule": "forward_fill_only_after_next_business_day",
        "excluded_forecasts": "wind,solar",
    }
    connection.executemany(
        "INSERT INTO clean_build_info(key, value) VALUES (?, ?)",
        values.items(),
    )


def _write_quality_checks(connection: sqlite3.Connection) -> None:
    checks: list[tuple[str, str, str, str, str]] = []

    def add_check(
        name: str,
        observed: Any,
        expected: Any,
        details: str,
        *,
        passed: bool,
    ) -> None:
        checks.append(
            (
                name,
                "PASS" if passed else "WARN",
                str(observed),
                str(expected),
                details,
            )
        )

    cities = [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT city FROM clean_weather_hourly ORDER BY city"
        )
    ]
    expected_cities = sorted(WEATHER_LOCATIONS)
    add_check(
        "weather_location_set",
        ", ".join(cities),
        ", ".join(expected_cities),
        "Clean weather must contain exactly the six selected locations.",
        passed=cities == expected_cities,
    )

    duplicate_weather = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT city, target_hour_utc
            FROM clean_weather_hourly
            GROUP BY city, target_hour_utc
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    add_check(
        "weather_duplicate_city_hours",
        duplicate_weather,
        0,
        "Duplicate key is city + target_hour_utc.",
        passed=duplicate_weather == 0,
    )

    weather_after_cutoff = connection.execute(
        """
        SELECT COUNT(*) FROM clean_weather_hourly
        WHERE forecast_run_time_utc >= decision_cutoff_utc
        """
    ).fetchone()[0]
    add_check(
        "weather_run_before_pre_dam_cutoff",
        weather_after_cutoff,
        0,
        "Every weather run must be initialized before the 09:55 America/Chicago cutoff.",
        passed=weather_after_cutoff == 0,
    )

    invalid_weather_lead = connection.execute(
        """
        SELECT COUNT(*) FROM clean_weather_hourly
        WHERE forecast_lead_hours NOT IN (24.0, 48.0)
        """
    ).fetchone()[0]
    add_check(
        "weather_hybrid_lead_values",
        invalid_weather_lead,
        0,
        "Hybrid Previous Runs weather must use only 24-hour or 48-hour leads.",
        passed=invalid_weather_lead == 0,
    )

    expected_feature_hours = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT target_hour_utc
            FROM clean_weather_hourly
            GROUP BY target_hour_utc
            HAVING COUNT(*) = 6
        )
        """
    ).fetchone()[0]
    feature_hours = connection.execute(
        "SELECT COUNT(*) FROM feature_weather_hourly"
    ).fetchone()[0]
    add_check(
        "weather_feature_hour_coverage",
        feature_hours,
        expected_feature_hours,
        "Each feature row requires all six city-hour observations.",
        passed=feature_hours == expected_feature_hours,
    )

    incomplete_rt = connection.execute(
        """
        SELECT COUNT(*) FROM clean_rt_price_hourly
        WHERE is_complete_hour = 0
        """
    ).fetchone()[0]
    add_check(
        "incomplete_rt_hours",
        incomplete_rt,
        0,
        "Warning rows remain queryable and are excluded from label view.",
        passed=incomplete_rt == 0,
    )

    incomplete_labels = connection.execute(
        """
        SELECT COUNT(*) FROM clean_price_hourly
        WHERE is_label_complete = 0
        """
    ).fetchone()[0]
    add_check(
        "incomplete_price_labels",
        incomplete_labels,
        0,
        "Use vw_complete_price_labels for training and backtesting.",
        passed=incomplete_labels == 0,
    )

    complete_labels = connection.execute(
        "SELECT COUNT(*) FROM vw_complete_price_labels"
    ).fetchone()[0]
    model_price_weather = connection.execute(
        "SELECT COUNT(*) FROM vw_model_price_weather_hourly"
    ).fetchone()[0]
    add_check(
        "price_weather_join_coverage",
        model_price_weather,
        complete_labels,
        "A difference indicates complete price labels outside weather UTC coverage.",
        passed=model_price_weather == complete_labels,
    )

    load_feature_rows = connection.execute(
        """
        SELECT COUNT(*) FROM vw_model_dataset_hourly
        WHERE load_forecast_mw IS NOT NULL
        """
    ).fetchone()[0]
    add_check(
        "load_asof_feature_coverage",
        load_feature_rows,
        complete_labels,
        "Current load history has only one publish time per target and is not full-vintage coverage.",
        passed=load_feature_rows == complete_labels,
    )

    gas_feature_rows = connection.execute(
        """
        SELECT COUNT(*) FROM vw_model_dataset_hourly
        WHERE gas_price_usd_per_mmbtu IS NOT NULL
        """
    ).fetchone()[0]
    add_check(
        "gas_forward_fill_coverage",
        gas_feature_rows,
        complete_labels,
        "Initial hours remain NULL because forward fill never uses future observations.",
        passed=gas_feature_rows == complete_labels,
    )

    split_counts = {
        split: count
        for split, count in connection.execute(
            """
            SELECT split_name, COUNT(*)
            FROM model_split_assignments
            GROUP BY split_name ORDER BY split_name
            """
        )
    }
    split_total = sum(split_counts.values())
    add_check(
        "chronological_split_assignments",
        json.dumps(split_counts, sort_keys=True),
        complete_labels,
        "Rows are assigned in UTC order using 70% train, 15% validation, 15% test.",
        passed=(
            set(split_counts) == {"train", "validation", "test"}
            and split_total == complete_labels
        ),
    )

    gas_missing = connection.execute(
        "SELECT COUNT(*) FROM clean_gas_daily WHERE is_missing = 1"
    ).fetchone()[0]
    add_check(
        "gas_missing_values",
        gas_missing,
        0,
        "Missing FRED values remain NULL and require an explicit lag/fill rule.",
        passed=gas_missing == 0,
    )

    utc_checks = (
        ("clean_da_price_hourly", "delivery_hour_utc"),
        ("clean_rt_price_15min", "interval_start_utc"),
        ("clean_rt_price_hourly", "delivery_hour_utc"),
        ("clean_price_hourly", "delivery_hour_utc"),
        ("clean_weather_hourly", "target_hour_utc"),
        ("clean_weather_hourly", "forecast_run_time_utc"),
        ("clean_weather_hourly", "decision_cutoff_utc"),
        ("feature_weather_hourly", "target_hour_utc"),
        ("feature_weather_hourly", "forecast_run_time_utc"),
        ("feature_weather_hourly", "decision_cutoff_utc"),
        ("feature_time_hourly", "delivery_hour_utc"),
        ("feature_time_hourly", "decision_time_utc"),
        ("feature_load_da_hourly", "delivery_hour_utc"),
        ("feature_load_da_hourly", "decision_time_utc"),
        ("clean_load_forecast", "target_time_utc"),
        ("clean_load_forecast", "publish_time_utc"),
    )
    invalid_utc = sum(
        connection.execute(
            f"SELECT COUNT(*) FROM {table} "
            f"WHERE {column} NOT GLOB '????-??-??T??:??:??Z'"
        ).fetchone()[0]
        for table, column in utc_checks
    )
    add_check(
        "canonical_utc_timestamps",
        invalid_utc,
        0,
        "Timestamp columns must use YYYY-MM-DDTHH:MM:SSZ.",
        passed=invalid_utc == 0,
    )

    excluded_tables = connection.execute(
        """
        SELECT COUNT(*) FROM sqlite_master
        WHERE type = 'table'
          AND (name LIKE '%wind%' OR name LIKE '%solar%')
        """
    ).fetchone()[0]
    add_check(
        "wind_solar_tables_excluded",
        excluded_tables,
        0,
        "Wind and solar forecasts are intentionally excluded from schema v2.",
        passed=excluded_tables == 0,
    )

    connection.executemany(
        """
        INSERT INTO quality_check_results(
            check_name, status, observed_value, expected_value, details
        ) VALUES (?, ?, ?, ?, ?)
        """,
        checks,
    )


def _validate_clean_database(
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"Analytics database integrity check failed: {integrity}")

    cities = tuple(
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT city FROM clean_weather_hourly ORDER BY city"
        )
    )
    if cities != tuple(sorted(WEATHER_LOCATIONS)):
        raise RuntimeError(f"Unexpected clean weather locations: {cities}")

    duplicate_weather = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT city, target_hour_utc
            FROM clean_weather_hourly
            GROUP BY city, target_hour_utc
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    if duplicate_weather:
        raise RuntimeError("Duplicate city-hour rows remain in clean weather")

    invalid_utc = 0
    utc_checks = (
        ("clean_da_price_hourly", "delivery_hour_utc"),
        ("clean_rt_price_15min", "interval_start_utc"),
        ("clean_rt_price_hourly", "delivery_hour_utc"),
        ("clean_price_hourly", "delivery_hour_utc"),
        ("clean_weather_hourly", "target_hour_utc"),
        ("clean_weather_hourly", "forecast_run_time_utc"),
        ("clean_weather_hourly", "decision_cutoff_utc"),
        ("feature_weather_hourly", "target_hour_utc"),
        ("feature_weather_hourly", "forecast_run_time_utc"),
        ("feature_weather_hourly", "decision_cutoff_utc"),
        ("feature_time_hourly", "delivery_hour_utc"),
        ("feature_time_hourly", "decision_time_utc"),
        ("feature_load_da_hourly", "delivery_hour_utc"),
        ("feature_load_da_hourly", "decision_time_utc"),
        ("clean_load_forecast", "target_time_utc"),
        ("clean_load_forecast", "publish_time_utc"),
    )
    for table, column in utc_checks:
        invalid_utc += connection.execute(
            f"SELECT COUNT(*) FROM {table} "
            f"WHERE {column} NOT GLOB '????-??-??T??:??:??Z'"
        ).fetchone()[0]
    if invalid_utc:
        raise RuntimeError(f"Found {invalid_utc} non-canonical UTC timestamps")

    return {
        "integrity_check": integrity,
        "weather_locations": list(cities),
        "table_rows": {
            table: connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            for table in TABLES
        },
        "quality_status_counts": {
            status: count
            for status, count in connection.execute(
                """
                SELECT status, COUNT(*)
                FROM quality_check_results
                GROUP BY status ORDER BY status
                """
            )
        },
    }


def build_clean_database(
    raw_database: Path = DEFAULT_RAW_DATABASE,
    analytics_database: Path = DEFAULT_ANALYTICS_DATABASE,
) -> dict[str, Any]:
    raw_database = raw_database.resolve()
    analytics_database = analytics_database.resolve()
    if not raw_database.exists():
        raise FileNotFoundError(f"Raw database not found: {raw_database}")
    if raw_database == analytics_database:
        raise ValueError("Raw and analytics database paths must be different")

    analytics_database.parent.mkdir(parents=True, exist_ok=True)
    temp_database = analytics_database.with_name(
        f"{analytics_database.stem}.tmp{analytics_database.suffix}"
    )
    temp_database.unlink(missing_ok=True)

    connection = sqlite3.connect(temp_database)
    try:
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA temp_store = FILE")
        _register_sql_functions(connection)
        _attach_raw(connection, raw_database)
        connection.executescript(SCHEMA_SQL)
        _write_build_info(connection, raw_database)
        connection.execute(PRICE_INSERT_SQL["clean_da_price_hourly"])
        connection.execute(PRICE_INSERT_SQL["clean_rt_price_15min"])
        connection.executescript(DERIVED_PRICE_SQL)
        connection.execute(WEATHER_INSERT_SQL)
        connection.execute(WEATHER_FEATURE_INSERT_SQL)
        connection.execute(GAS_INSERT_SQL)
        connection.execute(LOAD_INSERT_SQL)
        _build_time_features(connection)
        connection.execute(LOAD_FEATURE_INSERT_SQL)
        _build_gas_features(connection)
        _build_split_assignments(connection)
        _write_quality_checks(connection)
        connection.commit()
        connection.execute("ANALYZE")
        validation = _validate_clean_database(connection)
        connection.commit()
    except Exception:
        connection.close()
        temp_database.unlink(missing_ok=True)
        raise
    else:
        connection.close()

    os.replace(temp_database, analytics_database)
    validation["raw_database"] = str(raw_database)
    validation["analytics_database"] = str(analytics_database)
    validation["database_bytes"] = analytics_database.stat().st_size
    return validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build typed, deduplicated UTC clean tables from the raw SQLite "
            "database. Wind and solar forecasts are intentionally excluded."
        )
    )
    parser.add_argument(
        "--raw-database", type=Path, default=DEFAULT_RAW_DATABASE
    )
    parser.add_argument(
        "--analytics-database", type=Path, default=DEFAULT_ANALYTICS_DATABASE
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_clean_database(
        args.raw_database, args.analytics_database
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
