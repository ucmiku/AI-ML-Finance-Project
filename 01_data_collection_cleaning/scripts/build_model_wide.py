from __future__ import annotations

import argparse
import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from collectors.common import DATA_WORKSPACE


DEFAULT_ANALYTICS_DATABASE = (
    DATA_WORKSPACE / "interim" / "ercot_analytics.sqlite"
)
DEFAULT_EXPORT_CSV = (
    DATA_WORKSPACE / "processed" / "model_wide_hourly_2024_2026_final.csv"
)

MODEL_START_UTC = "2024-01-19T12:00:00Z"
# 2026-06-30 is in CDT (UTC-05), so this keeps the full local delivery day.
MODEL_END_EXCLUSIVE_UTC = "2026-07-01T05:00:00Z"

MODEL_TABLE = "model_wide_hourly_2024_2026"
SPLIT_TABLE = "model_split_assignments_2024_2026"
QC_TABLE = "model_wide_quality_check_results"


REQUIRED_OBJECTS = {
    "vw_complete_price_labels",
    "feature_time_hourly",
    "feature_weather_hourly",
    "feature_gas_da_daily",
    "feature_pre_dam_forecast_hourly",
}


MODEL_SQL = f"""
CREATE TABLE {MODEL_TABLE} AS
WITH forecast_merged AS (
    SELECT
        delivery_hour_utc,
        MAX(load_forecast_issue_time_utc)
            AS load_forecast_issue_time_utc,
        MAX(wind_forecast_issue_time_utc)
            AS wind_forecast_issue_time_utc,
        MAX(solar_forecast_issue_time_utc)
            AS solar_forecast_issue_time_utc,
        MAX(load_source_product_id) AS load_source_product_id,
        MAX(wind_source_product_id) AS wind_source_product_id,
        MAX(solar_source_product_id) AS solar_source_product_id,
        MAX(load_coast_mw) AS load_coast_mw,
        MAX(load_east_mw) AS load_east_mw,
        MAX(load_far_west_mw) AS load_far_west_mw,
        MAX(load_north_mw) AS load_north_mw,
        MAX(load_north_central_mw) AS load_north_central_mw,
        MAX(load_south_central_mw) AS load_south_central_mw,
        MAX(load_southern_mw) AS load_southern_mw,
        MAX(load_west_mw) AS load_west_mw,
        MAX(load_system_total_mw) AS load_system_total_mw,
        MAX(wind_stwpf_lz_north_mw) AS wind_stwpf_lz_north_mw,
        MAX(wind_stwpf_lz_south_houston_mw)
            AS wind_stwpf_lz_south_houston_mw,
        MAX(wind_stwpf_lz_west_mw) AS wind_stwpf_lz_west_mw,
        MAX(wind_stwpf_system_wide_mw)
            AS wind_stwpf_system_wide_mw,
        MAX(wind_wgrpp_lz_north_mw) AS wind_wgrpp_lz_north_mw,
        MAX(wind_wgrpp_lz_south_houston_mw)
            AS wind_wgrpp_lz_south_houston_mw,
        MAX(wind_wgrpp_lz_west_mw) AS wind_wgrpp_lz_west_mw,
        MAX(wind_wgrpp_system_wide_mw)
            AS wind_wgrpp_system_wide_mw,
        MAX(solar_pvgrpp_system_mw) AS solar_pvgrpp_system_mw,
        MAX(solar_stppf_system_mw) AS solar_stppf_system_mw,
        MAX(has_load_forecast) AS has_load_forecast,
        MAX(has_wind_forecast) AS has_wind_forecast,
        MAX(has_solar_forecast) AS has_solar_forecast,
        CASE
            WHEN MAX(has_load_forecast) = 1
             AND MAX(has_wind_forecast) = 1
             AND MAX(has_solar_forecast) = 1
            THEN 1 ELSE 0
        END AS has_all_three_forecasts,
        MAX(load_pre_dam_valid) AS load_pre_dam_valid,
        MAX(wind_pre_dam_valid) AS wind_pre_dam_valid,
        MAX(solar_pre_dam_valid) AS solar_pre_dam_valid,
        CASE
            WHEN MAX(load_pre_dam_valid) = 1
             AND MAX(wind_pre_dam_valid) = 1
             AND MAX(solar_pre_dam_valid) = 1
            THEN 1 ELSE 0
        END AS all_issue_times_pre_dam_valid
    FROM feature_pre_dam_forecast_hourly
    GROUP BY delivery_hour_utc
)
SELECT
    p.delivery_hour_utc,
    p.location,
    CAST(NULL AS TEXT) AS split_name,
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

    g.gas_observation_date,
    g.gas_price_usd_per_mmbtu,
    g.is_forward_filled AS gas_is_forward_filled,
    g.gas_availability_assumption,

    w.dfw_city_count,
    w.forecast_run_time_utc AS weather_forecast_run_time_utc,
    w.decision_cutoff_utc AS weather_decision_cutoff_utc,
    w.forecast_model AS weather_forecast_model,
    w.forecast_lead_hours AS weather_forecast_lead_hours,
    w.temperature_dfw_mean_c,
    w.temperature_dfw_min_c,
    w.temperature_dfw_max_c,
    w.temperature_wichita_c,
    w.temperature_wichita_minus_dfw_c,
    w.humidity_dfw_mean_pct,
    w.humidity_wichita_pct,
    w.humidity_wichita_minus_dfw_pct,
    w.wind_speed_dfw_mean_ms,
    w.wind_speed_dfw_max_ms,
    w.wind_speed_wichita_ms,
    w.wind_speed_wichita_minus_dfw_ms,
    w.wind_gust_dfw_mean_ms,
    w.wind_gust_dfw_max_ms,
    w.wind_gust_wichita_ms,
    w.wind_gust_wichita_minus_dfw_ms,
    w.cloud_cover_dfw_mean_pct,
    w.cloud_cover_wichita_pct,
    w.cloud_cover_wichita_minus_dfw_pct,
    w.radiation_dfw_mean_wm2,
    w.radiation_wichita_wm2,
    w.radiation_wichita_minus_dfw_wm2,
    w.precipitation_dfw_mean_mm,
    w.precipitation_dfw_max_mm,
    w.precipitation_wichita_mm,
    w.precipitation_wichita_minus_dfw_mm,
    w.north_temperature_min_c,
    w.north_temperature_max_c,
    w.north_wind_gust_max_ms,
    w.north_precipitation_max_mm,
    w.freezing_city_count,
    w.extreme_heat_city_count,
    w.high_wind_city_count,
    w.rainy_city_count,
    w.availability_assumption AS weather_availability_assumption,

    f.load_forecast_issue_time_utc,
    f.wind_forecast_issue_time_utc,
    f.solar_forecast_issue_time_utc,
    f.load_source_product_id,
    f.wind_source_product_id,
    f.solar_source_product_id,
    f.load_coast_mw,
    f.load_east_mw,
    f.load_far_west_mw,
    f.load_north_mw,
    f.load_north_central_mw,
    f.load_south_central_mw,
    f.load_southern_mw,
    f.load_west_mw,
    f.load_system_total_mw,
    f.load_north_mw + f.load_north_central_mw
        AS load_hb_north_proxy_mw,
    f.wind_stwpf_lz_north_mw,
    f.wind_stwpf_lz_south_houston_mw,
    f.wind_stwpf_lz_west_mw,
    f.wind_stwpf_system_wide_mw,
    f.wind_wgrpp_lz_north_mw,
    f.wind_wgrpp_lz_south_houston_mw,
    f.wind_wgrpp_lz_west_mw,
    f.wind_wgrpp_system_wide_mw,
    f.solar_pvgrpp_system_mw,
    f.solar_stppf_system_mw,
    f.wind_stwpf_system_wide_mw + f.solar_stppf_system_mw
        AS renewable_st_forecast_system_mw,
    f.load_system_total_mw
        - (f.wind_stwpf_system_wide_mw + f.solar_stppf_system_mw)
        AS net_load_st_forecast_system_mw,
    f.wind_wgrpp_system_wide_mw + f.solar_pvgrpp_system_mw
        AS renewable_potential_system_mw,
    f.load_system_total_mw
        - (f.wind_wgrpp_system_wide_mw + f.solar_pvgrpp_system_mw)
        AS net_load_potential_system_mw,
    (f.wind_stwpf_system_wide_mw + f.solar_stppf_system_mw)
        / NULLIF(f.load_system_total_mw, 0)
        AS renewable_st_share_of_load,
    f.wind_stwpf_lz_north_mw
        / NULLIF(f.load_north_mw + f.load_north_central_mw, 0)
        AS wind_north_share_of_north_load,
    f.has_load_forecast,
    f.has_wind_forecast,
    f.has_solar_forecast,
    f.has_all_three_forecasts,
    f.load_pre_dam_valid,
    f.wind_pre_dam_valid,
    f.solar_pre_dam_valid,
    f.all_issue_times_pre_dam_valid
FROM vw_complete_price_labels AS p
JOIN feature_time_hourly AS t
  ON t.delivery_hour_utc = p.delivery_hour_utc
JOIN feature_weather_hourly AS w
  ON w.target_hour_utc = p.delivery_hour_utc
JOIN forecast_merged AS f
  ON f.delivery_hour_utc = p.delivery_hour_utc
JOIN feature_gas_da_daily AS g
  ON g.decision_date_local = t.decision_date_local
WHERE p.delivery_hour_utc >= '{MODEL_START_UTC}'
  AND p.delivery_hour_utc < '{MODEL_END_EXCLUSIVE_UTC}'
  AND f.has_all_three_forecasts = 1
  AND f.all_issue_times_pre_dam_valid = 1
  AND g.gas_price_usd_per_mmbtu IS NOT NULL
ORDER BY p.delivery_hour_utc;
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _required_objects_exist(connection: sqlite3.Connection) -> None:
    existing = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
    }
    missing = sorted(REQUIRED_OBJECTS - existing)
    if missing:
        raise RuntimeError(
            "Analytics database is missing required objects: "
            + ", ".join(missing)
        )


def _create_split_assignments(connection: sqlite3.Connection) -> None:
    rows = [
        (row[0], row[1])
        for row in connection.execute(
            f"SELECT delivery_hour_utc, delivery_date_local "
            f"FROM {MODEL_TABLE} "
            "ORDER BY delivery_hour_utc"
        )
    ]
    if not rows:
        raise RuntimeError("No rows available for model split assignments")

    delivery_dates = list(dict.fromkeys(row[1] for row in rows))
    train_end = int(len(delivery_dates) * 0.70)
    validation_end = int(len(delivery_dates) * 0.85)
    date_splits: dict[str, str] = {}
    for index, delivery_date in enumerate(delivery_dates):
        date_splits[delivery_date] = (
            "train"
            if index < train_end
            else "validation"
            if index < validation_end
            else "test"
        )

    assignments: list[tuple[str, str]] = []
    for key, delivery_date in rows:
        assignments.append((key, date_splits[delivery_date]))

    connection.execute(
        f"""
        CREATE TABLE {SPLIT_TABLE} (
            delivery_hour_utc TEXT PRIMARY KEY,
            split_name TEXT NOT NULL,
            CHECK (split_name IN ('train', 'validation', 'test'))
        ) WITHOUT ROWID
        """
    )
    connection.executemany(
        f"INSERT INTO {SPLIT_TABLE}(delivery_hour_utc, split_name) "
        "VALUES (?, ?)",
        assignments,
    )
    connection.execute(
        f"""
        UPDATE {MODEL_TABLE}
        SET split_name = (
            SELECT s.split_name
            FROM {SPLIT_TABLE} AS s
            WHERE s.delivery_hour_utc = {MODEL_TABLE}.delivery_hour_utc
        )
        """
    )


def _write_qc(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE {QC_TABLE} (
            check_name TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            observed_value TEXT NOT NULL,
            expected_value TEXT NOT NULL,
            details TEXT NOT NULL,
            checked_at_utc TEXT NOT NULL,
            CHECK (status IN ('PASS', 'WARN'))
        ) WITHOUT ROWID
        """
    )

    checks: list[tuple[str, str, str, str, str, str]] = []

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
                _utc_now(),
            )
        )

    row_count = connection.execute(
        f"SELECT COUNT(*) FROM {MODEL_TABLE}"
    ).fetchone()[0]
    add_check(
        "model_row_count",
        row_count,
        "> 0",
        "Strict 2024-2026 rows after joining labels, time, weather, gas, and pre-DAM forecasts.",
        passed=row_count > 0,
    )

    duplicate_keys = connection.execute(
        f"""
        SELECT COUNT(*) FROM (
            SELECT delivery_hour_utc
            FROM {MODEL_TABLE}
            GROUP BY delivery_hour_utc
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    add_check(
        "unique_delivery_hour_utc",
        duplicate_keys,
        0,
        "Final model table uses delivery_hour_utc as the single canonical join key.",
        passed=duplicate_keys == 0,
    )

    invalid_utc = connection.execute(
        f"""
        SELECT COUNT(*) FROM {MODEL_TABLE}
        WHERE delivery_hour_utc NOT GLOB '????-??-??T??:??:??Z'
           OR decision_time_utc NOT GLOB '????-??-??T??:??:??Z'
        """
    ).fetchone()[0]
    add_check(
        "canonical_utc_keys",
        invalid_utc,
        0,
        "UTC timestamps must use YYYY-MM-DDTHH:MM:SSZ.",
        passed=invalid_utc == 0,
    )

    invalid_issue_utc = connection.execute(
        f"""
        SELECT COUNT(*) FROM {MODEL_TABLE}
        WHERE load_forecast_issue_time_utc NOT GLOB '????-??-??T??:??:??Z'
           OR wind_forecast_issue_time_utc NOT GLOB '????-??-??T??:??:??Z'
           OR solar_forecast_issue_time_utc NOT GLOB '????-??-??T??:??:??Z'
        """
    ).fetchone()[0]
    add_check(
        "canonical_forecast_issue_times",
        invalid_issue_utc,
        0,
        "Forecast issue timestamps are retained in canonical UTC form.",
        passed=invalid_issue_utc == 0,
    )

    min_max = connection.execute(
        f"SELECT MIN(delivery_hour_utc), MAX(delivery_hour_utc) "
        f"FROM {MODEL_TABLE}"
    ).fetchone()
    add_check(
        "model_date_scope",
        f"{min_max[0]} through {min_max[1]}",
        f"{MODEL_START_UTC} <= key < {MODEL_END_EXCLUSIVE_UTC}",
        "The final wide table is restricted to the 2024-2026 pre-DAM feature window.",
        passed=(
            min_max[0] is not None
            and min_max[0] >= MODEL_START_UTC
            and min_max[1] < MODEL_END_EXCLUSIVE_UTC
        ),
    )

    leakage_rows = connection.execute(
        f"""
        SELECT COUNT(*) FROM {MODEL_TABLE}
        WHERE all_issue_times_pre_dam_valid <> 1
        """
    ).fetchone()[0]
    add_check(
        "pre_dam_validity",
        leakage_rows,
        0,
        "All retained forecast issue times must be valid before the day-ahead decision point.",
        passed=leakage_rows == 0,
    )

    weather_leakage_rows = connection.execute(
        f"""
        SELECT COUNT(*) FROM {MODEL_TABLE}
        WHERE weather_forecast_run_time_utc >= weather_decision_cutoff_utc
        """
    ).fetchone()[0]
    add_check(
        "weather_pre_dam_validity",
        weather_leakage_rows,
        0,
        "Weather model runs must be initialized before the day-ahead decision cutoff.",
        passed=weather_leakage_rows == 0,
    )

    weather_lead_mismatches = connection.execute(
        f"""
        SELECT COUNT(*) FROM {MODEL_TABLE}
        WHERE (ercot_local_hour <= 8
               AND ABS(weather_forecast_lead_hours - 24.0) > 0.01)
           OR (ercot_local_hour >= 9
               AND ABS(weather_forecast_lead_hours - 48.0) > 0.01)
        """
    ).fetchone()[0]
    add_check(
        "weather_hybrid_lead_rule",
        weather_lead_mismatches,
        0,
        "Local hours 00-08 use day1; local hours 09-23 use day2.",
        passed=weather_lead_mismatches == 0,
    )

    missing_forecasts = connection.execute(
        f"""
        SELECT COUNT(*) FROM {MODEL_TABLE}
        WHERE has_all_three_forecasts <> 1
        """
    ).fetchone()[0]
    add_check(
        "complete_pre_dam_forecasts",
        missing_forecasts,
        0,
        "Rows missing load, wind, or solar forecast values are excluded from the strict model table.",
        passed=missing_forecasts == 0,
    )

    unmerged_fragments = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT delivery_hour_utc
            FROM feature_pre_dam_forecast_hourly
            WHERE delivery_hour_utc >= '{MODEL_START_UTC}'
              AND delivery_hour_utc < '{MODEL_END_EXCLUSIVE_UTC}'
            GROUP BY delivery_hour_utc
            HAVING COUNT(*) > 1
               AND MAX(has_load_forecast) = 1
               AND MAX(has_wind_forecast) = 1
               AND MAX(has_solar_forecast) = 1
               AND MAX(load_pre_dam_valid) = 1
               AND MAX(wind_pre_dam_valid) = 1
               AND MAX(solar_pre_dam_valid) = 1
        ) AS fragmented
        LEFT JOIN {MODEL_TABLE} AS model
          ON model.delivery_hour_utc = fragmented.delivery_hour_utc
        WHERE model.delivery_hour_utc IS NULL
        """
    ).fetchone()[0]
    add_check(
        "forecast_fragments_merged",
        unmerged_fragments,
        0,
        "Complete forecast products split across duplicate UTC rows are merged before the strict join.",
        passed=unmerged_fragments == 0,
    )

    incomplete_labels_retained = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {MODEL_TABLE} AS model
        JOIN clean_price_hourly AS price
          ON price.delivery_hour_utc = model.delivery_hour_utc
         AND price.location = model.location
        WHERE price.is_label_complete <> 1
        """
    ).fetchone()[0]
    add_check(
        "incomplete_rt_labels_excluded",
        incomplete_labels_retained,
        0,
        "Hours without all four 15-minute RT intervals are excluded rather than imputed.",
        passed=incomplete_labels_retained == 0,
    )

    missing_gas = connection.execute(
        f"""
        SELECT COUNT(*) FROM {MODEL_TABLE}
        WHERE gas_price_usd_per_mmbtu IS NULL
        """
    ).fetchone()[0]
    add_check(
        "gas_forward_fill_ready",
        missing_gas,
        0,
        "Henry Hub uses next-business-day availability and forward fill only.",
        passed=missing_gas == 0,
    )

    split_counts = {
        split: count
        for split, count in connection.execute(
            f"""
            SELECT split_name, COUNT(*)
            FROM {SPLIT_TABLE}
            GROUP BY split_name
            ORDER BY split_name
            """
        )
    }
    add_check(
        "chronological_split",
        split_counts,
        row_count,
        "Final rows are split in UTC order using 70% train, 15% validation, 15% test.",
        passed=(
            set(split_counts) == {"train", "validation", "test"}
            and sum(split_counts.values()) == row_count
        ),
    )

    split_delivery_dates = connection.execute(
        f"""
        SELECT COUNT(*) FROM (
            SELECT delivery_date_local
            FROM {MODEL_TABLE}
            GROUP BY delivery_date_local
            HAVING COUNT(DISTINCT split_name) > 1
        )
        """
    ).fetchone()[0]
    add_check(
        "whole_delivery_date_split",
        split_delivery_dates,
        0,
        "All UTC hours belonging to one ERCOT local delivery date remain in the same split.",
        passed=split_delivery_dates == 0,
    )

    connection.executemany(
        f"""
        INSERT INTO {QC_TABLE}(
            check_name, status, observed_value, expected_value,
            details, checked_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        checks,
    )


def _export_csv(connection: sqlite3.Connection, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cursor = connection.execute(f"SELECT * FROM {MODEL_TABLE}")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([description[0] for description in cursor.description])
        writer.writerows(cursor)


def build_model_wide(
    analytics_database: Path = DEFAULT_ANALYTICS_DATABASE,
    export_csv: Path | None = DEFAULT_EXPORT_CSV,
) -> dict[str, Any]:
    analytics_database = analytics_database.resolve()
    if not analytics_database.exists():
        raise FileNotFoundError(
            f"Analytics database not found: {analytics_database}"
        )

    connection = sqlite3.connect(analytics_database)
    try:
        _required_objects_exist(connection)
        connection.execute(f"DROP TABLE IF EXISTS {MODEL_TABLE}")
        connection.execute(f"DROP TABLE IF EXISTS {SPLIT_TABLE}")
        connection.execute(f"DROP TABLE IF EXISTS {QC_TABLE}")
        connection.execute(MODEL_SQL)
        connection.execute(
            f"CREATE UNIQUE INDEX idx_{MODEL_TABLE}_hour "
            f"ON {MODEL_TABLE}(delivery_hour_utc)"
        )
        _create_split_assignments(connection)
        _write_qc(connection)
        connection.commit()

        if export_csv is not None:
            _export_csv(connection, export_csv.resolve())

        row_count = connection.execute(
            f"SELECT COUNT(*) FROM {MODEL_TABLE}"
        ).fetchone()[0]
        split_counts = {
            split: count
            for split, count in connection.execute(
                f"""
                SELECT split_name, COUNT(*)
                FROM {MODEL_TABLE}
                GROUP BY split_name ORDER BY split_name
                """
            )
        }
        min_max = connection.execute(
            f"SELECT MIN(delivery_hour_utc), MAX(delivery_hour_utc) "
            f"FROM {MODEL_TABLE}"
        ).fetchone()
        qc_counts = {
            status: count
            for status, count in connection.execute(
                f"""
                SELECT status, COUNT(*)
                FROM {QC_TABLE}
                GROUP BY status ORDER BY status
                """
            )
        }
    finally:
        connection.close()

    return {
        "analytics_database": str(analytics_database),
        "model_table": MODEL_TABLE,
        "row_count": row_count,
        "delivery_hour_utc_min": min_max[0],
        "delivery_hour_utc_max": min_max[1],
        "split_counts": split_counts,
        "quality_status_counts": qc_counts,
        "export_csv": str(export_csv.resolve()) if export_csv else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the strict 2024-2026 model-wide SQLite table."
    )
    parser.add_argument(
        "--analytics-database",
        type=Path,
        default=DEFAULT_ANALYTICS_DATABASE,
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        default=DEFAULT_EXPORT_CSV,
        help="CSV export path. Use --no-export-csv to skip exporting.",
    )
    parser.add_argument(
        "--no-export-csv",
        action="store_true",
        help="Only update SQLite tables; do not write a processed CSV.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = build_model_wide(
        analytics_database=args.analytics_database,
        export_csv=None if args.no_export_csv else args.export_csv,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
