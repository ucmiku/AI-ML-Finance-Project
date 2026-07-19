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
    DATA_WORKSPACE / "processed" / "model_wide_hourly_2024_2026.csv"
)

MODEL_START_UTC = "2024-01-01T00:00:00Z"
MODEL_END_EXCLUSIVE_UTC = "2026-07-01T00:00:00Z"

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
JOIN feature_pre_dam_forecast_hourly AS f
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
    keys = [
        row[0]
        for row in connection.execute(
            f"SELECT delivery_hour_utc FROM {MODEL_TABLE} "
            "ORDER BY delivery_hour_utc"
        )
    ]
    if not keys:
        raise RuntimeError("No rows available for model split assignments")

    train_end = int(len(keys) * 0.70)
    validation_end = int(len(keys) * 0.85)
    assignments: list[tuple[str, str]] = []
    for index, key in enumerate(keys):
        split = (
            "train"
            if index < train_end
            else "validation"
            if index < validation_end
            else "test"
        )
        assignments.append((key, split))

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
