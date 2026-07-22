from __future__ import annotations

import argparse
import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from collectors.common import DATA_WORKSPACE


DEFAULT_ANALYTICS_DATABASE = DATA_WORKSPACE / "interim" / "ercot_analytics.sqlite"
TABLE_NAME = "feature_pre_dam_forecast_hourly"
TEMP_TABLE_NAME = f"{TABLE_NAME}_new"

FLOAT_COLUMNS = [
    "load_coast_mw",
    "load_east_mw",
    "load_far_west_mw",
    "load_north_mw",
    "load_north_central_mw",
    "load_south_central_mw",
    "load_southern_mw",
    "load_west_mw",
    "load_system_total_mw",
    "wind_stwpf_lz_north_mw",
    "wind_stwpf_lz_south_houston_mw",
    "wind_stwpf_lz_west_mw",
    "wind_stwpf_system_wide_mw",
    "wind_wgrpp_lz_north_mw",
    "wind_wgrpp_lz_south_houston_mw",
    "wind_wgrpp_lz_west_mw",
    "wind_wgrpp_system_wide_mw",
    "solar_pvgrpp_system_mw",
    "solar_stppf_system_mw",
]

BOOLEAN_COLUMNS = [
    "has_load_forecast",
    "has_wind_forecast",
    "has_solar_forecast",
    "has_all_three_forecasts",
    "load_pre_dam_valid",
    "wind_pre_dam_valid",
    "solar_pre_dam_valid",
    "all_issue_times_pre_dam_valid",
]

TEXT_COLUMNS = [
    "delivery_date",
    "hour_ending",
    "delivery_datetime",
    "dst_flag",
    "load_forecast_issue_time",
    "wind_forecast_issue_time",
    "solar_forecast_issue_time",
    "load_issue_date",
    "wind_issue_date",
    "solar_issue_date",
    "load_source_product_id",
    "wind_source_product_id",
    "solar_source_product_id",
]

TABLE_COLUMNS = [
    "delivery_hour_utc",
    "delivery_date",
    "hour_ending",
    "delivery_datetime",
    "dst_flag",
    "load_forecast_issue_time_utc",
    "load_forecast_issue_time",
    "wind_forecast_issue_time_utc",
    "wind_forecast_issue_time",
    "solar_forecast_issue_time_utc",
    "solar_forecast_issue_time",
    "load_issue_date",
    "wind_issue_date",
    "solar_issue_date",
    "load_source_product_id",
    "wind_source_product_id",
    "solar_source_product_id",
    *FLOAT_COLUMNS,
    *BOOLEAN_COLUMNS,
]


CREATE_TABLE_SQL = f"""
CREATE TABLE {TEMP_TABLE_NAME} (
    delivery_hour_utc TEXT NOT NULL,
    delivery_date TEXT NOT NULL,
    hour_ending TEXT NOT NULL,
    delivery_datetime TEXT NOT NULL,
    dst_flag TEXT NOT NULL,
    load_forecast_issue_time_utc TEXT,
    load_forecast_issue_time TEXT,
    wind_forecast_issue_time_utc TEXT,
    wind_forecast_issue_time TEXT,
    solar_forecast_issue_time_utc TEXT,
    solar_forecast_issue_time TEXT,
    load_issue_date TEXT,
    wind_issue_date TEXT,
    solar_issue_date TEXT,
    load_source_product_id TEXT,
    wind_source_product_id TEXT,
    solar_source_product_id TEXT,
    load_coast_mw REAL,
    load_east_mw REAL,
    load_far_west_mw REAL,
    load_north_mw REAL,
    load_north_central_mw REAL,
    load_south_central_mw REAL,
    load_southern_mw REAL,
    load_west_mw REAL,
    load_system_total_mw REAL,
    wind_stwpf_lz_north_mw REAL,
    wind_stwpf_lz_south_houston_mw REAL,
    wind_stwpf_lz_west_mw REAL,
    wind_stwpf_system_wide_mw REAL,
    wind_wgrpp_lz_north_mw REAL,
    wind_wgrpp_lz_south_houston_mw REAL,
    wind_wgrpp_lz_west_mw REAL,
    wind_wgrpp_system_wide_mw REAL,
    solar_pvgrpp_system_mw REAL,
    solar_stppf_system_mw REAL,
    has_load_forecast INTEGER CHECK (has_load_forecast IN (0, 1)),
    has_wind_forecast INTEGER CHECK (has_wind_forecast IN (0, 1)),
    has_solar_forecast INTEGER CHECK (has_solar_forecast IN (0, 1)),
    has_all_three_forecasts INTEGER CHECK (has_all_three_forecasts IN (0, 1)),
    load_pre_dam_valid INTEGER CHECK (load_pre_dam_valid IN (0, 1)),
    wind_pre_dam_valid INTEGER CHECK (wind_pre_dam_valid IN (0, 1)),
    solar_pre_dam_valid INTEGER CHECK (solar_pre_dam_valid IN (0, 1)),
    all_issue_times_pre_dam_valid INTEGER
        CHECK (all_issue_times_pre_dam_valid IN (0, 1)),
    PRIMARY KEY (delivery_date, hour_ending, dst_flag),
    UNIQUE (delivery_hour_utc)
) WITHOUT ROWID
"""


def _canonical_utc(value: str) -> str:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp has no UTC offset: {value!r}")
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _optional_utc(value: str) -> str | None:
    return _canonical_utc(value) if value.strip() else None


def _optional_float(value: str) -> float | None:
    return float(value) if value.strip() else None


def _boolean(value: str, column: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return 1
    if normalized in {"false", "0"}:
        return 0
    raise ValueError(f"Invalid boolean in {column}: {value!r}")


def _read_rows(source_csv: Path) -> list[tuple[Any, ...]]:
    with source_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {
            "delivery_datetime_utc",
            *TEXT_COLUMNS,
            *FLOAT_COLUMNS,
            *BOOLEAN_COLUMNS,
        }
        missing = sorted(required - fields)
        if missing:
            raise ValueError("Source CSV is missing columns: " + ", ".join(missing))

        output: list[tuple[Any, ...]] = []
        seen_utc: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            try:
                delivery_hour_utc = _canonical_utc(row["delivery_datetime_utc"])
                if delivery_hour_utc in seen_utc:
                    raise ValueError(f"duplicate delivery UTC {delivery_hour_utc}")
                seen_utc.add(delivery_hour_utc)

                values: dict[str, Any] = {
                    "delivery_hour_utc": delivery_hour_utc,
                    "load_forecast_issue_time_utc": _optional_utc(
                        row["load_forecast_issue_time"]
                    ),
                    "wind_forecast_issue_time_utc": _optional_utc(
                        row["wind_forecast_issue_time"]
                    ),
                    "solar_forecast_issue_time_utc": _optional_utc(
                        row["solar_forecast_issue_time"]
                    ),
                }
                values.update({column: row[column].strip() or None for column in TEXT_COLUMNS})
                values.update(
                    {column: _optional_float(row[column]) for column in FLOAT_COLUMNS}
                )
                values.update(
                    {column: _boolean(row[column], column) for column in BOOLEAN_COLUMNS}
                )
                output.append(tuple(values[column] for column in TABLE_COLUMNS))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid source row {line_number}: {exc}") from exc
    if not output:
        raise ValueError("Source CSV contains no data rows")
    return output


def import_pre_dam_features(source_csv: Path, database: Path) -> dict[str, Any]:
    source_csv = source_csv.resolve()
    database = database.resolve()
    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")
    if not database.exists():
        raise FileNotFoundError(f"Analytics database not found: {database}")

    rows = _read_rows(source_csv)
    placeholders = ", ".join("?" for _ in TABLE_COLUMNS)
    columns_sql = ", ".join(TABLE_COLUMNS)

    connection = sqlite3.connect(database)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(f"DROP TABLE IF EXISTS {TEMP_TABLE_NAME}")
        connection.execute(CREATE_TABLE_SQL)
        connection.executemany(
            f"INSERT INTO {TEMP_TABLE_NAME} ({columns_sql}) VALUES ({placeholders})",
            rows,
        )
        incomplete = connection.execute(
            f"""
            SELECT COUNT(*) FROM {TEMP_TABLE_NAME}
            WHERE has_all_three_forecasts <> 1
               OR all_issue_times_pre_dam_valid <> 1
            """
        ).fetchone()[0]
        if incomplete:
            raise ValueError(
                f"Source contains {incomplete} rows without complete pre-DAM forecasts"
            )

        connection.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        connection.execute(
            f"ALTER TABLE {TEMP_TABLE_NAME} RENAME TO {TABLE_NAME}"
        )
        connection.execute(
            f"CREATE INDEX idx_pre_dam_delivery_hour_utc "
            f"ON {TABLE_NAME}(delivery_hour_utc)"
        )
        connection.execute(
            f"CREATE INDEX idx_pre_dam_complete "
            f"ON {TABLE_NAME}(has_all_three_forecasts, all_issue_times_pre_dam_valid)"
        )
        connection.commit()

        min_max = connection.execute(
            f"SELECT MIN(delivery_hour_utc), MAX(delivery_hour_utc) "
            f"FROM {TABLE_NAME}"
        ).fetchone()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    return {
        "source_csv": str(source_csv),
        "database": str(database),
        "table": TABLE_NAME,
        "row_count": len(rows),
        "delivery_hour_utc_min": min_max[0],
        "delivery_hour_utc_max": min_max[1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import the complete ERCOT pre-DAM forecast feature CSV."
    )
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument(
        "--analytics-database",
        type=Path,
        default=DEFAULT_ANALYTICS_DATABASE,
    )
    args = parser.parse_args()
    print(import_pre_dam_features(args.source_csv, args.analytics_database))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
