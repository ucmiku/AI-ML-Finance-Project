from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = (
    PROJECT_ROOT
    / "01_data_collection_cleaning"
    / "processed"
    / "model_wide_hourly_2024_2026_final.csv"
)
DEFAULT_DB = (
    PROJECT_ROOT / "01_data_collection_cleaning" / "interim" / "ercot_data.sqlite"
)
DEFAULT_TABLE = "model_wide_hourly_2024_2026"


def import_model_wide(csv_path: Path, db_path: Path, table_name: str) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    df = pd.read_csv(csv_path)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_delivery_hour_utc "
            f"ON {table_name}(delivery_hour_utc)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_delivery_date_local "
            f"ON {table_name}(delivery_date_local)"
        )

    print(f"Imported {len(df):,} rows into {db_path}::{table_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    args = parser.parse_args()

    import_model_wide(args.csv, args.database, args.table)


if __name__ == "__main__":
    main()
