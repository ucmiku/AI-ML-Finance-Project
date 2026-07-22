from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

from collectors.common import DATA_WORKSPACE, PROJECT_ROOT, build_http_session
from collectors.openmeteo_collector import collect_openmeteo_single_run_weather


DEFAULT_RAW_DATABASE = DATA_WORKSPACE / "interim" / "ercot_data.sqlite"
DEFAULT_RAW_ROOT = DATA_WORKSPACE / "raw"
LOGGER = logging.getLogger("recover_single_run_weather")


INCOMPLETE_DATES_SQL = """
WITH file_dates AS (
    SELECT
        f.file_id,
        json_extract(r.record_json, '$.delivery_date_local') AS delivery_date,
        MIN(
            CASE
                WHEN json_extract(r.record_json, '$.temperature_2m') IS NOT NULL
                 AND json_extract(r.record_json, '$.relative_humidity_2m') IS NOT NULL
                 AND json_extract(r.record_json, '$.wind_speed_10m') IS NOT NULL
                 AND json_extract(r.record_json, '$.wind_gusts_10m') IS NOT NULL
                 AND json_extract(r.record_json, '$.cloud_cover') IS NOT NULL
                 AND json_extract(r.record_json, '$.shortwave_radiation') IS NOT NULL
                 AND json_extract(r.record_json, '$.precipitation') IS NOT NULL
                THEN 1 ELSE 0
            END
        ) AS file_is_complete
    FROM raw_records AS r
    JOIN raw_files AS f ON f.file_id = r.file_id
    WHERE f.source = 'openmeteo'
      AND f.dataset = 'single-run-ecmwf'
    GROUP BY f.file_id, delivery_date
), delivery_dates AS (
    SELECT delivery_date, MAX(file_is_complete) AS has_complete_file
    FROM file_dates
    WHERE delivery_date IS NOT NULL
    GROUP BY delivery_date
)
SELECT delivery_date
FROM delivery_dates
WHERE has_complete_file = 0
ORDER BY delivery_date
"""


def _load_project_environment() -> None:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")


def _incomplete_dates(database: Path) -> list[date]:
    connection = sqlite3.connect(database)
    try:
        return [
            date.fromisoformat(row[0])
            for row in connection.execute(INCOMPLETE_DATES_SQL)
        ]
    finally:
        connection.close()


def _complete_dates_on_disk(raw_root: Path) -> set[date]:
    directory = raw_root / "openmeteo" / "single-run-ecmwf"
    complete: set[date] = set()
    if not directory.exists():
        return complete
    for path in directory.rglob("*.metadata.json"):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("weather_values_complete") is not True:
            continue
        value = metadata.get("delivery_date_local")
        if value:
            complete.add(date.fromisoformat(str(value)))
    return complete


def recover(
    raw_database: Path = DEFAULT_RAW_DATABASE,
    raw_root: Path = DEFAULT_RAW_ROOT,
    workers: int = 4,
) -> dict[str, Any]:
    candidates = _incomplete_dates(raw_database.resolve())
    complete_on_disk = _complete_dates_on_disk(raw_root.resolve())
    pending = [item for item in candidates if item not in complete_on_disk]
    status_counts: dict[str, int] = {}
    unresolved: list[str] = []

    def recover_one(delivery_date: date) -> tuple[date, dict[str, Any]]:
        results = collect_openmeteo_single_run_weather(
            delivery_date,
            delivery_date,
            raw_root=raw_root.resolve(),
            session=build_http_session(),
            skip_existing=False,
        )
        return delivery_date, results[-1]

    completed_count = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(recover_one, delivery_date): delivery_date
            for delivery_date in pending
        }
        for future in as_completed(futures):
            delivery_date, result = future.result()
            status = str(result["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
            if status != "downloaded":
                unresolved.append(delivery_date.isoformat())
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == len(pending):
                LOGGER.info(
                    "Single Run recovery progress: %d/%d dates",
                    completed_count,
                    len(pending),
                )

    return {
        "database_incomplete_dates": len(candidates),
        "already_recovered_on_disk": len(candidates) - len(pending),
        "attempted_dates": len(pending),
        "workers": workers,
        "status_counts": status_counts,
        "unresolved_dates": sorted(unresolved),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover incomplete Open-Meteo Single Run delivery dates."
    )
    parser.add_argument(
        "--raw-database", type=Path, default=DEFAULT_RAW_DATABASE
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    _load_project_environment()
    print(
        json.dumps(
            recover(args.raw_database, args.raw_root, args.workers),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
