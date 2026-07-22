from __future__ import annotations

import argparse
import csv
import gzip
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from collectors.common import DATA_WORKSPACE, DEFAULT_RAW_ROOT, sha256_file


DEFAULT_DATABASE = DATA_WORKSPACE / "interim" / "ercot_data.sqlite"
SCHEMA_VERSION = 1


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_files (
    file_id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    dataset TEXT NOT NULL,
    file_path TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL,
    file_format TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    file_size_bytes INTEGER NOT NULL,
    source_row_count INTEGER,
    collected_at_utc TEXT,
    request_json TEXT,
    metadata_json TEXT NOT NULL,
    imported_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_records (
    record_id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES raw_files(file_id) ON DELETE CASCADE,
    record_number INTEGER NOT NULL,
    record_json TEXT NOT NULL,
    interval_start_utc TEXT,
    publish_time_utc TEXT,
    observation_date TEXT,
    location TEXT,
    UNIQUE(file_id, record_number)
);

CREATE INDEX IF NOT EXISTS idx_raw_records_dataset_time
    ON raw_records(file_id, interval_start_utc);
CREATE INDEX IF NOT EXISTS idx_raw_records_observation_date
    ON raw_records(file_id, observation_date);
CREATE INDEX IF NOT EXISTS idx_raw_files_dataset
    ON raw_files(source, dataset);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _text(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _find_value(record: dict[str, Any], *names: str) -> str | None:
    normalized = {
        str(key).strip().lower().replace(" ", "_"): value
        for key, value in record.items()
    }
    for name in names:
        value = normalized.get(name.lower().replace(" ", "_"))
        if value is not None:
            return _text(value)
    return None


def _read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv_records(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def _openmeteo_records(payload: dict[str, Any], dataset: str) -> Iterator[dict[str, Any]]:
    locations = payload.get("locations")
    if isinstance(locations, list):
        for location_payload in locations:
            if not isinstance(location_payload, dict):
                continue
            location_name = location_payload.get("location")
            hourly = location_payload.get("hourly")
            if not isinstance(hourly, dict) or not isinstance(
                hourly.get("time"), list
            ):
                continue
            times = hourly["time"]
            columns = {
                key: values
                for key, values in hourly.items()
                if key != "time" and isinstance(values, list)
            }
            context = {
                key: location_payload.get(key)
                for key in (
                    "forecast_run_time_utc",
                    "decision_cutoff_utc",
                    "forecast_model",
                    "forecast_lead_rule",
                    "availability_assumption",
                    "delivery_date_local",
                )
                if location_payload.get(key) is not None
            }
            for index, interval_start in enumerate(times):
                record: dict[str, Any] = {
                    "location": location_name,
                    "interval_start": interval_start,
                    **context,
                }
                for key, values in columns.items():
                    record[key] = values[index] if index < len(values) else None
                yield record
        return

    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
        yield payload
        return

    times = hourly["time"]
    columns = {
        key: values
        for key, values in hourly.items()
        if key != "time" and isinstance(values, list)
    }
    for index, interval_start in enumerate(times):
        record: dict[str, Any] = {
            "location": dataset.removeprefix("historical-forecast_")
            .removeprefix("forecast_"),
            "interval_start": interval_start,
        }
        for key, values in columns.items():
            record[key] = values[index] if index < len(values) else None
        yield record


def _json_records(payload: Any, source: str, dataset: str) -> Iterator[dict[str, Any]]:
    if source == "fred" and isinstance(payload, dict):
        observations = payload.get("observations")
        if isinstance(observations, list):
            yield from (item for item in observations if isinstance(item, dict))
            return
    if source == "openmeteo" and isinstance(payload, dict):
        yield from _openmeteo_records(payload, dataset)
        return
    if isinstance(payload, list):
        yield from (item if isinstance(item, dict) else {"value": item} for item in payload)
        return
    if isinstance(payload, dict):
        yield payload
        return
    yield {"value": payload}


def _source_dataset(metadata: dict[str, Any], path: Path) -> tuple[str, str]:
    source = _text(metadata.get("source"))
    dataset = _text(metadata.get("dataset"))
    if source and dataset:
        return source, dataset
    parts = path.parts
    try:
        raw_index = parts.index("raw")
        return parts[raw_index + 1], parts[raw_index + 2]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Cannot infer source and dataset for {path}") from exc


def _metadata_for(path: Path) -> dict[str, Any]:
    metadata_path = path.with_name(path.name + ".metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file for {path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata must be a JSON object: {metadata_path}")
    return metadata


def _record_context(record: dict[str, Any]) -> tuple[str | None, str | None, str | None, str | None]:
    interval = _find_value(record, "interval_start_utc", "interval_start", "time")
    publish = _find_value(record, "publish_time_utc", "publish_time")
    observation = _find_value(record, "observation_date", "date")
    location = _find_value(record, "location")
    return interval, publish, observation, location


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_SQL)
    connection.execute(
        "INSERT INTO schema_info(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        ("schema_version", str(SCHEMA_VERSION)),
    )
    connection.commit()


def ingest_file(connection: sqlite3.Connection, path: Path) -> tuple[str, int, bool]:
    metadata = _metadata_for(path)
    source, dataset = _source_dataset(metadata, path)
    digest = sha256_file(path)
    existing = connection.execute(
        "SELECT file_id FROM raw_files WHERE sha256 = ?", (digest,)
    ).fetchone()
    if existing:
        return dataset, 0, False

    file_format = "csv" if path.name.endswith(".csv.gz") else "json"
    imported_at = _utc_now()
    connection.execute(
        """INSERT INTO raw_files(
            source, dataset, file_path, file_name, file_format, sha256,
            file_size_bytes, source_row_count, collected_at_utc, request_json,
            metadata_json, imported_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            source,
            dataset,
            str(path.resolve()),
            path.name,
            file_format,
            digest,
            path.stat().st_size,
            metadata.get("row_count"),
            metadata.get("collected_at_utc"),
            json.dumps(metadata.get("request", metadata.get("request_params")), sort_keys=True),
            json.dumps(metadata, ensure_ascii=True, sort_keys=True),
            imported_at,
        ),
    )
    file_id = int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])

    if file_format == "csv":
        records = _read_csv_records(path)
    else:
        records = _json_records(_read_gzip_json(path), source, dataset)

    count = 0
    batch: list[tuple[Any, ...]] = []
    for record_number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            record = {"value": record}
        interval, publish, observation, location = _record_context(record)
        batch.append(
            (
                file_id,
                record_number,
                json.dumps(record, ensure_ascii=True, separators=(",", ":")),
                interval,
                publish,
                observation,
                location,
            )
        )
        if len(batch) >= 1000:
            connection.executemany(
                "INSERT INTO raw_records(file_id, record_number, record_json, "
                "interval_start_utc, publish_time_utc, observation_date, location) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
            count += len(batch)
            batch.clear()
    if batch:
        connection.executemany(
            "INSERT INTO raw_records(file_id, record_number, record_json, "
            "interval_start_utc, publish_time_utc, observation_date, location) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            batch,
        )
        count += len(batch)
    return dataset, count, True


def ingest_raw(raw_root: Path, database_path: Path) -> dict[str, int]:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        path
        for pattern in ("*.csv.gz", "*.json.gz")
        for path in raw_root.rglob(pattern)
        if not path.name.endswith(".metadata.json")
    )
    summary = {
        "files_seen": len(paths),
        "files_imported": 0,
        "records_imported": 0,
        "files_skipped_missing_metadata": 0,
    }
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = NORMAL")
        initialize_database(connection)
        for path in paths:
            if not path.with_name(path.name + ".metadata.json").exists():
                summary["files_skipped_missing_metadata"] += 1
                print(f"skipped missing metadata: {path}")
                continue
            dataset, count, imported = ingest_file(connection, path)
            if imported:
                connection.commit()
                summary["files_imported"] += 1
                summary["records_imported"] += count
                print(f"imported {dataset}: {count:,} records")
    finally:
        connection.close()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest immutable raw files into SQLite.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = ingest_raw(args.raw_root.resolve(), args.database.resolve())
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
