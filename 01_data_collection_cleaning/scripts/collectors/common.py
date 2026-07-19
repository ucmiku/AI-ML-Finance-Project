from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from collections.abc import Iterator
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DATA_WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT_ROOT = DATA_WORKSPACE.parent
DEFAULT_RAW_ROOT = DATA_WORKSPACE / "raw"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def filename_timestamp(value: datetime | None = None) -> str:
    value = value or utc_now()
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid date {value!r}; expected YYYY-MM-DD") from exc


def validate_date_range(start: date, end: date) -> None:
    if end < start:
        raise ValueError(f"End date {end} is before start date {start}")


def iter_date_chunks(
    start: date,
    end_inclusive: date,
    chunk_days: int,
) -> Iterator[tuple[date, date]]:
    """Yield half-open date ranges [start, end) covering an inclusive range."""
    validate_date_range(start, end_inclusive)
    if chunk_days < 1:
        raise ValueError("chunk_days must be at least 1")

    cursor = start
    stop = end_inclusive + timedelta(days=1)
    while cursor < stop:
        chunk_end = min(cursor + timedelta(days=chunk_days), stop)
        yield cursor, chunk_end
        cursor = chunk_end


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("._-") or "unknown"


def build_http_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update(
        {"User-Agent": "ercot-weather-arbitrage-research/0.1"}
    )
    return session


def partitioned_directory(
    raw_root: Path,
    source: str,
    dataset: str,
    partition_date: date,
) -> Path:
    return (
        raw_root
        / safe_slug(source)
        / safe_slug(dataset)
        / f"year={partition_date.year:04d}"
        / f"month={partition_date.month:02d}"
    )


def _atomic_replace(temp_path: Path, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp_path, final_path)


def write_gzip_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    try:
        with gzip.open(temp_path, "wb") as handle:
            handle.write(payload)
        _atomic_replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_path_for(data_path: Path) -> Path:
    return data_path.with_name(data_path.name + ".metadata.json")


def find_completed_file(
    directory: Path,
    *,
    file_prefix: str | tuple[str, ...],
    extension: str,
) -> tuple[Path, dict[str, Any]] | None:
    """Return the newest raw file with readable metadata for a request block."""
    if not directory.exists():
        return None
    prefixes = (file_prefix,) if isinstance(file_prefix, str) else file_prefix
    candidates = sorted(
        (
            path
            for prefix in prefixes
            for path in directory.glob(f"{prefix}*{extension}")
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    for data_path in candidates:
        metadata_path = metadata_path_for(data_path)
        if not metadata_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        return data_path, metadata
    return None


def write_metadata(data_path: Path, metadata: dict[str, Any]) -> Path:
    enriched = dict(metadata)
    enriched["file_name"] = data_path.name
    enriched["file_size_bytes"] = data_path.stat().st_size
    enriched["sha256"] = sha256_file(data_path)
    path = metadata_path_for(data_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, enriched)
    return path
