from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import requests

from .common import (
    DEFAULT_RAW_ROOT,
    build_http_session,
    filename_timestamp,
    find_completed_file,
    iso_utc,
    partitioned_directory,
    safe_slug,
    utc_now,
    validate_date_range,
    write_gzip_bytes,
    write_metadata,
)


FRED_OBSERVATIONS_URL = (
    "https://api.stlouisfed.org/fred/series/observations"
)


def collect_fred_series(
    start: date,
    end: date,
    *,
    series_id: str = "DHHNGSP",
    api_key: str | None = None,
    raw_root: Path = DEFAULT_RAW_ROOT,
    session: requests.Session | None = None,
    skip_existing: bool = True,
) -> list[dict[str, Any]]:
    """Download one FRED series and preserve the original JSON response."""
    validate_date_range(start, end)
    api_key = api_key or os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Add it to .env or the process environment."
        )

    directory = partitioned_directory(raw_root, "fred", series_id, start)
    file_prefix = f"{safe_slug(series_id)}_{start.isoformat()}_{end.isoformat()}_"
    if skip_existing:
        completed = find_completed_file(
            directory, file_prefix=file_prefix, extension=".json.gz"
        )
        if completed:
            data_path, metadata = completed
            return [
                {
                    "source": "fred",
                    "dataset": series_id,
                    "data_path": str(data_path),
                    "metadata_path": str(data_path) + ".metadata.json",
                    "row_count": metadata.get("row_count"),
                    "status": "skipped_existing",
                }
            ]

    collected_at = utc_now()
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start.isoformat(),
        "observation_end": end.isoformat(),
        "sort_order": "asc",
    }
    http = session or build_http_session()
    response = http.get(FRED_OBSERVATIONS_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if "observations" not in payload:
        raise RuntimeError(f"Unexpected FRED response: {json.dumps(payload)[:500]}")

    file_name = (
        f"{file_prefix}"
        f"{filename_timestamp(collected_at)}.json.gz"
    )
    data_path = directory / file_name
    write_gzip_bytes(data_path, response.content)

    public_params = {key: value for key, value in params.items() if key != "api_key"}
    metadata_path = write_metadata(
        data_path,
        {
            "source": "fred",
            "dataset": series_id,
            "endpoint": FRED_OBSERVATIONS_URL,
            "request_params": public_params,
            "collected_at_utc": iso_utc(collected_at),
            "row_count": len(payload["observations"]),
        },
    )
    return [
        {
            "source": "fred",
            "dataset": series_id,
            "data_path": str(data_path),
            "metadata_path": str(metadata_path),
            "row_count": len(payload["observations"]),
            "status": "downloaded",
        }
    ]
