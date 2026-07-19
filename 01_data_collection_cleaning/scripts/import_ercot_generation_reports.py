from __future__ import annotations

import argparse
import csv
import fnmatch
import gzip
import io
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from collectors.common import (
    DEFAULT_RAW_ROOT,
    filename_timestamp,
    iso_utc,
    partitioned_directory,
    safe_slug,
    sha256_file,
    utc_now,
    write_metadata,
)


ERCOT_TIMEZONE = ZoneInfo("America/Chicago")


@dataclass(frozen=True)
class ReportConfig:
    dataset: str
    report_type_id: int
    report_name: str


REPORTS = {
    "wind-system": ReportConfig(
        dataset="ercot_wind_actual_and_forecast_hourly",
        report_type_id=13028,
        report_name=(
            "Wind Power Production - Hourly Averaged Actual and "
            "Forecasted Values"
        ),
    ),
    "wind-region": ReportConfig(
        dataset="ercot_wind_actual_and_forecast_by_geographical_region_hourly",
        report_type_id=14787,
        report_name=(
            "Wind Power Production - Hourly Averaged Actual and "
            "Forecasted Values by Geographical Region"
        ),
    ),
    "solar-system": ReportConfig(
        dataset="ercot_solar_actual_and_forecast_hourly",
        report_type_id=13483,
        report_name=(
            "Solar Power Production - Hourly Averaged Actual and "
            "Forecasted Values"
        ),
    ),
    "solar-region": ReportConfig(
        dataset="ercot_solar_actual_and_forecast_by_geographical_region_hourly",
        report_type_id=21809,
        report_name=(
            "Solar Power Production - Hourly Averaged Actual and "
            "Forecasted Values by Geographical Region"
        ),
    ),
}


COLUMN_RENAMES = {
    "SYSTEM WIDE HSL": "HSL SYSTEM WIDE",
    "SYSTEM WIDE GEN": "GEN SYSTEM WIDE",
    "ACTUAL SYSTEM WIDE": "GEN SYSTEM WIDE",
    "ACTUAL LZ SOUTH HOUSTON": "GEN LZ SOUTH HOUSTON",
    "ACTUAL LZ WEST": "GEN LZ WEST",
    "ACTUAL LZ NORTH": "GEN LZ NORTH",
    "ACTUAL PANHANDLE": "GEN PANHANDLE",
    "ACTUAL COASTAL": "GEN COASTAL",
    "ACTUAL SOUTH": "GEN SOUTH",
    "ACTUAL WEST": "GEN WEST",
    "ACTUAL NORTH": "GEN NORTH",
}


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_header(value: Any) -> str:
    text = str(value).replace("\ufeff", "").strip()
    text = re.sub(r"\s+", " ", text)
    return COLUMN_RENAMES.get(text.upper(), text)


def _snake_case(value: str) -> str:
    value = _normalize_header(value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return value.strip("_").lower()


def _canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [_snake_case(column) for column in frame.columns]
    return frame


def _parse_time_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(
            ERCOT_TIMEZONE,
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    return parsed.dt.tz_convert(timezone.utc).dt.strftime(
        "%Y-%m-%d %H:%M:%S+00:00"
    )


def _parse_single_time(value: str) -> str:
    parsed = pd.to_datetime(pd.Series([value]), errors="coerce")
    if parsed.isna().iloc[0]:
        raise ValueError(f"Cannot parse publish time {value!r}")
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(
            ERCOT_TIMEZONE,
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    return str(
        parsed.dt.tz_convert(timezone.utc).dt.strftime(
            "%Y-%m-%d %H:%M:%S+00:00"
        ).iloc[0]
    )


def _infer_publish_time_from_name(name: str) -> str | None:
    candidates = [
        r"(?P<date>20\d{6})[._-]?(?P<time>\d{6})",
        r"(?P<date>20\d{6})[._-]?(?P<time>\d{4})",
    ]
    for pattern in candidates:
        matches = list(re.finditer(pattern, name))
        for match in reversed(matches):
            date_part = match.group("date")
            time_part = match.group("time")
            if len(time_part) == 4:
                time_part = f"{time_part}00"
            try:
                return _parse_single_time(f"{date_part} {time_part}")
            except ValueError:
                continue
    return None


def _ensure_time_columns(
    frame: pd.DataFrame,
    *,
    source_name: str,
    publish_time: str | None,
) -> pd.DataFrame:
    frame = frame.copy()
    for base in ("interval_start", "interval_end"):
        utc_name = f"{base}_utc"
        if utc_name in frame.columns:
            frame[utc_name] = _parse_time_series(frame[utc_name])
        elif base in frame.columns:
            frame[utc_name] = _parse_time_series(frame[base])
        else:
            raise ValueError(
                f"{source_name} has no {base!r} or {utc_name!r} column"
            )

    if "publish_time_utc" in frame.columns:
        frame["publish_time_utc"] = _parse_time_series(frame["publish_time_utc"])
    elif "publish_time" in frame.columns:
        frame["publish_time_utc"] = _parse_time_series(frame["publish_time"])
    else:
        inferred = publish_time or _infer_publish_time_from_name(source_name)
        if inferred is None:
            raise ValueError(
                f"{source_name} has no publish_time column and no publish "
                "timestamp could be inferred from the file name. Rename the "
                "file to include YYYYMMDD_HHMMSS, or pass --publish-time for "
                "single-report imports."
            )
        frame["publish_time_utc"] = inferred
    return frame


def _read_csv_payload(payload: bytes, *, source_name: str) -> pd.DataFrame:
    text = payload.decode("utf-8-sig", errors="replace")
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample)
        sep = dialect.delimiter
    except csv.Error:
        sep = ","
    return pd.read_csv(io.StringIO(text), sep=sep)


def _iter_source_frames(path: Path) -> Iterable[tuple[str, pd.DataFrame]]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if member.endswith("/") or not member.lower().endswith(".csv"):
                    continue
                payload = archive.read(member)
                yield f"{path.name}/{member}", _read_csv_payload(
                    payload,
                    source_name=f"{path.name}/{member}",
                )
        return
    if path.suffix.lower() == ".csv":
        yield path.name, pd.read_csv(path, encoding="utf-8-sig")
        return
    if path.name.lower().endswith(".csv.gz"):
        with gzip.open(path, "rt", encoding="utf-8-sig") as handle:
            yield path.name, pd.read_csv(handle)
        return
    raise ValueError(f"Unsupported input file type: {path}")


def _expand_inputs(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        if any(char in value for char in "*?[]"):
            parent = Path(value).parent
            pattern = Path(value).name
            if str(parent) == ".":
                parent = Path.cwd()
            paths.extend(
                path
                for path in parent.iterdir()
                if fnmatch.fnmatch(path.name, pattern)
            )
        else:
            paths.append(Path(value))
    return sorted(path.resolve() for path in paths)


def _existing_manual_imports(raw_root: Path, dataset: str) -> set[tuple[str, str]]:
    directory = raw_root / "ercot_manual" / safe_slug(dataset)
    if not directory.exists():
        return set()
    existing: set[tuple[str, str]] = set()
    for metadata_path in directory.rglob("*.metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source_sha = metadata.get("manual_source_sha256")
        member = metadata.get("manual_source_member", "")
        if source_sha:
            existing.add((str(source_sha), str(member)))
    return existing


def _write_frame(frame: pd.DataFrame, data_path: Path) -> None:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = data_path.with_name(data_path.name + ".part")
    try:
        with gzip.GzipFile(temp_path, "wb", mtime=0) as handle:
            payload = frame.to_csv(index=False).encode("utf-8")
            handle.write(payload)
        temp_path.replace(data_path)
    finally:
        temp_path.unlink(missing_ok=True)


def import_generation_reports(
    *,
    report_key: str,
    inputs: list[str],
    raw_root: Path = DEFAULT_RAW_ROOT,
    publish_time: str | None = None,
    skip_existing: bool = True,
) -> list[dict[str, Any]]:
    config = REPORTS[report_key]
    paths = _expand_inputs(inputs)
    if not paths:
        raise FileNotFoundError("No input files matched")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Input files do not exist: {missing}")

    existing = (
        _existing_manual_imports(raw_root, config.dataset)
        if skip_existing
        else set()
    )
    results: list[dict[str, Any]] = []
    for path in paths:
        source_sha = sha256_file(path)
        for member_name, frame in _iter_source_frames(path):
            if (source_sha, member_name) in existing:
                results.append(
                    {
                        "source": "ercot_manual",
                        "dataset": config.dataset,
                        "manual_source_path": str(path),
                        "manual_source_member": member_name,
                        "status": "skipped_existing",
                    }
                )
                continue
            collected_at = utc_now()
            normalized = _canonicalize_columns(frame)
            normalized = _ensure_time_columns(
                normalized,
                source_name=member_name,
                publish_time=publish_time,
            )
            partition_time = pd.to_datetime(
                normalized["interval_start_utc"].dropna().iloc[0],
                utc=True,
            ).date()
            directory = partitioned_directory(
                raw_root,
                "ercot_manual",
                config.dataset,
                partition_time,
            )
            file_name = (
                f"{config.dataset}_{safe_slug(Path(member_name).stem)}_"
                f"{filename_timestamp(collected_at)}.csv.gz"
            )
            data_path = directory / file_name
            _write_frame(normalized, data_path)
            metadata_path = write_metadata(
                data_path,
                {
                    "source": "ercot_manual",
                    "dataset": config.dataset,
                    "report_type_id": config.report_type_id,
                    "report_name": config.report_name,
                    "manual_source_path": str(path),
                    "manual_source_file": path.name,
                    "manual_source_member": member_name,
                    "manual_source_sha256": source_sha,
                    "collected_at_utc": iso_utc(collected_at),
                    "row_count": int(len(normalized)),
                    "columns": [str(column) for column in normalized.columns],
                    "status": "imported_manual_download",
                },
            )
            results.append(
                {
                    "source": "ercot_manual",
                    "dataset": config.dataset,
                    "data_path": str(data_path),
                    "metadata_path": str(metadata_path),
                    "row_count": int(len(normalized)),
                    "manual_source_path": str(path),
                    "manual_source_member": member_name,
                    "status": "imported",
                }
            )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert manually downloaded ERCOT generation report CSV/ZIP files "
            "into immutable raw csv.gz files with metadata."
        )
    )
    parser.add_argument(
        "--report",
        required=True,
        choices=sorted(REPORTS),
        help="ERCOT generation report type",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="CSV, csv.gz, zip, or glob patterns containing ERCOT report CSVs",
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--publish-time",
        help=(
            "Optional publish time for single-report files with no publish "
            "timestamp column. Naive values are interpreted as America/Chicago."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Import files even if their source SHA/member was already imported",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results = import_generation_reports(
        report_key=args.report,
        inputs=args.input,
        raw_root=args.raw_root.resolve(),
        publish_time=args.publish_time,
        skip_existing=not args.force,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
