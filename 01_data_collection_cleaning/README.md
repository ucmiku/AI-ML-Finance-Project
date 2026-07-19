# Data Collection and Cleaning

This workspace collects immutable raw responses before any normalization or
SQLite loading. Every raw file has a sibling `.metadata.json` containing its
request parameters, collection timestamp, row count, and SHA-256 digest.

## Setup

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-data.txt
Copy-Item .env.example .env
```

Set `FRED_API_KEY` and `GRIDSTATUS_API_KEY` in `.env`. The public Open-Meteo
API normally needs no key. API keys are never written to metadata files.

## Commands

Collect ERCOT North Hub day-ahead hourly and real-time 15-minute SPP data from
the GridStatus.io Hosted API:

```powershell
python 01_data_collection_cleaning/scripts/collect_data.py prices `
  --start-date 2024-01-01 --end-date 2024-01-07
```

The request filters `location=HB_NORTH` on the server and queries the official
`ercot_spp_day_ahead_hourly` and `ercot_spp_real_time_15_min` datasets. The
default 31-day request blocks support resuming and keep each response well below
the API safety limit. This path does not access ERCOT's Incapsula-protected MIS
website.

Collect archived hourly forecasts for the default North Texas locations
(Dallas, Fort Worth, Denton, McKinney, Arlington, and Wichita Falls):

```powershell
python 01_data_collection_cleaning/scripts/collect_data.py weather `
  --start-date 2024-01-01 --end-date 2024-01-31
```

Override locations by repeating `--weather-location`:

```powershell
python 01_data_collection_cleaning/scripts/collect_data.py weather `
  --start-date 2024-01-01 --end-date 2024-01-02 `
  --weather-location "Dallas,32.7767,-96.7970"
```

Collect ERCOT load, wind, and solar forecast datasets from GridStatus.io:

```powershell
python 01_data_collection_cleaning/scripts/collect_data.py forecasts `
  --start-date 2024-01-01 --end-date 2024-01-07
```

By default this uses `--forecast-mode as-of-da`, saving the latest forecast
vintage available at 10:00 America/Chicago one day before each delivery day.
The script queries candidates with `publish_time_end`, then keeps the latest
`publish_time_utc` per `interval_start_utc` locally. The saved folders use short
names like `ercot_load_da`, `ercot_wind_da`, and `ercot_solar_da`. Use
`--forecast-mode all-vintages` only when the
GridStatus.io account has enough row export quota for every historical forecast
revision.

The default forecast datasets are:

| CLI name | GridStatus.io dataset | Raw dataset folder |
|---|---|---|
| `SEVEN_DAY_LOAD_FORECAST` | `ercot_load_forecast` | `ercot_seven_day_load_forecast` |
| `WIND_PRODUCTION_FORECAST` | `ercot_wind_actual_and_forecast_hourly` | `ercot_wind_production_forecast` |
| `SOLAR_PRODUCTION_FORECAST` | `ercot_solar_actual_and_forecast_hourly` | `ercot_solar_production_forecast` |

These GridStatus.io datasets include `publish_time_utc`, so the cleaning layer
can later build as-of features that only use forecasts published before the
day-ahead decision time.

When GridStatus.io quota is not enough, manually download ERCOT generation
report CSV/ZIP files from the ERCOT data product pages and convert them into
the same immutable raw layout:

| Report flag | ERCOT report | Data product |
|---|---|---|
| `wind-system` | Wind hourly system-wide actual/forecast | `NP4-732-CD` |
| `wind-region` | Wind hourly regional actual/forecast | `NP4-742-CD` |
| `solar-system` | Solar hourly system-wide actual/forecast | `NP4-737-CD` |
| `solar-region` | Solar hourly regional actual/forecast | `NP4-745-CD` |

Example import after downloading files:

```powershell
python 01_data_collection_cleaning/scripts/import_ercot_generation_reports.py `
  --report wind-system `
  --input "01_data_collection_cleaning/manual_downloads/ercot_generation/wind_system/*.zip"

python 01_data_collection_cleaning/scripts/import_ercot_generation_reports.py `
  --report solar-system `
  --input "01_data_collection_cleaning/manual_downloads/ercot_generation/solar_system/*.zip"
```

If a downloaded CSV does not contain `Publish Time`, the importer tries to infer
it from a `YYYYMMDD_HHMMSS` or `YYYYMMDD_HHMM` timestamp in the file name. If
that is not possible, rename the file or pass `--publish-time` for a single
report file.

Collect Henry Hub daily spot prices from FRED (`DHHNGSP`):

```powershell
python 01_data_collection_cleaning/scripts/collect_data.py gas `
  --start-date 2024-01-01 --end-date 2024-12-31
```

Collect all sources for the same date range:

```powershell
python 01_data_collection_cleaning/scripts/collect_data.py all `
  --start-date 2024-01-01 --end-date 2024-01-07
```

Collectors resume by default: a request block with both a raw file and readable
metadata is skipped. Pass `--force` only when an existing block must be fetched
again.

Use `weather --weather-mode forecast` for forecasts available today. The
historical-forecast endpoint supplies archived modeled forecasts, but its raw
response does not identify a precise publication timestamp. The cleaning layer
must not invent `issued_at`; it should retain the collection timestamp and
document the endpoint's availability assumptions before model training.

## Raw layout

```text
raw/
  gridstatusio/<dataset>/year=YYYY/month=MM/*.csv.gz
  openmeteo/<dataset>/year=YYYY/month=MM/*.json.gz
  fred/DHHNGSP/year=YYYY/month=MM/*.json.gz
```

Run offline tests with:

```powershell
python -m unittest discover 01_data_collection_cleaning/tests -v
```

## SQLite raw database

Load all existing raw files into the SQLite raw layer:

```powershell
python 01_data_collection_cleaning/scripts/ingest_raw.py
```

The default database is `01_data_collection_cleaning/interim/ercot_data.sqlite`.
The `raw_files` table stores one row per immutable source file, including its
SHA-256 digest, metadata, request parameters, and collection time. The
`raw_records` table stores one JSON object per source record and includes
optional time, publication-time, observation-date, and location indexes. CSV
and JSON source fields are intentionally kept unchanged in this layer; typed
cleaning and joins belong in later interim/processed tables.

The importer is safe to rerun. A file whose SHA-256 digest is already in
`raw_files` is skipped, so newly collected files can be added incrementally.

## SQLite clean database

Build typed, deduplicated clean tables in UTC from the raw database:

```powershell
python 01_data_collection_cleaning/scripts/build_clean.py
```

The default output is
`01_data_collection_cleaning/interim/ercot_analytics.sqlite`. The build is
atomic and reproducible: it creates a temporary database, validates it, and
only then replaces the previous analytics database. The clean layer contains
day-ahead and real-time prices, hourly price spreads, the six selected North
Texas weather locations, Henry Hub vintages, and ERCOT load forecasts. Wind
and solar forecasts are intentionally excluded for now.

Schema v3 also builds `feature_weather_hourly`, which aggregates the five DFW
locations while retaining Wichita Falls and its differences from DFW. Use
`vw_complete_price_labels` for complete DA/RT labels and
`vw_model_price_weather_hourly` for the current price-plus-weather delivery
view. Query `quality_check_results` before any model export.

The model delivery view uses a 09:55 America/Chicago cutoff on the day before
delivery. UTC remains the join key; ERCOT local time is retained as a derived
feature. The default split is chronological 70% train, 15% validation, and 15%
test. Gas values are available on the next business day and are forward-filled
only. The current load forecast source has only 168 complete as-of hours, so
load coverage must be checked before using it as a model input.

Build the strict 2024-2026 model-wide table after the pre-DAM forecast feature
table has been imported:

```powershell
python 01_data_collection_cleaning/scripts/build_model_wide.py
```

This creates `model_wide_hourly_2024_2026`,
`model_split_assignments_2024_2026`, and
`model_wide_quality_check_results` in
`01_data_collection_cleaning/interim/ercot_analytics.sqlite`. It also exports
`01_data_collection_cleaning/processed/model_wide_hourly_2024_2026.csv`.
The final table uses `delivery_hour_utc` as the single UTC join key, keeps
ERCOT local time fields as model features, requires complete price labels and
pre-DAM-valid load/wind/solar forecasts, and uses a chronological 70/15/15
split computed only on the final 2024-2026 rows.

All timestamp columns use canonical UTC strings such as
`2025-01-01T06:00:00Z`. FRED observation and vintage values remain dates
because the source does not provide time-of-day precision. See
`CLEAN_DATA_SCHEMA.md` for table definitions, deduplication rules, and QC
results.

Useful checks with the SQLite CLI are:

```sql
SELECT source, dataset, COUNT(*) AS files
FROM raw_files GROUP BY source, dataset ORDER BY source, dataset;
SELECT f.source, f.dataset, COUNT(r.record_id) AS records
FROM raw_files AS f LEFT JOIN raw_records AS r ON r.file_id = f.file_id
GROUP BY f.source, f.dataset ORDER BY f.source, f.dataset;
```
