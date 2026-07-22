from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from collectors.common import DEFAULT_RAW_ROOT, PROJECT_ROOT, parse_date
from collectors.fred_collector import collect_fred_series
from collectors.gridstatus_collector import (
    DEFAULT_FORECASTS,
    DEFAULT_MARKETS,
    collect_ercot_asof_forecasts,
    collect_ercot_forecasts,
    collect_ercot_prices,
)
from collectors.openmeteo_collector import (
    DEFAULT_TEXAS_LOCATIONS,
    WeatherLocation,
    collect_openmeteo_previous_runs_weather,
    collect_openmeteo_weather,
    collect_openmeteo_single_run_weather,
    parse_location,
)


LOGGER = logging.getLogger("data_collection")


def _load_project_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        if (PROJECT_ROOT / ".env").exists():
            raise RuntimeError(
                "python-dotenv is required to read .env. Run: "
                "pip install -r requirements-data.txt"
            ) from exc
        return
    load_dotenv(PROJECT_ROOT / ".env")


def _add_date_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD, inclusive")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD, inclusive")


def _add_raw_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help=f"Raw output directory (default: {DEFAULT_RAW_ROOT})",
    )


def _add_force_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download request blocks even when complete raw files already exist",
    )


def _add_weather_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--weather-mode",
        choices=(
            "historical-forecast",
            "forecast",
            "single-run",
            "previous-runs-day2",
            "previous-runs-hybrid",
        ),
        default="historical-forecast",
        help="Open-Meteo endpoint to use",
    )
    parser.add_argument(
        "--weather-location",
        action="append",
        default=[],
        metavar="NAME,LAT,LON",
        help="Repeat to override the six default North Texas locations",
    )
    parser.add_argument(
        "--weather-chunk-days",
        type=int,
        default=31,
        help="Days per Open-Meteo request (default: 31)",
    )
    parser.add_argument(
        "--weather-model",
        default="ecmwf_ifs",
        help="Open-Meteo Single Runs model (default: ecmwf_ifs)",
    )
    parser.add_argument(
        "--weather-run-hour-utc",
        type=int,
        default=0,
        choices=(0, 12),
        help="Single Runs initialization hour UTC (default: 00Z)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect raw ERCOT, weather, and Henry Hub data."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed log messages"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prices = subparsers.add_parser("prices", help="Collect ERCOT SPP prices")
    _add_date_arguments(prices)
    _add_raw_root_argument(prices)
    _add_force_argument(prices)
    prices.add_argument(
        "--markets",
        nargs="+",
        default=list(DEFAULT_MARKETS),
        help="Market names: DAY_AHEAD_HOURLY and/or REAL_TIME_15_MIN",
    )
    prices.add_argument("--location", default="HB_NORTH")
    prices.add_argument("--location-type", default="Trading Hub")
    prices.add_argument("--chunk-days", type=int, default=31)

    forecasts = subparsers.add_parser(
        "forecasts", help="Collect ERCOT load, wind, and solar forecasts"
    )
    _add_date_arguments(forecasts)
    _add_raw_root_argument(forecasts)
    _add_force_argument(forecasts)
    forecasts.add_argument(
        "--forecast-datasets",
        nargs="+",
        default=list(DEFAULT_FORECASTS),
        help=(
            "Forecast names: SEVEN_DAY_LOAD_FORECAST, "
            "WIND_PRODUCTION_FORECAST, and/or SOLAR_PRODUCTION_FORECAST"
        ),
    )
    forecasts.add_argument(
        "--forecast-chunk-days",
        type=int,
        default=7,
        help="Days per GridStatus.io forecast request (default: 7)",
    )
    forecasts.add_argument(
        "--forecast-mode",
        choices=("as-of-da", "all-vintages"),
        default="as-of-da",
        help="Use one day-ahead as-of vintage per delivery day or all vintages",
    )
    forecasts.add_argument(
        "--asof-days-before",
        type=int,
        default=1,
        help="As-of date offset from delivery date for as-of-da mode",
    )
    forecasts.add_argument(
        "--asof-hour-local",
        type=int,
        default=10,
        help="America/Chicago hour used as the day-ahead as-of cutoff",
    )

    weather = subparsers.add_parser("weather", help="Collect Open-Meteo weather")
    _add_date_arguments(weather)
    _add_raw_root_argument(weather)
    _add_force_argument(weather)
    _add_weather_arguments(weather)

    gas = subparsers.add_parser("gas", help="Collect a FRED gas-price series")
    _add_date_arguments(gas)
    _add_raw_root_argument(gas)
    _add_force_argument(gas)
    gas.add_argument("--series-id", default="DHHNGSP")

    all_sources = subparsers.add_parser("all", help="Collect all three sources")
    _add_date_arguments(all_sources)
    _add_raw_root_argument(all_sources)
    _add_force_argument(all_sources)
    _add_weather_arguments(all_sources)
    all_sources.add_argument(
        "--markets", nargs="+", default=list(DEFAULT_MARKETS)
    )
    all_sources.add_argument("--location", default="HB_NORTH")
    all_sources.add_argument("--location-type", default="Trading Hub")
    all_sources.add_argument("--price-chunk-days", type=int, default=31)
    all_sources.add_argument(
        "--forecast-datasets",
        nargs="+",
        default=list(DEFAULT_FORECASTS),
    )
    all_sources.add_argument("--forecast-chunk-days", type=int, default=7)
    all_sources.add_argument(
        "--forecast-mode",
        choices=("as-of-da", "all-vintages"),
        default="as-of-da",
    )
    all_sources.add_argument("--asof-days-before", type=int, default=1)
    all_sources.add_argument("--asof-hour-local", type=int, default=10)
    all_sources.add_argument("--series-id", default="DHHNGSP")
    return parser


def _weather_locations(values: list[str]) -> tuple[WeatherLocation, ...]:
    if not values:
        return DEFAULT_TEXAS_LOCATIONS
    return tuple(parse_location(value) for value in values)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    raw_root = args.raw_root.resolve()
    results: list[dict[str, Any]] = []

    if args.command in {"prices", "all"}:
        LOGGER.info("Collecting ERCOT prices from %s through %s", start, end)
        results.extend(
            collect_ercot_prices(
                start,
                end,
                markets=tuple(args.markets),
                location=args.location,
                location_type=args.location_type,
                chunk_days=(
                    args.chunk_days
                    if args.command == "prices"
                    else args.price_chunk_days
                ),
                raw_root=raw_root,
                skip_existing=not args.force,
            )
        )

    if args.command in {"forecasts", "all"}:
        LOGGER.info("Collecting ERCOT forecasts from %s through %s", start, end)
        if args.forecast_mode == "as-of-da":
            results.extend(
                collect_ercot_asof_forecasts(
                    start,
                    end,
                    forecasts=tuple(args.forecast_datasets),
                    asof_days_before=args.asof_days_before,
                    asof_hour_local=args.asof_hour_local,
                    raw_root=raw_root,
                    skip_existing=not args.force,
                )
            )
        else:
            results.extend(
                collect_ercot_forecasts(
                    start,
                    end,
                    forecasts=tuple(args.forecast_datasets),
                    chunk_days=args.forecast_chunk_days,
                    raw_root=raw_root,
                    skip_existing=not args.force,
                )
            )

    if args.command in {"weather", "all"}:
        LOGGER.info("Collecting Open-Meteo weather from %s through %s", start, end)
        if args.weather_mode in {
            "previous-runs-day2",
            "previous-runs-hybrid",
        }:
            results.extend(
                collect_openmeteo_previous_runs_weather(
                    start,
                    end,
                    locations=_weather_locations(args.weather_location),
                    chunk_days=args.weather_chunk_days,
                    raw_root=raw_root,
                    skip_existing=not args.force,
                    lead_mode=(
                        "hybrid"
                        if args.weather_mode == "previous-runs-hybrid"
                        else "day2"
                    ),
                )
            )
        elif args.weather_mode == "single-run":
            results.extend(
                collect_openmeteo_single_run_weather(
                    start,
                    end,
                    locations=_weather_locations(args.weather_location),
                    model=args.weather_model,
                    run_hour_utc=args.weather_run_hour_utc,
                    raw_root=raw_root,
                    skip_existing=not args.force,
                )
            )
        else:
            results.extend(
                collect_openmeteo_weather(
                    start,
                    end,
                    locations=_weather_locations(args.weather_location),
                    mode=args.weather_mode,
                    chunk_days=args.weather_chunk_days,
                    raw_root=raw_root,
                    skip_existing=not args.force,
                )
            )

    if args.command in {"gas", "all"}:
        LOGGER.info("Collecting FRED series %s", args.series_id)
        results.extend(
            collect_fred_series(
                start,
                end,
                series_id=args.series_id,
                raw_root=raw_root,
                skip_existing=not args.force,
            )
        )
    return results


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        _load_project_environment()
        results = run(args)
    except Exception:
        LOGGER.exception("Collection failed")
        return 1

    print(json.dumps(results, ensure_ascii=False, indent=2))
    LOGGER.info("Collection completed: %d raw files", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
