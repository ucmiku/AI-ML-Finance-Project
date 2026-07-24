from __future__ import annotations

import argparse
import asyncio
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill ERCOT DA/RT hourly price actuals into realtime SQLite."
    )
    parser.add_argument("--start", default="2026-07-01", type=parse_date)
    parser.add_argument("--end", default="2026-07-22", type=parse_date)
    parser.add_argument("--location", default="HB_NORTH")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    # Import after load_dotenv and with PYTHONPATH/app-dir configured.
    from app.services.realtime_collectors import collect_price_actuals
    from app.services.realtime_store import init_realtime_tables

    init_realtime_tables()
    rows = collect_price_actuals(
        args.start,
        args.end + timedelta(days=1),
        location=args.location,
    )
    complete = [
        row
        for row in rows
        if row.get("da_price_usd_per_mwh") is not None
        and row.get("rt_price_usd_per_mwh") is not None
    ]
    print(
        f"Backfilled {len(rows)} hourly rows for {args.location}; "
        f"{len(complete)} rows have both DA and RT prices."
    )


if __name__ == "__main__":
    asyncio.run(main())
