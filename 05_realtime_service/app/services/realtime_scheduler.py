from __future__ import annotations

import asyncio
import logging

from app.services.realtime_collectors import collect_realtime_snapshot
from app.services.realtime_store import init_realtime_tables
from app.services.model_service import run_prediction


LOGGER = logging.getLogger(__name__)
COLLECTION_INTERVAL_SECONDS = 15 * 60


async def run_collection_once() -> dict:
    init_realtime_tables()
    result = await collect_realtime_snapshot()
    try:
        prediction = await asyncio.to_thread(
            run_prediction, result["delivery_date_local"]
        )
        result["prediction_status"] = prediction["status"]
        result["prediction_rows"] = prediction["row_count"]
    except Exception as exc:
        result["prediction_status"] = "failed"
        result["prediction_error"] = str(exc)
    return result


async def collection_loop(stop_event: asyncio.Event) -> None:
    init_realtime_tables()
    while not stop_event.is_set():
        try:
            result = await collect_realtime_snapshot()
            try:
                prediction = await asyncio.to_thread(
                    run_prediction, result["delivery_date_local"]
                )
                result["prediction_status"] = prediction["status"]
                result["prediction_rows"] = prediction["row_count"]
            except Exception as exc:
                result["prediction_status"] = "failed"
                result["prediction_error"] = str(exc)
            LOGGER.info("Realtime collection finished: %s", result)
        except Exception:
            LOGGER.exception("Realtime collection failed")

        try:
            await asyncio.wait_for(stop_event.wait(), COLLECTION_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            continue
