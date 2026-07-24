from fastapi import APIRouter, HTTPException

from app.schemas.realtime import (
    RealtimeCollectionResponse,
    RealtimeDayAheadResponse,
    RealtimeStatusResponse,
)
from app.services.realtime_query_service import (
    get_realtime_day_ahead,
    get_realtime_status,
)
from app.services.realtime_scheduler import run_collection_once


router = APIRouter()


@router.post("/v1/realtime/collect", response_model=RealtimeCollectionResponse)
async def collect_now() -> RealtimeCollectionResponse:
    try:
        return RealtimeCollectionResponse(**await run_collection_once())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Realtime collection failed: {exc}") from exc


@router.get("/v1/realtime/status", response_model=RealtimeStatusResponse)
def realtime_status() -> RealtimeStatusResponse:
    try:
        return RealtimeStatusResponse(**get_realtime_status())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Realtime status failed: {exc}") from exc


@router.get("/v1/realtime/day-ahead/{delivery_date}", response_model=RealtimeDayAheadResponse)
def realtime_day_ahead(delivery_date: str) -> RealtimeDayAheadResponse:
    try:
        return RealtimeDayAheadResponse(**get_realtime_day_ahead(delivery_date))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Realtime query failed: {exc}") from exc
