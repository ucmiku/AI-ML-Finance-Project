from fastapi import APIRouter, HTTPException

from app.schemas.market import DataStatusResponse, DayAheadForecastResponse
from app.services.market_service import get_data_status, get_day_ahead_forecast


router = APIRouter()


@router.get("/v1/forecasts/day-ahead/{delivery_date}", response_model=DayAheadForecastResponse)
def day_ahead_forecast(delivery_date: str) -> DayAheadForecastResponse:
    try:
        return DayAheadForecastResponse(**get_day_ahead_forecast(delivery_date))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecast query failed: {exc}") from exc


@router.get("/v1/data-status/{delivery_date}", response_model=DataStatusResponse)
def data_status(delivery_date: str) -> DataStatusResponse:
    try:
        return DataStatusResponse(**get_data_status(delivery_date))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Status query failed: {exc}") from exc
