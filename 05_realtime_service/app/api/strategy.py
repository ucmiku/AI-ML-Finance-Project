from fastapi import APIRouter, HTTPException

from app.schemas.strategy import StrategyRecommendationResponse
from app.services.strategy_service import get_extreme_weather_strategy_recommendation


router = APIRouter()


@router.get(
    "/v1/strategy/extreme-weather/{delivery_date}",
    response_model=StrategyRecommendationResponse,
)
def extreme_weather_strategy(delivery_date: str) -> StrategyRecommendationResponse:
    try:
        return StrategyRecommendationResponse(
            **get_extreme_weather_strategy_recommendation(delivery_date)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Strategy recommendation failed: {exc}",
        ) from exc


@router.get(
    "/v1/trading-advice/extreme-weather/{delivery_date}",
    response_model=StrategyRecommendationResponse,
)
def extreme_weather_trading_advice(delivery_date: str) -> StrategyRecommendationResponse:
    return extreme_weather_strategy(delivery_date)
