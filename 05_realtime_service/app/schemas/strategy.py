from __future__ import annotations

from pydantic import BaseModel


class StrategyRecommendationHour(BaseModel):
    delivery_hour_utc: str
    delivery_date_local: str
    delivery_time_local: str | None = None
    ercot_local_hour: int
    hour: int
    predicted_spread: float | None = None
    p_negative: float | None = None
    p_neutral: float | None = None
    p_positive: float | None = None
    confidence: float | None = None
    direction_confidence: float | None = None
    strategy_confidence: float | None = None
    feature_missing_count: int | None = None
    model_signal: str
    model_numeric_signal: int
    b2a_direction_pass: bool
    extreme_weather_pass: bool
    fixed_extreme_weather_flag: int
    freezing_hour_flag: int | None = None
    extreme_heat_hour_flag: int | None = None
    high_wind_hour_flag: int | None = None
    fixed_compound_extreme_count: int | None = None
    strategy_signal: str
    strategy_numeric_signal: int
    recommendation: str
    trade_strength: str
    reason: str


class StrategyRecommendationResponse(BaseModel):
    delivery_date: str
    strategy_id: int
    strategy_name: str
    strategy_description: str
    probability_threshold: float
    row_count: int
    trade_count: int
    dec_count: int
    inc_count: int
    no_trade_count: int
    avg_direction_confidence: float | None = None
    avg_strategy_confidence: float | None = None
    status: str
    message: str | None = None
    hours: list[StrategyRecommendationHour]
