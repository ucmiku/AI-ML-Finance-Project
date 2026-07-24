from __future__ import annotations

from pydantic import BaseModel


class ForecastHour(BaseModel):
    delivery_hour_utc: str
    delivery_date_local: str
    delivery_time_local: str
    ercot_local_hour: int
    hour: int | None = None
    is_dst: int
    decision_time_utc: str
    gas_price_usd_per_mmbtu: float | None = None
    temperature_dfw_mean_c: float | None = None
    wind_speed_dfw_mean_ms: float | None = None
    cloud_cover_dfw_mean_pct: float | None = None
    load_system_total_mw: float | None = None
    load_coast_mw: float | None = None
    load_east_mw: float | None = None
    load_far_west_mw: float | None = None
    load_north_mw: float | None = None
    load_north_central_mw: float | None = None
    load_south_central_mw: float | None = None
    load_southern_mw: float | None = None
    load_west_mw: float | None = None
    wind_stwpf_system_wide_mw: float | None = None
    solar_pvgrpp_system_mw: float | None = None
    spread_usd_per_mwh: float | None = None
    rt_above_da: int | None = None
    split_name: str | None = None
    predicted_spread: float | None = None
    confidence: float | None = None
    prediction_signal: str | None = None
    prediction_confidence: float | None = None
    feature_missing_count: int | None = None
    p_negative: float | None = None
    p_neutral: float | None = None
    p_positive: float | None = None
    model_name: str | None = None
    model_version: str | None = None
    predicted_at_utc: str | None = None
    gas_price_z30: float | None = None
    load_system_z30_same_hour: float | None = None
    net_load_z30_same_hour: float | None = None


class DayAheadForecastResponse(BaseModel):
    delivery_date: str
    table: str
    row_count: int
    hours: list[ForecastHour]


class DataStatusResponse(BaseModel):
    delivery_date: str
    table: str
    row_count: int
    expected_hours: int
    missing_hours: int
    complete_day: bool
    first_delivery_hour_utc: str | None = None
    last_delivery_hour_utc: str | None = None
    has_all_three_forecasts: bool
    load_pre_dam_valid: bool
    wind_pre_dam_valid: bool
    solar_pre_dam_valid: bool
    all_issue_times_pre_dam_valid: bool
