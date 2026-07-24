from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ModelInfoResponse(BaseModel):
    model_loaded: bool
    model_name: str
    model_version: str
    model_fold: str | None = None
    model_dir: str | None = None
    selected_agent: str | None = None
    target: str | None = None
    prediction_unit: str | None = None
    location: str | None = None
    feature_schema_version: str | None = None
    b2a_feature_count: int | None = None
    b2b_feature_count: int | None = None
    signal_probability_threshold: float | None = None
    realtime_enabled: bool
    realtime_refresh_interval_seconds: int
    prediction_horizon: str | None = None
    production_note: str | None = None
    feature_importances: list[dict[str, Any]] | None = None
    error: str | None = None


class PredictionRunResponse(BaseModel):
    delivery_date: str
    row_count: int
    status: str
    model_name: str | None = None
    model_version: str | None = None
    model_fold: str | None = None
    predicted_at_utc: str | None = None
    predictions: list[dict[str, Any]]


class PredictionListResponse(BaseModel):
    delivery_date: str
    row_count: int
    status: str | None = None
    message: str | None = None
    predictions: list[dict[str, Any]]
