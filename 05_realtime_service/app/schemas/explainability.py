from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ShapFeatureRankingRow(BaseModel):
    as_of_date: str
    window_type: str
    window_start_utc: str
    window_end_utc: str
    output_head: str
    feature_name: str
    feature: str | None = None
    feature_group: str
    mean_abs_shap: float
    importance: float | None = None
    mean_shap: float
    rank: int
    n_rows: int
    model_version: str
    feature_version: str


class ShapDependenceRow(BaseModel):
    as_of_date: str
    window_type: str
    output_head: str
    feature_name: str
    feature: str | None = None
    delivery_hour_utc: str
    feature_value: Any = None
    shap_value: float
    predicted_spread: float | None = None
    p_negative: float | None = None
    p_positive: float | None = None
    signal: str | None = None
    color_by: str | None = None


class ShapLocalExplanationRow(BaseModel):
    delivery_hour_utc: str
    as_of_date: str
    output_head: str
    feature_name: str
    feature: str | None = None
    feature_group: str
    feature_value: Any = None
    shap_value: float
    abs_shap: float
    rank_within_prediction: int
    predicted_spread: float | None = None
    p_negative: float | None = None
    p_neutral: float | None = None
    p_positive: float | None = None
    signal: str | None = None
