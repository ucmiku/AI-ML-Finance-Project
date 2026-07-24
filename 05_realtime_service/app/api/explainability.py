from __future__ import annotations

from datetime import date as Date
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.schemas.explainability import (
    ShapDependenceRow,
    ShapFeatureRankingRow,
    ShapLocalExplanationRow,
)
from app.services.explainability_service import (
    ExplainabilityNotFound,
    get_dependence,
    get_feature_ranking,
    get_local_explanation,
)


router = APIRouter()

WindowParam = Literal["daily", "weekly", "monthly"]
OutputHeadParam = Literal[
    "spread_regression",
    "negative_probability",
    "neutral_probability",
    "positive_probability",
]


@router.get(
    "/explainability/ranking",
    response_model=list[ShapFeatureRankingRow],
)
def shap_feature_ranking(
    window: WindowParam,
    date: Date,
    output_head: OutputHeadParam,
    top_n: int = Query(default=20, gt=0),
) -> list[ShapFeatureRankingRow]:
    try:
        return [
            ShapFeatureRankingRow(
                **{
                    **row,
                    "feature": row.get("feature_name"),
                    "importance": row.get("mean_abs_shap"),
                }
            )
            for row in get_feature_ranking(
                window=window,
                as_of_date=date.isoformat(),
                output_head=output_head,
                top_n=top_n,
            )
        ]
    except ExplainabilityNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/explainability/dependence",
    response_model=list[ShapDependenceRow],
)
def shap_dependence(
    feature_name: str,
    window: WindowParam,
    date: Date,
    output_head: OutputHeadParam,
    color_by: str | None = None,
) -> list[ShapDependenceRow]:
    try:
        return [
            ShapDependenceRow(**{**row, "feature": row.get("feature_name")})
            for row in get_dependence(
                feature_name=feature_name,
                window=window,
                as_of_date=date.isoformat(),
                output_head=output_head,
                color_by=color_by,
            )
        ]
    except ExplainabilityNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/explainability/local",
    response_model=list[ShapLocalExplanationRow],
)
def shap_local_explanation(
    delivery_hour_utc: str,
    output_head: OutputHeadParam,
    top_n: int = Query(default=10, gt=0),
) -> list[ShapLocalExplanationRow]:
    try:
        return [
            ShapLocalExplanationRow(**{**row, "feature": row.get("feature_name")})
            for row in get_local_explanation(
                delivery_hour_utc=delivery_hour_utc,
                output_head=output_head,
                top_n=top_n,
            )
        ]
    except ExplainabilityNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
