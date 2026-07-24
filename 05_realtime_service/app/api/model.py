from datetime import date as Date
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.schemas.model import (
    ModelInfoResponse,
    PredictionListResponse,
    PredictionRunResponse,
)
from app.services.explainability_service import (
    ExplainabilityNotFound,
    get_feature_ranking,
)
from app.services.model_service import get_model_info, get_predictions, run_prediction


router = APIRouter()
WindowParam = Literal["daily", "weekly", "monthly"]
OutputHeadParam = Literal[
    "spread_regression",
    "negative_probability",
    "neutral_probability",
    "positive_probability",
]


@router.get("/v1/model/info", response_model=ModelInfoResponse)
def model_info(
    window: WindowParam | None = None,
    date: Date | None = None,
    output_head: OutputHeadParam | None = None,
    top_n: int = Query(default=20, gt=0),
) -> ModelInfoResponse:
    payload = get_model_info()
    if window is not None or date is not None or output_head is not None:
        if window is None or date is None or output_head is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "window, date, and output_head must be provided together "
                    "when requesting feature_importances"
                ),
            )
        try:
            ranking = get_feature_ranking(
                window=window,
                as_of_date=date.isoformat(),
                output_head=output_head,
                top_n=top_n,
            )
            payload["feature_importances"] = [
                {
                    **row,
                    "feature": row.get("feature_name"),
                    "importance": row.get("mean_abs_shap"),
                }
                for row in ranking
            ]
        except ExplainabilityNotFound:
            payload["feature_importances"] = []
    return ModelInfoResponse(**payload)


@router.post("/v1/model/predict/{delivery_date}", response_model=PredictionRunResponse)
def predict_day(delivery_date: str) -> PredictionRunResponse:
    try:
        return PredictionRunResponse(**run_prediction(delivery_date))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@router.get("/v1/predictions/day-ahead/{delivery_date}", response_model=PredictionListResponse)
def predictions_day_ahead(delivery_date: str) -> PredictionListResponse:
    try:
        return PredictionListResponse(**get_predictions(delivery_date))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction query failed: {exc}") from exc
