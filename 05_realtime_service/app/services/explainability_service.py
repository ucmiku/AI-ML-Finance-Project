from __future__ import annotations

import math
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import xgboost as xgb

from app.services.model_service import (
    _offline_feature_frame,
    _transform_with_compatible_imputer,
    build_realtime_feature_frame,
    load_model_bundle,
    run_offline_prediction,
    run_prediction,
)


WindowType = Literal["daily", "weekly", "monthly"]
OutputHead = Literal[
    "spread_regression",
    "negative_probability",
    "neutral_probability",
    "positive_probability",
]

ERCOT_TZ = ZoneInfo("America/Chicago")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SHAP_PACKAGE_ROOT = PROJECT_ROOT / "02_model_training_validation"
SHAP_OUTPUT_ROOT = SHAP_PACKAGE_ROOT / "outputs" / "shap"
SHAP_SAMPLE_ROOT = SHAP_PACKAGE_ROOT / "sample_outputs"
SHAP_REPORT_SOURCE_ROOT = (
    SHAP_PACKAGE_ROOT
    / "02_model_training_validation"
    / "reports"
    / "shap_figures"
    / "source_tables"
)

OUTPUT_HEAD_CONFIG = {
    "spread_regression": {
        "model_key": "regressor",
        "features_key": "b2a_features",
        "class_index": None,
    },
    "negative_probability": {
        "model_key": "classifier",
        "features_key": "b2b_features",
        "class_index": 0,
    },
    "neutral_probability": {
        "model_key": "classifier",
        "features_key": "b2b_features",
        "class_index": 2,
    },
    "positive_probability": {
        "model_key": "classifier",
        "features_key": "b2b_features",
        "class_index": 4,
    },
}

OUTPUT_HEAD_ALIASES = {
    "spread_regression": {
        "spread_regression",
        "q50_continuous_spread",
    },
    "negative_probability": {
        "negative_probability",
        "negative_probability_C1_logit",
    },
    "neutral_probability": {
        "neutral_probability",
        "no_trade_probability_C3_logit",
    },
    "positive_probability": {
        "positive_probability",
        "positive_probability_C5_logit",
    },
}

RANKING_COLUMNS = [
    "as_of_date",
    "window_type",
    "window_start_utc",
    "window_end_utc",
    "output_head",
    "feature_name",
    "feature_group",
    "mean_abs_shap",
    "mean_shap",
    "rank",
    "n_rows",
    "model_version",
    "feature_version",
]

DEPENDENCE_COLUMNS = [
    "as_of_date",
    "window_type",
    "output_head",
    "feature_name",
    "delivery_hour_utc",
    "feature_value",
    "shap_value",
    "predicted_spread",
    "p_negative",
    "p_positive",
    "signal",
    "color_by",
]

LOCAL_COLUMNS = [
    "delivery_hour_utc",
    "as_of_date",
    "output_head",
    "feature_name",
    "feature_group",
    "feature_value",
    "shap_value",
    "abs_shap",
    "rank_within_prediction",
    "predicted_spread",
    "p_negative",
    "p_neutral",
    "p_positive",
    "signal",
]


class ExplainabilityNotFound(Exception):
    pass


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _ordered_records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    return [
        {column: _jsonable(record.get(column)) for column in columns}
        for record in frame[columns].to_dict(orient="records")
    ]


def _delivery_date_from_utc_hour(delivery_hour_utc: str) -> str:
    try:
        hour = datetime.fromisoformat(
            delivery_hour_utc.strip().replace("Z", "+00:00"),
        )
    except ValueError as exc:
        raise ExplainabilityNotFound(
            f"Invalid delivery_hour_utc: {delivery_hour_utc}",
        ) from exc
    return hour.astimezone(ERCOT_TZ).date().isoformat()


def _normalize_utc_hour(delivery_hour_utc: str) -> str:
    try:
        hour = datetime.fromisoformat(
            delivery_hour_utc.strip().replace("Z", "+00:00"),
        )
    except ValueError as exc:
        raise ExplainabilityNotFound(
            f"Invalid delivery_hour_utc: {delivery_hour_utc}",
        ) from exc
    if hour.tzinfo is None:
        hour = hour.replace(tzinfo=ZoneInfo("UTC"))
    return hour.astimezone(ZoneInfo("UTC")).replace(
        minute=0,
        second=0,
        microsecond=0,
    ).isoformat().replace("+00:00", "Z")


def _utc_window_bounds(frame: pd.DataFrame) -> tuple[str, str]:
    hours = pd.to_datetime(frame["delivery_hour_utc"], utc=True).sort_values()
    start = hours.iloc[0].isoformat().replace("+00:00", "Z")
    end = (hours.iloc[-1] + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    return start, end


@lru_cache(maxsize=1)
def _feature_group_map() -> dict[str, str]:
    candidates = [
        SHAP_PACKAGE_ROOT / "02_model_training_validation" / "metrics" / "shap_summary.csv",
        SHAP_REPORT_SOURCE_ROOT / "grouped_feature_importance_v3.csv",
    ]
    mapping: dict[str, str] = {}
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if {"feature_name", "feature_group"}.issubset(frame.columns):
            for row in frame[["feature_name", "feature_group"]].dropna().to_dict(
                orient="records",
            ):
                mapping[str(row["feature_name"])] = str(row["feature_group"])
    return mapping


def _fallback_feature_group(feature_name: str) -> str:
    if feature_name in _feature_group_map():
        return _feature_group_map()[feature_name]
    if feature_name.startswith(("load_", "net_load_", "renewable_")):
        return "Load"
    if feature_name.startswith("wind_"):
        return "Wind"
    if feature_name.startswith("solar_"):
        return "Solar and Net Load"
    if "temperature" in feature_name or "humidity" in feature_name:
        return "Raw Weather"
    if "weather" in feature_name or "wind_gust" in feature_name:
        return "Extreme Weather"
    if feature_name.startswith("gas_") or feature_name == "gas_price":
        return "Gas"
    if feature_name.startswith("spread_"):
        return "Historical Spread"
    if feature_name.startswith(("hour_", "dow_", "month_", "is_")):
        return "Calendar"
    return "Other"


def _prediction_context(delivery_date: str) -> dict[str, dict[str, Any]]:
    prediction = run_prediction(delivery_date)
    if prediction.get("row_count", 0) == 0:
        prediction = run_offline_prediction(delivery_date)
    if prediction.get("row_count", 0) == 0:
        raise ExplainabilityNotFound(
            f"No predictions available for delivery date {delivery_date}",
        )
    return {
        row["delivery_hour_utc"]: row
        for row in prediction.get("predictions", [])
    }


def _dynamic_contributions(
    delivery_date: str,
    output_head: OutputHead,
) -> tuple[pd.DataFrame, list[str], np.ndarray, dict[str, dict[str, Any]]]:
    if output_head not in OUTPUT_HEAD_CONFIG:
        raise ExplainabilityNotFound(f"Unknown output_head: {output_head}")
    frame = build_realtime_feature_frame(delivery_date)
    if frame.empty:
        frame = _offline_feature_frame(delivery_date)
    if frame.empty:
        raise ExplainabilityNotFound(
            f"No feature rows available for delivery date {delivery_date}",
        )

    bundle = load_model_bundle()
    config = OUTPUT_HEAD_CONFIG[output_head]
    features = list(bundle[config["features_key"]])
    pipe = bundle[config["model_key"]]
    model = pipe["model"] if isinstance(pipe, dict) else pipe
    matrix = (
        _transform_with_compatible_imputer(pipe["imputer"], frame[features])
        if isinstance(pipe, dict)
        else frame[features]
    )
    dmatrix = xgb.DMatrix(matrix, feature_names=features)
    contrib = model.get_booster().predict(dmatrix, pred_contribs=True)
    if config["class_index"] is None:
        shap_values = contrib[:, :-1]
    elif contrib.ndim == 3 and contrib.shape[1] == 5:
        shap_values = contrib[:, int(config["class_index"]), :-1]
    elif contrib.ndim == 3:
        shap_values = contrib[:, :-1, int(config["class_index"])]
    else:
        raise ExplainabilityNotFound(
            f"Unexpected SHAP contribution shape for {output_head}: {contrib.shape}",
        )
    context = _prediction_context(delivery_date)
    return frame, features, shap_values, context


def _read_csv_records(paths: list[Path], columns: list[str]) -> pd.DataFrame | None:
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if set(columns).issubset(frame.columns):
            return frame[columns].copy()
    return None


def _filter_output_head(frame: pd.DataFrame, output_head: OutputHead) -> pd.DataFrame:
    aliases = OUTPUT_HEAD_ALIASES[output_head]
    return frame[frame["output_head"].isin(aliases)].copy()


def _ranking_from_files(
    window: WindowType,
    as_of_date: str,
    output_head: OutputHead,
    top_n: int,
) -> list[dict[str, Any]] | None:
    paths = [
        SHAP_OUTPUT_ROOT / f"shap_{window}_feature_ranking.csv",
    ]
    if window == "daily":
        paths.append(SHAP_SAMPLE_ROOT / "shap_daily_feature_ranking_sample.csv")
    frame = _read_csv_records(paths, RANKING_COLUMNS)
    if frame is None:
        return None
    frame = _filter_output_head(frame, output_head)
    frame = frame[
        (frame["as_of_date"].astype(str) == as_of_date)
        & (frame["window_type"].astype(str) == window)
    ].copy()
    if frame.empty:
        return None
    frame = frame.sort_values("rank").head(top_n)
    frame["output_head"] = output_head
    return _ordered_records(frame, RANKING_COLUMNS)


def _dependence_from_files(
    feature_name: str,
    window: WindowType,
    as_of_date: str,
    output_head: OutputHead,
    color_by: str | None,
) -> list[dict[str, Any]] | None:
    paths = [
        SHAP_OUTPUT_ROOT / f"shap_dependence_{window}.csv",
    ]
    if window == "daily":
        paths.append(SHAP_SAMPLE_ROOT / "shap_dependence_sample.csv")
    frame = _read_csv_records(paths, DEPENDENCE_COLUMNS)
    if frame is None:
        return None
    frame = _filter_output_head(frame, output_head)
    frame = frame[
        (frame["as_of_date"].astype(str) == as_of_date)
        & (frame["window_type"].astype(str) == window)
        & (frame["feature_name"].astype(str) == feature_name)
    ].copy()
    if frame.empty:
        return None
    frame["output_head"] = output_head
    frame["color_by"] = color_by
    return _ordered_records(frame, DEPENDENCE_COLUMNS)


def _local_from_files(
    delivery_hour_utc: str,
    output_head: OutputHead,
    top_n: int,
) -> list[dict[str, Any]] | None:
    paths = [
        SHAP_OUTPUT_ROOT / "shap_local_explanations.csv",
        SHAP_SAMPLE_ROOT / "shap_local_explanations_sample.csv",
    ]
    frame = _read_csv_records(paths, LOCAL_COLUMNS)
    if frame is None:
        return None
    frame = _filter_output_head(frame, output_head)
    normalized_hour = _normalize_utc_hour(delivery_hour_utc)
    frame["_normalized_delivery_hour_utc"] = frame["delivery_hour_utc"].astype(
        str,
    ).map(_normalize_utc_hour)
    frame = frame[frame["_normalized_delivery_hour_utc"] == normalized_hour].copy()
    if frame.empty:
        return None
    frame["delivery_hour_utc"] = normalized_hour
    frame = frame.sort_values("rank_within_prediction").head(top_n)
    frame["output_head"] = output_head
    return _ordered_records(frame, LOCAL_COLUMNS)


def get_feature_ranking(
    *,
    window: WindowType,
    as_of_date: str,
    output_head: OutputHead,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    if top_n <= 0:
        raise ExplainabilityNotFound("top_n must be positive")

    file_records = _ranking_from_files(window, as_of_date, output_head, top_n)
    if file_records is not None:
        return file_records

    frame, features, shap_values, _ = _dynamic_contributions(as_of_date, output_head)
    window_start, window_end = _utc_window_bounds(frame)
    summary = pd.DataFrame(
        {
            "feature_name": features,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
        },
    )
    summary["feature_group"] = summary["feature_name"].map(_fallback_feature_group)
    summary = summary.sort_values("mean_abs_shap", ascending=False).reset_index(
        drop=True,
    )
    summary["rank"] = summary.index + 1
    summary = summary.head(top_n)
    bundle = load_model_bundle()
    summary["as_of_date"] = as_of_date
    summary["window_type"] = window
    summary["window_start_utc"] = window_start
    summary["window_end_utc"] = window_end
    summary["output_head"] = output_head
    summary["n_rows"] = len(frame)
    summary["model_version"] = bundle["metadata"].get("model_version", "v3")
    summary["feature_version"] = bundle["schema"].get(
        "schema_version",
        "feature_schema",
    )
    return _ordered_records(summary, RANKING_COLUMNS)


def get_dependence(
    *,
    feature_name: str,
    window: WindowType,
    as_of_date: str,
    output_head: OutputHead,
    color_by: str | None = None,
) -> list[dict[str, Any]]:
    file_records = _dependence_from_files(
        feature_name,
        window,
        as_of_date,
        output_head,
        color_by,
    )
    if file_records is not None:
        return file_records

    frame, features, shap_values, context = _dynamic_contributions(
        as_of_date,
        output_head,
    )
    if feature_name not in features:
        raise ExplainabilityNotFound(f"Unknown feature_name: {feature_name}")

    feature_index = features.index(feature_name)
    rows: list[dict[str, Any]] = []
    for row_index, row in frame.reset_index(drop=True).iterrows():
        hour = str(row["delivery_hour_utc"])
        prediction = context.get(hour, {})
        rows.append(
            {
                "as_of_date": as_of_date,
                "window_type": window,
                "output_head": output_head,
                "feature_name": feature_name,
                "delivery_hour_utc": hour,
                "feature_value": _jsonable(row.get(feature_name)),
                "shap_value": float(shap_values[row_index, feature_index]),
                "predicted_spread": _jsonable(prediction.get("predicted_spread")),
                "p_negative": _jsonable(prediction.get("p_negative")),
                "p_positive": _jsonable(prediction.get("p_positive")),
                "signal": prediction.get("signal"),
                "color_by": color_by,
            },
        )
    return rows


def get_local_explanation(
    *,
    delivery_hour_utc: str,
    output_head: OutputHead,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    if top_n <= 0:
        raise ExplainabilityNotFound("top_n must be positive")

    normalized_hour = _normalize_utc_hour(delivery_hour_utc)
    file_records = _local_from_files(normalized_hour, output_head, top_n)
    if file_records is not None:
        return file_records

    delivery_date = _delivery_date_from_utc_hour(normalized_hour)
    frame, features, shap_values, context = _dynamic_contributions(
        delivery_date,
        output_head,
    )
    hour_rows = frame.reset_index(drop=True)
    matches = hour_rows.index[hour_rows["delivery_hour_utc"] == normalized_hour]
    if len(matches) == 0:
        raise ExplainabilityNotFound(
            f"No SHAP local data available for {normalized_hour}",
        )

    row_index = int(matches[0])
    prediction = context.get(normalized_hour, {})
    feature_rows = []
    for feature_index, feature_name in enumerate(features):
        shap_value = float(shap_values[row_index, feature_index])
        feature_rows.append(
            {
                "delivery_hour_utc": normalized_hour,
                "as_of_date": delivery_date,
                "output_head": output_head,
                "feature_name": feature_name,
                "feature_group": _fallback_feature_group(feature_name),
                "feature_value": _jsonable(hour_rows.iloc[row_index].get(feature_name)),
                "shap_value": shap_value,
                "abs_shap": abs(shap_value),
                "predicted_spread": _jsonable(prediction.get("predicted_spread")),
                "p_negative": _jsonable(prediction.get("p_negative")),
                "p_neutral": _jsonable(prediction.get("p_neutral")),
                "p_positive": _jsonable(prediction.get("p_positive")),
                "signal": prediction.get("signal"),
            },
        )
    out = pd.DataFrame(feature_rows).sort_values("abs_shap", ascending=False).head(
        top_n,
    )
    out = out.reset_index(drop=True)
    out["rank_within_prediction"] = out.index + 1
    return _ordered_records(out, LOCAL_COLUMNS)
