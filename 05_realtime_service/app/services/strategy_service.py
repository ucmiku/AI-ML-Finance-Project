from __future__ import annotations

import math
from typing import Any

import pandas as pd

from app.services.model_service import (
    _offline_feature_frame,
    build_realtime_feature_frame,
    get_predictions,
    load_model_bundle,
)


STRATEGY_6_ID = 6
STRATEGY_6_NAME = "ExtremeWeather_Only"
STRATEGY_6_DESCRIPTION = (
    "B2B probability signal plus B2A spread direction confirmation, "
    "trading only when fixed_extreme_weather_flag equals 1."
)


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _clean_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _feature_frame(delivery_date: str) -> pd.DataFrame:
    frame = build_realtime_feature_frame(delivery_date)
    if frame.empty:
        frame = _offline_feature_frame(delivery_date)
    if frame.empty:
        return frame
    return frame.sort_values("delivery_hour_utc").reset_index(drop=True)


def _feature_lookup(delivery_date: str) -> dict[str, dict[str, Any]]:
    frame = _feature_frame(delivery_date)
    if frame.empty:
        return {}
    needed = [
        "delivery_hour_utc",
        "delivery_time_local",
        "ercot_local_hour",
        "freezing_hour_flag",
        "extreme_heat_hour_flag",
        "high_wind_hour_flag",
        "fixed_extreme_weather_flag",
        "fixed_compound_extreme_count",
    ]
    available = [column for column in needed if column in frame.columns]
    lookup: dict[str, dict[str, Any]] = {}
    for row in frame[available].to_dict(orient="records"):
        lookup[str(row["delivery_hour_utc"])] = row
    return lookup


def _model_signal_from_probabilities(
    p_positive: float | None,
    p_negative: float | None,
    threshold: float,
) -> tuple[str, int]:
    if p_positive is not None and p_negative is not None:
        if p_positive >= threshold and p_positive > p_negative:
            return "DEC", 1
        if p_negative >= threshold and p_negative > p_positive:
            return "INC", -1
    return "NO_TRADE", 0


def _recommendation(signal: str) -> str:
    if signal == "DEC":
        return "BUY_DA_SELL_RT"
    if signal == "INC":
        return "SELL_DA_BUY_RT"
    return "NO_TRADE"


def _trade_strength(strategy_confidence: float | None) -> str:
    if strategy_confidence is None or strategy_confidence <= 0:
        return "No Trade"
    if strategy_confidence >= 0.80:
        return "Strong Trade"
    if strategy_confidence >= 0.70:
        return "Trade"
    if strategy_confidence >= 0.60:
        return "Watch"
    return "No Trade"


def _reason(
    model_signal: str,
    b2a_direction_pass: bool,
    extreme_weather_pass: bool,
    strategy_signal: str,
) -> str:
    if strategy_signal != "NO_TRADE":
        return "Extreme weather hour; probability signal and predicted spread direction agree."
    if model_signal == "NO_TRADE":
        return "No trade because classification probability did not pass the threshold."
    if not b2a_direction_pass:
        return "No trade because predicted spread direction does not confirm the probability signal."
    if not extreme_weather_pass:
        return "No trade because this hour is not flagged as extreme weather."
    return "No trade after strategy filters."


def get_extreme_weather_strategy_recommendation(delivery_date: str) -> dict[str, Any]:
    bundle = load_model_bundle()
    threshold = float(bundle["thresholds"].get("probability_threshold", 0.60))
    predictions_payload = get_predictions(delivery_date)
    predictions = predictions_payload.get("predictions", [])
    if not predictions:
        return {
            "delivery_date": delivery_date,
            "strategy_id": STRATEGY_6_ID,
            "strategy_name": STRATEGY_6_NAME,
            "strategy_description": STRATEGY_6_DESCRIPTION,
            "probability_threshold": threshold,
            "row_count": 0,
            "trade_count": 0,
            "dec_count": 0,
            "inc_count": 0,
            "no_trade_count": 0,
            "avg_direction_confidence": None,
            "avg_strategy_confidence": None,
            "status": predictions_payload.get("status", "no_predictions"),
            "message": predictions_payload.get("message"),
            "hours": [],
        }

    features = _feature_lookup(delivery_date)
    hours: list[dict[str, Any]] = []
    for prediction in sorted(predictions, key=lambda row: str(row["delivery_hour_utc"])):
        hour_utc = str(prediction["delivery_hour_utc"])
        feature = features.get(hour_utc, {})
        p_positive = _clean_float(prediction.get("p_positive"))
        p_negative = _clean_float(prediction.get("p_negative"))
        p_neutral = _clean_float(prediction.get("p_neutral"))
        predicted_spread = _clean_float(prediction.get("predicted_spread"))

        model_signal, model_numeric_signal = _model_signal_from_probabilities(
            p_positive,
            p_negative,
            threshold,
        )
        if prediction.get("signal") in {"DEC", "INC", "NO_TRADE"}:
            model_signal = str(prediction["signal"])
            model_numeric_signal = int(prediction.get("numeric_signal") or 0)

        spread_direction = 0
        if predicted_spread is not None and predicted_spread > 0:
            spread_direction = 1
        elif predicted_spread is not None and predicted_spread < 0:
            spread_direction = -1
        b2a_direction_pass = model_numeric_signal != 0 and model_numeric_signal == spread_direction

        extreme_flag = _clean_int(feature.get("fixed_extreme_weather_flag"))
        if extreme_flag is None:
            freezing = _clean_int(feature.get("freezing_hour_flag")) or 0
            heat = _clean_int(feature.get("extreme_heat_hour_flag")) or 0
            wind = _clean_int(feature.get("high_wind_hour_flag")) or 0
            extreme_flag = int(bool(freezing or heat or wind))
        extreme_weather_pass = extreme_flag == 1

        strategy_numeric_signal = (
            model_numeric_signal if b2a_direction_pass and extreme_weather_pass else 0
        )
        strategy_signal = {
            1: "DEC",
            -1: "INC",
        }.get(strategy_numeric_signal, "NO_TRADE")
        direction_confidence = None
        if p_positive is not None and p_negative is not None:
            direction_confidence = max(p_positive, p_negative)
        strategy_confidence = (
            direction_confidence if strategy_signal != "NO_TRADE" else 0.0
        )

        local_hour = _clean_int(
            feature.get("ercot_local_hour")
            if feature
            else prediction.get("ercot_local_hour", prediction.get("hour"))
        )
        hours.append(
            {
                "delivery_hour_utc": hour_utc,
                "delivery_date_local": str(
                    prediction.get("delivery_date_local", delivery_date)
                ),
                "delivery_time_local": feature.get("delivery_time_local")
                or prediction.get("delivery_time_local"),
                "ercot_local_hour": local_hour if local_hour is not None else 0,
                "hour": local_hour if local_hour is not None else 0,
                "predicted_spread": predicted_spread,
                "p_negative": p_negative,
                "p_neutral": p_neutral,
                "p_positive": p_positive,
                "confidence": _clean_float(
                    prediction.get("confidence", prediction.get("prediction_confidence"))
                ),
                "direction_confidence": direction_confidence,
                "strategy_confidence": strategy_confidence,
                "feature_missing_count": _clean_int(prediction.get("feature_missing_count")),
                "model_signal": model_signal,
                "model_numeric_signal": model_numeric_signal,
                "b2a_direction_pass": bool(b2a_direction_pass),
                "extreme_weather_pass": bool(extreme_weather_pass),
                "fixed_extreme_weather_flag": extreme_flag,
                "freezing_hour_flag": _clean_int(feature.get("freezing_hour_flag")),
                "extreme_heat_hour_flag": _clean_int(feature.get("extreme_heat_hour_flag")),
                "high_wind_hour_flag": _clean_int(feature.get("high_wind_hour_flag")),
                "fixed_compound_extreme_count": _clean_int(
                    feature.get("fixed_compound_extreme_count")
                ),
                "strategy_signal": strategy_signal,
                "strategy_numeric_signal": strategy_numeric_signal,
                "recommendation": _recommendation(strategy_signal),
                "trade_strength": _trade_strength(strategy_confidence),
                "reason": _reason(
                    model_signal,
                    bool(b2a_direction_pass),
                    bool(extreme_weather_pass),
                    strategy_signal,
                ),
            }
        )

    trade_count = sum(1 for row in hours if row["strategy_signal"] != "NO_TRADE")
    dec_count = sum(1 for row in hours if row["strategy_signal"] == "DEC")
    inc_count = sum(1 for row in hours if row["strategy_signal"] == "INC")
    direction_values = [
        row["direction_confidence"]
        for row in hours
        if row["direction_confidence"] is not None
    ]
    strategy_values = [
        row["strategy_confidence"]
        for row in hours
        if row["strategy_signal"] != "NO_TRADE"
        and row["strategy_confidence"] is not None
    ]
    return {
        "delivery_date": delivery_date,
        "strategy_id": STRATEGY_6_ID,
        "strategy_name": STRATEGY_6_NAME,
        "strategy_description": STRATEGY_6_DESCRIPTION,
        "probability_threshold": threshold,
        "row_count": len(hours),
        "trade_count": trade_count,
        "dec_count": dec_count,
        "inc_count": inc_count,
        "no_trade_count": len(hours) - trade_count,
        "avg_direction_confidence": (
            sum(direction_values) / len(direction_values) if direction_values else None
        ),
        "avg_strategy_confidence": (
            sum(strategy_values) / len(strategy_values) if strategy_values else None
        ),
        "status": "success",
        "message": None,
        "hours": hours,
    }
