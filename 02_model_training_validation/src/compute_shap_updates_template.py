# -*- coding: utf-8 -*-
"""Template for future C1 SHAP update job.

This file is intentionally a template. It is not run during packaging.
It shows the required structure for daily/weekly/monthly SHAP production.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import shap
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install shap before running this production update job") from exc


def validate_features(frame: pd.DataFrame, feature_order: list[str]) -> None:
    missing = sorted(set(feature_order) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(feature_order))
    if missing:
        raise ValueError(f"Missing features: {missing[:20]}")
    # Extra columns may be allowed in raw feature table, but only feature_order is sent to model.


def compute_regression_shap(pipeline: dict, feature_frame: pd.DataFrame) -> pd.DataFrame:
    features = pipeline["features"]
    validate_features(feature_frame, features)
    x = pipeline["imputer"].transform(feature_frame[features])
    explainer = shap.TreeExplainer(pipeline["model"])
    values = explainer.shap_values(x)
    return pd.DataFrame(values, columns=features)


def compute_classifier_shap_for_class(pipeline: dict, feature_frame: pd.DataFrame, class_index: int) -> pd.DataFrame:
    features = pipeline["features"]
    validate_features(feature_frame, features)
    x = pipeline["imputer"].transform(feature_frame[features])
    explainer = shap.TreeExplainer(pipeline["model"])
    values = explainer.shap_values(x)
    if isinstance(values, list):
        class_values = values[class_index]
    else:
        class_values = values[:, :, class_index]
    return pd.DataFrame(class_values, columns=features)


def aggregate_ranking(shap_long: pd.DataFrame, window_type: str) -> pd.DataFrame:
    grouped = shap_long.groupby(["output_head", "feature_name", "feature_group"], as_index=False).agg(
        mean_abs_shap=("abs_shap", "mean"),
        mean_shap=("shap_value", "mean"),
        n_rows=("delivery_hour_utc", "nunique"),
    )
    grouped["rank"] = grouped.groupby("output_head")["mean_abs_shap"].rank(method="first", ascending=False).astype(int)
    grouped["window_type"] = window_type
    return grouped.sort_values(["output_head", "rank"])


def main() -> None:
    # Fill these paths in production scheduler config.
    model_dir = Path("models/c1_prediction_agent")
    feature_table = Path("daily_prediction_features.parquet")
    output_dir = Path("outputs/shap")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Example only: real deployment should load approved model/version for the prediction date.
    reg = joblib.load(model_dir / "b2a_xgboost_regressor.joblib")
    clf = joblib.load(model_dir / "b2b_xgboost_classifier.joblib")
    features = pd.read_parquet(feature_table)

    # Compute each output separately. Do not add different output heads together.
    # Save local long-format table, then aggregate daily/weekly/monthly.
    raise NotImplementedError("Wire this template to production model paths and scheduler inputs.")


if __name__ == "__main__":
    main()
