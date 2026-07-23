# -*- coding: utf-8 -*-
"""Run C1 XGBoost Prediction Agent inference from packaged artifacts.

The packaged fold models reproduce the 2025 OOF workflow. For research replay,
choose the fold model matching the validation period. For production, retrain
with the frozen config on the approved history window and keep the same output
schema and threshold rules.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


CLASS_COLUMNS = ["p_c1", "p_c2", "p_c3", "p_c4", "p_c5"]


def _load_features(schema_path: Path, component: str) -> list[str]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return list(schema[component]["feature_order"])


def _predict_proba_5(model, X) -> np.ndarray:
    raw = model.predict_proba(X)
    probabilities = np.zeros((len(X), 5), dtype=float)
    classes = getattr(model, "classes_", np.arange(5)).astype(int)
    # XGBoost classifier was trained on zero-based class ids.
    probabilities[:, classes] = raw
    return probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description="Run packaged C1 inference")
    parser.add_argument("--input", required=True, type=Path, help="Feature table CSV or Parquet")
    parser.add_argument("--output", required=True, type=Path, help="Output prediction CSV")
    parser.add_argument("--model-dir", default=Path("models/c1_prediction_agent"), type=Path)
    parser.add_argument("--fold", default="validation_fold_3", choices=["validation_fold_1", "validation_fold_2", "validation_fold_3"])
    args = parser.parse_args()

    frame = pd.read_parquet(args.input) if args.input.suffix.lower() == ".parquet" else pd.read_csv(args.input)
    schema_path = args.model_dir / "feature_schema.json"
    b2a_features = _load_features(schema_path, "b2a_xgboost_regressor")
    b2b_features = _load_features(schema_path, "b2b_xgboost_classifier")

    missing = sorted((set(b2a_features) | set(b2b_features)) - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required feature columns: {missing[:20]}")

    reg = joblib.load(args.model_dir / "b2a_xgboost_regressor" / f"{args.fold}_pipeline.joblib")
    clf = joblib.load(args.model_dir / "b2b_xgboost_classifier" / f"{args.fold}_pipeline.joblib")

    predicted_spread = reg["model"].predict(reg["imputer"].transform(frame[b2a_features])) if isinstance(reg, dict) else reg.predict(frame[b2a_features])
    if isinstance(clf, dict):
        X_clf = clf["imputer"].transform(frame[b2b_features])
        probabilities = _predict_proba_5(clf["model"], X_clf)
    else:
        probabilities = _predict_proba_5(clf, frame[b2b_features])

    out = pd.DataFrame({"predicted_spread": predicted_spread})
    if "delivery_hour_utc" in frame.columns:
        out.insert(0, "delivery_hour_utc", frame["delivery_hour_utc"].to_numpy())
    for idx, col in enumerate(CLASS_COLUMNS):
        out[col] = probabilities[:, idx]
    out["p_negative"] = out["p_c1"] + out["p_c2"]
    out["p_neutral"] = out["p_c3"]
    out["p_positive"] = out["p_c4"] + out["p_c5"]
    out["predicted_class"] = np.argmax(probabilities, axis=1) + 1
    out["confidence"] = probabilities.max(axis=1)
    out["signal"] = "NO_TRADE"
    out.loc[(out["p_positive"] >= 0.60) & (out["p_positive"] > out["p_negative"]), "signal"] = "DEC"
    out.loc[(out["p_negative"] >= 0.60) & (out["p_negative"] > out["p_positive"]), "signal"] = "INC"
    out["model_name"] = "C1_XGBoost_Prediction_Agent"
    out["model_version"] = "v3"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
