# -*- coding: utf-8 -*-
"""C1 Prediction Agent inference and SHAP interface for FastAPI integration."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    import shap
except Exception:  # pragma: no cover
    shap = None

CLASS_NAMES = ["C1", "C2", "C3", "C4", "C5"]


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


class C1PredictionAgent:
    def __init__(self, package_root: str | Path | None = None) -> None:
        if package_root is None:
            package_root = Path(__file__).resolve().parents[2]
        self.package_root = Path(package_root).resolve()
        self.model_dir = self.package_root / "models" / "c1_prediction_agent"
        self.feature_schema = self._read_json("feature_schema.json")
        self.class_mapping = self._read_json("class_mapping.json")
        self.thresholds = self._read_json("thresholds.json")
        self.model_metadata = self._read_json("model_metadata.json")
        self.deployment_status = self.model_metadata["deployment_status"]
        self.model_version = self.model_metadata["model_version"]
        self.b2a_features = list(self.feature_schema["b2a_regression"]["feature_order"])
        self.b2b_features = list(self.feature_schema["b2b_classifier"]["feature_order"])
        self.all_required_features = list(self.feature_schema["all_required_features"])
        self.regression_folds = []
        self.classifier_folds = []
        self._load_models()

    def _read_json(self, name: str) -> dict[str, Any]:
        return json.loads((self.model_dir / name).read_text(encoding="utf-8"))

    def _load_models(self) -> None:
        reg_dir = self.model_dir / "b2a_regression"
        clf_dir = self.model_dir / "b2b_classifier"
        for fold in [1, 2, 3]:
            reg_path = reg_dir / f"fold_{fold}_pipeline.joblib"
            clf_path = clf_dir / f"fold_{fold}_pipeline.joblib"
            if not reg_path.exists() or not clf_path.exists():
                raise FileNotFoundError(f"Missing model artifact: {reg_path} or {clf_path}")
            self.regression_folds.append(joblib.load(reg_path))
            self.classifier_folds.append(joblib.load(clf_path))

    def validate_features(self, features: dict[str, Any]) -> None:
        provided = set(features)
        required = set(self.all_required_features)
        missing = sorted(required - provided)
        extra = sorted(provided - required)
        if missing:
            raise ValueError(f"Missing required C1 features: {missing[:30]}")
        if extra:
            raise ValueError(f"Unexpected extra C1 features: {extra[:30]}")

    def _frame_from_features(self, features: dict[str, Any]) -> pd.DataFrame:
        self.validate_features(features)
        clean = {k: (np.nan if features[k] is None else features[k]) for k in self.all_required_features}
        return pd.DataFrame([clean])

    def _predict_regression_fold(self, artifact: dict[str, Any], row: pd.DataFrame) -> float:
        ordered = artifact.get("features", self.b2a_features)
        x = artifact["imputer"].transform(row[ordered])
        return float(artifact["model"].predict(x)[0])

    def _predict_classifier_fold(self, artifact: dict[str, Any], row: pd.DataFrame) -> np.ndarray:
        ordered = artifact.get("features", self.b2b_features)
        x = artifact["imputer"].transform(row[ordered])
        model = artifact["model"]
        raw = model.predict_proba(x)
        probs = np.zeros((1, 5), dtype=float)
        classes = getattr(model, "classes_", np.arange(5)).astype(int)
        probs[:, classes] = raw
        out = probs[0]
        total = out.sum()
        if not np.isclose(total, 1.0, atol=1e-5):
            raise ValueError(f"Classifier probability sum is not 1: {total}")
        return out

    def predict_one(self, features: dict[str, Any], delivery_hour_utc: str | None = None) -> dict[str, Any]:
        row = self._frame_from_features(features)
        spread_values = [self._predict_regression_fold(pipe, row) for pipe in self.regression_folds]
        prob_values = [self._predict_classifier_fold(pipe, row) for pipe in self.classifier_folds]
        predicted_spread = float(np.mean(spread_values))
        probabilities = np.mean(np.vstack(prob_values), axis=0)
        probabilities = probabilities / probabilities.sum()
        p_c1, p_c2, p_c3, p_c4, p_c5 = [float(v) for v in probabilities]
        p_negative = p_c1 + p_c2
        p_neutral = p_c3
        p_positive = p_c4 + p_c5
        threshold = float(self.thresholds["threshold"])
        if p_positive >= threshold and p_positive > p_negative:
            signal = "DEC"
        elif p_negative >= threshold and p_negative > p_positive:
            signal = "INC"
        else:
            signal = "NO_TRADE"
        predicted_index = int(np.argmax(probabilities))
        response = {
            "delivery_hour_utc": delivery_hour_utc,
            "predicted_spread": predicted_spread,
            "p_c1": p_c1,
            "p_c2": p_c2,
            "p_c3": p_c3,
            "p_c4": p_c4,
            "p_c5": p_c5,
            "p_negative": float(p_negative),
            "p_neutral": float(p_neutral),
            "p_positive": float(p_positive),
            "predicted_class": CLASS_NAMES[predicted_index],
            "confidence": float(np.max(probabilities)),
            "signal": signal,
            "model_version": self.model_version,
            "deployment_status": self.deployment_status,
        }
        return {k: _jsonable(v) for k, v in response.items()}

    def predict_batch(self, records: list[dict[str, Any]] | pd.DataFrame) -> list[dict[str, Any]]:
        rows = records.to_dict(orient="records") if isinstance(records, pd.DataFrame) else records
        results = []
        for record in rows:
            record = dict(record)
            delivery_hour_utc = record.pop("delivery_hour_utc", None)
            features = record["features"] if isinstance(record.get("features"), dict) else record
            results.append(self.predict_one(features, delivery_hour_utc=delivery_hour_utc))
        return results

    def _shap_values_for_artifact(self, artifact: dict[str, Any], row: pd.DataFrame, head: str) -> tuple[list[str], np.ndarray]:
        if shap is None:
            raise RuntimeError("The shap package is not installed. Install requirements-model.txt.")
        if head == "spread_regression":
            features = artifact.get("features", self.b2a_features)
            x = artifact["imputer"].transform(row[features])
            values = shap.TreeExplainer(artifact["model"]).shap_values(x)
            arr = np.asarray(values)
            if arr.ndim == 2:
                arr = arr[0]
            return list(features), np.asarray(arr, dtype=float)
        features = artifact.get("features", self.b2b_features)
        x = artifact["imputer"].transform(row[features])
        values = shap.TreeExplainer(artifact["model"]).shap_values(x)
        class_index = CLASS_NAMES.index(head)
        if isinstance(values, list):
            arr = np.asarray(values[class_index])[0]
        else:
            arr = np.asarray(values)
            if arr.ndim == 3:
                arr = arr[0, :, class_index]
            elif arr.ndim == 2:
                arr = arr[0]
            else:
                raise ValueError(f"Unexpected SHAP array shape: {arr.shape}")
        return list(features), np.asarray(arr, dtype=float)

    def explain_local(self, features: dict[str, Any], head: str = "predicted_class", top_k: int = 10) -> dict[str, Any]:
        row = self._frame_from_features(features)
        prediction = self.predict_one(features)
        if head == "predicted_class":
            resolved_head = prediction["predicted_class"]
        elif head in {"spread", "predicted_spread", "spread_regression"}:
            resolved_head = "spread_regression"
        elif head in set(CLASS_NAMES):
            resolved_head = head
        elif head in {"p_negative", "negative_probability", "p_positive", "positive_probability"}:
            raise ValueError("p_negative/p_positive are probability sums; request component class heads C1-C5 or predicted_class instead of fabricated additive SHAP.")
        else:
            raise ValueError(f"Unsupported SHAP head: {head}")

        fold_frames = []
        artifacts = self.regression_folds if resolved_head == "spread_regression" else self.classifier_folds
        for artifact in artifacts:
            feature_names, shap_values = self._shap_values_for_artifact(artifact, row, resolved_head)
            fold_frames.append(pd.DataFrame({"feature": feature_names, "shap_value": shap_values}))
        avg = pd.concat(fold_frames).groupby("feature", as_index=False)["shap_value"].mean()
        feature_values = row.iloc[0].to_dict()
        avg["feature_value"] = avg["feature"].map(feature_values)
        avg["abs_shap_value"] = avg["shap_value"].abs()
        avg = avg.sort_values("abs_shap_value", ascending=False).head(int(top_k)).reset_index(drop=True)
        avg["rank"] = np.arange(1, len(avg) + 1)
        records = [
            {
                "feature": str(r["feature"]),
                "feature_value": _jsonable(r["feature_value"]),
                "shap_value": _jsonable(r["shap_value"]),
                "abs_shap_value": _jsonable(r["abs_shap_value"]),
                "rank": int(r["rank"]),
            }
            for _, r in avg.iterrows()
        ]
        return {
            "requested_head": head,
            "resolved_head": resolved_head,
            "top_k": int(top_k),
            "explanations": records,
            "prediction_context": prediction,
            "warning": "SHAP is model explanation, not causal proof. Different output heads must be interpreted separately.",
        }

    def get_global_shap(self, head: str | None = None, top_k: int = 20) -> dict[str, Any]:
        path = self.model_dir / "shap" / "global" / "global_shap_summary.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if head is not None:
            if head not in data:
                raise ValueError(f"Unknown global SHAP head: {head}. Available: {sorted(data)}")
            return {head: data[head][: int(top_k)]}
        return {key: value[: int(top_k)] for key, value in data.items()}

    def metadata(self) -> dict[str, Any]:
        return {
            "model_metadata": self.model_metadata,
            "feature_schema": self.feature_schema,
            "class_mapping": self.class_mapping,
            "thresholds": self.thresholds,
        }


@lru_cache(maxsize=1)
def load_c1_agent(package_root: str | Path | None = None) -> C1PredictionAgent:
    return C1PredictionAgent(package_root=package_root)


def predict_one(features: dict[str, Any], delivery_hour_utc: str | None = None) -> dict[str, Any]:
    return load_c1_agent().predict_one(features, delivery_hour_utc=delivery_hour_utc)


def predict_batch(records: list[dict[str, Any]] | pd.DataFrame) -> list[dict[str, Any]]:
    return load_c1_agent().predict_batch(records)


def explain_local(features: dict[str, Any], head: str = "predicted_class", top_k: int = 10) -> dict[str, Any]:
    return load_c1_agent().explain_local(features, head=head, top_k=top_k)


def get_global_shap(head: str | None = None, top_k: int = 20) -> dict[str, Any]:
    return load_c1_agent().get_global_shap(head=head, top_k=top_k)
