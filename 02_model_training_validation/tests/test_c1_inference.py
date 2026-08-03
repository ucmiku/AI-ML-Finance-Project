# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.c1_inference import load_c1_agent


def main() -> None:
    agent = load_c1_agent(ROOT)
    assert len(agent.regression_folds) == 3
    assert len(agent.classifier_folds) == 3
    request = json.loads((ROOT / "examples" / "sample_request.json").read_text(encoding="utf-8"))
    prediction = agent.predict_one(request["features"], delivery_hour_utc=request.get("delivery_hour_utc"))
    batch = agent.predict_batch([{**request["features"], "delivery_hour_utc": request.get("delivery_hour_utc")}])
    prob_sum = sum(prediction[f"p_c{i}"] for i in range(1, 6))
    assert abs(prob_sum - 1.0) <= 1e-5
    assert abs(prediction["p_negative"] - (prediction["p_c1"] + prediction["p_c2"])) <= 1e-8
    assert abs(prediction["p_neutral"] - prediction["p_c3"]) <= 1e-8
    assert abs(prediction["p_positive"] - (prediction["p_c4"] + prediction["p_c5"])) <= 1e-8
    threshold = 0.60
    if prediction["p_positive"] >= threshold and prediction["p_positive"] > prediction["p_negative"]:
        expected_signal = "DEC"
    elif prediction["p_negative"] >= threshold and prediction["p_negative"] > prediction["p_positive"]:
        expected_signal = "INC"
    else:
        expected_signal = "NO_TRADE"
    assert prediction["signal"] == expected_signal
    global_shap = agent.get_global_shap(top_k=20)
    assert "spread_regression" in global_shap
    assert "positive_probability_C5" in global_shap
    local = agent.explain_local(request["features"], top_k=10)
    assert local["explanations"]
    assert {"feature", "feature_value", "shap_value", "abs_shap_value", "rank"}.issubset(local["explanations"][0])
    json.dumps(prediction)
    json.dumps(batch)
    json.dumps(global_shap)
    json.dumps(local)
    print(json.dumps({
        "models_loaded": 6,
        "prediction": prediction,
        "batch_count": len(batch),
        "global_shap_heads": sorted(global_shap),
        "local_shap_top_feature": local["explanations"][0],
    }, indent=2))


if __name__ == "__main__":
    main()
