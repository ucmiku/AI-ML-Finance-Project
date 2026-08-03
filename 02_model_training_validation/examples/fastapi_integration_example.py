# -*- coding: utf-8 -*-
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from src.inference.c1_inference import load_c1_agent

agent = load_c1_agent()
app = FastAPI(title="C1 Prediction Agent Integration Example")


class PredictRequest(BaseModel):
    delivery_hour_utc: str | None = None
    features: dict
    model_config = ConfigDict(extra="forbid")


class BatchRequest(BaseModel):
    records: list[PredictRequest]


@app.post("/predict")
def predict(payload: PredictRequest):
    return agent.predict_one(payload.features, delivery_hour_utc=payload.delivery_hour_utc)


@app.post("/predict-batch")
def predict_batch(payload: BatchRequest):
    rows = []
    for record in payload.records:
        row = dict(record.features)
        row["delivery_hour_utc"] = record.delivery_hour_utc
        rows.append(row)
    return {"predictions": agent.predict_batch(rows)}


@app.post("/explain/local")
def explain_local(payload: PredictRequest, head: str = "predicted_class", top_k: int = 10):
    return agent.explain_local(payload.features, head=head, top_k=top_k)


@app.get("/explain/global")
def explain_global(head: str | None = None, top_k: int = 20):
    return agent.get_global_shap(head=head, top_k=top_k)


@app.get("/model/metadata")
def metadata():
    return agent.metadata()
