from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "frontend" / "public" / "mock" / "demo_snapshot.json"

app = FastAPI(title="ERCOT Map Workbench v3 Mock API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5178", "http://localhost:5178"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _snapshot() -> dict:
    if SNAPSHOT.exists():
        return json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    # Small fallback mirrors frontend mock data without importing TypeScript.
    return {
        "metadata": {
            "hub": "HB_NORTH",
            "mode": "mock_api",
            "delivery_time": "2025-07-15 12:00:00-05:00",
            "forecast_issue_time": "2025-07-14 12:00:00+00:00",
            "data_status": "demo_only",
            "source": "integration/mock_api_server.py",
        },
        "locations": [
            {
                "location": "DFW",
                "latitude": 32.9,
                "longitude": -97.04,
                "risk_level": "high",
                "coordinate_status": "demo_only",
                "values": {
                    "temperature_dfw_mean_c": 38.2,
                    "humidity_dfw_mean_pct": 44,
                    "wind_speed_dfw_mean_ms": 5.6,
                    "wind_gust_dfw_mean_ms": 9.8,
                    "precipitation_dfw_mean_mm": 0.1,
                },
            }
        ],
        "weather": {},
        "load": {"load_system_total_mw": 70600, "net_load_st_forecast_system_mw": 48200},
        "wind": {"wind_gap_system_mw": 1700, "wind_stwpf_system_wide_mw": 12400},
        "solar": {"solar_gap_system_mw": 700, "solar_stppf_system_mw": 10600},
        "renewable": {"renewable_st_share_of_load": 0.326, "renewable_st_forecast_system_mw": 23000},
        "gas": {"gas_price": 2.72},
        "extreme_weather": {"fixed_extreme_weather_flag": 1},
        "prediction": {"predicted_spread": 11.8, "signal": "INC", "confidence": 82, "risk_level": "high"},
        "drivers": [],
        "timeseries_24h": [],
        "warnings": ["Mock API fallback data."],
    }


@app.get("/api/v1/health")
def health() -> dict:
    return {"status": "ok", "service": "ercot-map-workbench-v3-mock"}


@app.get("/api/v1/dashboard/snapshot")
def dashboard_snapshot() -> dict:
    return _snapshot()


@app.get("/api/v1/dashboard/timeseries")
def dashboard_timeseries() -> list[dict]:
    return _snapshot().get("timeseries_24h", [])


@app.get("/api/v1/map/layers")
def map_layers() -> list[dict]:
    return [
        {"id": "texas-reference", "name": "Texas reference boundary", "type": "reference", "enabled": True, "visible": True, "opacity": 0.55, "variable": "Administrative boundary from basemap", "range": "Basemap", "legend": ["Boundary / county lines"], "sourceStatus": "available"},
        {"id": "ercot-boundary", "name": "ERCOT boundary", "type": "boundary", "enabled": False, "visible": False, "opacity": 0.6, "variable": "Unavailable", "range": "N/A", "legend": ["No trusted ERCOT GeoJSON found"], "reason": "No trusted ERCOT boundary file found.", "sourceStatus": "unavailable"},
        {"id": "load-zones", "name": "Load Zones", "type": "polygon", "enabled": False, "visible": False, "opacity": 0.5, "variable": "Unavailable", "range": "N/A", "legend": ["No trusted Load Zone GeoJSON found"], "reason": "No Load Zone boundary GeoJSON found.", "sourceStatus": "unavailable"},
        {"id": "weather-zones", "name": "Weather Zones", "type": "polygon", "enabled": False, "visible": False, "opacity": 0.5, "variable": "Unavailable", "range": "N/A", "legend": ["No trusted Weather Zone GeoJSON found"], "reason": "No Weather Zone boundary GeoJSON found.", "sourceStatus": "unavailable"},
        {"id": "weather-points", "name": "Weather Forecast Points", "type": "point", "enabled": True, "visible": True, "opacity": 1, "variable": "Temperature", "range": "27-39 C", "legend": ["Cool", "Warm", "Hot"], "sourceStatus": "demo"},
        {"id": "extreme-risk", "name": "Extreme Weather Risk", "type": "risk", "enabled": True, "visible": True, "opacity": 0.65, "variable": "Low / Medium / High", "range": "Categorical", "legend": ["Low", "Medium", "High"], "sourceStatus": "demo"},
        {"id": "settlement-hub", "name": "Settlement Hub", "type": "hub", "enabled": True, "visible": True, "opacity": 1, "variable": "HB_NORTH", "range": "Demo hub marker", "legend": ["Settlement hub demo marker"], "sourceStatus": "demo"},
        {"id": "labels", "name": "Labels", "type": "label", "enabled": True, "visible": True, "opacity": 1, "variable": "City and point labels", "range": "Visible/Hidden", "legend": ["Location labels"], "sourceStatus": "available"},
    ]
