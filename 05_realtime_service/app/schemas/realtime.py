from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class RealtimeCollectionResponse(BaseModel):
    delivery_date_local: str
    collected_at_utc: str
    status: str
    counts: dict[str, int]
    messages: list[str]


class RealtimeStatusResponse(BaseModel):
    latest_run: dict[str, Any] | None
    row_counts: dict[str, int]


class RealtimeDayAheadResponse(BaseModel):
    delivery_date: str
    latest_collected_at_utc: str | None
    weather_rows: list[dict[str, Any]]
    gas_observation: dict[str, Any] | None
    ercot_rows: list[dict[str, Any]]
