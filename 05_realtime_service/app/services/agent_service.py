from __future__ import annotations

import json

from langchain.tools import tool
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

from app.services.market_service import get_day_ahead_forecast
from app.services.model_service import get_predictions


SYSTEM_PREFIX = """
You are an expert ERCOT energy trader assistant and a highly skilled SQL assistant.
You have access to a historical SQL database and tools to fetch live/realtime forecast data.

Business Rules based on data dictionary:
1. Main feature table: model_wide_hourly_2024_2026.
2. Target metric: spread_usd_per_mwh is the price spread RT - DA. A positive value means Real-Time price is higher than Day-Ahead price.
3. Time: Use hour, delivery_date_local, month, day_of_week, and is_dst for time grouping or filtering. Do not parse raw UTC strings for hour grouping unless the user explicitly asks for UTC analysis.
4. Weather events: For freezing weather, check freezing_city_count > 0. For heatwaves, use extreme_heat_city_count.
5. Fundamentals: Henry Hub gas price is gas_price.
6. Trading interpretation: positive spread means a long DA position would benefit from RT settlement; negative spread means the opposite.
7. Leakage rule: DA price, RT price, spread, and derived label columns are targets or settlement fields, not features available before the day-ahead deadline.
8. Historical SQL data mainly covers 2024 through 2026-06-30. For 2026-07-23 and later realtime predictions, or if the user asks about today, tomorrow, current prediction, live data, or the current dashboard, use the live forecast tool or the dashboard context.

Always write strictly valid SQLite/PostgreSQL queries. Prefer concise aggregate queries before detailed row inspection.
Output natural language explanations along with the data findings.
"""


@tool
def fetch_live_forecast(date_str: str) -> str:
    """Fetch day-ahead forecast rows and realtime model predictions for a specific ERCOT local date in YYYY-MM-DD format."""
    try:
        forecast = get_day_ahead_forecast(date_str)
        predictions = get_predictions(date_str)
        payload = {
            "forecast": forecast,
            "predictions": predictions,
        }
        return json.dumps(payload, ensure_ascii=False, default=str)
    except Exception as exc:
        return f"No prediction data available for {date_str}: {exc}"


def _build_agent_input(user_question: str, dashboard_context: str | None) -> str:
    context = (dashboard_context or "").strip()
    if len(context) > 12000:
        context = context[:12000] + "\n...[dashboard_context truncated]"
    return f"""
IMPORTANT CONTEXT FROM USER'S CURRENT SCREEN:
{context if context else "No dashboard context was provided."}

If the user asks about "today", "tomorrow", "current prediction", "live data",
"realtime data", or dates after 2026-06-30, rely heavily on the dashboard context
above or use the fetch_live_forecast tool before answering.

Question: {user_question}
""".strip()


def get_agent_answer(
    user_question: str,
    api_key: str,
    db_uri: str,
    dashboard_context: str | None = None,
) -> str:
    """
    Initialize the LLM-backed SQL agent, query the project database, and return the answer.
    """
    db = SQLDatabase.from_uri(db_uri)

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0,
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type="openai-tools",
        prefix=SYSTEM_PREFIX,
        extra_tools=[fetch_live_forecast],
    )

    response = agent_executor.invoke(
        {"input": _build_agent_input(user_question, dashboard_context)},
    )
    return response["output"]
