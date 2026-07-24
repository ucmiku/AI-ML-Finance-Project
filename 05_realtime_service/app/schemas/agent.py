from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1)
    api_key: str = Field(..., min_length=1)
    dashboard_context: str | None = None


class AgentResponse(BaseModel):
    answer: str
