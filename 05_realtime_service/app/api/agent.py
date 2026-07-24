from fastapi import APIRouter, HTTPException

from app.core.config import get_agent_db_uri
from app.schemas.agent import AgentRequest, AgentResponse
from app.services.agent_service import get_agent_answer


router = APIRouter()


@router.post("/v1/agent/ask", response_model=AgentResponse)
def ask_agent(request: AgentRequest) -> AgentResponse:
    """
    Receive a user question, run the LangChain SQL Agent, and return the insight.
    """
    try:
        answer = get_agent_answer(
            user_question=request.question,
            api_key=request.api_key,
            db_uri=get_agent_db_uri(),
            dashboard_context=request.dashboard_context,
        )
        return AgentResponse(answer=answer)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Agent Execution Error: {exc}",
        ) from exc
