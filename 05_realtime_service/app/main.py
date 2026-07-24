from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI
from dotenv import load_dotenv

from app.api import agent
from app.api import explainability
from app.api import market
from app.api import model
from app.api import realtime
from app.api import strategy
from app.services.realtime_scheduler import collection_loop


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    stop_event = asyncio.Event()
    task = asyncio.create_task(collection_loop(stop_event))
    try:
        yield
    finally:
        stop_event.set()
        await task


app = FastAPI(title="ERCOT Real-Time Features API", lifespan=lifespan)


@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
def ready() -> dict[str, str]:
    return {"status": "ready"}


app.include_router(agent.router, tags=["Agent"])
app.include_router(explainability.router, tags=["Explainability"])
app.include_router(market.router, tags=["Market"])
app.include_router(model.router, tags=["Model"])
app.include_router(realtime.router, tags=["Realtime"])
app.include_router(strategy.router, tags=["Strategy"])
