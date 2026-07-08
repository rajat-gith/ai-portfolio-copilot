# app/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.routes import router
from .config import Config
from .rag.chain import get_rag_chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler: runs on startup and shutdown.
    We initialize the RAG pipeline once and attach it to app.state.
    """
    config = Config.from_env()
    app.state.rag = get_rag_chain(config)
    yield
    # (optional) add cleanup code here if needed later


app = FastAPI(
    title="AI Portfolio Copilot",
    description="Ask questions about a profile using RAG over CMS data.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
