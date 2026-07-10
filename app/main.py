# app/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.routes import router
from .config import Config
from .rag.chain import get_rag_chain


app = FastAPI(
    title="AI Portfolio Copilot",
    description="Ask questions about a profile using RAG over CMS data.",
    version="0.1.0",
)

app.include_router(router)
