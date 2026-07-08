from typing import List

from fastapi import APIRouter, Header, HTTPException, Request

from ..config import Config
from ..exceptions import IngestError
from ..ingestion.pipeline import IngestionPipeline
from .schemas import AskRequest, AskResponse, IngestRequest, SourceSnippet

router = APIRouter()

# First N characters of a source document shown alongside an answer.
SNIPPET_PREVIEW_LENGTH = 200


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/ingest-profile")
def ingest_profile(
    payload: IngestRequest,
    x_api_key: str = Header(..., alias="X-API-Key"),
    x_api_secret: str = Header(..., alias="X-API-Secret"),
):
    """
    Fetches a profile's data from the CMS and (re)builds the vector store for it.

    CMS credentials are per-request (X-API-Key / X-API-Secret headers) rather
    than server-wide config, so one deployment can ingest data belonging to
    different CMS accounts/tenants. `profile_id` is in the body.
    """
    config = Config.from_env()
    pipeline = IngestionPipeline(config)

    try:
        stats = pipeline.run(
            profile_id=payload.profile_id,
            api_key=x_api_key,
            api_secret=x_api_secret,
        )
    except IngestError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return stats


@router.post("/ask-profile", response_model=AskResponse)
def ask_profile(req: AskRequest, request: Request):
    """
    Takes a natural language question and returns:
    - answer: the LLM's answer grounded in your profile data
    - sources: small snippets of the profile used to answer
    """
    rag = request.app.state.rag  # callable from get_rag_chain()

    result = rag(req.question)

    answer: str = result.get("result", "")
    source_docs = result.get("source_documents", []) or []

    sources: List[SourceSnippet] = []
    for doc in source_docs:
        meta = doc.metadata or {}
        section = meta.get("section")
        snippet = doc.page_content[:SNIPPET_PREVIEW_LENGTH]
        sources.append(SourceSnippet(section=section, snippet=snippet))

    return AskResponse(answer=answer, sources=sources)