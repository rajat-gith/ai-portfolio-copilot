from typing import List

from fastapi import APIRouter, Header, HTTPException

from ..config import Config
from ..exceptions import IngestError
from ..ingestion.pipeline import IngestionPipeline
from ..rag.chain import MIN_RELEVANCE_SCORE, get_rag_chain
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
def ask_profile(req: AskRequest):
    """
    Takes a natural language question + profile_id and returns:
    - answer: the LLM's answer grounded ONLY in that profile's data
    - sources: small snippets of the profile used to answer

    profile_id selects which Chroma collection to query, so two different
    profile_ids always retrieve from two different, isolated collections.
    """
    config = Config.from_env()
    rag = get_rag_chain(config, req.profile_id)

    result = rag(req.question)

    answer: str = result.get("result", "")
    source_docs = result.get("source_documents", []) or []
    source_scores = result.get("source_scores", []) or []

    sources: List[SourceSnippet] = []
    for doc, score in zip(source_docs, source_scores):
        meta = doc.metadata or {}
        sources.append(
            SourceSnippet(
                section=meta.get("section"),
                title=meta.get("title"),
                snippet=_snippet(doc.page_content),
                relevance_score=round(score, 3),
            )
        )

    confidence = "grounded" if any(s.relevance_score >= MIN_RELEVANCE_SCORE for s in sources) else "low_confidence"

    return AskResponse(answer=answer, sources=sources, confidence=confidence)


def _snippet(text: str, limit: int = SNIPPET_PREVIEW_LENGTH) -> str:
    """Truncate on a word boundary instead of slicing mid-word."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "…"