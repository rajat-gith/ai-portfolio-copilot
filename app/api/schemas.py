from typing import List, Optional

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    profile_id: str


class SourceSnippet(BaseModel):
    section: Optional[str] = None
    title: Optional[str] = None
    snippet: str
    relevance_score: float


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceSnippet]
    # "grounded" = at least one source cleared the relevance threshold;
    # "low_confidence" = only the best-effort fallback sources were used.
    # Lets a UI visually flag answers it should trust less.
    confidence: str


class IngestRequest(BaseModel):
    """Body for POST /ingest-profile. CMS credentials travel as headers instead
    (X-API-Key / X-API-Secret) — see app/api/routes.py."""
    profile_id: str