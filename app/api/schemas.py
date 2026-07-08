from typing import List, Optional

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class SourceSnippet(BaseModel):
    section: Optional[str] = None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceSnippet]


class IngestRequest(BaseModel):
    """Body for POST /ingest-profile. CMS credentials travel as headers instead
    (X-API-Key / X-API-Secret) — see app/api/routes.py."""
    profile_id: str