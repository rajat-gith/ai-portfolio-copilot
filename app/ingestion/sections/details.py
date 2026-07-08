from langchain_core.documents import Document

from ...clients.cms_client import CMSClient
from ..normalize import safe_get

ENDPOINT = "public/details"
SECTION = "details"


def build_document(cms_client: CMSClient, profile_id: str) -> Document:
    """Build the single profile "details" document (name, headline, location, summary).

    Not wired into the default ingestion run (see document_builder.build_all_documents),
    matching the original pipeline where this section was disabled. Kept here so it can
    be re-enabled by calling it from the orchestrator.
    """
    details = cms_client.fetch(ENDPOINT)
    text = (
        f"Name: {safe_get(details, 'name')}\n"
        f"Headline: {safe_get(details, 'headline')}\n"
        f"Location: {safe_get(details, 'location')}\n"
        f"Summary: {safe_get(details, 'summary')}\n"
    )
    return Document(
        page_content=text,
        metadata={"section": SECTION, "profile_id": profile_id},
    )
