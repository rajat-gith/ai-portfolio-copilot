from typing import List

from langchain_core.documents import Document

from ...clients.cms_client import CMSClient
from ..normalize import safe_get

ENDPOINT = "public/details"
SECTION = "details"


def build_documents(cms_client: CMSClient, profile_id: str) -> List[Document]:
    """Build the single profile "details" document (name, headline, location, summary).

    Returns a one-item list (matching every other section module's
    build_documents(cms_client, profile_id) -> List[Document] signature) so
    it plugs straight into document_builder.SECTION_BUILDERS.
    """
    details = cms_client.fetch(ENDPOINT)
    text = (
        "Profile Summary:\n"
        f"Name: {safe_get(details, 'name')}\n"
        f"Headline: {safe_get(details, 'headline')}\n"
        f"Location: {safe_get(details, 'location')}\n"
        f"Summary: {safe_get(details, 'summary')}\n"
    )
    return [
        Document(
            page_content=text,
            metadata={
                "section": SECTION,
                "profile_id": profile_id,
                "title": "Profile Summary",
            },
        )
    ]