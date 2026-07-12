from typing import List

from langchain_core.documents import Document

from ...clients.cms_client import CMSClient
from ..normalize import normalize_list_response, safe_get, format_duration

ENDPOINT = "public/extra/volunteering"
SECTION = "volunteering"


def build_documents(cms_client: CMSClient, profile_id: str) -> List[Document]:
    """Build one Document per volunteering entry.

    Expected raw shape:
    {
      "Id", "userId", "role", "organization", "cause",
      "startDate", "endDate", "isOngoing",
      "description", "location", "website",
      "createdAt", "updatedAt"
    }
    """
    raw = cms_client.fetch(ENDPOINT)
    entries = normalize_list_response(raw)

    docs: List[Document] = []
    for vol in entries:
        role = safe_get(vol, "role")
        organization = safe_get(vol, "organization")
        location = safe_get(vol, "location")
        duration = format_duration(
            start=safe_get(vol, "startDate"),
            end_raw=vol.get("endDate"),
            is_ongoing=bool(vol.get("isOngoing", False)),
        )
        description = safe_get(vol, "description")

        text = (
            "Volunteering:\n"
            f"Role: {role}\n"
            f"Organization: {organization}\n"
            f"Location: {location}\n"
            f"Duration: {duration}\n"
            f"Description: {description}\n"
        )

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "section": SECTION,
                    "profile_id": profile_id,
                    "volunteering_id": str(vol.get("Id", "")),
                    "title": f"{role} at {organization}",
                },
            )
        )

    return docs
