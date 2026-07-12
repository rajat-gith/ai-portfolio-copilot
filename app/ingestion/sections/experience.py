from typing import List

from langchain_core.documents import Document

from ...clients.cms_client import CMSClient
from ..normalize import normalize_list_response, safe_get, safe_list, format_duration

ENDPOINT = "public/experience"
SECTION = "experience"


def build_documents(cms_client: CMSClient, profile_id: str) -> List[Document]:
    """Build one Document per work experience entry.

    Expected raw shape:
    {
      "_id", "userId", "title", "company", "location",
      "employmentType",
      "period": { "startDate", "endDate", "ongoing" },
      "description", "technologiesUsed", "createdAt", "updatedAt"
    }
    """
    raw = cms_client.fetch(ENDPOINT)
    entries = normalize_list_response(raw)

    docs: List[Document] = []
    for exp in entries:
        title = safe_get(exp, "title")
        company = safe_get(exp, "company")
        location = safe_get(exp, "location")

        period = exp.get("period", {}) or {}
        duration = format_duration(
            start=str(period.get("startDate", "N/A")),
            end_raw=period.get("endDate"),
            is_ongoing=bool(period.get("ongoing", False)),
        )

        description = safe_get(exp, "description")
        employment_type = safe_get(exp, "employmentType")
        technologies_used = safe_list(exp, "technologiesUsed")

        text = (
            "Work Experience:\n"
            f"Role: {title}\n"
            f"Company: {company}\n"
            f"Location: {location}\n"
            f"Duration: {duration}\n"
            f"Employment Type: {employment_type}\n"
            f"Responsibilities: {description}\n"
            f"Technologies Used: {technologies_used}\n"
        )

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "section": SECTION,
                    "profile_id": profile_id,
                    "experience_id": str(exp.get("_id", "")),
                    "title": f"{title} at {company}",
                },
            )
        )

    return docs
