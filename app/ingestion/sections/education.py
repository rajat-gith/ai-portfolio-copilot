from typing import List

from langchain_core.documents import Document

from ...clients.cms_client import CMSClient
from ..normalize import normalize_list_response, safe_get, safe_list, format_duration

ENDPOINT = "public/education"
SECTION = "education"


def build_documents(cms_client: CMSClient, profile_id: str) -> List[Document]:
    """Build one Document per education entry.

    Expected raw shape:
    {
      "_id", "courseName", "institute",
      "periodOfCourse": { "startDate", "endDate", "isOngoing" },
      "degree", "skills", "courseworks",
      "grades": { "type", "value" },
      "userId", "createdAt", "updatedAt"
    }
    """
    raw = cms_client.fetch(ENDPOINT)
    entries = normalize_list_response(raw)

    docs: List[Document] = []
    for edu in entries:
        institute = safe_get(edu, "institute")
        degree = safe_get(edu, "degree")
        course_name = safe_get(edu, "courseName")

        period = edu.get("periodOfCourse", {}) or {}
        duration = format_duration(
            start=str(period.get("startDate", "N/A")),
            end_raw=period.get("endDate"),
            is_ongoing=bool(period.get("isOngoing", False)),
        )

        grades = edu.get("grades", {}) or {}
        grade_value = str(grades.get("value", "N/A"))
        grade_type = str(grades.get("type", "N/A"))

        skills = safe_list(edu, "skills")
        courseworks = safe_list(edu, "courseworks")
        description = safe_get(edu, "description")

        text = (
            "Education:\n"
            f"Institution: {institute}\n"
            f"Degree: {degree}\n"
            f"Course Name: {course_name}\n"
            f"Duration: {duration}\n"
            f"Grade: {grade_value} ({grade_type})\n"
            f"Skills: {skills}\n"
            f"Courseworks: {courseworks}\n"
            f"Description: {description}\n"
        )

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "section": SECTION,
                    "profile_id": profile_id,
                    "education_id": str(edu.get("_id", "")),
                },
            )
        )

    return docs
