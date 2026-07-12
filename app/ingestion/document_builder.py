import re
from typing import Dict, List, Optional

from langchain_core.documents import Document

from ..clients.cms_client import CMSClient
from ..config import Config
from ..exceptions import CMSFetchError
from ..logger import setup_logger
from .sections import blog, details, education, experience, volunteering

logger = setup_logger(__name__)

# Each entry: (section name, module with a build_documents(cms_client, profile_id) function).
# Add/remove sections here without touching the orchestration logic below.
#
# NOTE: "details" (name/headline/location/summary) used to be built but never
# wired in here, so basic "who is this / what do they do" questions had no
# context to retrieve from. It's included now.
SECTION_BUILDERS = (
    ("details", details),
    ("education", education),
    ("experience", experience),
    ("blogs", blog),
    ("volunteering", volunteering),
)

_TECH_LINE_RE = re.compile(r"(?:Technologies Used|Skills):\s*(.+)")


def _build_skills_document(docs: List[Document], profile_id: str) -> Optional[Document]:
    """
    Aggregate every technology/skill mentioned across experience and
    education entries into one dedicated chunk.

    Why this exists: a query like "does X know Node.js" is a short, keyword-
    heavy phrase. Embedded on its own, it can sit semantically closer to a
    single line buried inside a longer, narrative experience chunk than to
    that whole chunk's overall embedding — so with a small k the chunk that
    actually contains "Node.js" can lose out to more "generically relevant"
    chunks. A short, skills-only document gives that kind of query a direct,
    high-similarity target to retrieve, independent of how much surrounding
    text a given role/degree entry happens to have.
    """
    techs = set()
    for doc in docs:
        if doc.metadata.get("section") not in ("experience", "education"):
            continue
        match = _TECH_LINE_RE.search(doc.page_content)
        if not match:
            continue
        for tech in match.group(1).split(","):
            tech = tech.strip()
            if tech and tech.upper() != "N/A":
                techs.add(tech)

    if not techs:
        return None

    text = (
        "Skills & Technologies:\n"
        "The following skills and technologies appear across this person's "
        "experience and education: " + ", ".join(sorted(techs)) + "."
    )
    return Document(
        page_content=text,
        metadata={"section": "skills", "profile_id": profile_id, "title": "Skills & Technologies"},
    )


class DocumentBuilder:
    """Builds LangChain documents from CMS profile data.

    Delegates the actual per-section parsing to app/ingestion/sections/*,
    and just handles fetch orchestration + graceful degradation here.
    """

    def __init__(self, cms_client: CMSClient, config: Config):
        self.cms_client = cms_client
        self.config = config
        self.sections_status: Dict[str, str] = {}

    def build_all_documents(self, profile_id: str) -> List[Document]:
        """
        Build all profile documents with graceful degradation.

        Args:
            profile_id: the CMS profile to build documents for (per-request value)

        Returns:
            List of all successfully built documents

        Note:
            Individual (non-critical) sections that fail to fetch are logged
            and skipped rather than aborting the whole run.
        """
        docs: List[Document] = []

        for section_name, section_module in SECTION_BUILDERS:
            try:
                section_docs = section_module.build_documents(self.cms_client, profile_id)
                docs.extend(section_docs)
                self.sections_status[section_name] = "success"
                logger.info(f"Fetched {len(section_docs)} {section_name} entries")
            except CMSFetchError as e:
                self.sections_status[section_name] = "failed"
                logger.warning(f"Failed to fetch {section_name}: {e}")

        skills_doc = _build_skills_document(docs, profile_id)
        if skills_doc is not None:
            docs.append(skills_doc)
            self.sections_status["skills"] = "success"
            logger.info("Generated 1 aggregated skills entry")

        logger.info(f"Total documents created: {len(docs)}")
        logger.info(f"Sections status: {self.sections_status}")

        return docs

    def get_sections_status(self) -> Dict[str, str]:
        """Return status of all sections"""
        return self.sections_status.copy()