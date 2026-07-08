from typing import Dict, List

from langchain_core.documents import Document

from ..clients.cms_client import CMSClient
from ..config import Config
from ..exceptions import CMSFetchError
from ..logger import setup_logger
from .sections import blog, education, experience, volunteering

logger = setup_logger(__name__)

# Each entry: (section name, module with a build_documents(cms_client, profile_id) function).
# Add/remove sections here without touching the orchestration logic below.
SECTION_BUILDERS = (
    ("education", education),
    ("experience", experience),
    ("blogs", blog),
    ("volunteering", volunteering),
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

        logger.info(f"Total documents created: {len(docs)}")
        logger.info(f"Sections status: {self.sections_status}")

        return docs

    def get_sections_status(self) -> Dict[str, str]:
        """Return status of all sections"""
        return self.sections_status.copy()