import os
from typing import Dict, Any
from datetime import datetime

from ..config import Config
from ..clients.cms_client import CMSClient
from ..exceptions import IngestError, ConfigurationError
from ..logger import setup_logger
from .document_builder import DocumentBuilder
from .vectorstore_manager import VectorStoreManager

logger = setup_logger(__name__)


class IngestionPipeline:
    """Main orchestrator for the ingestion pipeline"""

    def __init__(self, config: Config):
        self.config = config
        self.stats = {
            'start_time': None,
            'end_time': None,
            'documents_fetched': 0,
            'chunks_created': 0,
            'errors': [],
        }

    def run(self, profile_id: str, api_key: str, api_secret: str) -> Dict[str, Any]:
        """
        Execute the complete ingestion pipeline for a single profile.

        Args:
            profile_id: which CMS profile to ingest (request body value)
            api_key: CMS X-API-Key credential (request header value)
            api_secret: CMS X-API-Secret credential (request header value)

        Returns:
            Statistics dictionary

        Raises:
            IngestError: If pipeline fails
        """
        self.stats['start_time'] = datetime.now()

        try:
            logger.info("=" * 60)
            logger.info("Starting profile data ingestion")
            logger.info(f"Profile ID: {profile_id}")
            logger.info(f"CMS Base URL: {self.config.cms_base_url}")
            logger.info("=" * 60)

            with CMSClient(self.config, api_key=api_key, api_secret=api_secret) as cms_client:
                document_builder = DocumentBuilder(cms_client, self.config)
                vectorstore_manager = VectorStoreManager(self.config)

                documents = document_builder.build_all_documents(profile_id)
                self.stats['documents_fetched'] = len(documents)

                if not documents:
                    raise IngestError("No documents were successfully fetched")

                split_docs = vectorstore_manager.split_documents(documents)
                self.stats['chunks_created'] = len(split_docs)

                vectorstore_manager.persist_to_chroma(split_docs, profile_id)

            self.stats['end_time'] = datetime.now()
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()

            logger.info("=" * 60)
            logger.info("Ingestion completed successfully!")
            logger.info(f"Duration: {duration:.2f} seconds")
            logger.info(f"Documents fetched: {self.stats['documents_fetched']}")
            logger.info(f"Chunks created: {self.stats['chunks_created']}")
            logger.info("=" * 60)

            return self.get_stats()

        except IngestError:
            raise

        except Exception as e:
            logger.error(f"Unexpected error during ingestion: {str(e)}", exc_info=True)
            raise IngestError(f"Ingestion failed: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Return ingestion statistics"""
        return self.stats.copy()


def main():
    """
    CLI entry point: run `python -m app.ingest`.

    profile_id/api_key/api_secret are per-request values everywhere else
    (see app/api/routes.py: POST /ingest-profile), but the CLI has no
    "request" to read them from, so it falls back to env vars here.
    """
    try:
        try:
            config = Config.from_env()
        except Exception as e:
            logger.error(f"Configuration error: {e}")
            raise ConfigurationError(f"Invalid configuration: {e}")

        profile_id = os.getenv('PROFILE_ID', 'default')
        api_key = os.getenv('X_API_KEY', '')
        api_secret = os.getenv('X_API_SECRET', '')
        if not api_key or not api_secret:
            raise ConfigurationError(
                "X_API_KEY and X_API_SECRET must be set in .env for CLI ingestion"
            )

        pipeline = IngestionPipeline(config)
        stats = pipeline.run(profile_id=profile_id, api_key=api_key, api_secret=api_secret)

        if stats['errors']:
            logger.warning("Ingestion completed with errors")
            exit(1)

    except IngestError as e:
        logger.error(f"Ingestion failed: {e}")
        exit(1)

    except KeyboardInterrupt:
        logger.info("Ingestion interrupted by user")
        exit(130)

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        exit(1)