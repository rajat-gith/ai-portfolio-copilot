import time
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from ..config import Config
from ..exceptions import VectorStoreError
from ..logger import setup_logger
from ..vectorstore_naming import namespace_for_profile

logger = setup_logger(__name__)


class VectorStoreManager:
    """Manages document splitting and vector store persistence (Pinecone)."""

    def __init__(self, config: Config):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks

        Args:
            documents: List of documents to split

        Returns:
            List of chunked documents
        """
        logger.info("Splitting documents into chunks...")
        split_docs = self.splitter.split_documents(documents)
        logger.info(f"Created {len(split_docs)} chunks")
        return split_docs

    def _ensure_index(self, pc: Pinecone) -> None:
        """Create the shared Pinecone index if it doesn't exist yet, and wait
        until it's ready to accept writes. A no-op on every ingestion after
        the first, since the index is shared across all profiles."""
        index_name = self.config.pinecone_index_name
        if pc.has_index(index_name):
            return

        logger.info(f"Pinecone index '{index_name}' not found — creating it...")
        pc.create_index(
            name=index_name,
            dimension=self.config.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=self.config.pinecone_cloud,
                region=self.config.pinecone_region,
            ),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
        logger.info(f"Pinecone index '{index_name}' is ready")

    def persist_to_pinecone(self, documents: List[Document], profile_id: str) -> None:
        """
        Embed and persist documents to this profile's own namespace inside
        the shared Pinecone index.

        Each profile_id gets its own namespace, so ingesting one profile
        never touches another profile's data, and /ask-profile queries only
        the right one. Unlike the old local-Chroma setup, this data survives
        container restarts/redeploys/spin-downs — it isn't on local disk.

        Args:
            documents: List of documents to persist
            profile_id: which profile's namespace to (re)build

        Raises:
            VectorStoreError: If persistence fails
        """
        namespace = namespace_for_profile(profile_id)
        try:
            pc = Pinecone(api_key=self.config.pinecone_api_key)
            self._ensure_index(pc)
            index = pc.Index(self.config.pinecone_index_name)

            # Wipe any previous vectors for this profile's namespace first,
            # so a re-ingest is a clean rebuild rather than old + new chunks
            # piling up together. delete_all is scoped to the namespace only
            # — other profiles' namespaces in the same index are untouched.
            # (A brand-new namespace has nothing to delete; Pinecone 404s on
            # that case, which we treat as a no-op rather than an error.)
            logger.info(f"Clearing existing namespace '{namespace}' (if any)...")
            try:
                index.delete(delete_all=True, namespace=namespace)
            except Exception as e:
                logger.info(f"Nothing to clear for namespace '{namespace}' ({e})")

            logger.info("Initializing embeddings model...")
            embeddings = GoogleGenerativeAIEmbeddings(
                model=self.config.embedding_model,
                output_dimensionality=self.config.embedding_dimension,
            )

            logger.info(
                f"Embedding and upserting {len(documents)} chunks into "
                f"index '{self.config.pinecone_index_name}', namespace '{namespace}'"
            )
            PineconeVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
                index_name=self.config.pinecone_index_name,
                namespace=namespace,
            )
            logger.info("Vector store persisted successfully")

        except Exception as e:
            error_msg = f"Failed to persist vector store for profile '{profile_id}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(error_msg)