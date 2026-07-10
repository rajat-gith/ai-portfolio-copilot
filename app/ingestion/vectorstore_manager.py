from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from ..config import Config
from ..exceptions import VectorStoreError
from ..logger import setup_logger
from ..vectorstore_naming import collection_name_for_profile

logger = setup_logger(__name__)


class VectorStoreManager:
    """Manages document splitting and vector store persistence"""

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

    def persist_to_chroma(self, documents: List[Document], profile_id: str) -> None:
        """
        Embed and persist documents to this profile's own Chroma collection.

        Each profile_id gets its own collection inside the shared persist
        directory, so ingesting one profile never touches another profile's
        data, and /ask-profile can query the right one.

        Args:
            documents: List of documents to persist
            profile_id: which profile's collection to (re)build

        Raises:
            VectorStoreError: If persistence fails
        """
        collection_name = collection_name_for_profile(profile_id)
        try:
            logger.info("Initializing embeddings model...")
            embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model)

            # Wipe any previous collection for this profile first, so a re-ingest
            # is a clean rebuild rather than old + new chunks piling up together.
            logger.info(f"Clearing existing collection '{collection_name}' (if any)...")
            existing = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=str(self.config.vectorstore_dir),
            )
            existing.delete_collection()

            logger.info(
                f"Building and persisting collection '{collection_name}' "
                f"to {self.config.vectorstore_dir}"
            )
            Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=str(self.config.vectorstore_dir),
            )
            logger.info("Vector store persisted successfully")

        except Exception as e:
            error_msg = f"Failed to persist vector store for profile '{profile_id}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(error_msg)