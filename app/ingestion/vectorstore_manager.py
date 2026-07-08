from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from ..config import Config
from ..exceptions import VectorStoreError
from ..logger import setup_logger

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

    def persist_to_chroma(self, documents: List[Document]) -> None:
        """
        Embed and persist documents to Chroma vector store

        Args:
            documents: List of documents to persist

        Raises:
            VectorStoreError: If persistence fails
        """
        try:
            logger.info("Initializing embeddings model...")
            embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model)

            logger.info(f"Building and persisting vector store to {self.config.vectorstore_dir}")
            Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=str(self.config.vectorstore_dir),
            )
            logger.info("Vector store persisted successfully")

        except Exception as e:
            error_msg = f"Failed to persist vector store: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(error_msg)
