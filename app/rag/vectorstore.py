from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import Config


def get_vectorstore(config: Config) -> Chroma:
    """
    Load the persisted Chroma vector store from disk.

    Uses the same embedding model name as ingestion (app.ingestion.vectorstore_manager),
    sourced from the shared Config, so retrieval always matches what was indexed.
    """
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(config.vectorstore_dir),
    )
