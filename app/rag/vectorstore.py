from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import Config
from ..vectorstore_naming import collection_name_for_profile


@lru_cache(maxsize=1)
def _get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Loading the embedding model is the expensive part — cache it process-wide
    instead of reloading it on every request/profile."""
    return HuggingFaceEmbeddings(model_name=model_name)


def get_vectorstore(config: Config, profile_id: str) -> Chroma:
    """
    Load this profile's own persisted Chroma collection.

    Uses the same embedding model name and collection-naming scheme as
    ingestion (app.ingestion.vectorstore_manager), so retrieval always reads
    back exactly what was indexed for that profile_id — never another one's.
    """
    embeddings = _get_embeddings(config.embedding_model)
    return Chroma(
        collection_name=collection_name_for_profile(profile_id),
        embedding_function=embeddings,
        persist_directory=str(config.vectorstore_dir),
    )