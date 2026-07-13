from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from ..config import Config
from ..vectorstore_naming import namespace_for_profile


@lru_cache(maxsize=1)
def _get_embeddings(model_name: str, output_dimensionality: int) -> GoogleGenerativeAIEmbeddings:
    """Loading the embedding model is process-wide cached.

    output_dimensionality must match what was used at ingestion time (and
    what the Pinecone index was created with) — Pinecone rejects any query
    vector whose dimension doesn't exactly match the index's.
    """
    return GoogleGenerativeAIEmbeddings(model=model_name, output_dimensionality=output_dimensionality)


@lru_cache(maxsize=1)
def _get_pinecone_index(api_key: str, index_name: str):
    """
    The Pinecone client + index handle are cheap to reuse across requests —
    cache process-wide, same pattern as _get_embeddings. Caching is keyed on
    (api_key, index_name) so it stays correct if either ever changes.
    """
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)


def get_vectorstore(config: Config, profile_id: str) -> PineconeVectorStore:
    """
    Load this profile's own namespace within the shared Pinecone index.

    Uses the same embedding model name and namespace-naming scheme as
    ingestion (app.ingestion.vectorstore_manager), so retrieval always reads
    back exactly what was indexed for that profile_id — never another one's.
    """
    embeddings = _get_embeddings(config.embedding_model, config.embedding_dimension)
    index = _get_pinecone_index(config.pinecone_api_key, config.pinecone_index_name)
    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=namespace_for_profile(profile_id),
    )