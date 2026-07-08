class IngestError(Exception):
    """Base exception for ingestion errors"""
    pass


class CMSFetchError(IngestError):
    """Raised when CMS API fetch fails"""
    pass


class VectorStoreError(IngestError):
    """Raised when vector store operations fail"""
    pass


class ConfigurationError(IngestError):
    """Raised when configuration is invalid"""
    pass