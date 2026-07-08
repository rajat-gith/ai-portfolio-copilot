import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv()


class Config(BaseModel):
    """Application configuration with validation"""

    # CMS Configuration
    # NOTE: profile_id, api_key and api_secret are intentionally NOT part of
    # server config. They are per-request values (api_key/api_secret arrive
    # as request headers, profile_id in the request body) so one deployment
    # can ingest data for many different CMS tenants/credentials.
    cms_base_url: str = Field(..., env='CMS_BASE_URL')

    # Vector store / Processing
    vectorstore_dir: Path = Field(default=Path('data/vectorstore'))
    chunk_size: int = Field(default=800, ge=100, le=2000)
    chunk_overlap: int = Field(default=150, ge=0, le=500)

    # Embedding / LLM model names, shared by ingestion and the RAG chain
    # so both sides always agree on which embedding model produced the vectors.
    embedding_model: str = Field(default='sentence-transformers/all-MiniLM-L6-v2')
    llm_model: str = Field(default='gemini-2.5-flash')
    retriever_k: int = Field(default=4, ge=1, le=20)

    # Network / Retry
    request_timeout: int = Field(default=30, ge=5, le=120)
    max_retries: int = Field(default=3, ge=0, le=10)

    # Optional LLM Key
    openai_api_key: Optional[str] = Field(default=None, env='OPENAI_API_KEY')

    # ---------------- VALIDATORS ---------------- #

    @field_validator('cms_base_url')
    @classmethod
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('CMS_BASE_URL must start with http:// or https://')
        return v.rstrip('/')

    @field_validator('vectorstore_dir')
    @classmethod
    def ensure_dir(cls, v):
        v.mkdir(parents=True, exist_ok=True)
        return v

    class ConfigDict:
        env_prefix = ''

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        return cls(
            cms_base_url=os.getenv('CMS_BASE_URL', ''),
            vectorstore_dir=Path(os.getenv('VECTORSTORE_DIR', 'data/vectorstore')),
            chunk_size=int(os.getenv('CHUNK_SIZE', 800)),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', 150)),
            embedding_model=os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
            llm_model=os.getenv('LLM_MODEL', 'gemini-2.5-flash'),
            retriever_k=int(os.getenv('RETRIEVER_K', 4)),
            request_timeout=int(os.getenv('REQUEST_TIMEOUT', 30)),
            max_retries=int(os.getenv('MAX_RETRIES', 3)),
            openai_api_key=os.getenv('OPENAI_API_KEY'),
        )