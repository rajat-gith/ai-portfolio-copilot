from functools import lru_cache
from typing import Any, Callable, Dict, List, Tuple

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

from ..config import Config
from .prompts import ANSWER_PROMPT
from .vectorstore import get_vectorstore

RagCallable = Callable[[str], Dict[str, Any]]

# Chunks scoring below this are treated as noise rather than evidence and
# excluded from the LLM's context.
MIN_RELEVANCE_SCORE = 0.55

# If nothing clears MIN_RELEVANCE_SCORE (e.g. a borderline phrasing like
# "does X know Node js"), fall back to this many best-scoring chunks instead
# of answering from empty context — keeps recall reasonable for real
# questions the profile *does* have data for, while ANSWER_PROMPT's own
# "I don't know" instruction still covers genuinely off-topic questions.
FALLBACK_TOP_N = 3


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


@lru_cache(maxsize=1)
def _get_llm(model_name: str) -> ChatGoogleGenerativeAI:
    """The LLM client is cheap to reuse across profiles/requests — cache it."""
    return ChatGoogleGenerativeAI(model=model_name, temperature=0.0)


def get_rag_chain(config: Config, profile_id: str) -> RagCallable:
    """
    Build a RAG pipeline scoped to a single profile's Pinecone namespace.

    Called per-request with the profile_id from the incoming question, so two
    different profile_ids always retrieve from two different namespaces and
    can never answer using each other's data. The embedding model and LLM
    client are cached process-wide (see _get_embeddings / _get_llm); only the
    lightweight retriever/chain wiring is rebuilt per call.

    Returns a callable: rag(question: str) -> {
        "result": answer,
        "source_documents": docs actually passed to the LLM,
        "source_scores": matching relevance score per doc (0-1, higher=better),
    }
    """
    vectorstore = get_vectorstore(config, profile_id)
    llm = _get_llm(config.llm_model)

    chain = ANSWER_PROMPT | llm | StrOutputParser()

    def rag(question: str) -> Dict[str, Any]:
        # Pull more candidates than we'll necessarily use (retriever_k acts as
        # a ceiling on fetch size), then filter/rank by actual relevance
        # rather than blindly trusting "top-k is always good enough" — this
        # is what was silently dropping real matches (e.g. a Node.js mention
        # buried in one experience entry) when other, less relevant chunks
        # crowded a too-small fixed k.
        #
        # NOTE: Pinecone's score (cosine metric) is not the same normalized
        # [0,1] "relevance score" Chroma computed for us — it's raw cosine
        # similarity, roughly in [0,1] for normalized embeddings like
        # text-embedding-004 but not guaranteed. If MIN_RELEVANCE_SCORE feels
        # too strict/loose after switching to Pinecone, log a few real scores
        # and retune it — it's not a universal constant.
        scored: List[Tuple[Document, float]] = vectorstore.similarity_search_with_score(
            question, k=max(config.retriever_k, 8)
        )
        scored.sort(key=lambda pair: pair[1], reverse=True)

        relevant = [(doc, score) for doc, score in scored if score >= MIN_RELEVANCE_SCORE]
        used = relevant if relevant else scored[:FALLBACK_TOP_N]

        docs = [doc for doc, _ in used]
        scores = [score for _, score in used]

        context = _format_docs(docs)
        answer = chain.invoke({"question": question, "context": context})

        return {
            "result": answer,
            "source_documents": docs,
            "source_scores": scores,
        }

    return rag