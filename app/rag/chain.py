from functools import lru_cache
from typing import Any, Callable, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from ..config import Config
from .prompts import ANSWER_PROMPT
from .vectorstore import get_vectorstore

RagCallable = Callable[[str], Dict[str, Any]]


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


@lru_cache(maxsize=1)
def _get_llm(model_name: str) -> ChatGoogleGenerativeAI:
    """The LLM client is cheap to reuse across profiles/requests — cache it."""
    return ChatGoogleGenerativeAI(model=model_name, temperature=0.0)


def get_rag_chain(config: Config, profile_id: str) -> RagCallable:
    """
    Build a RAG pipeline scoped to a single profile's Chroma collection.

    Called per-request with the profile_id from the incoming question, so two
    different profile_ids always retrieve from two different collections and
    can never answer using each other's data. The embedding model and LLM
    client are cached process-wide (see _get_embeddings / _get_llm); only the
    lightweight retriever/chain wiring is rebuilt per call.

    Returns a callable: rag(question: str) -> {"result": answer, "source_documents": docs}
    """
    vectorstore = get_vectorstore(config, profile_id)
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.retriever_k})
    llm = _get_llm(config.llm_model)

    chain = (
        {
            "question": RunnablePassthrough(),
            "context": retriever | _format_docs,
        }
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )

    def rag(question: str) -> Dict[str, Any]:
        docs = retriever.invoke(question)
        answer = chain.invoke(question)
        return {
            "result": answer,
            "source_documents": docs,
        }

    return rag