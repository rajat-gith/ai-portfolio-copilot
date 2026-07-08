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


def get_rag_chain(config: Config = None) -> RagCallable:
    """
    Create a simple RAG pipeline using Gemini as the LLM.

    Returns a callable: rag(question: str) -> {"result": answer, "source_documents": docs}
    """
    config = config or Config.from_env()

    vectorstore = get_vectorstore(config)
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.retriever_k})

    llm = ChatGoogleGenerativeAI(
        model=config.llm_model,
        temperature=0.0,
    )

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
