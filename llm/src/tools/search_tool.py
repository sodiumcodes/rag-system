from langchain_core.tools import tool
from llm.src.retrieval.retriever import DocumentRetriever
from functools import lru_cache

@lru_cache(maxsize=1)
def get_retriever() -> DocumentRetriever:
    return DocumentRetriever()

retriever = get_retriever()

@tool
def answer(query: str) -> list[dict]:
    """
    Retrieve relevant document chunks using dense similarity search.

    Use this tool when the user selects the standard Answer mode.
    Returns the top 5 semantically relevant chunks by default.
    """
    return retriever.similarity_search(
        query=query,
        top_k=5,
    )

@tool
def fast_answer(query: str) -> list[dict]:
    """
    Retrieve relevant document chunks using hybrid search.

    Use this tool when the user selects Fast Answer mode.
    Returns the top 5 fused results by default.
    """
    return retriever.hybrid_search(
        query=query,
        top_k=5,
        fetch_k=20,
        rrf_k=60,
    )

# Collect the tools for later use in the LLM pipeline
search_tools = [answer, fast_answer]
