"""LangChain-based embedders for Phase 2.

These extend the existing embedders in embeddings.py (MockEmbedder, LocalEmbedder,
OpenAIEmbedder) by wrapping LangChain's embedding interfaces. They are callable
with a single string (for compatibility with EmbeddingStore / PineconeStore) and
also expose embed_documents() for batch ingestion.

No existing files are removed or modified.
"""
from __future__ import annotations


class LangChainHuggingFaceEmbedder:
    """
    Embedder backed by LangChain's HuggingFaceEmbeddings (sentence-transformers).

    Drop-in replacement for LocalEmbedder but routed through LangChain, which:
    - Exposes the same embed_query / embed_documents interface used by LangChain
      vector stores (including langchain_pinecone.PineconeVectorStore).
    - Enables batch embedding via embed_documents() for faster ingestion.

    Requires: pip install langchain-huggingface sentence-transformers

    Default model: all-MiniLM-L6-v2 → 384-dimensional embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from langchain_huggingface import HuggingFaceEmbeddings

        self.model_name = model_name
        self._backend_name = f"langchain-huggingface/{model_name}"
        self._lc_embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def __call__(self, text: str) -> list[float]:
        """Embed a single query string (callable interface for store compatibility)."""
        return self._lc_embedder.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed a list of texts (faster than calling __call__ in a loop)."""
        return self._lc_embedder.embed_documents(texts)

    @property
    def lc_embedder(self):
        """Expose the underlying LangChain embedder (for langchain_pinecone integration)."""
        return self._lc_embedder


class LangChainOpenAIEmbedder:
    """
    Embedder backed by LangChain's OpenAIEmbeddings.

    Requires: pip install langchain-openai
    Requires: OPENAI_API_KEY in environment or .env file.

    Default model: text-embedding-3-small → 1536-dimensional embeddings.
    Note: when using with PineconeStore, set dimension=1536.
    """

    def __init__(self, model_name: str = "text-embedding-3-small") -> None:
        from langchain_openai import OpenAIEmbeddings

        self.model_name = model_name
        self._backend_name = f"langchain-openai/{model_name}"
        self._lc_embedder = OpenAIEmbeddings(model=model_name)

    def __call__(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._lc_embedder.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed a list of texts."""
        return self._lc_embedder.embed_documents(texts)

    @property
    def lc_embedder(self):
        """Expose the underlying LangChain embedder."""
        return self._lc_embedder
