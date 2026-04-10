from .agent import KnowledgeBaseAgent
from .chunking import (
    ChunkingStrategyComparator,
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
    compute_similarity,
)
from .embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    MockEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from .models import Document
from .store import EmbeddingStore

# ── Phase 2 additions ────────────────────────────────────────────────────────
# Imported lazily to avoid hard dependency errors when Phase 2 packages are not
# installed.  Import directly from the sub-modules when needed:
#   from src.langchain_chunking import LangChainMarkdownChunker, LangChainRecursiveChunker
#   from src.langchain_embeddings import LangChainHuggingFaceEmbedder, LangChainOpenAIEmbedder
#   from src.pinecone_store import PineconeStore

__all__ = [
    # Phase 1 — unchanged
    "Document",
    "FixedSizeChunker",
    "SentenceChunker",
    "RecursiveChunker",
    "ChunkingStrategyComparator",
    "compute_similarity",
    "EmbeddingStore",
    "KnowledgeBaseAgent",
    "MockEmbedder",
    "LocalEmbedder",
    "OpenAIEmbedder",
    "_mock_embed",
    "LOCAL_EMBEDDING_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "EMBEDDING_PROVIDER_ENV",
    # Phase 2 — available via direct sub-module imports
    # "LangChainMarkdownChunker",   # src.langchain_chunking
    # "LangChainRecursiveChunker",  # src.langchain_chunking
    # "LangChainHuggingFaceEmbedder", # src.langchain_embeddings
    # "LangChainOpenAIEmbedder",    # src.langchain_embeddings
    # "PineconeStore",              # src.pinecone_store
]
