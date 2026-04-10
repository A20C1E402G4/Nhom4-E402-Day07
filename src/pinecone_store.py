"""Pinecone-backed vector store for Phase 2.

Implements the same public interface as EmbeddingStore (add_documents, search,
search_with_filter, get_collection_size, delete_document) so it can be used
as a drop-in replacement in KnowledgeBaseAgent and benchmark scripts.

No existing files (store.py, embeddings.py, etc.) are removed or modified.

Requirements:
    pip install pinecone langchain-pinecone

Environment variables (in .env or shell):
    PINECONE_API_KEY=<your-key>
"""
from __future__ import annotations

import time
from typing import Any, Callable

from .models import Document


class PineconeStore:
    """
    Pinecone-backed vector store matching the EmbeddingStore interface.

    Design decisions:
    - content is stored inside Pinecone metadata (key: "_content") so it can be
      retrieved together with the vector in a single query call, avoiding a second
      round-trip to fetch document text.
    - doc_id is stored in metadata (key: "doc_id") to support delete_document and
      search_with_filter, mirroring how EmbeddingStore tracks document identity.
    - Upsert is used instead of insert so re-running a benchmark script on the same
      index is idempotent — duplicate chunk IDs are silently overwritten.
    - Pinecone metadata filter syntax uses {"key": {"$eq": value}} for equality,
      which is translated automatically from the plain {key: value} dicts used by
      EmbeddingStore's search_with_filter callers.

    Usage:
        from src.pinecone_store import PineconeStore
        from src.langchain_embeddings import LangChainHuggingFaceEmbedder

        store = PineconeStore(
            index_name="ml-rag-index",
            embedding_fn=LangChainHuggingFaceEmbedder(),
            dimension=384,
        )
        store.add_documents(docs)
        results = store.search("gradient descent", top_k=3)
    """

    # Pinecone metadata has a 40 KB per-vector limit.
    # We truncate stored content to avoid exceeding this when chunks are large.
    _MAX_CONTENT_CHARS = 8_000

    def __init__(
        self,
        index_name: str = "ml-rag-index",
        embedding_fn: Callable[[str], list[float]] | None = None,
        dimension: int = 384,
        metric: str = "cosine",
        create_if_not_exists: bool = True,
        namespace: str = "",
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        from pinecone import Pinecone, ServerlessSpec

        from .embeddings import _mock_embed

        self._embedding_fn = embedding_fn or _mock_embed
        self._index_name = index_name
        self._namespace = namespace
        self._next_index = 0

        self._pc = Pinecone()  # reads PINECONE_API_KEY from environment

        existing_names = [idx.name for idx in self._pc.list_indexes()]
        if index_name not in existing_names:
            if not create_if_not_exists:
                raise ValueError(
                    f"Pinecone index '{index_name}' does not exist. "
                    "Set create_if_not_exists=True to create it automatically."
                )
            self._pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            # Wait until the index is ready before returning
            self._wait_until_ready(index_name)

        self._index = self._pc.Index(index_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_until_ready(self, index_name: str, timeout: int = 60) -> None:
        """Poll until Pinecone reports the index as ready (max `timeout` seconds)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            description = self._pc.describe_index(index_name)
            if description.status.get("ready", False):
                return
            time.sleep(2)
        raise TimeoutError(f"Pinecone index '{index_name}' did not become ready within {timeout}s.")

    @staticmethod
    def _to_pinecone_filter(metadata_filter: dict) -> dict:
        """Convert {key: value} dict to Pinecone filter syntax {key: {"$eq": value}}."""
        return {k: {"$eq": v} for k, v in metadata_filter.items()}

    # ------------------------------------------------------------------
    # Public interface (mirrors EmbeddingStore)
    # ------------------------------------------------------------------

    def add_documents(self, docs: list[Document]) -> None:
        """Embed and upsert documents into Pinecone.

        How it works:
        1. Build a vector record per document: embed content, compose a unique ID
           from doc.id + internal counter, merge doc metadata with doc_id + content.
        2. Upsert in batches of 100 (Pinecone's recommended batch size).
        3. Increment _next_index per document so IDs never collide across multiple
           add_documents() calls.
        """
        vectors: list[dict[str, Any]] = []
        for doc in docs:
            embedding = self._embedding_fn(doc.content)
            vec_id = f"{doc.id}_{self._next_index}"
            content_stored = doc.content[: self._MAX_CONTENT_CHARS]
            metadata: dict[str, Any] = {
                **doc.metadata,
                "doc_id": doc.id,
                "_content": content_stored,
            }
            vectors.append({"id": vec_id, "values": embedding, "metadata": metadata})
            self._next_index += 1

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self._index.upsert(
                vectors=vectors[i : i + batch_size],
                namespace=self._namespace,
            )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Find top_k most similar documents to query.

        Returns a list of dicts with keys: content, score, metadata.
        This matches the EmbeddingStore.search() contract so KnowledgeBaseAgent
        works without modification.
        """
        query_vec = self._embedding_fn(query)
        response = self._index.query(
            vector=query_vec,
            top_k=top_k,
            include_metadata=True,
            namespace=self._namespace,
        )
        results: list[dict[str, Any]] = []
        for match in response.matches:
            meta = dict(match.metadata)
            content = meta.pop("_content", "")
            results.append(
                {
                    "content": content,
                    "score": float(match.score),
                    "metadata": meta,
                }
            )
        return results

    def search_with_filter(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Search with optional Pinecone server-side metadata filtering.

        Unlike EmbeddingStore's client-side filter, this delegates filtering to
        Pinecone's query API, which is more efficient for large indexes.

        metadata_filter format: plain {key: value} dict (same as EmbeddingStore).
        Internally converted to Pinecone's {"key": {"$eq": value}} syntax.
        """
        if metadata_filter is None:
            return self.search(query, top_k)

        query_vec = self._embedding_fn(query)
        response = self._index.query(
            vector=query_vec,
            top_k=top_k,
            include_metadata=True,
            namespace=self._namespace,
            filter=self._to_pinecone_filter(metadata_filter),
        )
        results: list[dict[str, Any]] = []
        for match in response.matches:
            meta = dict(match.metadata)
            content = meta.pop("_content", "")
            results.append(
                {
                    "content": content,
                    "score": float(match.score),
                    "metadata": meta,
                }
            )
        return results

    def get_collection_size(self) -> int:
        """Return total number of vectors stored in the namespace."""
        stats = self._index.describe_index_stats()
        if self._namespace:
            ns = stats.namespaces.get(self._namespace)
            return int(ns.vector_count) if ns else 0
        return int(stats.total_vector_count)

    def delete_document(self, doc_id: str) -> bool:
        """Delete all vectors belonging to a document identified by doc_id.

        Uses Pinecone's server-side metadata filter delete, which is more efficient
        than fetching IDs first. Returns True if at least one vector was removed.
        """
        before = self.get_collection_size()
        self._index.delete(
            filter={"doc_id": {"$eq": doc_id}},
            namespace=self._namespace,
        )
        after = self.get_collection_size()
        return after < before

    def delete_all(self, namespace: str | None = None) -> None:
        """Delete all vectors in the namespace (clean slate between benchmark runs).

        Silently ignores 404 'Namespace not found' errors — the namespace simply
        does not exist yet, which is equivalent to it already being empty.
        """
        ns = namespace if namespace is not None else self._namespace
        try:
            self._index.delete(delete_all=True, namespace=ns)
        except Exception as exc:
            # Pinecone returns 404 when the namespace has never been written to.
            # Treat this as a no-op: nothing to delete.
            if "Namespace not found" in str(exc) or "404" in str(exc):
                return
            raise
