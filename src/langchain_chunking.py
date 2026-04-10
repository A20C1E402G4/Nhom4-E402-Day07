"""LangChain-based chunking strategies for Phase 2.

These extend the existing chunking strategies in chunking.py with LangChain's
text splitters, providing richer metadata handling and better Markdown support.
Existing files (chunking.py, store.py, agent.py) are not modified.
"""
from __future__ import annotations

from .models import Document


class LangChainMarkdownChunker:
    """
    Split Markdown text by headers using LangChain's MarkdownHeaderTextSplitter,
    then apply RecursiveCharacterTextSplitter to enforce a max chunk size.

    Why this strategy fits machine_learning.md:
    - The document has a clear header hierarchy (##, ###, ####) mapping to
      topics, sections, and subsections (e.g. "Linear Regression", "LMS algorithm").
    - Splitting at header boundaries keeps each chunk semantically self-contained
      within its section, so retrieved chunks will always be on-topic.
    - Section/subsection titles are stored in metadata, enabling filtered search
      by topic (e.g. filter by section="Logistic regression").

    Design rationale:
    1. MarkdownHeaderTextSplitter produces LangChain Documents with header metadata.
    2. RecursiveCharacterTextSplitter then sub-splits any section that is still
       too long, with overlap to preserve cross-boundary context.
    3. Both header metadata and chunk-level metadata (source, index, strategy)
       are merged into the returned Document objects.
    """

    DEFAULT_HEADERS = [
        ("#", "topic"),
        ("##", "section"),
        ("###", "subsection"),
        ("####", "subsubsection"),
    ]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        headers_to_split_on: list[tuple[str, str]] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.headers_to_split_on = headers_to_split_on or self.DEFAULT_HEADERS

    def chunk(self, text: str, source: str = "unknown") -> list[Document]:
        """Chunk Markdown text, returning Documents with section metadata."""
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False,
        )
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        md_sections = md_splitter.split_text(text)
        docs: list[Document] = []
        for section_idx, lc_doc in enumerate(md_sections):
            sub_texts = char_splitter.split_text(lc_doc.page_content)
            for sub_idx, sub_text in enumerate(sub_texts):
                if not sub_text.strip():
                    continue
                metadata = {
                    **lc_doc.metadata,          # topic, section, subsection, ...
                    "source": source,
                    "section_index": section_idx,
                    "chunk_index": section_idx * 1000 + sub_idx,
                    "strategy": "markdown_header",
                }
                docs.append(
                    Document(
                        id=f"{source}_mh_{section_idx}_{sub_idx}",
                        content=sub_text.strip(),
                        metadata=metadata,
                    )
                )
        return docs

    def chunk_text(self, text: str) -> list[str]:
        """Convenience wrapper: return plain strings (compatible with comparators)."""
        return [doc.content for doc in self.chunk(text)]


class LangChainRecursiveChunker:
    """
    Wrap LangChain's RecursiveCharacterTextSplitter.

    Compared to the custom RecursiveChunker in chunking.py, this version:
    - Uses LangChain's production-grade implementation with a richer default
      separator list (handles code blocks, paragraphs, sentences, words).
    - Supports overlap natively (the existing RecursiveChunker does not use overlap).
    - Returns Documents with consistent metadata for downstream filtering.

    Design rationale:
    - chunk_size and chunk_overlap are the primary knobs.
    - Optional `separators` override lets you tune for specific content (e.g. LaTeX).
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators  # None → LangChain defaults

    def chunk(self, text: str, source: str = "unknown") -> list[Document]:
        """Chunk text, returning Documents with index metadata."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        kwargs: dict = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        if self.separators is not None:
            kwargs["separators"] = self.separators

        splitter = RecursiveCharacterTextSplitter(**kwargs)
        texts = splitter.split_text(text)
        return [
            Document(
                id=f"{source}_rc_{i}",
                content=t.strip(),
                metadata={
                    "source": source,
                    "chunk_index": i,
                    "strategy": "lc_recursive",
                },
            )
            for i, t in enumerate(texts)
            if t.strip()
        ]

    def chunk_text(self, text: str) -> list[str]:
        """Convenience wrapper: return plain strings."""
        return [doc.content for doc in self.chunk(text)]
