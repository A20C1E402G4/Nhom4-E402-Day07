"""Phase 2 — Group: Comparing Retrieval Strategies

Data   : data/machine_learning.md  (CS229 lecture notes by Andrew Ng)
Chunker: LangChain MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter
Embedder: LangChain HuggingFaceEmbeddings (all-MiniLM-L6-v2, 384-dim)
Store  : Pinecone (cosine metric, serverless on AWS us-east-1)

Two strategies are compared on the same set of benchmark queries:
  Strategy A — Markdown Header Chunking  (respects section boundaries)
  Strategy B — LC Recursive Chunking     (fixed character budget, with overlap)

Usage:
    # Set up .env with PINECONE_API_KEY, then run:
    python phase2_benchmark.py

    # Override queries from command line (space-separated quoted string):
    python phase2_benchmark.py "What is gradient descent?"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)

from src.agent import KnowledgeBaseAgent
from src.langchain_chunking import LangChainMarkdownChunker, LangChainRecursiveChunker
from src.langchain_embeddings import LangChainHuggingFaceEmbedder
from src.models import Document
from src.pinecone_store import PineconeStore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_FILE = Path("data/machine_learning.md")

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "vinlab7")

# Embedding dimension — must match the Pinecone index.
# vinlab7 was created with dim=1024; we use bge-large-en-v1.5 (1024-dim).
# Override via PINECONE_INDEX_NAME + EMBEDDING_DIM env vars for other indexes.
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# Benchmark queries (5 queries agreed on by the group).
# Each covers a different part of the document to test breadth of retrieval.
BENCHMARK_QUERIES: list[tuple[str, str]] = [
    (
        "What is gradient descent and how does it minimize the cost function?",
        "Gradient descent iteratively updates θ by moving in the direction of steepest decrease "
        "of J(θ). The update rule is θ_j := θ_j − α * ∂J/∂θ_j where α is the learning rate.",
    ),
    (
        "What is the cost function used in linear regression?",
        "The least-squares cost function J(θ) = ½ Σ (h_θ(x^(i)) − y^(i))² summed over all "
        "training examples.",
    ),
    (
        "How does logistic regression handle classification problems?",
        "Logistic regression uses the sigmoid function g(z) = 1/(1+e^−z) to output a probability "
        "between 0 and 1, which is then thresholded to produce a binary class label.",
    ),
    (
        "What is the normal equation and when is it used instead of gradient descent?",
        "The normal equation θ = (X^T X)^−1 X^T y solves for the optimal θ analytically, "
        "without iteration. It is preferred for small feature sets (n < ~10,000) because it avoids "
        "choosing a learning rate.",
    ),
    (
        "What is locally weighted linear regression and how does it differ from standard LR?",
        "LWLR fits a new model for each query point x by weighting training examples with "
        "w^(i) = exp(−(x^(i)−x)² / 2τ²), giving higher weight to nearby examples. "
        "Unlike standard LR, it is a non-parametric method that re-trains per query.",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_text() -> str:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    return DATA_FILE.read_text(encoding="utf-8")


def print_separator(char: str = "-", width: int = 72) -> None:
    print(char * width)


def run_strategy(
    label: str,
    docs: list[Document],
    store: PineconeStore,
    queries: list[tuple[str, str]],
    top_k: int = 3,
) -> dict[str, list[dict]]:
    """Upload docs to Pinecone and run benchmark queries. Returns results dict."""
    print_separator()
    print(f"Strategy: {label}  |  chunks: {len(docs)}")
    avg_len = sum(len(d.content) for d in docs) / len(docs) if docs else 0
    print(f"  avg chunk length: {avg_len:.0f} chars")
    print_separator(".")

    store.delete_all()
    store.add_documents(docs)
    actual_size = store.get_collection_size()
    print(f"  Pinecone index size after upsert: {actual_size} vectors")

    results: dict[str, list[dict]] = {}
    for query, gold in queries:
        hits = store.search(query, top_k=top_k)
        results[query] = hits
        print(f"\n  Q: {query}")
        for rank, hit in enumerate(hits, 1):
            section = hit["metadata"].get("section") or hit["metadata"].get("subsection", "")
            print(
                f"    [{rank}] score={hit['score']:.4f}"
                + (f" | section={section!r}" if section else "")
            )
            print(f"         {hit['content'][:120].replace(chr(10), ' ')!r}")
    return results


def demo_llm(prompt: str) -> str:
    """Mock LLM for agent demo (replace with real LLM in production)."""
    preview = prompt[:500].replace("\n", " ")
    return f"[DEMO LLM] Based on context: {preview}..."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(custom_query: str | None = None) -> None:
    text = load_text()
    print(f"Loaded {DATA_FILE}  ({len(text):,} chars)\n")

    # Build embedder once (shared between both strategies for fair comparison).
    # BAAI/bge-large-en-v1.5 produces 1024-dim vectors — matches the vinlab7 index.
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    print(f"Initialising LangChain HuggingFace embedder ({model_name})...")
    embedder = LangChainHuggingFaceEmbedder(model_name=model_name)
    print(f"  backend: {embedder._backend_name}\n")

    # Chunk documents with both strategies
    source_id = "machine_learning"

    chunker_a = LangChainMarkdownChunker(chunk_size=500, chunk_overlap=50)
    docs_a = chunker_a.chunk(text, source=source_id)

    chunker_b = LangChainRecursiveChunker(chunk_size=500, chunk_overlap=50)
    docs_b = chunker_b.chunk(text, source=source_id)

    print(f"Strategy A  (Markdown Header): {len(docs_a):3d} chunks")
    print(f"Strategy B  (LC Recursive):    {len(docs_b):3d} chunks\n")

    # Create Pinecone store
    print(f"Connecting to Pinecone index '{INDEX_NAME}' (dim={EMBEDDING_DIM})...")
    store = PineconeStore(
        index_name=INDEX_NAME,
        embedding_fn=embedder,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        create_if_not_exists=True,
    )
    print("  Connected.\n")

    queries = BENCHMARK_QUERIES
    if custom_query:
        queries = [(custom_query, "(no gold answer)")]

    # --- Run benchmarks -----------------------------------------------------
    results_a = run_strategy("A - Markdown Header", docs_a, store, queries)
    results_b = run_strategy("B - LC Recursive", docs_b, store, queries)

    # --- Summary table ------------------------------------------------------
    print_separator("=")
    print("SUMMARY -- Top-1 scores per query")
    print_separator("=")
    print(f"{'Query':<55} {'A':>6} {'B':>6}  Winner")
    print_separator(".")
    for query, gold in queries:
        score_a = results_a[query][0]["score"] if results_a[query] else 0.0
        score_b = results_b[query][0]["score"] if results_b[query] else 0.0
        winner = "A" if score_a >= score_b else "B"
        short_q = query[:52] + "..." if len(query) > 55 else query
        print(f"{short_q:<55} {score_a:>6.3f} {score_b:>6.3f}  {winner}")

    # --- Metadata filter demo ------------------------------------------------
    print_separator()
    print("Metadata filter demo (strategy A -- filter by section='Supervised learning'):")
    store.delete_all()
    store.add_documents(docs_a)
    filtered_results = store.search_with_filter(
        "linear regression hypothesis",
        top_k=3,
        metadata_filter={"section": "Supervised learning"},
    )
    for rank, hit in enumerate(filtered_results, 1):
        print(
            f"  [{rank}] score={hit['score']:.4f} | "
            f"section={hit['metadata'].get('section')!r} | "
            f"{hit['content'][:100].replace(chr(10), ' ')!r}"
        )

    # --- Agent demo ---------------------------------------------------------
    print_separator()
    print("KnowledgeBaseAgent demo (strategy A):")
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    demo_q = "Explain the LMS algorithm update rule."
    print(f"  Q: {demo_q}")
    print(f"  A: {agent.answer(demo_q, top_k=3)[:300]}")

    print_separator("=")
    print("Phase 2 benchmark complete.")


if __name__ == "__main__":
    custom = " ".join(sys.argv[1:]).strip() or None
    main(custom_query=custom)
