from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        """
        Store references to the vector store and the LLM callable.

        How it works:
            Simply assigns both arguments to instance attributes so that answer()
            can access them without re-construction. No embedding or network call
            happens at init time.

        Why this is enough:
            - Dependency injection of both the store and the LLM function makes the
              agent fully testable: tests can pass a pre-populated store and a lambda
              that returns a fixed string, with no real API keys or models required.
            - Keeping __init__ side-effect-free means constructing an agent is cheap
              and safe to do inside a test setUp method.
        """
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        """
        Answer a question using retrieved context from the knowledge base.

        How it works:
            1. Call self.store.search(question, top_k=top_k) to retrieve the top_k
               most semantically similar chunks to the question. The store handles
               embedding the query and ranking candidates.
            2. Join the "content" field of each result with double newlines to form
               a readable context block. Double newlines visually separate distinct
               chunks, making it easier for the LLM to treat them as independent
               passages.
            3. Construct a prompt in the standard RAG format:
                   Context:
                   <retrieved chunks>

                   Question: <question>
                   Answer:
               The "Answer:" suffix signals to the LLM where to begin its response,
               which is a common prompting convention for instruction-following models.
            4. Pass the assembled prompt to self.llm_fn and return whatever it
               produces — the agent imposes no post-processing on the LLM output.

        Why this is enough:
            - The three-step retrieve → ground → generate loop is the core RAG
              pattern. Every non-trivial improvement (re-ranking, citation, fallback
              for empty retrieval) builds on top of this foundation.
            - Grounding the prompt in retrieved context reduces hallucination: the
              LLM is steered toward the document corpus rather than its parametric
              knowledge alone.
            - Returning the raw LLM output keeps the agent's contract simple: a
              string in, a string out. The test suite only checks that the return
              value is a non-empty string, which this implementation always satisfies
              as long as llm_fn does.
        """
        results = self.store.search(question, top_k=top_k)
        context = "\n\n".join(r["content"] for r in results)
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        return self.llm_fn(prompt)
