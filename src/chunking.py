from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start: start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        """
        Split text into sentence-boundary-aware chunks.

        How it works:
            1. Use a regex lookbehind to split after sentence-ending punctuation
               (. ! ?) followed by a space, or after a period followed by a newline.
               Lookbehinds preserve the punctuation in the preceding token rather
               than discarding it, so sentences stay intact.
            2. Strip and filter empty strings that arise from multiple spaces or
               trailing punctuation at the end of the text.
            3. Group consecutive sentences into chunks of at most
               max_sentences_per_chunk, joining them with a single space.

        Why this is enough:
            - The regex covers the three standard English sentence terminators in
              the two most common positions (before a space or before a newline),
              which handles virtually all plain-prose inputs used in this lab.
            - Grouping by a fixed sentence count is deterministic and easy to tune:
              smaller values give finer-grained retrieval; larger values give more
              context per chunk.
            - Joining with a space produces clean, readable chunks with no leading
              or trailing whitespace, satisfying the "strip extra whitespace" rule
              from the class docstring.
        """
        if not text:
            return []
        sentences = re.split(r'(?<=[.!?]) |(?<=\.)\n', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i:i + self.max_sentences_per_chunk]
            chunks.append(" ".join(group))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        """
        Public entry point: split text into chunks no larger than chunk_size.

        Delegates immediately to _split with the full separator list.
        Handling the empty-text guard here keeps _split's base cases clean.
        """
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        """
        Recursively split current_text using the highest-priority separator that
        actually appears in the text, then re-merge small adjacent pieces.

        Algorithm (one recursive call):
            1. Base cases — stop recursing when:
               a. The text already fits within chunk_size → return [current_text].
               b. No separators left → character-level slice as last resort.
               c. Current separator is "" (empty string) → same character-level
                  slice (the empty separator signals "split every character", so
                  we treat it identically to the no-separators-left case).
            2. If the current separator does not appear in the text, skip it and
               recurse with the next separator — no splitting needed at this level.
            3. Split by the current separator and greedily merge pieces back
               together as long as the merged candidate stays within chunk_size.
               When adding a piece would exceed the limit:
                 - Flush the accumulated chunk to results.
                 - If the new piece itself fits, start a fresh accumulator.
                 - If the new piece is too large on its own, recurse into it
                   with the remaining (lower-priority) separators.
            4. Strip whitespace from every emitted chunk; drop empty strings.

        Why this is enough:
            - The separator priority list mirrors document structure from coarse
              (paragraph breaks) to fine (spaces, then characters). Each level only
              runs when the coarser split was insufficient, so the algorithm
              naturally prefers semantically meaningful boundaries.
            - Greedy merging maximises chunk density without ever exceeding
              chunk_size, which keeps retrieval context as rich as possible.
            - The character-level fallback guarantees termination even for pathological
              inputs (one very long word with no separators), satisfying the
              "falls back gracefully" test requirement.
            - Stripping and filtering at the very end avoids accumulating leading/
              trailing whitespace that appears when separators like "\n\n" are used.
        """
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text]
        if not remaining_separators:
            return [current_text[i:i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        if separator == "":
            return [current_text[i:i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        if separator not in current_text:
            return self._split(current_text, next_separators)

        parts = current_text.split(separator)
        result = []
        current_chunk = ""

        for part in parts:
            if not part:
                continue
            candidate = current_chunk + separator + part if current_chunk else part
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    result.append(current_chunk)
                    current_chunk = ""
                if len(part) <= self.chunk_size:
                    current_chunk = part
                else:
                    result.extend(self._split(part, next_separators))

        if current_chunk:
            result.append(current_chunk)

        return [r.strip() for r in result if r.strip()]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.

    How it works:
        1. Compute the magnitude (L2 norm) of each vector by taking the square
           root of its dot product with itself: ||v|| = sqrt(dot(v, v)).
        2. Guard against division by zero — a zero vector has no direction and
           therefore no meaningful similarity with anything, so we return 0.0.
        3. Return the normalised dot product, which ranges from -1 (opposite
           directions) through 0 (orthogonal) to +1 (identical directions).

    Why this is enough:
        - Cosine similarity is the standard metric for comparing dense embedding
          vectors because it is scale-invariant: only the direction of the vector
          matters, not its magnitude. This is exactly what embedding models
          optimise for.
        - The zero-magnitude guard is the only edge case that can cause a runtime
          error (ZeroDivisionError); all other inputs produce a valid float in
          [-1, 1].
        - Reusing the existing _dot helper keeps the implementation DRY and
          consistent with how similarity is computed elsewhere in the store.
    """
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        """
        Run all three chunking strategies on the same text and return statistics.

        How it works:
            1. Instantiate FixedSizeChunker, SentenceChunker, and RecursiveChunker
               with the given chunk_size (SentenceChunker uses its default sentence
               count because it does not take a character limit).
            2. Run each chunker's .chunk() method on the input text.
            3. For each strategy, compute:
               - count     : total number of chunks produced.
               - avg_length: mean character length across all chunks; 0 when there
                             are no chunks (avoids ZeroDivisionError).
               - chunks    : the raw list of chunk strings for deeper inspection.
            4. Return a dict keyed by strategy name so callers can compare results
               side-by-side.

        Why this is enough:
            - The three keys — "fixed_size", "by_sentences", "recursive" — are
              exactly what the test suite asserts, so the structure is complete.
            - count and avg_length are the two most informative summary statistics
              for comparing chunking strategies: count shows granularity, avg_length
              shows density. Together they reveal trade-offs without requiring the
              caller to iterate over raw chunks.
            - Storing the raw chunks list lets callers (and the report) inspect
              actual content when the summary statistics alone are not enough.
            - Passing chunk_size through to FixedSizeChunker and RecursiveChunker
              makes the comparison fair: both character-based strategies operate
              under the same size budget, so differences in count/avg_length reflect
              strategy behaviour rather than parameter differences.
        """
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size).chunk(text),
            "by_sentences": SentenceChunker().chunk(text),
            "recursive": RecursiveChunker(chunk_size=chunk_size).chunk(text),
        }
        result = {}
        for name, chunks in strategies.items():
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count else 0
            result[name] = {"count": count, "avg_length": avg_length, "chunks": chunks}
        return result
