
"""
Private Hybrid RAG engine for What If.

Encrypted SQLite remains the source of truth.
The vector index exists only in application memory.

Pipeline:
    diary -> chunks -> embeddings
          -> semantic search + BM25
          -> Reciprocal Rank Fusion
          -> relevance filtering
          -> retrieved evidence
"""

import re
import math
from dataclasses import dataclass

import numpy as np

from .embed import embed


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 350
CHUNK_OVERLAP = 70

CANDIDATE_K = 12
FINAL_K = 4

# Reciprocal Rank Fusion constant
RRF_K = 60

# Semantic relevance floor.
# This is a starting value, not a universal BGE threshold.
MIN_SEMANTIC_SCORE = 0.30

# Common words that don't help identify diary topics.
STOPWORDS = {
    "a", "an", "the", "i", "me", "my", "mine",
    "you", "your", "yours", "we", "our", "ours",
    "he", "she", "it", "they", "them", "their",
    "is", "am", "are", "was", "were", "be", "been",
    "being", "do", "does", "did", "have", "has", "had",
    "can", "could", "would", "should", "will",
    "what", "when", "where", "why", "how", "which",
    "if", "then", "than", "and", "or", "but",
    "to", "of", "in", "on", "at", "for", "from",
    "with", "about", "into", "by", "as", "this",
    "that", "these", "those", "now", "today",
    "think", "thinking", "idea",
}


# ============================================================
# QUERY EXPANSION
# ============================================================

# Initial domain vocabulary for motorcycle-related questions.
# Expand this carefully as the application grows.




def expand_query(question):
    """
    Domain-independent query processing.

    Keep the original meaningful terms.
    Do not inject topic-specific vocabulary.
    """
    return _words(question)

# ============================================================
# CHUNK
# ============================================================

@dataclass
class RAGChunk:
    chunk_id: str
    day: str
    text: str
    embedding: np.ndarray


# ============================================================
# TEXT PROCESSING
# ============================================================

def _tokens(text):
    return re.findall(r"\S+", str(text).strip())


def _word_tokens(text):
    words = re.findall(
        r"[a-z0-9']+",
        str(text).lower()
    )

    return [
        word for word in words
        if word not in STOPWORDS and len(word) > 1
    ]


def _words(text):
    return set(_word_tokens(text))


def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):
    """Split long diary entries into overlapping word chunks."""

    tokens = _tokens(text)

    if not tokens:
        return []

    if len(tokens) <= chunk_size:
        return [" ".join(tokens)]

    chunks = []
    start = 0

    while start < len(tokens):

        end = min(start + chunk_size, len(tokens))

        chunk = " ".join(tokens[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(tokens):
            break

        start = end - overlap

    return chunks


# ============================================================
# BM25 KEYWORD RETRIEVAL
# ============================================================

def bm25_search(query_words, documents, top_k=CANDIDATE_K):
    """
    Pure Python BM25 implementation.

    No external search library required.
    """

    if not documents or not query_words:
        return {}

    tokenized_docs = [
        _word_tokens(document)
        for document in documents
    ]

    doc_count = len(tokenized_docs)

    avg_length = (
        sum(len(doc) for doc in tokenized_docs) / doc_count
        if doc_count else 0
    )

    if avg_length == 0:
        return {}

    # Document frequency
    document_frequency = {}

    for document in tokenized_docs:
        for word in set(document):
            document_frequency[word] = (
                document_frequency.get(word, 0) + 1
            )

    k1 = 1.5
    b = 0.75

    scores = {}

    for index, document in enumerate(tokenized_docs):

        if not document:
            continue

        doc_length = len(document)
        score = 0.0

        for word in query_words:

            frequency = document.count(word)

            if frequency == 0:
                continue

            df = document_frequency.get(word, 0)

            idf = math.log(
                1 + (doc_count - df + 0.5) / (df + 0.5)
            )

            numerator = frequency * (k1 + 1)

            denominator = frequency + k1 * (
                1 - b + b * doc_length / avg_length
            )

            score += idf * (numerator / denominator)

        if score > 0:
            scores[index] = score

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return dict(ranked[:top_k])


# ============================================================
# KEYWORD SCORE
# ============================================================

def keyword_score(query_words, document):

    document_words = _words(document)

    if not query_words or not document_words:
        return 0.0

    return (
        len(query_words & document_words)
        / len(query_words)
    )


# ============================================================
# RAG INDEX
# ============================================================

class RAGIndex:

    def __init__(self):

        self.chunks = []
        self.matrix = None

    # --------------------------------------------------------
    # BUILD INDEX
    # --------------------------------------------------------

    def build(self, entries):

        chunks = []

        for day, entry, _old_vector in entries:

            text = entry.get("text", "").strip()

            if not text:
                continue

            pieces = chunk_text(text)

            for index, piece in enumerate(pieces):

                vector = embed(
                    piece,
                    is_query=False
                )

                chunks.append(
                    RAGChunk(
                        chunk_id=f"{day}:{index}",
                        day=day,
                        text=piece,
                        embedding=vector,
                    )
                )

        self.chunks = chunks

        if not chunks:
            self.matrix = None
            return 0

        self.matrix = np.vstack(
            [chunk.embedding for chunk in chunks]
        ).astype(np.float32)

        return len(chunks)

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    def search(self, question, k=FINAL_K):

        if not self.chunks or self.matrix is None:
            return []

        query_vector = embed(
            question,
            is_query=True
        )

        if query_vector.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                "Query and document embedding dimensions do not match."
            )

        # ----------------------------------------------------
        # 1. SEMANTIC SEARCH
        # ----------------------------------------------------

        semantic_scores = np.dot(
            self.matrix,
            query_vector
        )

        semantic_order = np.argsort(
            semantic_scores
        )[::-1]

        semantic_candidates = semantic_order[:CANDIDATE_K]

        semantic_ranks = {
            int(index): rank
            for rank, index in enumerate(
                semantic_order,
                start=1
            )
        }

        # ----------------------------------------------------
        # 2. BM25 SEARCH
        # ----------------------------------------------------

        query_words = expand_query(question)

        documents = [
            chunk.text
            for chunk in self.chunks
        ]

        bm25_scores = bm25_search(
            query_words,
            documents,
            top_k=CANDIDATE_K
        )

        bm25_ranks = {
            index: rank
            for rank, (index, _score) in enumerate(
                bm25_scores.items(),
                start=1
            )
        }

        # ----------------------------------------------------
        # 3. RECIPROCAL RANK FUSION
        # ----------------------------------------------------

        candidate_indices = set(
            int(index)
            for index in semantic_candidates
        )

        candidate_indices.update(bm25_scores.keys())

        results = []

        for index in candidate_indices:

            chunk = self.chunks[index]

            semantic = float(semantic_scores[index])

            lexical = keyword_score(
                query_words,
                chunk.text
            )

            bm25_score = bm25_scores.get(index, 0.0)

            semantic_rank = semantic_ranks[index]
            lexical_rank = bm25_ranks.get(index)

            # Semantic ranking always contributes.
            rrf_score = (
                1.0 / (RRF_K + semantic_rank)
            )

            # BM25 contributes only when a real lexical match exists.
            if lexical_rank is not None:
                rrf_score += (
                    1.0 / (RRF_K + lexical_rank)
                )

            results.append(
                {
                    "chunk": chunk,
                    "semantic_score": semantic,
                    "keyword_score": lexical,
                    "bm25_score": bm25_score,
                    "semantic_rank": semantic_rank,
                    "bm25_rank": lexical_rank,
                    "score": rrf_score,
                    "rrf_score": rrf_score,
                }
            )

        # ----------------------------------------------------
        # 4. RANK RESULTS
        # ----------------------------------------------------

        results.sort(
            key=lambda item: (
                item["score"],
                item["semantic_score"]
            ),
            reverse=True
        )

        # ----------------------------------------------------
        # 5. RELEVANCE FILTER
        # ----------------------------------------------------

        filtered = []

        best_semantic = max(
            (item["semantic_score"] for item in results),
            default=0.0
        )

        for item in results:

            semantic = item["semantic_score"]
            lexical = item["keyword_score"]

            # Reject entries that are both semantically weak
            # and have no meaningful keyword overlap.
            if (
                semantic < MIN_SEMANTIC_SCORE
                and lexical == 0
            ):
                continue

            # Keep entries reasonably close to the best
            # semantic match, or entries with useful lexical evidence.
            if (
                semantic < best_semantic - 0.20
                and lexical < 0.20
            ):
                continue

            filtered.append(item)

        # Don't return more than FINAL_K results.
        return filtered[:min(k, FINAL_K)]

    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    def debug_search(self, question, k=FINAL_K):

        results = self.search(
            question,
            k=k
        )

        print()
        print("=" * 75)
        print("[RAG] HYBRID QUERY")
        print(question)
        print("=" * 75)

        for index, result in enumerate(results, start=1):

            chunk = result["chunk"]

            print(
                f"\n#{index}"
                f" | day={chunk.day}"
                f" | RRF={result['rrf_score']:.4f}"
                f" | semantic={result['semantic_score']:.3f}"
                f" | semantic_rank={result['semantic_rank']}"
                f" | BM25={result['bm25_score']:.3f}"
                f" | BM25_rank={result['bm25_rank']}"
                f" | keyword={result['keyword_score']:.3f}"
            )

            print(f"   {chunk.text[:400]}")

        print("=" * 75)

        return results
