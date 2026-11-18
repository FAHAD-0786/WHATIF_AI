"""Local semantic embeddings for diary retrieval.

Uses BAAI/bge-small-en-v1.5 through fastembed.
Falls back to a word-hash embedding only if fastembed cannot load.
"""

import hashlib
import re
import traceback

import numpy as np

_model = None
_tried = False
_load_error = None

DIM_FALLBACK = 512
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def _load():
    global _model, _tried, _load_error

    if _tried:
        return _model

    _tried = True

    try:
        from fastembed import TextEmbedding

        print(f"[RAG] Loading embedding model: {MODEL_NAME}")
        _model = TextEmbedding(MODEL_NAME)
        print("[RAG] FastEmbed loaded successfully.")

    except Exception as e:
        _model = None
        print("fastembed could not load:", e)
        _load_error = repr(e)

        print("[RAG] WARNING: FastEmbed failed to load.")
        print(f"[RAG] Reason: {_load_error}")
        traceback.print_exc()

    return _model


def backend():
    _load()
    return "fastembed" if _model is not None else "hash-fallback"


def backend_error():
    _load()
    return _load_error


def _hash_embed(text):
    v = np.zeros(DIM_FALLBACK, dtype=np.float32)

    for w in re.findall(r"[a-z0-9']+", text.lower()):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % DIM_FALLBACK] += 1.0 if (h >> 100) & 1 else -1.0

    return v


def _normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)

    if n == 0:
        return v

    return v / n


def embed(text: str, is_query=False) -> np.ndarray:
    """Create a normalized embedding.

    For BGE retrieval, queries get a retrieval-oriented prefix.
    """

    text = str(text).strip()

    if not text:
        return np.zeros(DIM_FALLBACK, dtype=np.float32)

    m = _load()

    if m is None:
        return _normalize(_hash_embed(text))

    # BGE models work better when queries are explicitly identified.
    # Do not artificially modify the query.
    # Keep the user's semantic meaning intact.

    try:
        if is_query and hasattr(m, "query_embed"):
            v = next(iter(m.query_embed([text])))
        else:
            v = next(iter(m.embed([text])))

        return _normalize(v)

    except Exception as e:
        print("[RAG] Embedding failed:", repr(e))
        return _normalize(_hash_embed(text))