"""
rag_store.py — lightweight local RAG layer for the chatbot product.

Chunk a client's long-form document, embed the chunks with a small local
model, keep them in a plain JSON file. At chat time, embed the visitor's
message and return the closest chunks by cosine similarity, to ground the
reply in the client's actual document instead of only the hand-written
business_data.json.

WHY NO VECTOR DATABASE (chromadb was removed 2026-08-01 after it broke
production repeatedly):
chromadb 1.5.9 could not be made to work on Render's Starter instance. Its
Rust-backed client hung indefinitely — never returned, no error, no
traceback — on the first real request in a deployed gunicorn worker.
Confirmed by two live faulthandler stack dumps: once inside
list_collections(), once inside PersistentClient.__init__ -> get_tenant().
Four separate fixes were tried and all failed live: disabling its telemetry
thread, pinning onnxruntime to threads=1, a thread-local client per worker
thread, and switching gunicorn from gthread to the sync worker. Each cost a
deploy and left the demo site crash-looping (gunicorn WORKER TIMEOUT every
~2 minutes, with zero incoming traffic).

The realization that ended it: this app stores ~18 chunks per client. A
vector *database* — with its tenant/collection/persistence machinery, its
Rust extension, and its 23MB wheel — is enormously more infrastructure than
"cosine similarity over a small list" requires. numpy already ships as a
fastembed dependency, so the entire store is now a few lines of dot product
over a JSON file: no database, no native extension, no background threads,
no per-thread state, nothing that can hang. Revisit only if a client's
document is large enough that a linear scan is genuinely too slow — for
reference, that's likely somewhere north of 50,000 chunks, versus 18 today.
"""
import os
import json
import numpy as np

_STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_db")

_embedder = None


def _get_embedder():
    """Lazy singleton — the model loads from disk/downloads on first call,
    not at import time. Keeps startup cheap for demos that never call RAG.

    threads=1 avoids onnxruntime's default busy-spin thread pool, which is a
    real risk on this CPU-throttled instance (0.5 vCPU)."""
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", threads=1)
    return _embedder


def _store_path(client_slug: str) -> str:
    return os.path.join(_STORE_DIR, f"{client_slug}.json")


def has_collection(client_slug: str) -> bool:
    """Whether client_slug has an ingested document on disk. Render's
    filesystem is ephemeral across deploys and restarts, so this is checked
    fresh on every boot rather than assumed persistent."""
    return os.path.exists(_store_path(client_slug))


def chunk_text(text: str, max_words: int = 150) -> list[str]:
    """Paragraph-aware chunking: split on blank lines, then hard-wrap any
    paragraph longer than max_words. Good enough for policy docs and FAQs;
    revisit if a client's doc is one giant unbroken paragraph with no
    blank-line structure at all."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for p in paragraphs:
        words = p.split()
        if len(words) <= max_words:
            chunks.append(p)
        else:
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i:i + max_words]))
    return chunks


def ingest_document(client_slug: str, text: str, source_name: str = "document") -> int:
    """Chunk + embed + write to disk. Re-running for the same client_slug
    replaces that client's whole file rather than appending — onboarding is
    a full re-index, not an incremental update, since these docs are small
    enough that re-embedding everything is cheap and simple beats a
    stale-chunk bookkeeping problem."""
    chunks = chunk_text(text)
    if not chunks:
        return 0
    embeddings = [e.tolist() for e in _get_embedder().embed(chunks)]
    os.makedirs(_STORE_DIR, exist_ok=True)
    with open(_store_path(client_slug), "w", encoding="utf-8") as f:
        json.dump(
            {"source": source_name, "chunks": chunks, "embeddings": embeddings},
            f,
        )
    return len(chunks)


def retrieve(client_slug: str, query: str, k: int = 4) -> list[str]:
    """Top-k closest chunks for this query by cosine similarity. Empty list
    if the client has no ingested document yet — callers must treat that as
    "no RAG context available" and fall back to business_data.json alone,
    not as an error."""
    if not has_collection(client_slug):
        return []
    with open(_store_path(client_slug), "r", encoding="utf-8") as f:
        store = json.load(f)
    doc_vectors = np.array(store["embeddings"], dtype=np.float32)
    if doc_vectors.size == 0:
        return []
    query_vector = np.array(
        list(_get_embedder().embed([query]))[0], dtype=np.float32
    )
    # Cosine similarity: normalize both sides, then dot. Guard against a
    # zero-norm vector so a degenerate embedding can't raise here — RAG
    # failing must degrade to "no context," never break the whole reply.
    doc_norms = np.linalg.norm(doc_vectors, axis=1)
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0 or not np.any(doc_norms):
        return []
    scores = (doc_vectors @ query_vector) / (doc_norms * query_norm + 1e-10)
    top_indices = np.argsort(scores)[::-1][:k]
    return [store["chunks"][i] for i in top_indices]
