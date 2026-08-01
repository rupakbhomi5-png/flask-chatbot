"""
rag_store.py — lightweight local RAG layer for the chatbot product.

Chunk a client's long-form document once (via ingest_doc.py), embed the
chunks with a small local model, store them in a per-client Chroma
collection on disk. At chat time, embed the visitor's message and retrieve
the closest chunks to ground the reply in the client's actual document
instead of only the hand-written business_data.json.

Deliberately NOT sentence-transformers/torch: that stack is several hundred
MB and this app targets Render's Hobby tier, where deploy size and cold-start
memory are real constraints. fastembed ships small quantized ONNX models
(~130MB for the default model below), no GPU, no torch, pip installable.

Off by default from the app's side (see RAG_ENABLED in app_memory.py) —
importing this module costs nothing until ingest_document() or retrieve()
is actually called, and the embedding model only loads on first use.
"""
import os
import chromadb

_CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_db")
_client = chromadb.PersistentClient(path=_CHROMA_DIR)

_embedder = None


def _get_embedder():
    """Lazy singleton — the model loads from disk/downloads on first call,
    not at import time. Keeps startup cheap for demos that never call RAG."""
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _embedder


def _collection(client_slug: str):
    return _client.get_or_create_collection(name=client_slug)


def has_collection(client_slug: str) -> bool:
    """Whether client_slug already has an ingested collection on disk. Render's
    free/Starter filesystem is ephemeral across deploys and restarts, so this
    is checked fresh on every boot rather than assumed persistent."""
    return client_slug in {c.name for c in _client.list_collections()}


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
    """Chunk + embed + store. Re-running for the same client_slug replaces
    that client's whole collection rather than duplicating or appending —
    onboarding is a full re-index, not an incremental update, since these
    docs are small enough that re-embedding everything is cheap and simple
    beats a stale-chunk bookkeeping problem."""
    chunks = chunk_text(text)
    if not chunks:
        return 0
    if has_collection(client_slug):
        _client.delete_collection(name=client_slug)
    col = _collection(client_slug)
    embeddings = list(_get_embedder().embed(chunks))
    col.add(
        ids=[f"{client_slug}_{i}" for i in range(len(chunks))],
        embeddings=[e.tolist() for e in embeddings],
        documents=chunks,
        metadatas=[{"source": source_name, "chunk": i} for i in range(len(chunks))],
    )
    return len(chunks)


def retrieve(client_slug: str, query: str, k: int = 4) -> list[str]:
    """Top-k closest chunks for this query. Empty list if the client has no
    ingested collection yet — callers must treat that as "no RAG context
    available" and fall back to business_data.json alone, not as an error."""
    if not has_collection(client_slug):
        return []
    col = _collection(client_slug)
    query_embedding = list(_get_embedder().embed([query]))[0].tolist()
    results = col.query(query_embeddings=[query_embedding], n_results=k)
    docs = results.get("documents")
    return docs[0] if docs else []
