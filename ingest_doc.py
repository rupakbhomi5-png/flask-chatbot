"""
ingest_doc.py — one-time onboarding: turn a client's real document into a
queryable RAG collection for the chatbot to answer from.

Usage:
    python ingest_doc.py <client_slug> <path_to_doc.txt>

<client_slug> must match what app_memory.py derives from DATA_FILE (the
filename with "_data.json" stripped and lowercased — e.g. HVAC_data.json
-> "hvac") so the chat route's retrieval step finds the right collection.

Re-run any time the source document changes. Each run replaces that
client's collection entirely rather than appending to it.

Only .txt/.md input for now — plain text, no PDF parsing. Convert a PDF to
text first (there's no PDF library in requirements.txt and adding one is a
separate decision, not bundled into this).
"""
import sys
from rag_store import ingest_document


def main():
    if len(sys.argv) != 3:
        print("Usage: python ingest_doc.py <client_slug> <path_to_doc.txt>")
        sys.exit(1)
    client_slug, path = sys.argv[1], sys.argv[2]
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    n = ingest_document(client_slug, text, source_name=path)
    print(f"Ingested {n} chunks for '{client_slug}' from {path}")


if __name__ == "__main__":
    main()
