# Adding RAG to a chatbot — playbook

Built 2026-08-01 for the AI Agency's `flask_app` chatbot product. Two parts:
Part 1 is generic — copy it into any future AI project's own notes as a
starting checklist. Part 2 documents exactly what was built here, as a
worked example of Part 1.

---

## PART 1 — the generic pattern (copy this into any future project)

**What RAG actually is, mechanically:** a document store, a chunking step,
an embedding step (text → vectors), a retrieval step (find the vectors
closest to the question), and a generation step (feed those chunks to the
LLM as grounding). Five moving parts, no more.

**When you need it vs. when you don't:**
- Don't build RAG if all your content fits in a normal prompt (a short FAQ,
  a handful of products/services). Just put it in the prompt. RAG adds
  moving parts for no benefit at that size.
- Build RAG when the real content is a document too long to paste into a
  prompt every request — a policy doc, a long FAQ, a manual, a catalog with
  hundreds of entries — AND you want answers grounded in that document
  instead of the model guessing or the developer hand-transcribing it into
  structured JSON.

**The build, in order:**
1. **Pick a local, free embedding model.** Avoid `sentence-transformers` +
   `torch` unless you have room for it (several hundred MB, meaningful
   cold-start memory) — fine for a beefy server, bad for a free/hobby-tier
   host. `fastembed` (quantized ONNX models, ~100-150MB, no GPU, no torch)
   is the lighter default for anything deploy-size-constrained. Both are
   free and run locally — no API key, no per-call cost, no new vendor.
2. **Pick a local vector store.** `chromadb` running embedded/persistent
   (no separate server process, just a folder on disk) is the free default.
   Only reach for a hosted vector DB (Pinecone, etc.) once you have a real
   scale or multi-server reason to — it's a new bill and a new dependency
   for something a folder on disk already does at this size.
3. **Write a chunker.** Paragraph-aware splitting (blank-line boundaries),
   hard-wrap anything longer than ~150 words. Good enough for policy docs
   and FAQs. Revisit only if your source documents don't have paragraph
   structure at all (e.g. one wall of text with no blank lines).
4. **Write an ingestion function/script.** Chunk → embed → store, keyed by
   a client/project identifier so multiple documents/clients don't collide
   in one collection. Make re-running it replace the collection, not
   append — avoids stale-chunk bookkeeping for what's usually a small
   enough document that a full re-embed is cheap.
5. **Write a retrieval function.** Embed the incoming question, query the
   vector store for the top-k closest chunks (k=3-5 is a reasonable
   default), return the raw chunk text. Must return an empty result
   gracefully (no ingested collection yet, or nothing relevant found) —
   never let this be a hard failure that breaks the whole response.
6. **Wire retrieval into the actual request path, gated behind a feature
   flag.** Default OFF. Build the per-request prompt as:
   `static_system_prompt + retrieved_chunks_block`, where the block is an
   empty string when RAG is off or nothing was found — so the existing
   behavior is byte-for-byte unchanged unless the flag is explicitly on.
   This is what makes it safe to build without touching anything already
   live.
7. **Test with paraphrased questions, not exact document wording.** The
   whole point of embeddings over keyword search is that "do you charge
   extra for emergency calls at night" should still find a chunk that says
   "after-hours emergency calls carry a $75 dispatch fee." If it doesn't,
   something's wrong with chunking or the embedding model, not the
   retrieval logic.

**Things that will bite you if skipped:**
- Gitignore the vector store's on-disk folder — it's generated data, not
  source, and can get large.
- Don't call the embedding model in a hot path more than once per request.
  Embed the question once, reuse it if the request needs to loop
  (tool-calling, retries) — don't re-embed on every loop iteration.
- Lazy-load the embedding model (only on first actual use), not at process
  startup — otherwise every deploy pays the model-load cost even for
  requests that never use RAG.
- A RAG bot answers from whatever's in the source document, wrong prices
  and all. Don't point it at data you know is unconfirmed/placeholder —
  fix the source data first, RAG doesn't launder bad input into good
  output.
- **Retrieval is scoped per collection, and only per collection.** Keying
  every collection by `client_slug` isn't just organizational — it's the
  entire isolation boundary. A roofing client's bot can only ever retrieve
  from the `"roofing"` collection; it has no path to any other client's
  data, your own business's data, or anything outside whatever document you
  explicitly ran `ingest_doc.py` on for that slug. If a bot ever says
  something it shouldn't have access to, the cause is a data-entry mistake
  (ingested the wrong file under that slug), not a leak in the retrieval
  logic itself.

---

## PART 2 — what was actually built here (flask_app, 2026-08-01)

**Files added:**
- `rag_store.py` — `chunk_text()`, `ingest_document()`, `retrieve()`. Uses
  `fastembed` (`BAAI/bge-small-en-v1.5`, lazy-loaded) + `chromadb`
  (persistent, local, folder: `rag_db/` — gitignored).
- `ingest_doc.py` — CLI onboarding script:
  `python ingest_doc.py <client_slug> <path_to_doc.txt>`. Plain text/
  markdown only, no PDF parsing yet.

**Files changed:**
- `requirements.txt` — added `fastembed`, `chromadb`. (File is UTF-16 LE
  with CRLF line endings, unusual but pre-existing — edited with
  PowerShell `Add-Content -Encoding Unicode` to match, not plain
  UTF-8 append, which would have produced a mixed-encoding file.)
- `app_memory.py` — added `RAG_ENABLED` env var (default `false`, same
  pattern as the existing `MCP_ENABLED` flag) and `_RAG_CLIENT_SLUG`
  (derived from `DATA_FILE`, e.g. `HVAC_data.json` → `"hvac"`). Added
  `build_rag_context()`, called once per `/chat` request on the raw
  visitor message, appended to `SYSTEM_PROMPT` to build a per-request
  `request_system_prompt` used in both `client.messages.create` calls in
  the `/chat` route. Empty string when RAG is off — zero behavior change
  for any of the 6 live demo templates unless `RAG_ENABLED=true` is set
  explicitly on that specific deployment.
- `.gitignore` — added `rag_db/`.

**Tested locally, working:** ingested a 3-section test HVAC policy doc (30
lines), ran 3 paraphrased test questions against it (emergency-call
pricing, return policy on an installed thermostat, compressor warranty
length) — all three retrieved the correct source chunk despite no literal
word overlap with the question. Test collection deleted after — nothing
fake left in `rag_db/`.

**Deployed 2026-08-01, commit `5734d69`, pushed with explicit go-ahead.**
Confirmed live on both actual Render services ("Rupak-chatbot" and
"Roofing-demo" — there are only 2, not 6, see STATE.md correction),
clean build, no failures. `RAG_ENABLED` is unset on both — still a no-op
in production.

**Pilot round 2, 2026-08-01: auto-ingest-at-startup added.** Gap closed:
Render's free/Starter disk is ephemeral, so anything `ingest_doc.py` wrote
to local `rag_db/` would vanish on the next redeploy or restart, and
there's no Render shell access in this workflow to re-run it by hand every
time. Fix — `rag_seed_docs/<client_slug>.txt` is a checked-in (not
gitignored) directory of seed documents. On boot, if `RAG_ENABLED=true`
and `rag_store.has_collection(slug)` is false, `app_memory.py` auto-ingests
`rag_seed_docs/<slug>.txt` if it exists, so the collection self-heals on
every cold start with zero manual steps after a deploy. `ingest_doc.py`
still exists for onboarding a real client's document by hand later — the
two paths aren't mutually exclusive, auto-ingest is just the bootstrap for
whatever's already checked in.

`rag_seed_docs/roofing.txt` is the pilot's seed doc — a realistic but
explicitly fictional roofing company policy doc (warranty, emergency/storm
response, financing, service area, insurance-claim assistance, materials,
satisfaction policy), labeled as a demo document in its own header since
no real roofing client exists yet. Tested locally: auto-ingest ran end to
end (18 chunks), 3 paraphrased questions (after-hours call fee, financing,
storm insurance help) all retrieved the correct source chunk. Test
collection deleted after, same as round 1.

**Not done yet / deliberately stopped short:**
- Not wired to any real client — the pipeline and the auto-ingest bootstrap
  are both live, but `rag_seed_docs/roofing.txt` is a demo document, not a
  real client's content. Swap that file for a real client's document
  (`ingest_doc.py` still works for a one-off manual ingest too) whenever
  one exists.
- No PDF ingestion — text/markdown only. Add a PDF-to-text step separately
  if a real client's document only exists as a PDF.
- Chunking is naive (paragraph + word-count only). Fine for the tested
  policy-doc shape; revisit if a real client document has a very different
  structure (tables, deeply nested sections).
