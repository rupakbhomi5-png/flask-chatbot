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
2. **Count your chunks before you pick a vector store — you probably don't
   need one.** This is the single most expensive lesson from this build.
   A vector *database* (chromadb, and its peers) brings a native/Rust
   extension, a tenant/collection layer, background threads and a ~23MB
   wheel. Below roughly 10,000 chunks, all of that buys you nothing that
   `numpy` doesn't already do in five lines: store `{"chunks": [...],
   "embeddings": [[...]]}` as JSON, load it, and rank by cosine similarity
   (`(vectors @ query) / (norms * query_norm)`). numpy already ships as a
   `fastembed` dependency, so this adds **zero** new packages. Reach for an
   actual vector DB only when a linear scan is measurably too slow —
   realistically north of ~50,000 chunks. This project stores 18.
   See "the chromadb incident" in Part 2 for what ignoring this cost.
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
- **When a deployed hang can't be reproduced locally, get a real stack
  trace before changing any more code.** `faulthandler.dump_traceback_later(
  15, exit=False, file=sys.stderr)` at the top of the suspect function,
  cancelled in a `finally`, prints every thread's actual C/Python stack to
  the logs if the call doesn't finish in time. Two of those dumps ended a
  debugging session that four "plausible fix, redeploy, still broken"
  cycles had not. Timing prints tell you *that* something is slow; the
  stack tells you *where* it is stopped. Reach for it on hang #1, not
  hang #4 — every guess-and-redeploy round costs a deploy and, if the
  service is customer-facing, real downtime.
- **A hang is not a slowdown, and the difference tells you where to look.**
  Slow-and-finishes points at CPU/throttling/network. Never-returns points
  at a lock, a native extension, or a threading-model mismatch — no amount
  of raising timeouts or trimming thread counts will fix it. Check whether
  the call *ever* completes given unlimited time before theorizing about
  performance.
- **Confirm Render's actual Start Command matches the Procfile before
  blaming application code for timeouts.** Render only reads `Procfile` when
  the service's Settings → Start Command field is empty. If a Start Command
  was ever set by hand (even months ago, even for an unrelated reason), it
  silently wins forever and the Procfile is dead text nobody's reading. This
  service had Start Command `gunicorn app_memory:app` — no
  `--worker-class`, no `--timeout` — while the Procfile said `gthread`
  threads=8 timeout=120. Every request ran on the gunicorn default (sync
  worker, 30s timeout) with nobody aware of it, until a slower feature
  (RAG's embedding step) finally exceeded 30s and exposed it. Check
  Settings → Start Command directly, don't assume the Procfile is what's
  actually running, on any Render service before debugging a timeout.

---

## PART 2 — what was actually built here (flask_app, 2026-08-01)

**Files added:**
- `rag_store.py` — `chunk_text()`, `ingest_document()`, `retrieve()`. Uses
  `fastembed` (`BAAI/bge-small-en-v1.5`, lazy-loaded) for embeddings and a
  plain JSON file per client in `rag_db/` (gitignored) for storage, ranked
  with `numpy` cosine similarity. **No vector database** — chromadb was
  tried first and removed, see "the chromadb incident" below.
- `ingest_doc.py` — CLI onboarding script:
  `python ingest_doc.py <client_slug> <path_to_doc.txt>`. Plain text/
  markdown only, no PDF parsing yet.

**Files changed:**
- `requirements.txt` — added `fastembed` only. (File is UTF-16 LE
  with CRLF line endings, unusual but pre-existing — read/rewrite it via
  Python with explicit `utf-16` encoding, not a plain UTF-8 append, which
  would produce a mixed-encoding file. `numpy` needs no entry; it already
  arrives as a fastembed dependency.)
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

**Pilot round 3, 2026-08-01: the chromadb incident.** The first
`RAG_ENABLED=true` deploy broke the live demo, and it took roughly two
hours and a dozen deploys to actually fix. Written up honestly because
the wrong turns are the useful part.

*The one real, separate bug (fixed early, unrelated to RAG):*
Roofing-demo's Render **Start Command had silently diverged from the
Procfile** — bare `gunicorn app_memory:app`, no flags — so every request
had always been running on gunicorn's 30s-timeout default instead of the
Procfile's `--timeout 120`. Nobody noticed until RAG made one request slow
enough to cross 30s. Fixed in Render Settings. See the Part 1 bullet.

*The actual blocker:* `chromadb` 1.5.9 does not work on this Render
Starter instance, at all. Its Rust-backed client **hung forever** — never
returned, no error, no traceback — on the first real request in a deployed
gunicorn worker, until gunicorn's own `WORKER TIMEOUT` killed the process
and the cycle repeated every ~2 minutes with zero incoming traffic.

*Four fixes were tried and all failed live*, each costing a deploy:
1. `Settings(anonymized_telemetry=False)` — chromadb spawns a background
   posthog thread by default. Worth doing regardless, did not fix this.
2. `fastembed(threads=1)` — onnxruntime's busy-spin thread pool was the
   suspect. Also worth keeping, also not the cause.
3. A **thread-local** chromadb client (one per worker thread).
4. Switching gunicorn from `gthread` to the `sync` worker.

*What finally gave the answer:* `faulthandler.dump_traceback_later()`
around `retrieve()`, which printed real stacks from the stuck process.
Dump 1 caught it inside `chromadb/api/rust.py: list_collections()`. Dump 2,
after the thread-local fix, caught it inside
`PersistentClient.__init__ → get_tenant()` — i.e. it now hung on
*constructing* the client rather than using it. That second dump is what
killed the whole threading theory: fixes 3 and 4 were both elaborate ways
of managing an object that couldn't be safely created in the first place.

*The fix that worked: delete the vector database.* This app stores **18
chunks**. `rag_store.py` now writes `{"chunks": [...], "embeddings":
[[...]]}` to a JSON file and ranks with a numpy dot product. chromadb was
dropped from `requirements.txt` entirely; numpy was already present via
fastembed, so the dependency count went *down*.

**Confirmed working live 2026-08-01 10:09 PM**, worker `2l89h`, commit
`002fe9d`: `/chat` responds in **2.2s** (was: hung past 130s), no crash
loop. Verified RAG is genuinely grounding answers, not just not-crashing —
asked the live bot about emergency tarping fees and it answered "$150
dispatch fee… additional $75 after-hours fee… credited back toward your
final repair invoice." Those figures appear **only** in
`rag_seed_docs/roofing.txt`, nowhere in `roofing_data.json`.

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
