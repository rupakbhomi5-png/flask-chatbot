"""
preflight_rag.py — run this BEFORE turning RAG_ENABLED=true on any service.

    python preflight_rag.py <client_slug>      e.g.  python preflight_rag.py hvac

Why this exists: on 2026-08-01 the roofing RAG pilot burned ~2 hours and a
dozen production deploys on failures that were all detectable locally in
seconds. This script checks every one of them. If it prints ALL CHECKS
PASSED, the only remaining step is setting RAG_ENABLED=true on that one
Render service.

It is deliberately dependency-free beyond what the app already needs, and
it never touches the network or any live service — safe to run any time.
"""
import os
import sys

OK = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"

HERE = os.path.dirname(os.path.abspath(__file__))
failures = []
warnings = []


def check(passed: bool, label: str, detail: str = "", fatal: bool = True):
    if passed:
        print(f"{OK} {label}")
    elif fatal:
        print(f"{FAIL} {label}")
        if detail:
            print(f"       {detail}")
        failures.append(label)
    else:
        print(f"{WARN} {label}")
        if detail:
            print(f"       {detail}")
        warnings.append(label)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        print("ERROR: pass exactly one client_slug, e.g. 'hvac' or 'pm'.")
        sys.exit(2)
    slug = sys.argv[1].strip().lower()
    print(f"\nRAG preflight for client_slug: '{slug}'\n" + "-" * 52)

    # 1. chromadb must not be back. It hangs forever on Render's instances
    #    (two live faulthandler stack dumps, 2026-08-01). If someone
    #    reintroduces it, every RAG request on every service dies silently.
    req_path = os.path.join(HERE, "requirements.txt")
    try:
        with open(req_path, "rb") as f:
            raw = f.read()
        # This file is UTF-16 LE with a BOM — pre-existing quirk, not a bug.
        reqs = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8")
    except Exception as e:
        reqs = ""
        check(False, "requirements.txt readable", str(e))
    check(
        "chromadb" not in reqs.lower(),
        "chromadb absent from requirements.txt",
        "chromadb hangs forever on Render (never returns, no error). It was "
        "removed 2026-08-01 and must stay removed. See RAG-IMPLEMENTATION.md.",
    )

    # 2. The slug must match what app_memory.py derives from DATA_FILE, or
    #    the app looks for a document that was never ingested and RAG
    #    silently does nothing.
    expected_data_file = f"{slug}_data.json"
    data_file_exists = os.path.exists(os.path.join(HERE, expected_data_file))
    if not data_file_exists:
        # Case-insensitive retry: the real file is HVAC_data.json, and
        # app_memory.py lowercases the slug, so this is expected and fine.
        matches = [
            fn for fn in os.listdir(HERE)
            if fn.lower() == expected_data_file.lower()
        ]
        data_file_exists = bool(matches)
        if matches:
            expected_data_file = matches[0]
    check(
        data_file_exists,
        f"DATA_FILE exists for this slug ({expected_data_file})",
        f"app_memory.py derives the slug from DATA_FILE by stripping "
        f"'_data.json' and lowercasing. For slug '{slug}' the Render env var "
        f"DATA_FILE must be '{expected_data_file}'.",
    )

    # 3. The seed doc must exist and be checked in, or auto-ingest-at-boot
    #    finds nothing. Render's disk is ephemeral: anything ingested by
    #    hand vanishes on the next deploy, so the repo copy is the only
    #    thing that survives.
    seed_path = os.path.join(HERE, "rag_seed_docs", f"{slug}.txt")
    seed_exists = os.path.exists(seed_path)
    check(
        seed_exists,
        f"seed doc exists (rag_seed_docs/{slug}.txt)",
        "Create it. Render's filesystem is ephemeral, so the checked-in seed "
        "doc is what gets re-ingested on every boot.",
    )

    if seed_exists:
        with open(seed_path, "r", encoding="utf-8") as f:
            seed_text = f.read()
        check(
            len(seed_text.strip()) > 200,
            "seed doc has real content",
            f"only {len(seed_text.strip())} chars — probably a stub.",
        )
        # Placeholder content reaching a real prospect is a credibility
        # problem, not a technical one. Warn loudly, don't block.
        looks_demo = any(
            marker in seed_text.upper()
            for marker in ("DEMO DOCUMENT", "PLACEHOLDER", "FICTIONAL", "EXAMPLE ONLY")
        )
        check(
            not looks_demo,
            "seed doc is real client content (not a labeled demo)",
            "This doc is labeled as a demo/placeholder. Fine for a portfolio "
            "demo; swap it for the client's real document before this is "
            "presented as their live bot. A RAG bot repeats whatever is in "
            "the source, wrong prices and all.",
            fatal=False,
        )

    # 4. The real end-to-end test: ingest and retrieve for this exact slug,
    #    from a worker thread. Threading is how the chromadb hang surfaced,
    #    so exercising a non-main thread is the point, not incidental.
    if seed_exists and not failures:
        try:
            import threading
            import time
            import rag_store

            t0 = time.monotonic()
            n = rag_store.ingest_document(slug, seed_text, source_name=seed_path)
            check(n > 0, f"ingest produced chunks ({n} chunks, {time.monotonic()-t0:.1f}s)")

            result = {}

            def worker():
                try:
                    t = time.monotonic()
                    result["chunks"] = rag_store.retrieve(slug, "what are your prices and fees", k=3)
                    result["elapsed"] = time.monotonic() - t
                except Exception as exc:
                    result["error"] = repr(exc)

            th = threading.Thread(target=worker)
            th.start()
            th.join(timeout=30)

            if th.is_alive():
                check(
                    False,
                    "retrieve() returns from a worker thread",
                    "HUNG past 30s. This is the exact chromadb failure mode. "
                    "Do NOT enable RAG on any service until this passes.",
                )
            elif "error" in result:
                check(False, "retrieve() returns from a worker thread", result["error"])
            else:
                check(True, f"retrieve() from worker thread ({result['elapsed']:.2f}s)")
                check(
                    len(result.get("chunks", [])) > 0,
                    "retrieve() returned actual chunks",
                )
                check(
                    result.get("elapsed", 99) < 5,
                    "retrieve() is fast (<5s)",
                    f"took {result.get('elapsed')}s — slow enough to risk a "
                    f"gunicorn timeout under load.",
                    fatal=False,
                )
        except Exception as e:
            check(False, "end-to-end ingest + retrieve", repr(e))

    print("-" * 52)
    if failures:
        print(f"\n{len(failures)} BLOCKING FAILURE(S). Do not enable RAG for '{slug}' yet:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("\nALL CHECKS PASSED.")
    if warnings:
        print(f"({len(warnings)} warning(s) above — read them, none are blocking.)")
    print(
        f"\nRemaining manual steps for '{slug}', both in the Render dashboard:\n"
        f"  1. Settings -> Start Command must be:\n"
        f"       gunicorn app_memory:app --timeout 120\n"
        f"     Render IGNORES the Procfile whenever this field is non-empty,\n"
        f"     so a stale value here silently overrides it. Check it, don't assume.\n"
        f"  2. Environment -> add  RAG_ENABLED = true  (that ONE service only).\n"
        f"\nThen confirm live, don't assume:\n"
        f"  - Logs should show: 'RAG auto-ingested N chunks for {slug}'\n"
        f"  - Ask the live bot something answerable ONLY by the seed doc and\n"
        f"    check it quotes the real figure. 'It didn't crash' is not a pass.\n"
    )


if __name__ == "__main__":
    main()
