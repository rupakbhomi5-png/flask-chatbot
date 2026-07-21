import os
import sys
import json
import asyncio
import threading
import urllib.request
import anthropic
from flask import Flask, request, jsonify, render_template, Response, stream_with_context, redirect
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client, get_default_environment

load_dotenv(override=True)
assert os.environ.get("ANTHROPIC_API_KEY"), "Missing ANTHROPIC_API_KEY — set it before starting"

app = Flask(__name__)

@app.before_request
def redirect_to_https():
    if request.headers.get("X-Forwarded-Proto") == "http":
        url = request.url.replace("http://", "https://", 1)
        # 308, not 301: 301 permits clients to convert POST→GET, which would
        # silently break /chat for any visitor arriving over http.
        return redirect(url, code=308)

@app.after_request
def set_security_headers(response):
    # Templates use inline <script>/<style> (no external JS/CSS files) and
    # one external favicon from fav.farm — CSP below matches that as-is.
    # Revisit if templates move to external assets or nonces.
    #
    # EMBED WIDGET EXCEPTION: "/" with ?embed=1 is the iframe the /embed.js
    # loader creates so a client can embed the chatbot on their own site.
    # That specific response needs to be frameable from any third-party
    # domain (the client's site is never known in advance) — every other
    # response keeps the locked-down default. Scoped narrowly on purpose,
    # per CSP FAILSAFE's "only per specific domain/feature when built" rule.
    is_embed_iframe = request.path == "/" and request.args.get("embed") == "1"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    if not is_embed_iframe:
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    frame_ancestors = "frame-ancestors *; " if is_embed_iframe else "frame-ancestors 'none'; "
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' https://fav.farm data:; "
        "connect-src 'self'; "
        f"{frame_ancestors}"
        "base-uri 'self'; "
        "form-action 'self'"
    )
    return response

MAX_HISTORY_MESSAGES = 6
MODEL_NAME = "claude-haiku-4-5-20251001"
MAX_TOOL_ITERATIONS = 3

# ── Self-test IP allowlist ─────────────────────────────────────────────────────
# Set MY_IPS as a comma-separated env var on Render (e.g. "1.2.3.4,5.6.7.8").
# Env var, not hardcoded: no redeploy when your IP changes, and no shipping
# a placeholder string that silently never matches. Best-effort only.
MY_KNOWN_IPS = [
    ip.strip() for ip in os.environ.get("MY_IPS", "").split(",") if ip.strip()
]

_redis_url = os.environ.get("REDIS_URL")
if not _redis_url:
    print("⚠ REDIS_URL not set — rate limiter uses memory (single-worker only, resets on restart)")

def get_real_ip():
    """Render's proxy makes request.remote_addr always 127.0.0.1 — read the
    real visitor IP from X-Forwarded-For instead, or the limiter treats every
    visitor as the same client.

    SECURITY: use the LAST entry, not the first. The first entries are
    client-supplied and trivially spoofable (an attacker could mint a fresh
    fake IP per request and bypass rate limiting entirely, running up the
    API bill). The last entry is the one appended by Render's own proxy and
    is the only one you can trust. If you ever put another proxy (e.g.
    Cloudflare) in front of Render, this must change again."""
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[-1].strip()
    return request.remote_addr or "127.0.0.1"

limiter = Limiter(
    get_real_ip,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=_redis_url or "memory://",
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── Stateless conversation history ────────────────────────────────────────────
# The browser holds the conversation: it sends its text-only history with every
# /chat request and receives the updated history back in the SSE "done" event.
# No server-side state — survives Render restarts and works across gunicorn
# workers. (Templates already speak this protocol: {message, history} in,
# payload.history out.)
def sanitize_history(raw) -> list:
    """Validate client-supplied history: text-only user/assistant turns.
    Anything malformed is dropped — the client is not trusted."""
    if not isinstance(raw, list):
        return []
    clean = []
    for item in raw[-(MAX_HISTORY_MESSAGES * 2):]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        clean.append({"role": role, "content": content[:2000]})
    clean = clean[-MAX_HISTORY_MESSAGES:]
    while clean and clean[0]["role"] != "user":
        clean.pop(0)  # Claude API requires the first message to be from the user
    return clean

# ── Data & system prompt — loaded ONCE at startup ─────────────────────────────
def load_data() -> dict:
    # Default MUST match mcp_server.py exactly. They previously defaulted to
    # different files (pm_data vs business_data) — deploy without DATA_FILE
    # set and the prompt describes one business while MCP tools serve
    # another's inventory. Silent, no error, customer-visible.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.environ.get("DATA_FILE", "business_data.json")
    if not os.environ.get("DATA_FILE"):
        print(f"⚠ DATA_FILE not set — defaulting to {data_file}. Set it explicitly in production.")
    with open(os.path.join(base_dir, data_file), "r") as f:
        data = json.load(f)
    print(f"✓ Data loaded: {data_file} → {data.get('store_name', '?')}")
    return data

def build_system_prompt(data: dict) -> str:
    currency = data.get("currency", "$")
    sections = []
    if "products" in data:
        product_list = "\n".join(
            f"- {p['name']}: {currency} {p['price']:,}"
            for p in data["products"]
        )
        sections.append(f"PRODUCTS:\n{product_list}")
    if "services" in data:
        service_list = "\n".join(
            f"- {s['name']}: {currency} {s['price']} ({s['duration']})"
            for s in data["services"]
        )
        sections.append(f"SERVICES:\n{service_list}")
    if "faq" in data:
        faq_list = "\n".join(f"Q: {item['q']}\nA: {item['a']}" for item in data["faq"])
        sections.append(f"FAQ:\n{faq_list}")

    if "custom_sections" in data:
        for sec in data["custom_sections"]:
            item_list = "\n".join(f"- {i}" for i in sec["items"])
            sections.append(f"{sec['title']}: \n{item_list}")

    business_info = "\n\n".join(sections)
    return_policy = f"\n- Return Policy: {data['return_policy']}" if "return_policy" in data else ""
    language_instruction = f"\nLANGUAGE: {data['language']}" if "language" in data else ""
    lead_rules = (
        f"\n\nLEAD CAPTURE RULES:\n{data['lead_qualification']}"
        if "lead_qualification" in data else ""
    )
    return f"""You are {data['bot_name']}, a customer service agent for {data['store_name']}, a {data['business_type']} in {data['location']}.

BUSINESS INFO:
- Hours: {data['hours']}
- Location: {data['location']}
- Contact: {data['contact']}{return_policy}

{business_info}

RULES: Never invent products, services, or prices not listed above. Never promise stock or availability. For anything not listed, direct to {data['contact']}. Never volunteer prices, service options, or package lists unless the customer directly asks "how much" or "what do you offer" — unsolicited pitching kills trust.

LEAD CAPTURE RULES:
- Your only goal is to get a name and a way to reach them (phone or email). That is it.
- Do NOT require them to explain or diagnose their problem. "Something's broken" or "need help" is enough — capture the lead.
- Parse name and contact from whatever they write naturally. If they write "ram 9876543210", that is name=Ram, contact=9876543210. Do not ask them to reformat. Do not ask them to say "my name is" or "my number is".
- A 10-digit number is a phone. Anything with @ is an email. A word that is not a number is a name.
- The moment you have a name AND a contact, call capture_lead. No confirmation, no follow-up questions.
- After capturing, tell them: "Got it — someone will call you shortly." Nothing else.
- service_interest: summarize what they mentioned in a few words. If they said nothing specific, use "General inquiry".

Keep replies to 1-2 sentences. Be warm, not salesy.{language_instruction}{lead_rules}"""

DATA = load_data()
SYSTEM_PROMPT = build_system_prompt(DATA)

# ── Lead capture tool ──────────────────────────────────────────────────────────
LEAD_CAPTURE_TOOL = {
    "name": "capture_lead",
    "description": "Call when visitor gives both a name AND contact (phone or email). Notifies owner immediately.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "contact": {"type": "string"},
            "service_interest": {"type": "string"},
        },
        "required": ["name", "contact"],
    },
}

def send_lead_email(name: str, contact: str, service_interest: str = "", visitor_ip: str = "") -> str:
    """Notify the business owner via SendGrid when a lead is captured."""
    api_key = os.environ.get("SENDGRID_API_KEY")
    owner_email = os.environ.get("OWNER_EMAIL")
    from_email = os.environ.get("FROM_EMAIL", "noreply@example.com")
    if not all([api_key, owner_email]):
        print(f"⚠ LEAD (SendGrid not configured): {name} | {contact} | {service_interest} | IP: {visitor_ip}")
        return f"Lead captured: {name} ({contact})"
    store_name = DATA.get("store_name", "Your Business")
    tag = "[SELF-TEST] " if visitor_ip in MY_KNOWN_IPS else "[REAL LEAD] "
    payload = json.dumps({
        "personalizations": [{"to": [{"email": owner_email}]}],
        "from": {"email": from_email},
        "subject": f"{tag}New inquiry from {store_name}",
        "content": [{"type": "text/plain", "value": (
            f"New lead from your website chatbot.\n\n"
            f"Name:      {name}\n"
            f"Contact:   {contact}\n"
            f"Interested in: {service_interest or 'not specified'}\n"
            f"IP:        {visitor_ip or 'unknown'}\n\n"
            f"Reply within the hour — they are still on your site."
        )}],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.sendgrid.com/v3/mail/send",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            print(f"✓ Lead email sent: {name} | {contact} (status {resp.status})")
        return f"Lead captured and owner notified: {name} ({contact})"
    except urllib.error.HTTPError as e:
        print(f"⚠ Lead email failed ({e}): {name} | {contact} | {e.read().decode('utf-8')}")
        return f"Lead captured: {name} ({contact})"
    except Exception as e:
        print(f"⚠ Lead email failed ({e}): {name} | {contact}")
        return f"Lead captured: {name} ({contact})"

# ── MCP — persistent worker keeps one subprocess alive ────────────────────────
def _mcp_params() -> StdioServerParameters:
    # Two production traps fixed here:
    # 1. The MCP SDK does NOT inherit the parent environment — it builds a
    #    minimal safe env (PATH, HOME, ...) unless you pass env explicitly.
    #    Without this, DATA_FILE never reaches mcp_server.py and the tools
    #    serve the DEFAULT business's data no matter what the app loaded.
    # 2. sys.executable, not "python" — some images only ship python3;
    #    "python" missing on PATH means MCP silently fails at every boot.
    return StdioServerParameters(
        command=sys.executable,
        args=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")],
        env={
            **get_default_environment(),
            "DATA_FILE": os.environ.get("DATA_FILE", "business_data.json"),
        },
    )

_bg_loop = asyncio.new_event_loop()
threading.Thread(target=_bg_loop.run_forever, daemon=True, name="mcp-loop").start()
_mcp_call_queue = None
_mcp_ready = threading.Event()

async def _mcp_worker():
    global _mcp_call_queue
    _mcp_call_queue = asyncio.Queue()
    _mcp_ready.set()
    while True:
        try:
            async with stdio_client(_mcp_params()) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    print("✓ MCP worker: session ready")
                    while True:
                        item = await _mcp_call_queue.get()
                        if item is None:
                            return
                        name, args, fut = item
                        try:
                            result = await session.call_tool(name, args)
                            fut.set_result(result.content[0].text)
                        except Exception as exc:
                            fut.set_exception(exc)
                        _mcp_call_queue.task_done()
        except Exception as e:
            print(f"⚠ MCP worker crashed ({e}), restarting in 2s")
            while not _mcp_call_queue.empty():
                try:
                    item = _mcp_call_queue.get_nowait()
                    if item is not None:
                        _, _, fut = item
                        if not fut.done():
                            fut.set_exception(RuntimeError(f"MCP worker restarting: {e}"))
                        _mcp_call_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            await asyncio.sleep(2)

async def _fetch_mcp_tools_once() -> list[dict]:
    async with stdio_client(_mcp_params()) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            result = await s.list_tools()
            return [
                {"name": t.name, "description": t.description or "", "input_schema": t.inputSchema}
                for t in result.tools
            ]

def _call_mcp_tool_sync(name: str, args: dict) -> str:
    """Submit a tool call to the persistent MCP worker and block until done."""
    _mcp_ready.wait(timeout=10)
    loop = _bg_loop
    async def _dispatch():
        f = loop.create_future()
        await _mcp_call_queue.put((name, args, f))
        return await f
    return asyncio.run_coroutine_threadsafe(_dispatch(), loop).result(timeout=30)

# ── Tool startup ───────────────────────────────────────────────────────────────
_mcp_available = False
# Default OFF. Every product/service/FAQ is already baked into the system
# prompt at startup, so for these catalog sizes MCP adds a subprocess, a
# background event loop, and blocking 30s bridges inside request handlers —
# for answers the prompt already gives. Turn it on (MCP_ENABLED=true) only
# when a catalog is too large to fit in the prompt.
_MCP_ENABLED = os.environ.get("MCP_ENABLED", "false").lower() == "true"

if _MCP_ENABLED:
    try:
        _mcp_tools = asyncio.run_coroutine_threadsafe(
            _fetch_mcp_tools_once(), _bg_loop
        ).result(timeout=15)
        TOOLS = _mcp_tools + [LEAD_CAPTURE_TOOL]
        asyncio.run_coroutine_threadsafe(_mcp_worker(), _bg_loop)
        _mcp_ready.wait(timeout=5)
        _mcp_available = True
        print(f"✓ MCP: loaded {[t['name'] for t in TOOLS]}")
    except Exception as _e:
        TOOLS = [LEAD_CAPTURE_TOOL]
        _mcp_available = False
        print(f"⚠ MCP unavailable ({_e})")
        print("  → Products, services, and FAQ are covered by the system prompt.")
        print("  → Only lead capture requires a tool — that is still active.")
else:
    TOOLS = [LEAD_CAPTURE_TOOL]
    print("⚠ MCP_ENABLED=false — products/services/FAQ answered from system prompt, lead capture active.")

def run_tool(tool_name: str, tool_input: dict, visitor_ip: str = "") -> str:
    if tool_name == "capture_lead":
        return send_lead_email(
            name=tool_input.get("name", ""),
            contact=tool_input.get("contact", ""),
            service_interest=tool_input.get("service_interest", ""),
            visitor_ip=visitor_ip,
        )
    if _mcp_available:
        try:
            return _call_mcp_tool_sync(tool_name, tool_input)
        except Exception as e:
            # MCP worker crashed or timed out mid-conversation. Return a graceful
            # string so Claude answers from the system prompt instead of the
            # exception killing the SSE stream.
            print(f"⚠ MCP tool call failed ({tool_name}): {e}")
            return (
                f"Tool '{tool_name}' is temporarily unavailable. "
                "Answer from the business information you already have."
            )
    return f"Tool unavailable: {tool_name}"

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    template = os.environ.get("TEMPLATE_FILE", "index.html")
    return render_template(template, bot_name=DATA["bot_name"], store_name=DATA["store_name"])

# EMBEDDABLE WIDGET LOADER — a client pastes <script src=".../embed.js"></script>
# on their own site. This creates a small floating iframe pointing back at "/"
# with ?embed=1 (which the templates render as a bubble-then-panel widget
# instead of a full page) and resizes that iframe via postMessage. No CORS
# needed: the iframe's own /chat and /reset calls stay same-origin with it —
# only the outer <script> tag and the postMessage bridge cross into the
# client's page at all.
_EMBED_LOADER_JS = """(function () {
  var currentScript = document.currentScript;
  var base = currentScript.src.replace(/\\/embed\\.js.*$/, "");

  var CLOSED_CSS =
    "position:fixed;bottom:20px;right:20px;width:64px;height:64px;" +
    "border:none;border-radius:50%;box-shadow:0 4px 20px rgba(0,0,0,.25);" +
    "z-index:2147483000;background:transparent;" +
    "transition:width .2s ease,height .2s ease,border-radius .2s ease;";

  var iframe = document.createElement("iframe");
  iframe.src = base + "/?embed=1";
  iframe.title = "Chat";
  iframe.setAttribute("allow", "microphone");
  iframe.style.cssText = CLOSED_CSS;
  document.body.appendChild(iframe);

  var isOpen = false;

  window.addEventListener("message", function (e) {
    if (e.source !== iframe.contentWindow) return;
    if (!e.data || e.data.type !== "rupakco-widget-resize") return;
    isOpen = e.data.state === "open";
    var mobile = window.matchMedia("(max-width: 480px)").matches;
    if (isOpen) {
      iframe.style.cssText = mobile
        ? "position:fixed;bottom:0;right:0;width:100vw;height:100dvh;border:none;border-radius:0;box-shadow:none;z-index:2147483000;background:transparent;transition:width .2s ease,height .2s ease,border-radius .2s ease;"
        : "position:fixed;bottom:20px;right:20px;width:380px;height:min(640px,80vh);border:none;border-radius:16px;box-shadow:0 8px 32px rgba(0,0,0,.3);z-index:2147483000;background:transparent;transition:width .2s ease,height .2s ease,border-radius .2s ease;";
    } else {
      iframe.style.cssText = CLOSED_CSS;
    }
  });

  // Clicking anywhere on the host page outside the iframe collapses the
  // widget back to the bubble. A click landing inside the iframe never
  // reaches this listener (it's a separate document), so any mousedown
  // seen here is, by definition, an outside click — no coordinate math needed.
  document.addEventListener("mousedown", function () {
    if (!isOpen) return;
    isOpen = false;
    iframe.style.cssText = CLOSED_CSS;
    iframe.contentWindow.postMessage({ type: "rupakco-widget-collapse" }, "*");
  });
})();"""

@app.route("/embed.js")
def embed_js():
    return Response(_EMBED_LOADER_JS, mimetype="application/javascript")

@app.route("/chat", methods=["POST"])
@limiter.limit("10 per minute")
def chat():
    # get_json(silent=True) instead of request.json: wrong/missing
    # Content-Type otherwise raises 415 and Flask returns an HTML error page
    # the chat widget can't parse.
    body = request.get_json(silent=True)
    if not body or not body.get("message"):
        return jsonify({"error": "Message cannot be empty."}), 400
    user_message = body["message"].strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400
    if len(user_message) > 500:
        return jsonify({"error": "Message too long. Please keep it under 500 characters."}), 400

    # Same trusted-IP logic as the rate limiter (last XFF entry — the one
    # Render's proxy appended). Keeps [SELF-TEST] tagging unspoofable too.
    visitor_ip = get_real_ip()

    history = sanitize_history(body.get("history"))
    history.append({"role": "user", "content": user_message})
    client_history = list(history)  # text-only view returned to the browser

    def generate():
        nonlocal history
        try:
            used_tools = False

            for _ in range(MAX_TOOL_ITERATIONS):
                response = client.messages.create(
                    model=MODEL_NAME,
                    # 500, not 300: the PM config packs unit/issue/severity/
                    # entry-permission into service_interest — 300 tokens can
                    # truncate the tool_use JSON mid-argument (stop_reason
                    # "max_tokens" → malformed capture).
                    max_tokens=500,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=history,
                )
                if response.stop_reason != "tool_use":
                    break
                used_tools = True
                asst_content = []
                for block in response.content:
                    if block.type == "tool_use":
                        asst_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })
                    elif block.type == "text":
                        asst_content.append({"type": "text", "text": block.text})
                history.append({"role": "assistant", "content": asst_content})
                tool_results = []
                lead_captured = False
                for block in response.content:
                    if block.type == "tool_use":
                        result = run_tool(block.name, block.input, visitor_ip)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                        if block.name == "capture_lead":
                            lead_captured = True
                history.append({"role": "user", "content": tool_results})
                if lead_captured:
                    break

            if used_tools:
                # History now contains tool_use/tool_result blocks — the API
                # rejects such requests unless `tools` is defined. Pass tools
                # but forbid further calls so this is guaranteed a text turn.
                full_reply: list[str] = []
                with client.messages.stream(
                    model=MODEL_NAME,
                    max_tokens=300,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    tool_choice={"type": "none"},
                    messages=history,
                ) as stream:
                    for text in stream.text_stream:
                        full_reply.append(text)
                        yield f"data: {json.dumps({'token': text})}\n\n"
                final_text = "".join(full_reply)
            else:
                final_text = "".join(b.text for b in response.content if hasattr(b, "text"))
                # Word chunks, not per-character events — a 2-sentence reply
                # was previously ~150 SSE events for zero UX gain.
                words = final_text.split(" ")
                for i, word in enumerate(words):
                    token = word if i == len(words) - 1 else word + " "
                    yield f"data: {json.dumps({'token': token})}\n\n"

            client_history.append({"role": "assistant", "content": final_text})
            trimmed = client_history[-MAX_HISTORY_MESSAGES:]
            while trimmed and trimmed[0]["role"] != "user":
                trimmed = trimmed[1:]
            yield f"data: {json.dumps({'done': True, 'history': trimmed})}\n\n"

        except anthropic.AuthenticationError:
            yield f"data: {json.dumps({'error': 'Invalid API key. Check your configuration.'})}\n\n"
        except anthropic.RateLimitError:
            yield f"data: {json.dumps({'error': 'Rate limit reached. Please wait and try again.'})}\n\n"
        except anthropic.APIConnectionError:
            yield f"data: {json.dumps({'error': 'Could not connect to AI service. Check your internet.'})}\n\n"
        except anthropic.APIStatusError as e:
            yield f"data: {json.dumps({'error': f'API error: {e.status_code}'})}\n\n"
        except Exception as e:
            print(f"Unhandled error: {e}")
            yield f"data: {json.dumps({'error': 'Something went wrong. Please try again.'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.route("/reset", methods=["POST"])
def reset():
    # Stateless server — the browser clears its own history; nothing stored here.
    return jsonify({"message": "Conversation reset."})

# --- VOICE LEAD WEBHOOK (Projects A/B) — dormant unless VOICE_ENABLED=true ---
@app.route("/voice_lead", methods=["POST"])
def voice_lead():
    if os.environ.get("VOICE_ENABLED", "false").lower() != "true":
        return jsonify({"status": "disabled"}), 404
    secret = os.environ.get("VOICE_WEBHOOK_SECRET", "")
    if not secret or request.headers.get("X-Webhook-Secret") != secret:
        return jsonify({"status": "unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    contact = (data.get("contact") or "").strip()
    service_interest = (data.get("service_interest") or "").strip()
    # 0eb5003 guard, voice edition: no real callback contact = no lead email
    if not name or not contact or contact in ("unknown", "0000000000"):
        print(f"⚠ VOICE LEAD rejected by guard: {data}")
        return jsonify({"status": "rejected"}), 200
    send_lead_email(name, contact, service_interest)
    return jsonify({"status": "ok"}), 200

@app.errorhandler(429)
def rate_limit_handler(e):
    return jsonify({"error": "Too many requests. Please slow down."}), 429

if __name__ == "__main__":
    app.run(debug=True)