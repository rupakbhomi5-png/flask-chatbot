import os
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
from mcp.client.stdio import stdio_client

load_dotenv(override=True)
assert os.environ.get("ANTHROPIC_API_KEY"), "Missing ANTHROPIC_API_KEY — set it before starting"

app = Flask(__name__)

@app.before_request
def redirect_to_https():
    if request.headers.get("X-Forwarded-Proto") == "http":
        url = request.url.replace("http://", "https://", 1)
        return redirect(url, code=301)

@app.after_request
def set_security_headers(response):
    # Templates use inline <script>/<style> (no external JS/CSS files) and
    # one external favicon from fav.farm — CSP below matches that as-is.
    # Revisit if templates move to external assets or nonces.
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' https://fav.farm data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )
    return response

MAX_HISTORY_MESSAGES = 6
MODEL_NAME = "claude-haiku-4-5-20251001"
MAX_TOOL_ITERATIONS = 3

_redis_url = os.environ.get("REDIS_URL")
if not _redis_url:
    print("⚠ REDIS_URL not set — rate limiter uses memory (single-worker only, resets on restart)")

limiter = Limiter(
    get_remote_address,
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.environ.get("DATA_FILE", "business_data.json")
    with open(os.path.join(base_dir, data_file), "r") as f:
        return json.load(f)

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

def send_lead_email(name: str, contact: str, service_interest: str = "") -> str:
    """Notify the business owner via SendGrid when a lead is captured."""
    api_key = os.environ.get("SENDGRID_API_KEY")
    owner_email = os.environ.get("OWNER_EMAIL")
    from_email = os.environ.get("FROM_EMAIL", "noreply@example.com")
    if not all([api_key, owner_email]):
        print(f"⚠ LEAD (SendGrid not configured): {name} | {contact} | {service_interest}")
        return f"Lead captured: {name} ({contact})"
    store_name = DATA.get("store_name", "Your Business")
    payload = json.dumps({
        "personalizations": [{"to": [{"email": owner_email}]}],
        "from": {"email": from_email},
        "subject": f"New inquiry from {store_name}",
        "content": [{"type": "text/plain", "value": (
            f"New lead from your website chatbot.\n\n"
            f"Name:      {name}\n"
            f"Contact:   {contact}\n"
            f"Interested in: {service_interest or 'not specified'}\n\n"
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
    return StdioServerParameters(
        command="python",
        args=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")],
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
_MCP_ENABLED = os.environ.get("MCP_ENABLED", "true").lower() == "true"

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

def run_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "capture_lead":
        return send_lead_email(
            name=tool_input.get("name", ""),
            contact=tool_input.get("contact", ""),
            service_interest=tool_input.get("service_interest", ""),
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

@app.route("/chat", methods=["POST"])
@limiter.limit("10 per minute")
def chat():
    body = request.json
    if not body or not body.get("message"):
        return jsonify({"error": "Message cannot be empty."}), 400
    user_message = body["message"].strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400
    if len(user_message) > 500:
        return jsonify({"error": "Message too long. Please keep it under 500 characters."}), 400

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
                    max_tokens=300,
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
                        result = run_tool(block.name, block.input)
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
                full_reply: list[str] = []
                with client.messages.stream(
                    model=MODEL_NAME,
                    max_tokens=300,
                    system=SYSTEM_PROMPT,
                    messages=history,
                ) as stream:
                    for text in stream.text_stream:
                        full_reply.append(text)
                        yield f"data: {json.dumps({'token': text})}\n\n"
                final_text = "".join(full_reply)
            else:
                final_text = "".join(b.text for b in response.content if hasattr(b, "text"))
                for char in final_text:
                    yield f"data: {json.dumps({'token': char})}\n\n"

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

@app.errorhandler(429)
def rate_limit_handler(e):
    return jsonify({"error": "Too many requests. Please slow down."}), 429

if __name__ == "__main__":
    app.run(debug=True)
