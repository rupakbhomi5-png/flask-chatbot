import os
import json
import asyncio
import urllib.request
import anthropic
from flask import Flask, request, jsonify, render_template, Response, stream_with_context, redirect
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv(override=True)

app = Flask(__name__)

@app.before_request
def redirect_to_https():
    if request.headers.get("X-Forwarded-Proto") == "http":
        url = request.url.replace("http://", "https://", 1)
        return redirect(url, code=301)

MAX_HISTORY_MESSAGES = 6
MODEL_NAME = "claude-haiku-4-5-20251001"
MAX_TOOL_ITERATIONS = 3

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

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
        # Full Q&A pairs — bot answers FAQ directly without a tool call
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

RULES: Never invent products, services, or prices not listed above. Never promise stock or availability. For anything not listed, direct to {data['contact']}.

LEAD CAPTURE: The moment a visitor gives both a name AND a contact (phone or email), call capture_lead immediately — no confirmation, no follow-up questions. Then tell them someone will be in touch.

3 sentences max. Be friendly but precise.{language_instruction}{lead_rules}"""

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

# ── MCP plumbing ───────────────────────────────────────────────────────────────
def _mcp_params() -> StdioServerParameters:
    return StdioServerParameters(
        command="python",
        args=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")],
    )

async def _fetch_mcp_tools() -> list[dict]:
    async with stdio_client(_mcp_params()) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            result = await s.list_tools()
            return [
                {"name": t.name, "description": t.description or "", "input_schema": t.inputSchema}
                for t in result.tools
            ]

async def _call_mcp_tool(name: str, args: dict) -> str:
    async with stdio_client(_mcp_params()) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            result = await s.call_tool(name, args)
            return result.content[0].text

# ── Tool startup ───────────────────────────────────────────────────────────────
_mcp_available = False
_MCP_ENABLED = os.environ.get("MCP_ENABLED", "true").lower() == "true"

if _MCP_ENABLED:
    try:
        TOOLS = asyncio.run(_fetch_mcp_tools())
        TOOLS.append(LEAD_CAPTURE_TOOL)
        _mcp_available = True
        print(f"✓ MCP: loaded {[t['name'] for t in TOOLS]}")
    except Exception as _e:
        # MCP unavailable: product/service/FAQ already in system prompt.
        # Only lead capture needs a tool.
        TOOLS = [LEAD_CAPTURE_TOOL]
        _mcp_available = False
        print(f"⚠ MCP unavailable ({_e}) — lead capture only")
else:
    TOOLS = [LEAD_CAPTURE_TOOL]
    _mcp_available = False
    print("⚠ MCP_ENABLED=false — lead capture only")

def run_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "capture_lead":
        return send_lead_email(
            name=tool_input.get("name", ""),
            contact=tool_input.get("contact", ""),
            service_interest=tool_input.get("service_interest", ""),
        )
    if _mcp_available:
        return asyncio.run(_call_mcp_tool(tool_name, tool_input))
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
    if len(user_message) > 500:
        return jsonify({"error": "Message too long. Please keep it under 500 characters."}), 400
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    history = body.get("history", [])
    history.append({"role": "user", "content": user_message})
    history = history[-MAX_HISTORY_MESSAGES:]

    def generate():
        nonlocal history
        try:
            final_response = None

            for _ in range(MAX_TOOL_ITERATIONS):
                response = client.messages.create(
                    model=MODEL_NAME,
                    max_tokens=300,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=history,
                )

                if response.stop_reason != "tool_use":
                    final_response = response
                    break

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

            if final_response is None:
                final_response = client.messages.create(
                    model=MODEL_NAME,
                    max_tokens=300,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=history,
                )

            # Stream text locally — zero extra API cost
            final_text = "".join(b.text for b in final_response.content if hasattr(b, "text"))
            for char in final_text:
                yield f"data: {json.dumps({'token': char})}\n\n"

            history.append({"role": "assistant", "content": final_text})
            trimmed = history[-MAX_HISTORY_MESSAGES:]
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
    return jsonify({"message": "Conversation reset."})

@app.errorhandler(429)
def rate_limit_handler(e):
    return jsonify({"error": "Too many requests. Please slow down."}), 429

if __name__ == "__main__":
    app.run(debug=True)