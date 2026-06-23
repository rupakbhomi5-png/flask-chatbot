import os
import json
import uuid
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import anthropic
from flask import Flask, request, jsonify, render_template, session, Response, stream_with_context, redirect
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv(override=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")

@app.before_request
def redirect_to_https():
    if request.headers.get('X-Forwarded-Proto') == 'http':
        url = request.url.replace('http://', 'https://', 1)
        return redirect(url, code=301)


MAX_HISTORY_MESSAGES = 10  # last 5 user+assistant exchanges — bounds cookie size and token usage
MODEL_NAME = "claude-haiku-4-5-20251001"

# Rate Limiter setup
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

 
# ── In-memory conversation store ──────────────────────────────────────────────
# Cookie stores a UUID. History lives here, not in the cookie.
# This sidesteps the "cookie can't be set mid-stream" problem.
# Data is lost on dyno restart — acceptable for a portfolio demo.
conversation_store: dict[str, list] = {}


def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.environ.get("DATA_FILE", "business_data.json")
    with open(os.path.join(base_dir, data_file), "r") as f:
        return json.load(f)

def build_system_prompt():
    data = load_data()

    sections = []

    if "products" in data:
        product_list = "\n".join(
            f"- {p['name']}: NPR {p['price']:,}"
            for p in data["products"]
        )
        sections.append(f"PRODUCTS WE SELL:\n{product_list}")

    if "services" in data:
        service_list = "\n".join(
            f"- {s['name']}: ${s['price']} ({s['duration']})"
            for s in data["services"]
        )
        sections.append(f"SERVICES WE OFFER:\n{service_list}")

    if "faq" in data:
        faq_list = "\n".join(
            f"- {item['q']}"
            for item in data["faq"]
        )
        sections.append(f"COMMON QUESTIONS I CAN ANSWER:\n{faq_list}")

    business_info = "\n".join(sections)

    return_policy = ""
    if "return_policy" in data:
        return_policy = f"\n- Return Policy: {data['return_policy']}"
    language_instruction = f"\nLANGUAGE: {data['language']}" if "language" in data else ""
    return f"""You are {data['bot_name']}, a customer service agent for {data['store_name']}, \
a {data['business_type']} in {data['location']}.

BUSINESS INFORMATION:
- Hours: {data['hours']}
- Location: {data['location']}
- Contact: {data['contact']}

{business_info}

WHAT YOU MUST NEVER DO:
- Never invent products or services not listed above
- Never quote prices not listed above - say "please contact us at {data['contact']} for current pricing"
- Never promise stock or appointment availability
- For specialized items not on this list, always direct customer to call {data['contact']}

LEAD CAPTURE — THIS IS IMPORTANT:
When a visitor asks about pricing, availability, booking, scheduling, or shows any interest in a service or product, ask for their name and best contact number or email. Once you have BOTH their name and their contact, call the capture_lead tool immediately. Do not ask for anything else — name and contact is enough. After calling capture_lead, confirm to the visitor that someone will reach out shortly.

Keep response under 3 sentences. Be friendly but precise.{language_instruction}"""


SYSTEM_PROMPT = build_system_prompt()

# ── Lead capture tool definition (always available, handled locally) ──────────
LEAD_CAPTURE_TOOL = {
    "name": "capture_lead",
    "description": "Call this as soon as you have collected a visitor's name and contact information (phone or email). This immediately notifies the business owner so they can follow up while the lead is warm.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The visitor's name."
            },
            "contact": {
                "type": "string",
                "description": "The visitor's phone number or email address."
            },
            "service_interest": {
                "type": "string",
                "description": "What the visitor is interested in, in their own words."
            }
        },
        "required": ["name", "contact"]
    }
}


def send_lead_email(name: str, contact: str, service_interest: str = "") -> str:
    """Send an instant email notification to the business owner when a lead is captured."""
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    owner_email = os.environ.get("OWNER_EMAIL")

    if not all([smtp_host, smtp_user, smtp_pass, owner_email]):
        # SMTP not configured — log it, still return success to the bot
        print(f"⚠ LEAD (email not configured): {name} | {contact} | {service_interest}")
        return f"Lead captured: {name} ({contact})"

    data = load_data()
    store_name = data.get("store_name", "Your Business")

    subject = f"New lead: {name} — {store_name}"
    body = (
        f"New lead from your website chatbot.\n\n"
        f"Name:      {name}\n"
        f"Contact:   {contact}\n"
        f"Interested in: {service_interest or 'not specified'}\n\n"
        f"Reply within the hour — they are still on your site."
    )

    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = owner_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, owner_email, msg.as_string())
        print(f"✓ Lead email sent: {name} | {contact}")
        return f"Lead captured and owner notified: {name} ({contact})"
    except Exception as e:
        print(f"⚠ Lead email failed ({e}): {name} | {contact}")
        # Don't surface SMTP errors to the visitor — still confirm gracefully
        return f"Lead captured: {name} ({contact})"


def _mcp_params() -> StdioServerParameters:
    """Points to mcp_server.py sitting next to this file."""
    return StdioServerParameters(
        command="python",
        args=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")]
    )

async def _fetch_mcp_tools() -> list[dict]:
    """
    Connect to MCP server via stdio, list its tools, convert to Anthropic tool_definition format.
    Called once at startup."""
    async with stdio_client(_mcp_params()) as (read, write):
        async with ClientSession(read,write) as s:
            await s.initialize()
            result = await s.list_tools()
            return [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.inputSchema,
                }
                for t in result.tools
            ]
async def _call_mcp_tool(name: str, args: dict) -> str:
    """
    Open a fresh stdio connection to the MCP server, call one tool,
    return its text output. Once connection per call - fine for a demo """
    
    async with stdio_client(_mcp_params()) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            result = await s.call_tool(name, args)
            return result.content[0].text
        


# Fallback: if MCP server fails at startup, use a hardcoded definition
# so the app still boots. Tool execution also falls back to local function.
_FALLBACK_TOOLS = []
data_check = load_data()

if "products" in data_check:
    _FALLBACK_TOOLS.append({
        "name": "search_products",
        "description": "Search inventory by product category.",    
         "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Product category to search."}    
            },
            "required": ["category"]
        }
    })

if "services" in data_check:
    _FALLBACK_TOOLS.append({
        "name": "search_services",
        "description": "Search services by category.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Service category to search."}
            },
            "required": ["category"]
        }
    })

if "faq" in data_check:
    _FALLBACK_TOOLS.append({
        "name": "get_faq",
        "description": "Answer common customer questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Customer question."}
            },
            "required": ["question"]
        }
    })

_mcp_available = False
try:
    TOOLS = asyncio.run(_fetch_mcp_tools())
    _mcp_available = True
    print(f"✓ MCP: loaded {[t['name'] for t in TOOLS]}")
except Exception as _e:
    TOOLS = _FALLBACK_TOOLS
    _mcp_available = False
    print(f"⚠ MCP unavailable ({_e}), using fallback definitions")

# Always append capture_lead — it is handled locally, not via MCP
TOOLS.append(LEAD_CAPTURE_TOOL)
print(f"✓ capture_lead tool registered")


def _fallback_search_products(category: str) -> str:
    """Direct call - used only when MCP server is unavailable."""   
    data = load_data()
    matches = [
        p for p in data.get("products", [])
        if p.get("category") == category.lower()
    ]
    if not matches:
        return f"No products found in category '{category}'."
    lines = "\n".join(f"- {p['name']}: NPR {p['price']:,}" for p in matches)
    return f"Products in '{category}':\n{lines}"

def _fallback_search_services(category: str) -> str:
    data = load_data()
    matches = [s for s in data.get("services", []) if s.get("category") == category.lower()]
    if not matches:
        return f"No services found in category '{category}'."
    lines = "\n".join(f"- {s['name']}: ${s['price']} ({s['duration']})" for s in matches)
    return f"Services in '{category}':\n{lines}"

def _fallback_get_faq(question: str) -> str:
    data = load_data()
    for item in data.get("faq", []):
        if any(word in question.lower() for word in item["q"].lower().split()):
            return item["a"]
    return f"For that question please contact us at {data['contact']}."

def run_tool(tool_name: str, tool_input: dict) -> str:
    # capture_lead is always handled locally — never routed through MCP
    if tool_name == "capture_lead":
        return send_lead_email(
            name=tool_input.get("name", ""),
            contact=tool_input.get("contact", ""),
            service_interest=tool_input.get("service_interest", "")
        )
    if _mcp_available:
        return asyncio.run(_call_mcp_tool(tool_name, tool_input))
    if tool_name == "search_products":
        return _fallback_search_products(tool_input.get("category", ""))
    if tool_name == "search_services":
        return _fallback_search_services(tool_input.get("category", ""))
    if tool_name == "get_faq":
        return _fallback_get_faq(tool_input.get("question", ""))
    return f"Unknown tool: {tool_name}"

@app.route("/")
def index():
    data = load_data()
    template = os.environ.get("TEMPLATE_FILE", "index.html")
    return render_template(template, bot_name=data["bot_name"], store_name=data["store_name"])


@app.route("/chat", methods=["POST"])
@limiter.limit("10 per minute")
def chat():
    # Get or create a stable session UUID.
    # Cookie stores only the UUID; conversation lives in conversation_store.
    # This means the session cookie is set in the response HEADERS (before streaming
    # starts), and we can freely write to conversation_store from inside the generator.
    sid = session.get("sid")
    if not sid:
        sid = str(uuid.uuid4())
        session["sid"] = sid
    
    history = list(conversation_store.get(sid, []))
   
    data = request.json
    if not data or not data.get("message"):
        return jsonify({"error": "Message cannot be empty."}), 400

    user_message = data["message"].strip()

    # === BACKEND LENGTH VALIDATION ===
    if len(user_message) > 500:
        return jsonify({"error": "Message too long. Please keep it under 500 characters."}), 400

    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    history.append({
        "role": "user",
        "content": user_message
    })

    history = history[-MAX_HISTORY_MESSAGES:]

    def generate():
        nonlocal history

        try:
         # ── Phase 1: Tool resolution (non-streaming) ──────────────────────
            # Runs invisibly. Each iteration: Claude calls a tool → we execute it
            # → append result → loop again until Claude stops using tools.
            while True:
                response = client.messages.create(
                    model=MODEL_NAME,
                    max_tokens=300,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=history
                )            

                if response.stop_reason != "tool_use":
                    break #Claude is ready to give the final text answer

                #serializable assistant content (SDK object -> plain dicts)
                asst_content = []
                for block in response.content:
                    if block.type =="tool_use":
                        asst_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input
                        })
                    elif block.type == "text":
                        asst_content.append({
                            "type": "text",
                            "text": block.text
                        })
                        
                history.append({
                    "role": "assistant",
                    "content": asst_content
                })

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = run_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })

                history.append({
                    "role": "user",
                    "content": tool_results
                })
            
            # ── Phase 2: Stream final response ───────────────────────────────
            # history now contains all tool calls + results.
            # Claude generates its final answer; we stream each token to the client.
            full_reply: list[str] = []

            with client.messages.stream(
                model=MODEL_NAME,
                max_tokens=300,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=history,
            ) as stream:
                for text in stream.text_stream:
                    full_reply.append(text)
                    yield f"data: {json.dumps({'token': text})}\n\n"
        # Persist completed history to the in-memory store

            final_text = "".join(full_reply)
            history.append({"role": "assistant", "content": final_text})
            conversation_store[sid] = history[-MAX_HISTORY_MESSAGES:]
        

            yield f"data: {json.dumps({'done': True})}\n\n"
                      
               
        except anthropic.AuthenticationError:
            yield f"data: {json.dumps({'error': 'Invalid API key. Check your configuration.'})}\n\n"
    
        except anthropic.RateLimitError:
            yield f"data: {json.dumps({'error': 'Rate limit reached. Please wait and try again'})}\n\n"
        except anthropic.APIConnectionError:
            yield f"data: {json.dumps({'error': 'Could not connect to AI service. Check your internet.'})}\n\n"

        except anthropic.APIStatusError as e:
            yield f"data: {json.dumps({'error': f'API error: {e.status_code}'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': 'Something went wrong. Please try again.'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},)

@app.route('/reset', methods=['POST'])
def reset():
    sid = session.get("sid")
    if sid and sid in conversation_store:
        del conversation_store[sid]
    return jsonify({"message": "Conversation reset."})


@app.errorhandler(429)
def rate_limit_handler(e):
    return jsonify({"error": "Too many requests. Please slow down."}), 429


if __name__ == "__main__":
    app.run(debug=True)