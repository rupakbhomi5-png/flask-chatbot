import os
import json
import anthropic
from flask import Flask, request, jsonify, render_template, session
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

load_dotenv(override=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")

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


def build_system_prompt():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "products.json"), "r") as f:
        data = json.load(f)

    product_list = "\n".join(
        f"- {p['name']}: NPR {p['price']:,}"
        for p in data["products"]
    )

    return f'''You are Ramcha, a customer service agent for {data['store_name']}, \
an electronics retail store in {data['location']}.

STORE INFORMATION:
- Hours: {data['hours']}
- Location: {data['location']}
- Contact: {data['contact']}
- Return Policy: {data['return_policy']}

PRODUCTS WE SELL:
{product_list}

WHAT YOU MUST NEVER DO:
- Never invent products we don't sell
- Never quote prices not listed above - say "please contact us at {data['contact']} for current pricing"
- Never promise stock availability
- For specialized items not on this list, always direct customer to call {data['contact']}

Keep response under 3 sentences. Be friendly but precise.'''


SYSTEM_PROMPT = build_system_prompt()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
@limiter.limit("10 per minute")
def chat():
    if "history" not in session:
        session["history"] = []

    conversation_history = session["history"]
    data = request.json

    if not data or not data.get("message"):
        return jsonify({"error": "Message cannot be empty."}), 400

    user_message = data["message"].strip()

    # === BACKEND LENGTH VALIDATION ===
    if len(user_message) > 500:
        return jsonify({"error": "Message too long. Please keep it under 500 characters."}), 400

    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    conversation_history = conversation_history[-MAX_HISTORY_MESSAGES:]

    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=conversation_history
        )

        assistant_reply = response.content[0].text

        conversation_history.append({
            "role": "assistant",
            "content": assistant_reply
        })

        conversation_history = conversation_history[-MAX_HISTORY_MESSAGES:]

        session["history"] = conversation_history
        session.modified = True

        return jsonify({"reply": assistant_reply})

    except anthropic.AuthenticationError:
        if conversation_history: conversation_history.pop()
        return jsonify({"error": "Invalid API key. Check your configuration."}), 401

    except anthropic.RateLimitError:
        if conversation_history: conversation_history.pop()
        return jsonify({"error": "Rate limit reached. Please wait and try again."}), 429

    except anthropic.APIConnectionError:
        if conversation_history: conversation_history.pop()
        return jsonify({"error": "Could not connect to AI service. Check your internet."}), 503

    except anthropic.APIStatusError as e:
        if conversation_history: conversation_history.pop()
        return jsonify({"error": f"API error: {e.status_code}"}), 500

    except Exception as e:
        if conversation_history: conversation_history.pop()
        return jsonify({"error": "Something went wrong. Please try again."}), 500


@app.route('/reset', methods=['POST'])
def reset():
    session.clear()
    return jsonify({"message": "Conversation reset."})


@app.errorhandler(429)
def rate_limit_handler(e):
    return jsonify({"error": "Too many requests. Please slow down."}), 429


if __name__ == "__main__":
    app.run(debug=True)