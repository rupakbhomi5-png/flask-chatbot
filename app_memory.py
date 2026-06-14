import os
import anthropic
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

load_dotenv(override=True)

app = Flask (__name__)

#Rate Limiter setup
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "You are a helpful assistant.")

conversation_history = []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
@limiter.limit("10 per minute")
def chat():
    data = request.json

    if not data or not data.get("message"):
        return jsonify({"error": "Message cannot be empty."}), 400
    
    user_message = data["message"].strip()

    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400
    
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=conversation_history
        )

        assistant_reply = response.content[0].text

        conversation_history.append({
            "role": "assistant",
            "content": assistant_reply
        })

        return jsonify({"reply": assistant_reply})
    
    except anthropic.AuthenticationError:
        conversation_history.pop()
        return jsonify({"error": "Invalid API key. Check you configuration."}), 401
    
    except anthropic.RateLimitError:
        conversation_history.pop()
        return jsonify({"error": "Rate limit reached. Please wait and try again."}), 429
    
    except anthropic.APIConnectionError:
        conversation_history.pop()
        return jsonify({"error": "Could not connect to AI service. Check your internet."}), 503
    
    except anthropic.APIStatusError as e:
        conversation_history.pop()
        return jsonify({"error": f"API error: {e.status_code}"}), 500
    
    except Exception as e:
        conversation_history.pop()
        return jsonify({"error": "Something went wrong. Please try again."}), 500
    
@app.errorhandler(429)
def rate_limit_handler(e):
    return jsonify({"error": "Too many requests. Please slow down."}), 429

if __name__ == "__main__":
    app.run(debug=True)

    



