import os 
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import anthropic

load_dotenv(override=True)

app = Flask(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "You are a helpful assistant.")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

conversation_history = []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    # --- LAYER 1 : Input Validation ---
    data = request.json

    if not data or "message" not in data:
        return jsonify({"error": "No message received"}), 400

    user_message = data["message"].strip()

    if not user_message:
        return jsonify({"error": "Message cannot be empty"}), 400

    # --- Build conversation history ---
    conversation_history.append({
        "role": "user",
        "content": user_message
    }) 

    # --- LAYER 2 & 3: API Call with error handling ---
    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1000,
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
        return jsonify({"error": "API key is invalid. Contact support."}), 401
    
    except anthropic.RateLimitError:
        conversation_history.pop()
        return jsonify({"error": "Too many requests. Please wait a moment and try again."}), 429
    
    except anthropic.APIConnectionError:
        conversation_history.pop()
        return jsonify({"error": "Cannot reach the AI service. Check your internet connection."}), 503
    
    except anthropic.APIStatusError as e:
        conversation_history.pop()
        return jsonify({"error": f"AI Service returned an error ({e.status_code}). Try again."}), 502
    
    except Exception as e:
        conversation_history.pop()
        print(f"Unexpected error : {e}")
        return jsonify({"error": "Something went wrong. Please try again."}), 500
    
if __name__ == "__main__":
    app.run(debug=True)

    

