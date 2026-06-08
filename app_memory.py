import os
from flask import Flask, render_template, request, jsonify
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# System prompt now comes from environment, with a fallback default
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "You are a helpful assistant.")

conversation_history = []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]

    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=conversation_history
    )

    assistant_message = response.content[0].text

    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })

    return jsonify({"response": assistant_message})

if __name__ == "__main__":
    app.run(debug=True)
