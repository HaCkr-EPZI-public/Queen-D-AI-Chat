from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize Flask app
app = Flask(__name__)

# Load Microsoft DialoGPT-medium model
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# AI Information response without newlines
AI_INFORMATION = (
    "Hey! I'm Queen D, a pre-trained AI model developed by Asmodeus Epzi. "
    "I am designed to assist you with a wide range of tasks including answering questions, "
    "providing information, generating text, and more. "
    "I am powered by cutting-edge technology and can handle complex language processing tasks. "
    "My purpose is to make your experience smarter and more efficient!"
)

# Special response for "Queen D" or related queries
SPECIAL_RESPONSES = {
    "🌸 Dinathma 😚 She's Mine Bruh 🌸\n\n"
    "A shy beauty with a heart so pure, her smile can melt the coldest days, and her eyes hold the universe. "
    "She’s my everything—my muse, my dream, and the reason I believe in love. 💖 Hands off, she's taken! 😏": [
        "queend", "queen d", "d", "dinathma", "who is queen d", 
        "epzis lover", "epzis lover", "queen-d"
    ]
}

# Serve the HTML page at the root URL
@app.route("/")
def home():
    return render_template("index.html")

# Handle the /chat POST request
@app.route("/chat", methods=["POST"])
def chat():
    # Get user input
    data = request.json
    user_input = data.get("message", "").strip().lower()
    
    if not user_input:
        return jsonify({"error": "Message is required"}), 400

    # Check for special responses
    for response, triggers in SPECIAL_RESPONSES.items():
        if user_input in triggers:
            return jsonify({
                "Powered By": "Queen D",
                "response": response
            })
    
    # Handle specific queries like "Who are you?" or "Tell me about yourself?"
    if "who are you" in user_input or "tell me about yourself" in user_input:
        return jsonify({
            "Powered By": "Queen D",
            "response": AI_INFORMATION
        })
    
    # Generate response using the DialoGPT model for other queries
    try:
        # Encode the user input and generate a response
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
        chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        reply = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        # Return response with custom format
        return jsonify({
            "Powered By": "Queen D",
            "response": reply
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
