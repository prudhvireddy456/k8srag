from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rag import KubernetesRAG

app = Flask(__name__, template_folder=".")
CORS(app)  # Enable CORS for all routes

# Initialize KubernetesRAG
rag = KubernetesRAG(docs_dir="./k8s_docs", store_dir="./k8s_rag_store")
# rag.build_index()  # Build the index once during initialization

@app.route("/")
def home():
    """
    Serve the index.html file.
    """
    return render_template("index.html")

@app.route("/api/ask", methods=["POST"])
def ask():
    """
    Endpoint to handle Kubernetes-related questions.
    Expects a JSON payload with a 'question' field.
    """
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        answer = rag.generate_answer(question)
        return jsonify({"question": question, "answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")