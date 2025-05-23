from flask import Flask, request, jsonify
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from rag_utils import retrieve_relevant_docs
from sklearn.metrics.pairwise import cosine_similarity
from flask import render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def rdocs(blog_text, query, top_k =2):
    doc = [line.strip() for line in blog_text.split('\n') if line.strip()]
    
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(doc)

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, doc_vectors).flatten()
    top_indices = scores.argsort()[::-1][:top_k]

    return [doc[i] for i in top_indices]

@app.route('/')
def serve_ui():
    return render_template('index.html')

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400

    context_docs = retrieve_relevant_docs(user_message)
    context = "\n".join(context_docs)
    full_prompt = f"Context:\n{context}\n\nUser: {user_message}\nAssistant:"

    try:
        result = subprocess.run(
            ["ollama", "run", "mistral", full_prompt],
            capture_output=True, text=True, check=True
        )
        response = result.stdout.strip()
        return jsonify({'reply': response})
    except subprocess.CalledProcessError as e:
        return jsonify({'error': e.stderr}), 500

if __name__ == "__main__":
    app.run(debug=True)
