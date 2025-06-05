from flask import Flask, request, jsonify
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from rag_utils import retrieve_relevant_docs
from sklearn.metrics.pairwise import cosine_similarity
from flask import render_template 
from flask_cors import CORS
import os
import requests

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


def generate_promt(content, task):
     if task == "summarize":
        return f"Summarize this blog post:\n\n{content}"
     elif task == "answer":
        return f"Answer the question based on this blog post:\n\n{content}"
     elif task == "read":
        return f"Convert this blog post into a clear spoken script:\n\n{content}"
     return "Invalid task"


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
            capture_output=True, text=True, check=True, encoding='utf-8'
            )
        response = result.stdout.strip()
        return jsonify({'reply': response})
    except subprocess.CalledProcessError as e:
        return jsonify({'error': e.stderr}), 500

@app.route('/analyze', methods = ['POST'])
def analyze():
    data = request.json
    content = data['content']
    task = data['task']
    if task == 'summarize':
        response = requests.post('http://localhost:11434/api/generate',json={
            "model" : "llama3",
            "prompt" : f"Summarize this blog post:\n\n{content}",
            "stream": False
        })
        return jsonify({'summary': response.json()['response']})
    
    if not content or not task:
        return jsonify({'error': 'Content and task are required'}), 400
    if task not in ['summarize', 'answer', 'read']:
        return jsonify({'error': 'Invalid task'}), 400
    if task == 'answer':
        user_input = data.get('question','').strip()
        response = requests.post('http://localhost:11434/api/generate',json={
            "model" : "llama3",
            "prompt" : f"Answer the question based on this blog post:\n\n{content}",
            "stream": False
        })
        answer = response.json()['response']
        return jsonify({'answer':answer})
'''
@app.route('/speak', methods=['POST'])
def speak():
    text = request.json['text']
    voice_id = os.getenv("ZUrEGyu8GFMwnHbvLhv2")
    api_key = os.getenv("Vsk_ce785a28ed7e13d3ff5f2333330dfefbcb9a230854239c43")

    response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        },
        json={"text": text, "voice_settings": {"stability": 0.7, "similarity_boost": 0.7}},
    )
    with open("static/tts.mp3", "wb") as f:
        f.write(response.content)

    return jsonify({"audio": "/static/tts.mp3"})
'''


if __name__ == "__main__":
    app.run(debug=True)

