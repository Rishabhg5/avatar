# app.py
from flask import Flask, request, jsonify
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
import os

app = Flask(__name__)
os.environ["OPENAI_API_KEY"] = "your-openai-key"

# Load and split data
loader = TextLoader("data/sampledata.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create vector store
embedding = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embedding)

# Create retriever
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    try:
        answer = qa_chain.run(user_message)
        return jsonify({"reply": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
