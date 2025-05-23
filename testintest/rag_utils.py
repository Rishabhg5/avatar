from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open("data/sampledata.txt", "r", encoding = "utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]

vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)

def retrieve_relevant_docs(query, top_k=2):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, doc_vectors).flatten()
    ranked_indices = similarities.argsort()[::-1][:top_k]
    return [documents[i] for i in ranked_indices]
