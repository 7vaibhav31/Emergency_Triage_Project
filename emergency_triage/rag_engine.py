import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RAGEngine:
    def __init__(self):
        print("[RAG] Dynamic engine initialized. Awaiting document.")
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.chunks = []
        self.chunk_texts = []
        self.embeddings = None

    def ingest(self, text: str):
        """ Splitting document dynamically and building the index """
        # Split by double newlines or basic layout block assumptions
        raw_chunks = [c.strip() for c in re.split(r'\n\s*\n', text) if len(c.strip()) > 10]
        
        # Fallback if too few chunks (e.g. dense single block of text), chunk by sentences
        if len(raw_chunks) < 3:
            raw_chunks = [c.strip() + "." for c in text.split('. ') if len(c.strip()) > 10]

        if not raw_chunks:
            return 0
        
        self.chunks = []
        self.chunk_texts = []
        
        for i, chunk in enumerate(raw_chunks):
            self.chunks.append({"id": f"chunk_{i}", "text": chunk})
            self.chunk_texts.append(chunk)

        print(f"[RAG] Ingesting {len(self.chunks)} dynamic chunks...")
        self.embeddings = self.vectorizer.fit_transform(self.chunk_texts)
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """ Retreive the top chunks for a given user query """
        if self.embeddings is None or not self.chunks:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.embeddings).flatten()
        
        # Sort and keep top_k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for i in top_indices:
            if scores[i] > 0.01:  # Simple threshold
                chunk = self.chunks[i].copy()
                chunk["score"] = float(scores[i])
                results.append(chunk)

        return results

    def format_context(self, retrieved_chunks: list[dict]) -> str:
        """ Prepare the string to feed into the LLM prompt """
        if not retrieved_chunks:
            return "No specific matching excerpts retrieved from document."

        lines = ["EXCERPTS FROM UPLOADED DOCUMENT (ranked by contextual relevance):\n"]
        for i, chunk in enumerate(retrieved_chunks, 1):
            lines.append(
                f"[Excerpt {i} | Relevance Score: {chunk['score']:.2f}]\n"
                f"{chunk['text']}\n"
            )
        return "\n".join(lines)
