"""
rag_engine.py
=============
Lightweight, serverless-optimized RAG Engine using Scikit-Learn TF-IDF.
Designed specifically for Vercel Serverless deployment within the 250MB limit.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.chunks = []
        self.chunk_texts = []
        self.embeddings = None
        self.active_filename = "document.txt"
        self.category = "General Medical Document"

    def ingest(self, text: str, filename: str = "document.txt", category: str = "General Medical Document"):
        """Splits document text dynamically and builds the TF-IDF vector index."""
        self.active_filename = filename
        self.category = category

        # Split by double newlines or paragraphs
        raw_chunks = [c.strip() for c in re.split(r'\n\s*\n', text) if len(c.strip()) > 10]

        # Fallback if text is dense: split by sentences
        if len(raw_chunks) < 3:
            raw_chunks = [c.strip() + "." for c in text.split('. ') if len(c.strip()) > 10]

        if not raw_chunks:
            return 0

        self.chunks = []
        self.chunk_texts = []

        for i, chunk in enumerate(raw_chunks):
            self.chunks.append({
                "id": f"chunk_{i}",
                "text": chunk,
                "source": filename,
                "page": (i // 4) + 1,
            })
            self.chunk_texts.append(chunk)

        self.embeddings = self.vectorizer.fit_transform(self.chunk_texts)
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Retrieves the top chunks for a given user query using Cosine Similarity."""
        if self.embeddings is None or not self.chunks:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.embeddings).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in top_indices:
            if scores[i] > 0.005:
                chunk = self.chunks[i].copy()
                chunk["score"] = round(float(scores[i]), 3)
                results.append(chunk)

        return results

    def format_context(self, retrieved_chunks: list) -> str:
        """Formats retrieved chunks into clean evidence excerpts for the LLM prompt."""
        if not retrieved_chunks:
            return "No specific matching excerpts retrieved from document."

        lines = ["EXCERPTS FROM UPLOADED DOCUMENT (ranked by contextual relevance):"]
        for i, chunk in enumerate(retrieved_chunks, 1):
            source_info = f"{chunk.get('source', 'Document')}, Page {chunk.get('page', 1)}"
            lines.append(
                f"[Excerpt {i} | Source: {source_info} | Relevance Score: {chunk['score']:.2f}]\n"
                f"{chunk['text']}\n"
            )
        return "\n".join(lines)
