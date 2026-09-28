"""
rag_engine.py
=============
Multi-document, serverless-optimized RAG Engine using Scikit-Learn TF-IDF.
Supports indexing and querying multiple documents simultaneously with source citations.
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
        self.documents = []  # list of {"filename": str, "category": str, "chunks": int}

    def clear(self):
        """Clears all indexed documents and vector index."""
        self.chunks = []
        self.chunk_texts = []
        self.embeddings = None
        self.documents = []

    def remove_document(self, filename: str):
        """Removes a specific document and rebuilds the vector index."""
        self.documents = [d for d in self.documents if d["filename"] != filename]
        self.chunks = [c for c in self.chunks if c.get("source") != filename]
        self.chunk_texts = [c["text"] for c in self.chunks]
        if self.chunk_texts:
            self.embeddings = self.vectorizer.fit_transform(self.chunk_texts)
        else:
            self.embeddings = None

    def ingest(self, text: str, filename: str = "document.txt", category: str = "General Medical Document", append: bool = True):
        """
        Splits document text and indexes into the TF-IDF vector space.
        Appends to existing documents so multiple documents can be queried together.
        """
        # If document with the same filename already exists, remove old chunks first
        if filename in [d["filename"] for d in self.documents]:
            self.remove_document(filename)

        # Split into logical chunks
        raw_chunks = [c.strip() for c in re.split(r'\n\s*\n', text) if len(c.strip()) > 10]
        if len(raw_chunks) < 3:
            raw_chunks = [c.strip() + "." for c in text.split('. ') if len(c.strip()) > 10]

        if not raw_chunks:
            return 0

        doc_chunks = []
        for i, chunk in enumerate(raw_chunks):
            chunk_obj = {
                "id": f"{filename}_chunk_{i}",
                "text": chunk,
                "source": filename,
                "category": category,
                "page": (i // 4) + 1,
            }
            doc_chunks.append(chunk_obj)
            self.chunks.append(chunk_obj)
            self.chunk_texts.append(chunk)

        self.documents.append({
            "filename": filename,
            "category": category,
            "chunks": len(doc_chunks)
        })

        self.embeddings = self.vectorizer.fit_transform(self.chunk_texts)
        return len(doc_chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Retrieves top semantically relevant chunks across all indexed documents."""
        if self.embeddings is None or not self.chunks:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.embeddings).flatten()

        # Dynamically scale top_k if multiple documents are indexed
        num_docs = len(self.documents)
        effective_k = min(max(top_k, num_docs * 2), 6)
        top_indices = np.argsort(scores)[::-1][:effective_k]

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
            return "No specific matching excerpts retrieved from documents."

        lines = ["EXCERPTS FROM MEDICAL DOCUMENTS (ranked by relevance):"]
        for i, chunk in enumerate(retrieved_chunks, 1):
            source_info = f"{chunk.get('source', 'Document')}, Page {chunk.get('page', 1)}"
            cat = chunk.get("category", "")
            cat_str = f" | Category: {cat}" if cat else ""
            lines.append(
                f"[Excerpt {i} | Source: {source_info}{cat_str} | Relevance Score: {chunk['score']:.2f}]\n"
                f"{chunk['text']}\n"
            )
        return "\n".join(lines)
