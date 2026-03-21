# ============================================================
# rag_engine.py
#
# WHAT IS THIS FILE?
# This is the heart of the RAG system.
# RAG = Retrieval-Augmented Generation
#
# WHAT IT DOES STEP BY STEP:
#
# Step 1 — EMBED: Convert every knowledge chunk into a sparse vector
#          using TF-IDF. This is extremely fast and lightweight.
#
# Step 2 — RETRIEVE: When patient symptoms arrive, convert
#          them into a vector too, then find the knowledge
#          chunks whose vectors are closest (cosine similarity).
#
# Step 3 — PRUNE (ICP): Keep only the top-K most relevant
#          chunks. Drop the rest. This keeps responses fast.
#
# Step 4 — INJECT: Return the pruned chunks as a formatted
#          string that gets added to Claude's prompt.
#
# WHY TF-IDF OVER SENTENCE TRANSFORMERS?
# SentenceTransformers requires PyTorch, which is a massive ~900MB
# library. Vercel Serverless Functions have a 250MB strict limit.
# Switching to scikit-learn avoids this limit entirely.
# ============================================================

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rag_data.knowledge_base import KNOWLEDGE_CHUNKS

class RAGEngine:

    def __init__(self):
        # ── STEP 1: Load the embedding model ──────────────────
        print("[RAG] Loading TF-IDF model...")
        self.vectorizer = TfidfVectorizer(stop_words='english')

        self.chunks = KNOWLEDGE_CHUNKS
        self.chunk_texts = [c["text"] for c in self.chunks]

        print("[RAG] Embedding knowledge base...")
        # Fit the vectorizer on our medical chunks
        self.embeddings = self.vectorizer.fit_transform(self.chunk_texts)
        print(f"[RAG] Ready — {len(self.chunks)} chunks embedded.")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Given a patient symptoms string, find the top_k most
        relevant knowledge chunks from our medical database.
        """

        # ── STEP 2: Embed the incoming query ──────────────────
        query_vec = self.vectorizer.transform([query])

        # ── STEP 3: Cosine similarity ──────────────────────────
        scores = cosine_similarity(query_vec, self.embeddings).flatten()

        # ── STEP 4: ICP — Intelligent Context Pruning ─────────
        # Sort by score descending, keep only top_k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in top_indices:
            chunk = self.chunks[i].copy()
            chunk["score"] = float(scores[i])
            results.append(chunk)

        return results

    def format_context(self, retrieved_chunks: list[dict]) -> str:
        """
        Format the retrieved chunks into a clean string
        that gets injected into Claude's system prompt.
        """
        if not retrieved_chunks:
            return "No specific protocols retrieved."

        lines = ["RETRIEVED MEDICAL PROTOCOLS (ranked by relevance):\n"]
        for i, chunk in enumerate(retrieved_chunks, 1):
            lines.append(
                f"[Protocol {i} | Category: {chunk['category'].upper()} "
                f"| Relevance: {chunk['score']:.2f}]\n"
                f"{chunk['text']}\n"
            )
        return "\n".join(lines)
