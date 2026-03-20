# ============================================================
# rag_engine.py
#
# WHAT IS THIS FILE?
# This is the heart of the RAG system.
# RAG = Retrieval-Augmented Generation
#
# WHAT IT DOES STEP BY STEP:
#
# Step 1 — EMBED: Convert every knowledge chunk into a vector
#          (a list of numbers that represents its meaning).
#          Similar texts will have similar vectors.
#
# Step 2 — RETRIEVE: When patient symptoms arrive, convert
#          them into a vector too, then find the knowledge
#          chunks whose vectors are closest (cosine similarity).
#
# Step 3 — PRUNE (ICP): Keep only the top-K most relevant
#          chunks. Drop the rest. This is "Intelligent Context
#          Pruning" — we don't dump everything into Claude,
#          only what's useful. This keeps responses fast.
#
# Step 4 — INJECT: Return the pruned chunks as a formatted
#          string that gets added to Claude's prompt.
#
# WHY SENTENCE TRANSFORMERS?
# It's a local model that runs on your CPU — no API key,
# no cost, no internet needed for the embedding step.
# The model "all-MiniLM-L6-v2" is small (80MB) but very
# good at understanding semantic similarity.
# ============================================================

import numpy as np
from sentence_transformers import SentenceTransformer
from rag_data.knowledge_base import KNOWLEDGE_CHUNKS


class RAGEngine:

    def __init__(self):
        # ── STEP 1: Load the embedding model ──────────────────
        # SentenceTransformer converts text → vector (384 numbers).
        # "all-MiniLM-L6-v2" is fast, small, and works well for
        # medical text similarity. Downloads once, cached locally.
        print("[RAG] Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Pre-compute embeddings for every knowledge chunk
        # We do this ONCE at startup so retrieval is instant later.
        self.chunks = KNOWLEDGE_CHUNKS
        self.chunk_texts = [c["text"] for c in self.chunks]

        print("[RAG] Embedding knowledge base...")
        # encode() returns a 2D numpy array: shape = (num_chunks, 384)
        self.embeddings = self.model.encode(
            self.chunk_texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        print(f"[RAG] Ready — {len(self.chunks)} chunks embedded.")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Given a patient symptoms string, find the top_k most
        relevant knowledge chunks from our medical database.

        Args:
            query:  The patient's symptoms / presentation text.
            top_k:  How many chunks to return (ICP — keep it small!).

        Returns:
            List of chunk dicts ordered by relevance (best first).
        """

        # ── STEP 2: Embed the incoming query ──────────────────
        # Convert the symptom text into the same vector space
        # as our knowledge chunks so we can compare them.
        query_vec = self.model.encode(query, convert_to_numpy=True)

        # ── STEP 3: Cosine similarity ──────────────────────────
        # Cosine similarity measures the ANGLE between two vectors.
        # Score of 1.0 = identical meaning. 0.0 = completely different.
        #
        # Formula: similarity = (A · B) / (|A| × |B|)
        # We normalize first so we only need the dot product.

        # Normalize query vector to unit length
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)

        # Normalize all chunk embeddings (row-wise)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        chunks_norm = self.embeddings / (norms + 1e-9)

        # Dot product → similarity score for each chunk
        scores = chunks_norm @ query_norm  # shape: (num_chunks,)

        # ── STEP 4: ICP — Intelligent Context Pruning ─────────
        # Sort by score descending, keep only top_k.
        # This is where we throw away irrelevant medical info.
        # E.g. for a chest pain case, we DON'T inject the
        # opioid overdose protocol — it would just confuse Claude.
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in top_indices:
            chunk = self.chunks[i].copy()
            chunk["score"] = float(scores[i])   # attach relevance score
            results.append(chunk)

        return results

    def format_context(self, retrieved_chunks: list[dict]) -> str:
        """
        Format the retrieved chunks into a clean string
        that gets injected into Claude's system prompt.

        Args:
            retrieved_chunks: Output from retrieve().

        Returns:
            A formatted string of medical context.
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
