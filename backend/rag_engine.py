"""
rag_engine.py
=============
Fast Semantic RAG Engine using HuggingFace Embeddings & FAISS Vector Store.
Operates natively with LangChain Document objects.
""" 

from typing import List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Pre-load lightweight 384-dim semantic embedding model once in memory
_EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


class RAGEngine:
    """
    Semantic RAG engine powered by HuggingFace embeddings and FAISS vector index.
    """

    def __init__(self):
        self.vectorstore: FAISS | None = None
        self.documents: List[Document] = []
        self.document_category: str = "General Medical Document"
        self.active_filename: str = ""

    @property
    def has_documents(self) -> bool:
        """Returns True if documents have been indexed."""
        return bool(self.documents and self.vectorstore is not None)
    # runs once 
    def ingest_documents(self, documents: List[Document], document_category: str = "General Medical Document") -> int:
        """
        Indexes a list of LangChain Document chunks into FAISS vector store.
        """
        if not documents:
            self.documents = []
            self.vectorstore = None
            return 0

        self.documents = documents
        self.document_category = document_category
        self.active_filename = documents[0].metadata.get("source", "document.txt")

        self.vectorstore = FAISS.from_documents(documents, _EMBEDDINGS)
        print(f"[RAG] Indexed {len(documents)} document chunks into FAISS for {self.active_filename} ({self.document_category})")
        return len(documents)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieves top_k most semantically relevant chunks via FAISS similarity search with scores.
        """
        if self.vectorstore is None or not self.documents:
            return []

        results_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)

        results = []
        for doc, distance in results_with_scores:
            # Convert FAISS L2 distance to 0.0 - 1.0 similarity score
            similarity = round(max(0.0, 1.0 - (float(distance) / 2.0)), 3)
            chunk = {
                "text": doc.page_content,
                "id": doc.metadata.get("id", ""),
                "page": doc.metadata.get("page", 1),
                "source": doc.metadata.get("source", "document"),
                "section": doc.metadata.get("section", "General"),
                "score": similarity,
            }
            results.append(chunk)

        return results

    def format_context(self, retrieved_chunks: List[Dict]) -> str:
        """
        Formats retrieved chunks into clean evidence text for the LLM prompt.
        """
        if not retrieved_chunks:
            return "No specific matching excerpts retrieved from the document."

        lines = ["EXCERPTS FROM MEDICAL DOCUMENT:"]
        for i, chunk in enumerate(retrieved_chunks, 1):
            source_info = f"{chunk.get('source', 'Document')}, Page {chunk.get('page', 1)}"
            lines.append(
                f"[Excerpt {i} | Source: {source_info} | Section: {chunk.get('section', 'General')}]\n"
                f"{chunk['text']}\n"
            )
        return "\n".join(lines)
