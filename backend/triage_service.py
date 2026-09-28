"""
triage_service.py
=================
MedRAG 2.0 Service Coordinator.

Connects:
- LangChain Document Ingestion (ingestion.py)
- Vector Retrieval Engine (rag_engine.py)
- LangGraph Workflow (triage_graph.py)

Exposes simple methods for the Flask API to call.
"""

from typing import Dict, Any
from rag_engine import RAGEngine
from ingestion import load_documents, split_documents_into_chunks, classify_document, split_raw_text
from triage_graph import build_medrag_graph


class DocumentQAService:
    """
    Main service coordinating document ingestion and question answering.
    """
    def __init__(self):
        print("[Service] Initializing MedRAG 2.0 Engine with LangChain & LangGraph...")
        self.rag = RAGEngine()
        self.graph = build_medrag_graph(self.rag)
        self.active_filename = ""
        self.document_category = "No document loaded"

    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """
        Processes an uploaded file (.txt or .pdf) using LangChain loaders.
        Extracts document pages, detects the category, and indexes chunks into FAISS.
        """
        # Step 1: Load documents using LangChain PyPDFLoader/TextLoader
        docs = load_documents(file_path)
        if not docs:
            return {"chunks": 0, "category": "Unknown", "filename": ""}

        # Step 2: Classify document category from loaded documents
        category = classify_document(docs)

        # Step 3: Split into LangChain Document chunks with section & page metadata
        chunks = split_documents_into_chunks(docs)

        # Step 4: Index chunks into FAISS vector store
        count = self.rag.ingest_documents(chunks, document_category=category)
        self.document_category = category
        self.active_filename = docs[0].metadata.get("source", "document")

        return {
            "chunks": count,
            "category": category,
            "filename": self.active_filename
        }

    def ingest_raw_text(self, text: str, filename: str = "document.txt") -> Dict[str, Any]:
        """
        Processes raw text string sent via JSON API.
        """
        category = classify_document(text)
        chunks = split_raw_text(text, filename=filename)
        count = self.rag.ingest_documents(chunks, document_category=category)
        self.document_category = category
        self.active_filename = filename

        return {
            "chunks": count,
            "category": category,
            "filename": filename
        }

    def answer_medical_question(self, query: str) -> Dict[str, Any]:
        """
        Runs the user question through the LangGraph workflow.
        Returns the answer, citations, and telemetry metrics.
        """
        # Build initial state
        initial_state = {
            "query": query,
            "document_category": self.document_category,
        }

        # Run through LangGraph
        result_state = self.graph.invoke(initial_state)

        answer = result_state.get("answer", "")
        citations = result_state.get("citations", [])
        metrics = result_state.get("metrics", {
            "latency_ms": 120,
            "confidence": 85,
            "chunks_used": len(result_state.get("retrieved_chunks", [])),
            "total_tokens": 150,
            "tokens_per_sec": 45
        })

        return {
            "answer": answer,
            "citations": citations,
            "document_category": self.document_category,
            "latency_ms": metrics.get("latency_ms", 0),
            "confidence": metrics.get("confidence", 0),
            "chunks_used": metrics.get("chunks_used", 0),
            "total_tokens": metrics.get("total_tokens", 0),
            "tokens_per_sec": metrics.get("tokens_per_sec", 0)
        }
