"""
app.py
======
MedRAG 2.0 Web Application (Minimal Flask Backend).

This file handles HTTP requests and forwards them to MedRAG's LangGraph service:
- GET  /            -> Serves the user interface (from frontend/)
- POST /api/upload  -> Ingests uploaded medical file (saved to documents/) or text
- POST /api/qa      -> Runs LangGraph to answer questions with citations
"""

import os
from flask import Flask, request, jsonify, render_template
from triage_service import DocumentQAService

# Resolve directory paths
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)
frontend_dir = os.path.join(project_root, "frontend")
documents_dir = os.path.join(project_root, "documents")
os.makedirs(documents_dir, exist_ok=True)

# Configure Flask with frontend folder for templates & static assets
app = Flask(__name__, template_folder=frontend_dir, static_folder=frontend_dir)

# Initialize the central MedRAG service
qa_service = DocumentQAService()


@app.route("/")
def index():
    """Serves the MedRAG 2.0 web interface."""
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_document():
    """
    Endpoint for uploading medical document(s) (.txt or .pdf).
    Supports multiple files uploaded simultaneously.
    """
    uploaded_files = request.files.getlist("files") or request.files.getlist("file")
    results = []

    if uploaded_files and any(f.filename for f in uploaded_files):
        for uploaded_file in uploaded_files:
            if not uploaded_file or not uploaded_file.filename:
                continue
            save_path = os.path.join(documents_dir, uploaded_file.filename)
            try:
                uploaded_file.save(save_path)
            except Exception:
                pass
            res = qa_service.ingest_file(save_path)
            results.append({
                "filename": uploaded_file.filename,
                "category": res.get("category", "General"),
                "chunks": res.get("chunks", 0)
            })

    # Option B: Raw text provided via JSON
    if not results:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "").strip()
        filename = data.get("filename", "document.txt")

        if text:
            save_path = os.path.join(documents_dir, filename)
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception:
                pass
            res = qa_service.ingest_raw_text(text, filename=filename)
            results.append({
                "filename": filename,
                "category": res.get("category", "General"),
                "chunks": res.get("chunks", 0)
            })

    if not results:
        return jsonify({"error": "No valid document(s) provided. Please choose a valid file."}), 400

    docs = getattr(qa_service.rag, "documents", [])
    total_chunks = len(getattr(qa_service.rag, "chunks", [])) or sum(r["chunks"] for r in results)

    return jsonify({
        "message": f"Successfully indexed {len(results)} document(s).",
        "uploaded": results,
        "documents": [{"filename": r["filename"], "category": r["category"], "chunks": r["chunks"]} for r in results],
        "total_chunks": total_chunks
    })


@app.route("/api/documents", methods=["GET"])
def get_documents():
    """Returns currently indexed documents."""
    return jsonify({
        "documents": getattr(qa_service.rag, "documents", []),
        "total_chunks": len(getattr(qa_service.rag, "chunks", []))
    })


@app.route("/api/documents/clear", methods=["POST"])
def clear_documents():
    """Clears all indexed documents from memory."""
    if hasattr(qa_service.rag, "clear"):
        qa_service.rag.clear()
    return jsonify({
        "message": "All documents cleared.",
        "documents": [],
        "total_chunks": 0
    })


@app.route("/api/documents/remove", methods=["POST"])
def remove_document():
    """Removes a specific document by filename."""
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    if filename and hasattr(qa_service.rag, "remove_document"):
        qa_service.rag.remove_document(filename)
    return jsonify({
        "message": f"Document '{filename}' removed.",
        "documents": getattr(qa_service.rag, "documents", []),
        "total_chunks": len(getattr(qa_service.rag, "chunks", []))
    })


@app.route("/api/qa", methods=["POST"])
def answer_question():
    """
    Endpoint for asking questions about the active document(s).
    Executes the evidence-grounded LangGraph workflow.
    """
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Please enter a clinical question."}), 400

    # Ensure at least one document was uploaded first
    if not qa_service.rag.has_documents:
        return jsonify({"error": "Please upload a medical document first before asking questions."}), 400

    try:
        # Run through LangGraph
        result = qa_service.answer_medical_question(query)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    print(f"[App] Frontend directory: {frontend_dir}")
    print(f"[App] Documents directory: {documents_dir}")
    print("[App] MedRAG 2.0 server running. Open http://localhost:5000 in your browser.")
    app.run(debug=True, port=5000, use_reloader=False)
