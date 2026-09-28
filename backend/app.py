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
    Endpoint for uploading a medical document (.txt or .pdf).
    Saves the file into documents/ folder, extracts text, detects category, and indexes into RAG.
    """
    # Option A: File uploaded via standard form
    if "file" in request.files:
        uploaded_file = request.files["file"]
        if uploaded_file.filename:
            save_path = os.path.join(documents_dir, uploaded_file.filename)
            uploaded_file.save(save_path)
            result = qa_service.ingest_file(save_path)
            result["filename"] = uploaded_file.filename
            return jsonify({
                "message": f"Saved to documents/ and indexed into {result['chunks']} chunks.",
                "chunks": result["chunks"],
                "category": result["category"],
                "filename": uploaded_file.filename
            })

    # Option B: Raw text provided via JSON
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    filename = data.get("filename", "document.txt")

    if not text:
        return jsonify({"error": "No document provided. Please choose a valid file."}), 400

    # Save raw text document into documents_dir
    save_path = os.path.join(documents_dir, filename)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)

    result = qa_service.ingest_raw_text(text, filename=filename)
    return jsonify({
        "message": f"Saved to documents/ and indexed into {result['chunks']} chunks.",
        "chunks": result["chunks"],
        "category": result["category"],
        "filename": result["filename"]
    })


@app.route("/api/qa", methods=["POST"])
def answer_question():
    """
    Endpoint for asking questions about the active document.
    Executes the evidence-grounded LangGraph workflow.
    """
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Please enter a clinical question."}), 400

    # Ensure a document was uploaded first
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
