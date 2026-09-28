"""
app.py
======
MedRAG 2.0 Web Application & API Entry Point for Vercel Serverless.
Supports multi-document upload and simultaneous retrieval across documents.
"""

import io
import os
from flask import Flask, request, jsonify, render_template
from triage_service import DocumentQAService

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, "templates")

app = Flask(__name__, template_folder=template_dir, static_folder=template_dir)
qa_service = DocumentQAService()


def detect_document_category(text: str) -> str:
    """Classifies uploaded clinical text into standardized healthcare document types."""
    t = text.lower()
    if any(k in t for k in ["rx", "prescription", "take 1 tablet", "dosage", "sig:", "mg daily", "refills"]):
        return "Prescription"
    elif any(k in t for k in ["hemoglobin", "wbc", "platelet", "serum", "creatinine", "lipid", "lab report", "specimen"]):
        return "Lab Report"
    elif any(k in t for k in ["discharge", "hospital admission", "discharge summary", "chief complaint"]):
        return "Discharge Summary"
    elif any(k in t for k in ["mri", "ct scan", "ultrasound", "x-ray", "echocardiogram"]):
        return "Medical Imaging / Diagnostic Report"
    return "Clinical Document"


@app.route("/")
def index():
    """Serves the MedRAG Single-Page Application."""
    index_html = os.path.join(template_dir, "index.html")
    if os.path.exists(index_html):
        try:
            return render_template("index.html")
        except Exception:
            with open(index_html, "r", encoding="utf-8") as f:
                return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}
    return "MedRAG 2.0 Assistant is online.", 200


@app.route("/api/upload", methods=["POST"])
def upload_document():
    """
    Ingests one or more uploaded medical documents (.pdf or .txt) in memory.
    Safe for read-only serverless filesystems like Vercel.
    """
    uploaded_files = request.files.getlist("files") or request.files.getlist("file")
    results = []

    if uploaded_files and any(f.filename for f in uploaded_files):
        for uploaded_file in uploaded_files:
            if not uploaded_file or not uploaded_file.filename:
                continue
            filename = uploaded_file.filename
            file_bytes = uploaded_file.read()
            extracted_text = ""

            if filename.lower().endswith(".pdf"):
                try:
                    import pypdf
                    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                    pages_text = []
                    for page_idx, page in enumerate(reader.pages, 1):
                        p_txt = page.extract_text() or ""
                        if p_txt.strip():
                            pages_text.append(f"--- Page {page_idx} ---\n{p_txt}")
                    extracted_text = "\n\n".join(pages_text)
                except Exception as e:
                    return jsonify({"error": f"Failed to extract PDF '{filename}': {str(e)}"}), 400
            else:
                try:
                    extracted_text = file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    extracted_text = file_bytes.decode("latin-1", errors="ignore")

            if extracted_text.strip():
                category = detect_document_category(extracted_text)
                count = qa_service.ingest_document(extracted_text, filename=filename, category=category)
                results.append({
                    "filename": filename,
                    "category": category,
                    "chunks": count
                })

    # Option B: Raw text payload fallback
    if not results:
        data = request.get_json(silent=True) or {}
        extracted_text = data.get("text", "").strip()
        filename = data.get("filename", "document.txt")
        if extracted_text:
            category = detect_document_category(extracted_text)
            count = qa_service.ingest_document(extracted_text, filename=filename, category=category)
            results.append({
                "filename": filename,
                "category": category,
                "chunks": count
            })

    if not results:
        return jsonify({"error": "No valid text or documents found to upload."}), 400

    total_chunks = len(qa_service.rag.chunks)
    all_docs = qa_service.get_documents()

    return jsonify({
        "message": f"Successfully indexed {len(results)} document(s) into {total_chunks} total context fragments.",
        "uploaded": results,
        "documents": all_docs,
        "total_chunks": total_chunks
    })


@app.route("/api/documents", methods=["GET"])
def get_documents():
    """Returns currently indexed documents."""
    return jsonify({
        "documents": qa_service.get_documents(),
        "total_chunks": len(qa_service.rag.chunks)
    })


@app.route("/api/documents/clear", methods=["POST"])
def clear_documents():
    """Clears all indexed documents from memory."""
    qa_service.clear_documents()
    return jsonify({
        "message": "All indexed documents have been cleared.",
        "documents": [],
        "total_chunks": 0
    })


@app.route("/api/documents/remove", methods=["POST"])
def remove_document():
    """Removes a specific document by filename."""
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    if filename:
        qa_service.remove_document(filename)
    return jsonify({
        "message": f"Document '{filename}' removed.",
        "documents": qa_service.get_documents(),
        "total_chunks": len(qa_service.rag.chunks)
    })


@app.route("/api/qa", methods=["POST"])
def qa_document():
    """Answers a medical query grounded across all indexed documents."""
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    api_key = data.get("api_key") or request.headers.get("X-Api-Key")

    if not query:
        return jsonify({"error": "No question provided."}), 400

    result = qa_service.answer_medical_question(query, api_key=api_key)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
