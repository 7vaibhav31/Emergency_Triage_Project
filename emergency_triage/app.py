"""
app.py
======
MedRAG 2.0 Web Application & API Entry Point for Vercel Serverless.
"""

import io
import os
import re
from flask import Flask, request, jsonify, render_template, render_template_string
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
    return "Clinical Medical Document"


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
    Ingests an uploaded medical document (.pdf or .txt) completely in memory.
    Safe for read-only serverless filesystems like Vercel.
    """
    extracted_text = ""
    filename = "document.txt"

    # Option A: Multi-part file upload
    if "file" in request.files:
        uploaded_file = request.files["file"]
        if uploaded_file and uploaded_file.filename:
            filename = uploaded_file.filename
            file_bytes = uploaded_file.read()

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
                    return jsonify({"error": f"Failed to extract PDF text: {str(e)}"}), 400
            else:
                try:
                    extracted_text = file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    extracted_text = file_bytes.decode("latin-1", errors="ignore")

    # Option B: JSON payload
    if not extracted_text:
        data = request.get_json(silent=True) or {}
        extracted_text = data.get("text", "").strip()
        filename = data.get("filename", filename)

    if not extracted_text.strip():
        return jsonify({"error": "No readable text found in uploaded document."}), 400

    category = detect_document_category(extracted_text)
    chunks_count = qa_service.ingest_document(extracted_text, filename=filename, category=category)

    return jsonify({
        "message": f"Successfully indexed into {chunks_count} context fragments.",
        "chunks": chunks_count,
        "category": category,
        "filename": filename
    })


@app.route("/api/qa", methods=["POST"])
def qa_document():
    """Answers a medical query grounded in the uploaded document."""
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "No question provided."}), 400

    result = qa_service.answer_medical_question(query)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
