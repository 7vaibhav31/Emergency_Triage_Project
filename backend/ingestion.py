"""
ingestion.py
============
Handles reading and parsing medical documents (.txt, .pdf, and raw text) using
LangChain Document Loaders and Text Splitters.
Enriches chunks with page and section metadata, and classifies the document type.
"""

import os
from typing import List, Union
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(file_path: str) -> List[Document]:
    """
    Loads documents using LangChain's native document loaders (PyPDFLoader, TextLoader).
    Returns a list of LangChain Document objects.
    """
    filename = os.path.basename(file_path)
    if not os.path.exists(file_path):
        print(f"[Ingestion Error] File not found: {file_path}")
        return []

    try:
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for i, doc in enumerate(docs, start=1):
                # Ensure page number is 1-indexed and source is clean filename
                doc.metadata["page"] = doc.metadata.get("page", i - 1) + 1
                doc.metadata["source"] = filename
            return docs
        else:
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["page"] = 1
                doc.metadata["source"] = filename
            return docs
    except Exception as e:
        print(f"[Ingestion Error] Failed to load {filename} with LangChain loader: {e}")
        # Fallback reading
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                if content:
                    return [Document(page_content=content, metadata={"source": filename, "page": 1})]
        except Exception as inner_e:
            print(f"[Ingestion Error] Fallback reading failed for {filename}: {inner_e}")
        return []


def classify_document(content: Union[str, List[Document]]) -> str:
    """
    Simple, fast rule-based classifier that determines the category of the medical document.
    Works on raw string or a list of LangChain Documents.
    """
    if isinstance(content, list):
        text = " ".join(doc.page_content for doc in content)
    else:
        text = str(content)

    text_lower = text.lower()

    # Rule 1: Prescription check
    rx_keywords = ["rx", "prescription", "sig:", "take 1 tablet", "dosage", "refill", "dispense", "mg daily"]
    if any(k in text_lower for k in rx_keywords):
        return "Prescription"

    # Rule 2: Lab report check
    lab_keywords = ["hemoglobin", "wbc", "rbc", "platelet", "reference range", "specimen", "lab results", "fasting blood sugar", "lipid panel"]
    if any(k in text_lower for k in lab_keywords):
        return "Lab Report"

    # Rule 3: Discharge summary check
    discharge_keywords = ["discharge summary", "hospital course", "admission date", "discharge date", "condition on discharge"]
    if any(k in text_lower for k in discharge_keywords):
        return "Discharge Summary"

    # Rule 4: Medical test / Imaging report
    test_keywords = ["mri", "ct scan", "x-ray", "ultrasound", "impression:", "findings:", "radiology report", "ecg", "ekg"]
    if any(k in text_lower for k in test_keywords):
        return "Medical Test Report"

    # Fallback default
    return "General Medical Document"


def split_documents_into_chunks(
    docs: List[Document],
    chunk_size: int = 450,
    chunk_overlap: int = 60
) -> List[Document]:
    """
    Splits LangChain Document objects into smaller semantic chunks using
    RecursiveCharacterTextSplitter and enriches each chunk with section headers and IDs.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    split_docs = splitter.split_documents(docs)
    enriched_chunks: List[Document] = []

    for index, doc in enumerate(split_docs, start=1):
        content = doc.page_content.strip()
        if len(content) < 15:
            continue

        # Extract clinical section header if available (e.g., "DIAGNOSIS:", "MEDICATIONS:")
        first_line = content.split("\n")[0][:40]
        section = first_line if ":" in first_line else "General"

        metadata = dict(doc.metadata)
        metadata.update({
            "id": f"chunk_{index}",
            "page": metadata.get("page", 1),
            "source": metadata.get("source", "document"),
            "section": section
        })

        enriched_chunks.append(Document(page_content=content, metadata=metadata))

    return enriched_chunks


def load_and_split_file(file_path: str, chunk_size: int = 450, chunk_overlap: int = 60) -> List[Document]:
    """
    Convenience pipeline: loads file via LangChain DocumentLoader and splits into enriched chunks.
    """
    docs = load_documents(file_path)
    if not docs:
        return []
    return split_documents_into_chunks(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def split_raw_text(text: str, filename: str = "document.txt", chunk_size: int = 450, chunk_overlap: int = 60) -> List[Document]:
    """
    Wraps raw text into a LangChain Document and splits into enriched chunks.
    """
    doc = Document(page_content=text, metadata={"source": filename, "page": 1})
    return split_documents_into_chunks([doc], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
