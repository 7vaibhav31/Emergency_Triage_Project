# 🩺 MedRAG 2.0 — Code Architecture & Project Structure Guide

Welcome to **MedRAG 2.0: Evidence-Grounded Medical Document Assistant**!

The project is structured into three clean, dedicated folders:

```
RAG/
├── backend/            ➔ All Python RAG, LangChain, LangGraph, and Flask API code
│   ├── app.py          ➔ Flask Web Server & REST Endpoints
│   ├── config.py       ➔ Configuration & Environment (.env) Loader
│   ├── ingestion.py    ➔ LangChain Document Loaders (PyPDFLoader, TextLoader) & Splitters
│   ├── rag_engine.py   ➔ FAISS Vector Store & Semantic Retrieval (HuggingFaceEmbeddings)
│   ├── triage_graph.py ➔ LangGraph Decision Workflow & LCEL Chat Chain
│   ├── triage_service.py➔ Central Service Coordinator
│   ├── requirements.txt➔ Python dependencies
│   └── .env            ➔ API Keys & Environment Variables
├── frontend/           ➔ Single-page web application
│   └── index.html      ➔ Responsive Glassmorphism UI (HTML/CSS/JS)
├── documents/          ➔ Stored and uploaded medical documents
│   ├── prescription.txt
│   ├── sample_lab_report.pdf
│   └── document.txt
├── requirements.txt    ➔ Root-level dependency manifest
└── README.md           ➔ Project Overview
```

---

## 🧭 The 5-Step Backend Execution Sequence

```
┌─────────────────────────────────┐
│ 1. backend/ingestion.py         │ ➔ LangChain Document Loaders & Splitters (PyPDFLoader, TextLoader)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ 2. backend/rag_engine.py        │ ➔ FAISS Vector Store & Semantic Retrieval (HuggingFaceEmbeddings)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ 3. backend/triage_graph.py      │ ➔ LangGraph Decision Workflow (Guardrail, Retrieve, LCEL Generate)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ 4. backend/triage_service.py    │ ➔ Central Service Coordinator
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ 5. backend/app.py               │ ➔ Minimal Flask REST API (Serves frontend/ and saves to documents/)
└─────────────────────────────────┘
```

---

## 1. [`backend/ingestion.py`](file:///d:/project/RAG/backend/ingestion.py) — Document Ingestion & Classification
- **`load_documents(file_path)`**: Uses LangChain's native **`PyPDFLoader`** (for `.pdf` files) and **`TextLoader`** (for `.txt` files) to load standard LangChain `Document` objects with page and source metadata.
- **`classify_document(content)`**: A fast, keyword-based classifier that identifies whether the document is a:
  - *Prescription* (e.g. mentions "Rx", "dosage", "tablet")
  - *Lab Report* (e.g. mentions "hemoglobin", "platelet", "reference range")
  - *Discharge Summary* (e.g. mentions "hospital course", "admission date")
  - *Medical Test Report* (e.g. mentions "mri", "x-ray", "findings")
  - *General Medical Document* (fallback)
- **`split_documents_into_chunks(docs)`**: Splits LangChain `Document` objects using `RecursiveCharacterTextSplitter.split_documents()` and enriches metadata with clinical section titles and chunk IDs.

---

## 2. [`backend/rag_engine.py`](file:///d:/project/RAG/backend/rag_engine.py) — Semantic Vector Retrieval Engine (HuggingFace + FAISS)
- **`HuggingFaceEmbeddings`**: Uses `sentence-transformers/all-MiniLM-L6-v2` to convert text chunks into 384-dimensional dense semantic vectors.
- **`FAISS` (Facebook AI Similarity Search)**: In-memory vector database that performs fast similarity search over chunk embeddings.
- **`RAGEngine`**:
  - **`ingest_documents(documents)`**: Takes LangChain `Document` objects directly and builds an in-memory index with `FAISS.from_documents()`.
  - **`retrieve(query, top_k=3)`**: Calls FAISS `similarity_search_with_score(query, k=top_k)` to retrieve top semantic chunks and calculates similarity confidence scores.
  - **`format_context(retrieved_chunks)`**: Formats evidence excerpts with citations:
    `[Excerpt 1 | Source: report.pdf, Page 2 | Section: Lab Results] ...`

---

## 3. [`backend/triage_graph.py`](file:///d:/project/RAG/backend/triage_graph.py) — The LangGraph Workflow & LCEL Chain
Organizes the medical Q&A process into a clean 3-node LangGraph state graph:

```
[START] ──► [guardrail] ────(vague / diagnostic request)────► [END] (Refusal / Clarify)
                 │
            (valid query)
                 ▼
            [retrieve]  (semantic vector retrieval via FAISS)
                 ▼
            [generate]  (PromptTemplate | ChatOpenAI | StrOutputParser)
                 ▼
               [END]
```

### State Fields (`TriageState`):
- `query`: The user's input question.
- `document_category`: The detected document type (e.g., Prescription).
- `route`: Query routing status (`retrieve` or `stop`).
- `retrieved_chunks`: List of matching chunk dictionaries.
- `formatted_context`: Formatted evidence excerpts passed to the LLM.
- `answer`: Generated response.
- `citations`: Unique source and page citations (`["prescription.txt, Page 1"]`).
- `metrics`: Telemetry dictionary (latency ms, tokens, confidence score, tokens/sec).

### Nodes:
1. **`guardrail_node`**: Catches vague queries ("help") or diagnosis requests ("diagnose me") instantly, avoiding unnecessary LLM calls.
2. **`retrieve_node`**: Searches FAISS for top semantic chunks and extracts unique citations.
3. **`generate_node`**: Executes the LangChain LCEL runnable pipeline (`QA_PROMPT | _LLM | StrOutputParser()`), invoking the model and computing telemetry metrics.

---

## 4. [`backend/triage_service.py`](file:///d:/project/RAG/backend/triage_service.py) — Service Coordinator
- `ingest_file(file_path)`: Runs `ingestion.py` and indexes chunks into `rag_engine`.
- `ingest_raw_text(text)`: Parses raw text input into LangChain `Document` chunks and indexes into `rag_engine`.
- `answer_medical_question(query)`: Prepares the initial state and invokes the LangGraph pipeline (`graph.invoke(...)`).

---

## 5. [`backend/app.py`](file:///d:/project/RAG/backend/app.py) — Minimal Flask Web Application
- `GET /`: Serves the user interface directly from `frontend/index.html`.
- `POST /api/upload`: Saves uploaded `.txt`/`.pdf` files into the `documents/` folder and indexes them.
- `POST /api/qa`: Executes the LangGraph Q&A workflow and returns answer, citations, and telemetry.
