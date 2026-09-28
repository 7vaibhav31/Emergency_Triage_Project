# 🩺 MedRAG 2.0 — Evidence-Grounded Medical Document Assistant

[![Live Demo](https://img.shields.io/badge/Live-Deployment-ff6b9d?style=for-the-badge&logo=vercel&logoColor=white)](https://emergency-triage-project.vercel.app/#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://python.langchain.com/)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-0468FF?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![HuggingFace](https://img.shields.io/badge/Embeddings-HuggingFace%20MiniLM-yellow?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![LLM](https://img.shields.io/badge/Inference-NVIDIA%20NIM%20%2F%20Llama%203.2-76b900?style=for-the-badge&logo=nvidia&logoColor=white)](https://build.nvidia.com/)

An advanced, evidence-grounded Medical Retrieval-Augmented Generation (RAG) assistant designed for healthcare workflows. MedRAG 2.0 allows users to upload clinical documents (such as lab reports, prescriptions, and discharge summaries) and query an AI that is **strictly constrained to the provided source text**, eliminating hallucinations with an automated verification loop.

---

## 🌟 Key Capabilities

- 📄 **Multi-Format Ingestion & Classification:** Ingests both `.pdf` (multi-page extraction via `pypdf`) and `.txt` files, automatically categorizing them into *Prescription*, *Lab Report*, *Discharge Summary*, or *Medical Test Report*.
- ⚡ **Semantic Vector Search with FAISS & HuggingFace Embeddings:** Documents are split using LangChain's `RecursiveCharacterTextSplitter` (`chunk_size=450`, `chunk_overlap=60`) to preserve clinical context. Dense 384-dimensional semantic embeddings are generated via HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` and indexed into an in-memory **FAISS** vector store, enabling accurate conceptual matching across medical synonyms.
- 🧠 **LangGraph Agentic State Workflow:** Manages clinical inquiry through an explicit state graph:
  - **Query Understanding & Routing:** Intelligently routes inputs into document Q&A (`rag`), user clarification requests (`clarify`), or unsupported diagnostic questions (`unsupported`).
  - **Evidence Grounding:** Enforces strict excerpt-based answers with granular source and page citations (`[Source: file.pdf, Page 1]`).
  - **Evidence-Grounded Prompting:** Each LLM call is formatted through a **LangChain `PromptTemplate`** for consistent, structured clinical Q&A.
  - **Self-Verification Loop:** Validates generated responses against retrieved document evidence; safely retries or falls back to medical disclaimers if ungrounded claims are detected.
  - **Clinical Safety Refusals:** Refuses to issue unqualified medical diagnoses, directing patients to licensed practitioners.
- 📊 **Real-Time Telemetry & Metrics:** Tracks end-to-end latency (ms), token generation speed, and algorithmic retrieval confidence for every response.
- 🎨 **Modern SPA Interface:** Built with responsive vanilla HTML5/CSS3 (warm gradient glassmorphism palette) and vanilla JavaScript with dynamic typing indicators and Markdown rendering.

---

## 🏗️ System Architecture & Workflow

```
                        User Clinical Query
                               │
                               ▼
                   ┌───────────────────────┐
                   │ understand_and_route  │
                   └───────────┬───────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
      (clarification)    (unsupported)          (rag)
            │                  │                  │
            │                  ▼                  ▼
            │           ┌──────────────┐   ┌──────────────┐
            │           │ safe_refusal │   │   retrieve   │
            │           └──────┬───────┘   └──────┬───────┘
            │                  │                  ▼
            │                  │           ┌──────────────┐
            │                  │           │generate_ans  │
            │                  │           └──────┬───────┘
            │                  │                  ▼
            │                  │           ┌──────────────┐
            │                  │           │verify_evid.  │
            │                  │           └──────┬───────┘
            │                  │              /        \
            │                  │          (valid)    (invalid)
            │                  │            │            │
            │                  │            │       [retry once]
            │                  │            │            │
            │                  │            │      (still invalid)
            │                  │            │            ▼
            │                  └────────────┼────> [safe_refusal]
            ▼                               ▼            ▼
   User Clarification                  Final Answer  Safe Medical Notice
   (e.g., Vague Query)                 + Citations   + Doctor Disclaimer
```

---

## 📁 Repository Structure

```text
RAG/
├── backend/                       # Python Flask, RAG, and LangGraph modules
│   ├── app.py                     # Minimal Flask REST API
│   ├── config.py                  # API keys & configuration loader
│   ├── ingestion.py               # LangChain loaders (PyPDFLoader, TextLoader) & splitters
│   ├── rag_engine.py              # FAISS vector store & HuggingFace semantic embeddings
│   ├── triage_graph.py            # LangGraph decision workflow & LCEL chat chain
│   ├── triage_service.py          # Central service coordinator connecting RAG + LangGraph
│   └── requirements.txt           # Python backend dependencies
├── frontend/                      # Web user interface
│   └── index.html                 # Single-Page Application interface
├── documents/                     # Sample clinical documents for testing
│   ├── document.txt
│   ├── prescription.txt
│   └── sample_lab_report.pdf
├── requirements.txt               # Root Python dependency manifest
├── understanding_code.md          # In-depth architectural and code-reading guide
└── README.md                      # Primary project documentation
```

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend** | HTML5, CSS3, JavaScript | Modern glassmorphism SPA, markdown rendering, responsive layouts |
| **Backend API** | Flask | REST endpoints (`/api/upload`, `/api/qa`) |
| **Orchestration** | LangGraph | State machine managing query routing, RAG retrieval, and verification |
| **Document Splitting & Prompts** | LangChain (`langchain-text-splitters`, `langchain-core`) | Overlap-aware chunking (`RecursiveCharacterTextSplitter`) & standardized prompt templates |
| **Vector Retrieval** | FAISS + HuggingFace Embeddings (`all-MiniLM-L6-v2`) | High-speed dense semantic vector similarity search |
| **PDF Extraction** | PyPDF | Multi-page text and metadata extraction from clinical PDFs |
| **LLM Inference** | NVIDIA NIM API (`meta/llama-3.2-11b-vision-instruct`) | High-speed, evidence-grounded clinical reasoning |

---

## 🚀 Getting Started Locally

### 1. Clone the Repository
```bash
git clone https://github.com/7vaibhav31/Emergency_Triage_Project.git
cd Emergency_Triage_Project
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in `backend/` (or root):

```env
NVIDIA_API_KEY="your_nvidia_api_key_here"
NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"
MODEL_NAME="meta/llama-3.2-11b-vision-instruct"
```

### 5. Launch the Server
```bash
cd backend
python app.py
```

Open your browser and visit: **[http://localhost:5000](http://localhost:5000)**

---

## 🧪 Testing the Application

1. **Upload a Sample Document:**
   - Drag & drop or select `documents/sample_lab_report.pdf` or `documents/prescription.txt`.
   - The system automatically classifies the document (e.g., *Prescription* or *Lab Report*) and chunks it with page references.
2. **Ask Questions:**
   - *"What is the prescribed dosage for Amoxicillin?"*
   - *"What was the patient's Hemoglobin level?"*
3. **Test Safety & Refusals:**
   - Ask an ungrounded or diagnostic question: *"Diagnose my chest pain and tell me what surgery I need."*
   - Observe the LangGraph routing trigger a safe medical disclaimer advising the user to consult a licensed physician.

---

## 👥 Contributors

- **[Vaibhav Sharma](https://github.com/7vaibhav31)** — B.Tech CSE (AI/ML) | RAG Architectures & LLM Engineering
- **[Bhaskar Mishra](https://github.com/Bhaskar7462)** — B.Tech CSE | Backend Development & Machine Learning
