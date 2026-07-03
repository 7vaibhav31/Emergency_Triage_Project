# MedRAG - AI-Powered Doctor Assistant

## Project Overview
This project is an advanced Medical Retrieval-Augmented Generation (RAG) system designed to act as an intelligent assistant for doctors. Built with a Flask backend and a modern vanilla HTML/CSS/JS frontend, the application allows medical professionals to upload clinical documents (like patient histories or prescriptions) and query an AI to get instant, highly accurate answers grounded strictly in the provided document, preventing AI hallucinations.

---

## 🏗️ Architecture & Flow (Start to End)

### 1. The Frontend (User Interface)
- **Visual Design & Typography:** Engineered with a warm, premium gradient design system (pink, yellow, and sky blue hues), the application leverages modern glassmorphism and the `Inter` font family to ensure an inviting yet highly professional aesthetic.
- **Dynamic Multi-Page SPA:** An interactive Single-Page Application (SPA) natively handles navigation across 4 distinct views with zero page reloads:
  1. **Landing Hero:** A full-viewport introduction with animated visual elements and feature highlights.
  2. **Document Upload:** A drag-and-drop interface for ingesting `.txt` medical notes and prescriptions.
  3. **RAG Chat Interface:** The core Q&A dashboard containing the chat window, real-time performance metrics, and dynamic typing indicators.
  4. **About Team:** A dedicated, full-screen profile page showcasing the primary architects.
- **Creator Profiles:**
  - **Vaibhav Sharma** (Final-year B.Tech CSE AI/ML Engineer passionate about RAG systems and LLMs)
  - **Bhaskar Mishra** (3rd Year B.Tech Backend Developer and ML enthusiast)
- **Data Entry & Flow:** Users start by uploading a clinical document. The file is sent via AJAX to the backend for indexing. Once successful, the UI transitions to the Chat Interface where the doctor can sequentially ask questions about the patient's data.

### 2. The Backend (Flask Application)
- **What it is:** A Python Flask server (`app.py`) acting as the API layer.
- **How it works:** 
  - The `@app.route("/")` endpoint serves the SPA application.
  - The `@app.route("/api/upload")` endpoint receives the document text and triggers the RAG ingestion process.
  - The `@app.route("/api/qa")` endpoint receives the user's chat queries, retrieves the answer, and calculates performance metrics.

### 3. The Retrieval-Augmented Generation (RAG) Engine
Once a document is uploaded, the magic of RAG ensures the AI only references the provided medical text:

- **The Problem:** General LLMs often hallucinate incorrect medical advice or invent patient data.
- **The RAG Solution:** We force the LLM to act purely as a reading-comprehension engine over the specific document the doctor provided.

**Step-by-Step RAG Flow (`rag_engine.py`):**
1. **Dynamic Ingestion:** The uploaded document is split into smaller, logical text "chunks" to prevent overwhelming the AI's context window.
2. **TF-IDF Vectorization:** The system passes these chunks through `scikit-learn`'s `TfidfVectorizer` to convert them into mathematical vectors (sparse arrays), storing them in memory for the session.
3. **Retrieval (Cosine Similarity):** When the doctor asks a question, that question is vectorized. The engine calculates the Cosine Similarity between the question vector and all document chunk vectors.
4. **Intelligent Context Pruning:** We retrieve only the Top 3 most relevant chunks to the question, ensuring maximum accuracy and minimizing API latency.

### 4. The Large Language Model (LLM) Inference (`triage_service.py`)
- We use the `OpenAI` client connected to **OpenRouter**, utilizing a completely free, fast model (`arcee-ai/trinity-large-preview:free`).
- **The Prompt:** The `DocumentQAService` takes the retrieved RAG chunks and the doctor's question, wrapping them in a strict system prompt. The prompt commands the AI to answer *only* based on the provided excerpts and clearly state if the information is missing.
- **The API Call:** This highly constrained payload is sent securely to OpenRouter.

### 5. Final Delivery & Metrics
- The LLM streams back the answer.
- The Flask app calculates critical real-time performance metrics:
  - **Latency (ms):** Total round-trip time of the API call.
  - **Token Speed:** Words generated per second.
  - **Algorithmic Confidence:** The average relevance score of the retrieved chunks.
- The response and metrics are sent back to the frontend, displaying the answer in the chat bubble while rendering the speed/confidence stats dynamically in the side panel.

---

## 🚀 Deployment Considerations (Vercel)
To allow the project to be fully deployable for free on Vercel Serverless Functions, we bypassed heavy embedding models (like PyTorch/SentenceTransformers, which exceed Vercel's 250MB limit). By utilizing Scikit-Learn's TF-IDF vectorization, the RAG engine achieves blazing-fast, lightweight similarity matching natively in Python, keeping the build size extraordinarily small while maintaining high retrieval accuracy for text-based clinical notes.
