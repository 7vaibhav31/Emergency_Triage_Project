# 🩺 MedRAG — AI Doctor Assistant

**Live Deployment:** [https://emergency-triage-project.vercel.app/#](https://emergency-triage-project.vercel.app/#)

An advanced medical Retrieval-Augmented Generation (RAG) system built to act as an intelligent assistant for healthcare professionals. Upload clinical notes or prescriptions, and ask an AI questions grounded *strictly* in your uploaded document.

---

## 🌟 Features

- **Document Analysis:** Upload `.txt` clinical documents (prescriptions, patient histories).
- **Zero Hallucination RAG:** The AI acts as a reading-comprehension engine over your specific document using local TF-IDF vectorization and Cosine Similarity.
- **Interactive Multi-Page Chat:** A beautiful Single-Page Application (SPA) with a warm modern gradient UI, markdown rendering, and animated transitions.
- **Live Telemetry:** Real-time metrics dashboard tracking latency (ms), token speed (tokens/sec), and algorithmic confidence.
- **Free Vercel Deployment:** Embeddings are generated using blazing-fast, lightweight `scikit-learn` algorithms natively in Python to bypass Vercel's strict 250MB serverless constraints.

---

## 🏗️ Architecture Stack

- **Frontend:** Vanilla HTML5, CSS3 (Glassmorphism), JavaScript
- **Backend:** Python, Flask
- **RAG Engine:** Scikit-Learn (TF-IDF Vectorization)
- **AI Inference:** OpenRouter API (`arcee-ai/trinity-large-preview:free`)
- **Hosting:** Vercel Serverless

---

## 👥 Meet the Team

Built with ❤️ by passionate engineers focusing on real-world AI applications:
- **[Vaibhav Sharma](https://github.com/7vaibhav31)** (B.Tech CSE AI/ML) — RAG & LLMs
- **[Bhaskar Mishra](https://github.com/Bhaskar7462)** (3rd Year B.Tech) — Backend & ML

---

## 🚀 How to Run Locally

### 1. Clone & Open
Clone the repository and open the folder in your terminal.

### 2. Virtual Environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
The OpenRouter API key is securely handled by the application configuration. Simply start the server:
```bash
python app.py
```

### 5. Open in Web Browser
Go to **http://localhost:5000** and upload `prescription.txt` to start testing!
