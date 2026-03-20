# 🚑 Emergency Triage AI — RAG + Claude Project

A real-time emergency triage assistant using RAG (Retrieval-Augmented Generation)
and Claude Sonnet 4. Built for local development in Antigravity IDE.

---

## PROJECT STRUCTURE

```
emergency_triage/
│
├── app.py                    ← Flask web server + frontend HTML
├── triage_service.py         ← RAG → Claude pipeline logic
├── rag_engine.py             ← Embedding + retrieval + ICP
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
│
└── rag_data/
    └── knowledge_base.py     ← Medical protocol chunks (your "database")
```

---

## HOW TO RUN (step by step)

### Step 1 — Open the folder in Antigravity
Open the `emergency_triage/` folder in Antigravity IDE.

### Step 2 — Create a virtual environment
In the terminal inside Antigravity:
```
python -m venv venv
```

### Step 3 — Activate the venv
Windows:
```
venv\Scripts\activate
```
Mac/Linux:
```
source venv/bin/activate
```

### Step 4 — Install dependencies
```
pip install -r requirements.txt
```
(First time takes 2-3 minutes — downloads the embedding model ~80MB)

### Step 5 — Set your Anthropic API key
Windows:
```
set ANTHROPIC_API_KEY=your_api_key_here
```
Mac/Linux:
```
export ANTHROPIC_API_KEY=your_api_key_here
```

### Step 6 — Run the app
```
python app.py
```

### Step 7 — Open in browser
Go to: http://localhost:5000

---

## HOW THE RAG PIPELINE WORKS

```
Patient Symptoms (text)
        ↓
[RAG Engine] — embed symptoms with sentence-transformer model
        ↓
[Cosine Similarity] — compare against 15 embedded protocol chunks
        ↓
[ICP] — keep top 3 most relevant chunks, discard the rest
        ↓
[Prompt Assembly] — system prompt + retrieved protocols + patient data
        ↓
[Claude Sonnet 4 API] — generates structured JSON triage
        ↓
[Flask] — returns JSON to browser
        ↓
[Frontend] — renders priority card + actions + RAG debug panel
```

---

## WHAT IS RAG? (simple explanation)

Without RAG:
  Claude answers from training data only → generic responses

With RAG:
  1. We have a knowledge base of medical protocols
  2. When symptoms come in, we find the MOST RELEVANT protocols
  3. We inject those into Claude's prompt
  4. Claude now has specific, grounded guidelines to reference
  → More accurate, more specific, less hallucination

## WHAT IS ICP? (Intelligent Context Pruning)

We have 15+ protocol chunks, but we only send 3 to Claude.
Why? Sending all 15 would:
  - Waste tokens (slower, more expensive)
  - Confuse Claude with irrelevant info

ICP = retrieve many → score by relevance → keep only the best → inject
This is what keeps responses fast (target < 500ms).

---

## ADDING MORE MEDICAL PROTOCOLS

Edit `rag_data/knowledge_base.py` and add a new dict to KNOWLEDGE_CHUNKS:

```python
{
    "id": "your_id",
    "category": "your_category",
    "text": "Your protocol text here..."
}
```

Restart the app — it will automatically embed the new chunk on startup.
