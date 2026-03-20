# ============================================================
# triage_service.py  (OpenRouter / Llama version)
#
# Uses the openai library pointed at OpenRouter's API,
# so we can use free models like meta-llama/llama-3.3-70b-instruct.
#
# YOUR OPENROUTER API KEY:
# Get it from https://openrouter.ai/keys
# Then set it:
#   Windows:   set OPENROUTER_API_KEY=your_key_here
#   Mac/Linux: export OPENROUTER_API_KEY=your_key_here
# ============================================================

import json
import time
import os
from openai import OpenAI
from rag_engine import RAGEngine


# ── Configure OpenRouter client ───────────────────────────────
client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

MODEL = "arcee-ai/trinity-large-preview:free"


# ── System prompt ─────────────────────────────────────────────
BASE_SYSTEM_PROMPT = """You are an emergency medical triage AI assistant embedded in a real-time clinical decision support system. You assist qualified medical professionals only.

RULES:
1. Always respond with ONLY a valid JSON object — no text before or after, no markdown fences.
2. When symptoms are ambiguous, assign the HIGHER priority — never downgrade due to missing data.
3. Actions must be ordered by urgency using standard clinical terminology.
4. Do not recommend specific drug doses — state drug name and route only.
5. Confidence (0–100) reflects data completeness. Always respond even at low confidence.
6. Keep rationale under 60 words.
7. You are a decision-support tool. All clinical decisions rest with the treating physician.

OUTPUT FORMAT (strict — return ONLY this JSON, nothing else):
{
  "priority": "CRITICAL" | "MODERATE" | "LOW",
  "diagnosis": "Most probable diagnosis — 1 to 2 sentences.",
  "actions": ["action 1", "action 2", "..."],
  "confidence": <integer 0-100>,
  "rationale": "Brief clinical reasoning, max 60 words."
}"""


class TriageService:

    def __init__(self):
        # Initialize RAG engine (unchanged)
        self.rag = RAGEngine()

    def run_triage(self, symptoms: str, history: str = "") -> dict:
        """
        Full triage pipeline: RAG retrieval → prompt → Llama via OpenRouter → parse.
        """

        start_time = time.time()

        # ── STEP 1: RAG Retrieval (unchanged) ─────────────────
        retrieved = self.rag.retrieve(query=symptoms, top_k=3)
        rag_context = self.rag.format_context(retrieved)

        # ── STEP 2: Build the full prompt ─────────────────────
        history_section = (
            f"\n<patient_history>\n{history.strip()}\n</patient_history>"
            if history.strip() else ""
        )

        full_prompt = f"""{rag_context}

---

Use the retrieved protocols above to inform your triage assessment.
Now analyze this patient:

<symptoms>
{symptoms.strip()}
</symptoms>
{history_section}

Respond with ONLY the JSON object. No extra text."""

        # ── STEP 3: Call Llama via OpenRouter ─────────────────
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": BASE_SYSTEM_PROMPT},
                {"role": "user",   "content": full_prompt},
            ],
        )

        # ── STEP 4: Parse JSON from response ──────────────────
        raw_text = response.choices[0].message.content
        clean_text = (
            raw_text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        result = json.loads(clean_text)

        # ── STEP 5: Add metadata ───────────────────────────────
        latency_ms = int((time.time() - start_time) * 1000)
        result["latency_ms"] = latency_ms
        result["retrieved_protocols"] = [
            {"category": c["category"], "score": round(c["score"], 3)}
            for c in retrieved
        ]

        return result
