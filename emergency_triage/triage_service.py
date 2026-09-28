"""
triage_service.py
=================
MedRAG multi-document inference and retrieval coordinator.
Powered by Google Gemini Free Tier via OpenAI-compatible endpoint.
"""

import os
import time
from openai import OpenAI
from rag_engine import RAGEngine

QA_SYSTEM_PROMPT = """You are MedRAG, an intelligent medical document assistant.
Your job is to explain and summarize information strictly based on the provided document excerpts.

IMPORTANT RULES:
1. Ground every statement strictly in the provided excerpts.
2. If an excerpt mentions a value (e.g. hemoglobin, dosage, blood pressure), report it accurately.
3. If information is NOT mentioned in the excerpts, clearly state that it is not available in the documents.
4. If multiple documents are provided (e.g. prescription and lab report), synthesize findings across them and cite the specific source document and page number.
5. Do NOT diagnose diseases, prescribe medication, or give definitive medical advice.
6. Always advise the patient to consult their licensed doctor for clinical decisions.
7. Present your answer cleanly with bullet points if helpful."""


def resolve_ai_client(api_key: str = None):
    """
    Returns (client, model_name, provider_name).
    Configured for Google Gemini Free Tier with fallback support for NVIDIA NIM.
    """
    # 1. Check for Gemini Key (passed from UI or environment)
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key and (key.startswith("AIza") or not key.startswith("nvapi-")):
        return (
            OpenAI(
                api_key=key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            ),
            os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            "Google Gemini 1.5 Flash (Free Tier)"
        )

    # 2. Check for NVIDIA NIM key
    nvidia_key = api_key if (api_key and api_key.startswith("nvapi-")) else os.getenv("NVIDIA_API_KEY")
    if nvidia_key:
        return (
            OpenAI(
                api_key=nvidia_key,
                base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
            ),
            os.getenv("MODEL_NAME", "meta/llama-3.2-11b-vision-instruct"),
            "NVIDIA NIM"
        )

    # 3. Default to Google Gemini endpoint
    return (
        OpenAI(
            api_key=key or "",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        ),
        "gemini-1.5-flash",
        "Google Gemini 1.5 Flash"
    )


class DocumentQAService:
    def __init__(self):
        self.rag = RAGEngine()

    def ingest_document(self, text: str, filename: str = "document.txt", category: str = "General Medical Document") -> int:
        return self.rag.ingest(text, filename=filename, category=category)

    def get_documents(self) -> list:
        return self.rag.documents

    def clear_documents(self):
        self.rag.clear()

    def remove_document(self, filename: str):
        self.rag.remove_document(filename)

    def answer_medical_question(self, query: str, api_key: str = None) -> dict:
        start_time = time.time()

        # Guardrail: Check if documents have been uploaded
        if not self.rag.chunks:
            return {
                "answer": "📄 **No documents uploaded yet.**\n\nPlease use the **Upload Documents** area in the left sidebar to upload your medical files (.pdf or .txt, such as prescriptions or lab reports). Once uploaded, I'll be ready to answer your questions!",
                "latency_ms": 10,
                "confidence": 100,
                "chunks_used": 0,
                "total_tokens": 25,
                "tokens_per_sec": 100.0,
                "citations": []
            }

        # Guardrail: Handle vague greetings
        q_clean = query.strip()
        if len(q_clean) < 3 or q_clean.lower() in ["hi", "hello", "hey", "help"]:
            doc_names = ", ".join([d["filename"] for d in self.rag.documents])
            return {
                "answer": f"Hello! I am MedRAG, powered by Google Gemini. I currently have **{len(self.rag.documents)} document(s)** indexed: *{doc_names}*.\n\nAsk me any question about medications, dosages, lab results, or clinical observations!",
                "latency_ms": 15,
                "confidence": 95,
                "chunks_used": 0,
                "total_tokens": 30,
                "tokens_per_sec": 120.0,
                "citations": []
            }

        client, model_name, provider_name = resolve_ai_client(api_key)

        retrieved = self.rag.retrieve(query=query, top_k=3)
        rag_context = self.rag.format_context(retrieved)

        citations = []
        for c in retrieved:
            src = c.get("source", "Document")
            pg = c.get("page", 1)
            citations.append(f"{src} (Page {pg})")

        full_prompt = f"""{QA_SYSTEM_PROMPT}

{rag_context}

---
Based strictly on the medical excerpts above, answer the following clinical query:
{query}"""

        try:
            response = client.chat.completions.create(
                model=model_name,
                max_tokens=500,
                temperature=0.2,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
            )
            answer = response.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            if "API key" in err_str or "400" in err_str or "401" in err_str or "403" in err_str or "INVALID_ARGUMENT" in err_str:
                answer = (
                    "⚠️ **Google Gemini Free Tier Key Required:**\n\n"
                    "To enable live medical responses for your documents:\n"
                    "1. Get your 100% free API key from **[Google AI Studio (aistudio.google.com)](https://aistudio.google.com/app/apikey)**.\n"
                    "2. Enter it in the **Google Gemini Key** box in the left sidebar (or set `GEMINI_API_KEY` in Vercel Environment Variables).\n"
                    "3. Submit your question to get instant, evidence-grounded answers!"
                )
            else:
                answer = f"⚠️ AI Service Notice: {err_str}"

        latency_ms = max(1, int((time.time() - start_time) * 1000))
        avg_score = sum(c.get("score", 0) for c in retrieved) / max(len(retrieved), 1)
        confidence = min(max(int(avg_score * 100) + 40, 50), 99)

        total_tokens = int(len(full_prompt.split()) * 1.3) + int(len(answer.split()) * 1.3)
        tokens_per_sec = round((total_tokens / latency_ms) * 1000, 2) if latency_ms > 0 else 0

        return {
            "answer": answer,
            "latency_ms": latency_ms,
            "confidence": confidence,
            "chunks_used": len(retrieved),
            "total_tokens": total_tokens,
            "tokens_per_sec": tokens_per_sec,
            "citations": citations,
            "provider": provider_name
        }
