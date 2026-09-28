"""
triage_service.py
=================
MedRAG multi-document inference and retrieval coordinator for Vercel deployment.
Connects the lightweight multi-document RAG engine with NVIDIA NIM API.
"""

import os
import time
from openai import OpenAI
from rag_engine import RAGEngine

API_KEY = os.getenv("NVIDIA_API_KEY", "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")
BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL = os.getenv("MODEL_NAME", "meta/llama-3.2-11b-vision-instruct")

QA_SYSTEM_PROMPT = """You are MedRAG, an intelligent medical document assistant.
Your job is to explain and summarize information strictly based on the provided document excerpts.

IMPORTANT RULES:
1. Ground every statement strictly in the provided excerpts.
2. If an excerpt mentions a value (e.g. hemoglobin, dosage, blood pressure), report it accurately.
3. If information is NOT mentioned in the excerpts, clearly state that it is not available in the documents.
4. If multiple documents are provided (e.g. prescription and lab report), synthesize the findings across them and cite the specific source document.
5. Do NOT diagnose diseases, prescribe medication, or give definitive medical advice.
6. Always advise the patient to consult their licensed doctor for clinical decisions.
7. Present your answer cleanly with bullet points if helpful."""


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

    def answer_medical_question(self, query: str) -> dict:
        start_time = time.time()

        # Check if any documents have been uploaded
        if not self.rag.chunks:
            return {
                "answer": "📄 **No documents uploaded yet.**\n\nPlease use the **Upload Documents** area in the left sidebar to upload one or more medical files (.pdf or .txt, such as prescriptions or lab reports). Once uploaded, I'll be ready to answer your questions!",
                "latency_ms": 12,
                "confidence": 100,
                "chunks_used": 0,
                "total_tokens": 30,
                "tokens_per_sec": 120.0,
                "citations": []
            }

        # Handle vague or short greetings
        q_clean = query.strip()
        if len(q_clean) < 3 or q_clean.lower() in ["hi", "hello", "hey", "help"]:
            doc_names = ", ".join([d["filename"] for d in self.rag.documents])
            return {
                "answer": f"Hello! I am MedRAG, your medical document assistant. I currently have **{len(self.rag.documents)} document(s)** indexed: *{doc_names}*.\n\nAsk me any question about medications, dosages, lab results, or clinical observations!",
                "latency_ms": 20,
                "confidence": 95,
                "chunks_used": 0,
                "total_tokens": 35,
                "tokens_per_sec": 130.0,
                "citations": []
            }

        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )

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
                model=MODEL,
                max_tokens=400,
                temperature=0.2,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
            )
            answer = response.choices[0].message.content
        except Exception as e:
            try:
                response = client.chat.completions.create(
                    model="meta/llama-3.1-8b-instruct",
                    max_tokens=400,
                    temperature=0.2,
                    messages=[
                        {"role": "user", "content": full_prompt},
                    ],
                )
                answer = response.choices[0].message.content
            except Exception as e2:
                answer = f"Error communicating with AI service: {str(e2)}"

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
            "citations": citations
        }
