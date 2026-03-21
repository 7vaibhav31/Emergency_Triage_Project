import time
from openai import OpenAI
from rag_engine import RAGEngine

# Key hardcoded directly — no environment variable can override this
API_KEY = "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"
MODEL   = "meta/llama-3.1-8b-instruct"

QA_SYSTEM_PROMPT = """You are an advanced medical document assistant. 
Answer the user's questions strictly based on the provided document excerpts.
Be thorough, accurate, and professional. Structure your answer well using bullet points if helpful. Do NOT hallucinate data not found in the text."""

class DocumentQAService:
    def __init__(self):
        self.rag = RAGEngine()

    def ingest_document(self, text: str) -> int:
        return self.rag.ingest(text)
        
    def answer_medical_question(self, query: str) -> dict:
        start_time = time.time()
        
        # Single clean client — no duplicate Authorization header
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://integrate.api.nvidia.com/v1"
        )
        
        # Reduced top_k to 2 to decrease context window processing time (lower latency)
        retrieved   = self.rag.retrieve(query=query, top_k=2)
        rag_context = self.rag.format_context(retrieved)

        full_prompt = f"""{QA_SYSTEM_PROMPT}

{rag_context}

---
Based strictly on the medical excerpts above, answer the following clinical query:
{query}"""

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=350,  # Increased to allow detailed, structured responses
            messages=[
                {"role": "user", "content": full_prompt},
            ],
        )

        answer     = response.choices[0].message.content
        latency_ms = int((time.time() - start_time) * 1000)
        
        avg_score  = sum(c.get("score", 0) for c in retrieved) / max(len(retrieved), 1)
        confidence = min(int(avg_score * 100) + 50, 99) 
        
        total_tokens   = int(len(full_prompt.split()) * 1.3) + int(len(answer.split()) * 1.3)
        tokens_per_sec = round((total_tokens / latency_ms) * 1000, 2) if latency_ms > 0 else 0

        return {
            "answer":        answer,
            "latency_ms":    latency_ms,
            "confidence":    confidence,
            "chunks_used":   len(retrieved),
            "total_tokens":  total_tokens,
            "tokens_per_sec": tokens_per_sec
        }
