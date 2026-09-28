"""
triage_graph.py
===============
MedRAG LangGraph Decision Workflow.

Streamlined 3-step state graph powered by LangChain & LangGraph:
1. Guardrail -> Handles vague queries and diagnostic safety refusals
2. Retrieve  -> Fetches top semantic chunks & source citations from FAISS
3. Generate  -> Invokes LangChain ChatOpenAI LCEL chain with telemetry metrics
"""

import time
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from config import API_KEY, BASE_URL, MODEL_NAME, QA_SYSTEM_PROMPT
from rag_engine import RAGEngine

# Initialize LangChain Chat Model configured for NVIDIA NIM OpenAI-compatible API
_LLM = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.2,
    max_tokens=400,
)

# Standardized LangChain prompt template for evidence-grounded medical Q&A
QA_PROMPT = PromptTemplate.from_template(
    """{system_prompt}

Document Category: {category}

{context}

---
USER QUESTION: {query}

Provide a clear, factual answer based strictly on the excerpts above.
If the excerpts do not contain the answer, state: "Information not found in the uploaded document."
"""
)

# LangChain LCEL Runnable pipeline
QA_CHAIN = QA_PROMPT | _LLM | StrOutputParser()


# ==========================================
# 1. GRAPH STATE DEFINITION
# ==========================================
class TriageState(TypedDict, total=False):
    """
    The shared state dictionary flowing through each LangGraph node.
    Every node reads needed inputs from here and returns updated fields.
    """
    query: str                       # The raw clinical question submitted by the user
    document_category: str           # Auto-detected document type (e.g., 'Prescription', 'Lab Report')
    route: str                       # Routing flag: 'retrieve' (proceed to search) or 'stop' (fast exit/refusal)
    retrieved_chunks: List[Dict]     # Top matching chunks from FAISS with page, section, and text data
    formatted_context: str           # Formatted evidence excerpts with citations injected into prompt
    answer: str                      # Final generated response text from the LLM or refusal guardrail
    citations: List[str]             # List of unique document citations (e.g., ['prescription.txt, Page 1'])
    metrics: Dict[str, Any]          # Telemetry stats (latency_ms, confidence %, total_tokens, tokens_per_sec)


# ==========================================
# 2. WORKFLOW NODES
# ==========================================
def guardrail_node(state: TriageState) -> Dict[str, Any]:
    """
    Node 1: Clinical Guardrail & Input Validator.
    - Intercepts vague questions (e.g. 'help', 'explain') to ask for clarification.
    - Blocks unsafe medical requests (e.g. asking for medical diagnosis or new prescriptions).
    - Returns 'stop' route for fast responses without wasting LLM API calls.
    """
    q = state.get("query", "").strip().lower()

    # Check for overly vague inputs
    if len(q.split()) <= 2 or q in ["explain this", "help", "summary", "details", "explain"]:
        return {
            "route": "stop",
            "answer": "Could you please specify which section, test, medicine, or value you would like explained from the document?",
            "citations": [],
        }

    # Check for dangerous diagnostic / prescribing requests
    forbidden = [
        "diagnose", "what disease", "what illness", "prescribe",
        "which medicine should i take", "will i die", "predict future"
    ]
    if any(k in q for k in forbidden):
        return {
            "route": "stop",
            "answer": "I cannot provide a medical diagnosis or prescribe treatment. MedRAG only explains information present in your uploaded documents. Please consult a qualified healthcare professional.",
            "citations": [],
        }

    # Query is valid for document retrieval
    return {"route": "retrieve"}


def retrieve_node(state: TriageState, rag_engine: RAGEngine) -> Dict[str, Any]:
    """
    Node 2: Semantic Vector Retrieval.
    - Searches the FAISS vector index using dense HuggingFace embeddings.
    - Compiles deduplicated citation labels with exact page numbers.
    """
    chunks = rag_engine.retrieve(state.get("query", ""), top_k=3)
    citations = list({f"{c.get('source', 'Document')}, Page {c.get('page', 1)}" for c in chunks})
    return {
        "retrieved_chunks": chunks,
        "formatted_context": rag_engine.format_context(chunks),
        "citations": citations,
    }


def generate_node(state: TriageState) -> Dict[str, Any]:
    """
    Node 3: Answer Generation & Real-time Telemetry via LangChain Chain.
    - Uses the LangChain LCEL pipeline (QA_PROMPT | _LLM | StrOutputParser).
    - Computes real-time latency (ms), algorithmic confidence (%), and token generation speed.
    """
    start = time.time()
    chunks = state.get("retrieved_chunks", [])
    category = state.get("document_category", "General Medical Document")
    context = state.get("formatted_context", "")
    query = state.get("query", "")

    input_payload = {
        "system_prompt": QA_SYSTEM_PROMPT,
        "category": category,
        "context": context,
        "query": query,
    }

    try:
        answer = QA_CHAIN.invoke(input_payload).strip()
    except Exception as e:
        answer = f"Error during AI processing: {str(e)}"

    # Calculate real-time performance telemetry
    latency_ms = int((time.time() - start) * 1000)
    avg_score = sum(c.get("score", 0) for c in chunks) / max(len(chunks), 1)
    confidence = min(int(avg_score * 100) + 40, 99) if chunks else 0
    prompt_estimate_len = len(QA_SYSTEM_PROMPT) + len(category) + len(context) + len(query)
    tokens = int((prompt_estimate_len + len(answer)) / 4)
    tps = round((tokens / latency_ms) * 1000, 2) if latency_ms > 0 else 0

    return {
        "answer": answer,
        "metrics": {
            "latency_ms": latency_ms,
            "confidence": confidence,
            "chunks_used": len(chunks),
            "total_tokens": tokens,
            "tokens_per_sec": tps,
        },
    }

# ROUTE FUNCTION 
def route_guardrail(state: TriageState) -> str:
    """
    Decides the next node: exits to END if guardrail stopped, otherwise proceeds to retrieve.
    """
    return END if state.get("route") == "stop" else "retrieve"


# ==========================================
# 3. GRAPH COMPILER
# ==========================================
def build_medrag_graph(rag_engine: RAGEngine):
    """
    Constructs and compiles the 3-node LangGraph workflow:
    [START] -> [guardrail] -> (retrieve or END) -> [retrieve] -> [generate] -> [END]
    """
    graph = StateGraph(TriageState)

    # 1. Register nodes
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("retrieve", lambda s: retrieve_node(s, rag_engine))
    graph.add_node("generate", generate_node)

    # 2. Define workflow transitions
    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges("guardrail", route_guardrail)
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
