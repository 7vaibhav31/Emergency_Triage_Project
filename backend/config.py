"""
config.py
=========
Central configuration for MedRAG 2.0.
Loads API keys and settings from environment variables or .env file.
"""

import os
from dotenv import load_dotenv

# Current backend directory & project root directory
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)

# Load environment variables from backend/.env or root .env
load_dotenv(os.path.join(backend_dir, ".env"))
load_dotenv(os.path.join(project_root, ".env"))

# API Key & Model Configuration
API_KEY = os.getenv("NVIDIA_API_KEY")
BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta/llama-3.2-11b-vision-instruct")

if not API_KEY:
    raise ValueError(
        "NVIDIA_API_KEY is not set. Create a .env file in backend/ with:\n"
        "NVIDIA_API_KEY=your_key_here"
    )

# System prompt enforcing strict evidence-grounding and medical safety
QA_SYSTEM_PROMPT = """You are MedRAG, an intelligent medical document assistant.
Your job is to explain and summarize information strictly based on the provided document excerpts.

IMPORTANT RULES:
1. Ground every statement strictly in the provided excerpts.
2. If an excerpt mentions a value (e.g. hemoglobin, dosage, blood pressure), report it accurately.
3. If information is NOT mentioned in the excerpts, clearly state that it is not available in the document.
4. Do NOT diagnose diseases, prescribe medication, or give definitive medical advice.
5. Always advise the patient to consult their licensed doctor for clinical decisions.
6. Present your answer cleanly with bullet points if helpful."""
