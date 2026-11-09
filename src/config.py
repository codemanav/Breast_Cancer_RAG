"""
Project-wide configuration.

Values can be overridden with environment variables or a local `.env` file
(see `.env.example`).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Data locations
ESSAY_DIR = PROJECT_ROOT / "data" / "essays"
EVIDENCE_FILE = PROJECT_ROOT / "data" / "evidence" / "cards.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCUMENTS_FILE = PROCESSED_DIR / "story_cards.jsonl"
RESULTS_FILE = PROCESSED_DIR / "rag_results.json"

# Essays: (file name, display name)
ESSAYS = [("maryanne.docx", "MaryAnne"), ("sue.docx", "Sue")]

# Vector database
CHROMA_DIR = str(PROJECT_ROOT / "db" / "chroma")
COLLECTION_NAME = "breast_cancer_rag"

# Models
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")
RETRIEVAL_K = 4


def require_api_key() -> None:
    """Fail early with a clear message when no OpenAI key is configured."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it or copy .env.example to .env "
            "and add your key."
        )
