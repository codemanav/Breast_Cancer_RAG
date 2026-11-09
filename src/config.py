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

# Source data
DATA_DIR = PROJECT_ROOT / "data"
ESSAY_DIR = DATA_DIR / "essays"
EVIDENCE_FILE = DATA_DIR / "evidence" / "cards.json"
PAPERS_DIR = DATA_DIR / "papers"  # research-paper PDFs (not committed)
PARTICIPANTS_FILE = DATA_DIR / "participants.csv"
RESIDENCES_FILE = DATA_DIR / "residences.csv"
# PFAS Contamination Site Tracker (Northeastern University PFAS Project Lab).
# Its license forbids redistribution, so it lives outside version control.
PFAS_FILE = Path(os.getenv("PFAS_FILE", DATA_DIR / "external" / "pfas_sites.xlsx"))

# Generated outputs
PROCESSED_DIR = DATA_DIR / "processed"
DOCUMENTS_FILE = PROCESSED_DIR / "documents.jsonl"
LITERATURE_FILE = PROCESSED_DIR / "literature.jsonl"
STORY_CARDS_FILE = PROCESSED_DIR / "story_cards.jsonl"
EXPOSURE_FILE = PROCESSED_DIR / "exposure.json"
COMPARISONS_FILE = PROCESSED_DIR / "comparisons.jsonl"
SISTER_COMPARISONS_FILE = PROCESSED_DIR / "sister_comparisons.jsonl"
RANKING_FILE = PROCESSED_DIR / "ranking.json"
RESULTS_FILE = PROCESSED_DIR / "rag_results.json"

# Essays used by the single-question demo (main.py): (file name, display name)
ESSAYS = [("maryanne.docx", "MaryAnne"), ("sue.docx", "Sue")]

# Vector database
CHROMA_DIR = str(PROJECT_ROOT / "db" / "chroma")
COLLECTION_NAME = "breast_cancer_rag"  # main.py demo: essays + evidence
LITERATURE_COLLECTION = "literature"  # pipeline.py judge: evidence cards + papers

# Models
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")
RETRIEVAL_K = 4

# Exposure
PFAS_RADIUS_KM = float(os.getenv("PFAS_RADIUS_KM", "5"))


def require_api_key() -> None:
    """Fail early with a clear message when no OpenAI key is configured."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it or copy .env.example to .env "
            "and add your key."
        )
