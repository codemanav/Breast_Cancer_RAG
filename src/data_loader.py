"""
Load essays (Word documents) and evidence cards (JSON) as LangChain Documents,
and save/load processed documents as JSON Lines.
"""

import json
from pathlib import Path

from docx import Document as Docx
from langchain_core.documents import Document

from src.config import DOCUMENTS_FILE, ESSAY_DIR, ESSAYS, EVIDENCE_FILE


def load_essay(path, source_name):
    """Load a Word (.docx) essay and return it as a LangChain Document."""
    doc = Docx(str(path))
    text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    return Document(page_content=text, metadata={"source": source_name, "type": "essay"})


def load_cards(path):
    """
    Load evidence cards from a JSON array, a single JSON object, or JSON Lines.
    Returns one Document per card.
    """
    content = Path(path).read_text(encoding="utf-8").strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in content.splitlines() if line.strip()]
    if isinstance(data, dict):
        data = [data]

    return [
        Document(
            page_content=json.dumps(card, indent=2, ensure_ascii=False),
            metadata={"source": card.get("title", "unknown"), "type": "evidence"},
        )
        for card in data
    ]


def load_all_documents(essay_dir=ESSAY_DIR, evidence_file=EVIDENCE_FILE):
    """Load both essays and all evidence cards."""
    essays = [load_essay(Path(essay_dir) / fname, name) for fname, name in ESSAYS]
    return essays + load_cards(evidence_file)


def save_documents(docs, path=DOCUMENTS_FILE):
    """Write Documents to a JSON Lines file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            record = {"page_content": d.page_content, "metadata": d.metadata}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_documents(path=DOCUMENTS_FILE):
    """Read Documents back from a JSON Lines file written by save_documents."""
    with Path(path).open(encoding="utf-8") as f:
        return [Document(**json.loads(line)) for line in f if line.strip()]
