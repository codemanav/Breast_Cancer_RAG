"""
Load essays (Word documents), evidence cards (JSON), research papers (PDF),
and the participant/residence tables, and save/load documents as JSON Lines.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from docx import Document as Docx
from langchain_core.documents import Document

from src.config import (
    DOCUMENTS_FILE,
    ESSAY_DIR,
    ESSAYS,
    EVIDENCE_FILE,
    PAPERS_DIR,
    PARTICIPANTS_FILE,
    RESIDENCES_FILE,
)

# Literature document types: what the pairwise judge may cite as evidence.
LITERATURE_TYPES = ("evidence", "paper")


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


def load_papers(papers_dir=PAPERS_DIR, chunk_size=1500, chunk_overlap=200):
    """
    Load every PDF in `papers_dir` and split it into overlapping chunks.
    Each chunk keeps its paper (file stem) and page number for citation.
    Returns [] when the folder is missing or empty.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pypdf import PdfReader

    papers_dir = Path(papers_dir)
    if not papers_dir.is_dir():
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for pdf in sorted(papers_dir.glob("*.pdf")):
        pages = [
            Document(
                page_content=page.extract_text() or "",
                metadata={"source": pdf.stem, "type": "paper", "page": number},
            )
            for number, page in enumerate(PdfReader(str(pdf)).pages, start=1)
        ]
        chunks.extend(c for c in splitter.split_documents(pages) if c.page_content.strip())
    return chunks


def load_literature(evidence_file=EVIDENCE_FILE, papers_dir=PAPERS_DIR):
    """Evidence cards plus research-paper chunks: what the pairwise judge cites."""
    return load_cards(evidence_file) + load_papers(papers_dir)


def load_all_documents(essay_dir=ESSAY_DIR, evidence_file=EVIDENCE_FILE, papers_dir=PAPERS_DIR):
    """Load the demo essays, all evidence cards, and any research-paper chunks."""
    essays = [load_essay(Path(essay_dir) / fname, name) for fname, name in ESSAYS]
    return essays + load_cards(evidence_file) + load_papers(papers_dir)


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


@dataclass(frozen=True)
class Participant:
    participant_id: str
    name: str
    pair_id: str
    affected: bool | None  # None when not yet recorded
    essay_file: str

    def load_text(self, essay_dir=ESSAY_DIR):
        return load_essay(Path(essay_dir) / self.essay_file, self.participant_id).page_content


def _parse_affected(value):
    value = (value or "").strip().lower()
    if value in ("", "unknown", "na"):
        return None
    if value in ("1", "true", "yes", "y"):
        return True
    if value in ("0", "false", "no", "n"):
        return False
    raise ValueError(f"affected must be yes/no/blank, got {value!r}")


def load_participants(path=PARTICIPANTS_FILE):
    """Read the participant table. `affected` is True, False, or None (unknown)."""
    with Path(path).open(encoding="utf-8", newline="") as f:
        return [
            Participant(
                participant_id=row["participant_id"],
                name=row.get("name", ""),
                pair_id=row["pair_id"],
                affected=_parse_affected(row.get("affected")),
                essay_file=row["essay_file"],
            )
            for row in csv.DictReader(f)
        ]


def load_residences(path=RESIDENCES_FILE):
    """
    Read residence history rows. Blank years become None (unknown);
    an end_year of "present" marks a current residence.
    """

    def year(v):
        v = (v or "").strip()
        return int(v) if v and v.lower() != "present" else None

    with Path(path).open(encoding="utf-8", newline="") as f:
        return [
            {
                "participant_id": row["participant_id"],
                "place": row["place"],
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "start_year": year(row.get("start_year")),
                "end_year": year(row.get("end_year")),
                "current": (row.get("end_year") or "").strip().lower() == "present",
            }
            for row in csv.DictReader(f)
        ]
