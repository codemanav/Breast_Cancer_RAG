import json

import pytest

from src.config import ESSAYS
from src.data_loader import load_cards, load_documents, save_documents

CARD_A = {"title": "Card A", "factors": [{"factor": "alcohol"}]}
CARD_B = {"title": "Card B", "factors": []}


def test_loads_both_essays_and_all_cards(documents):
    essays = [d for d in documents if d.metadata["type"] == "essay"]
    cards = [d for d in documents if d.metadata["type"] == "evidence"]

    assert [d.metadata["source"] for d in essays] == [name for _, name in ESSAYS]
    assert all(len(d.page_content) > 1000 for d in essays)
    assert len(cards) == 2
    assert all(json.loads(d.page_content)["factors"] for d in cards)


@pytest.mark.parametrize(
    "content",
    [
        json.dumps([CARD_A, CARD_B]),  # JSON array
        "\n".join(json.dumps(c) for c in [CARD_A, CARD_B]),  # JSON Lines
    ],
    ids=["array", "jsonl"],
)
def test_load_cards_accepts_array_and_jsonl(tmp_path, content):
    path = tmp_path / "cards.json"
    path.write_text(content, encoding="utf-8")

    docs = load_cards(path)

    assert [d.metadata["source"] for d in docs] == ["Card A", "Card B"]
    assert json.loads(docs[0].page_content) == CARD_A


def test_load_cards_accepts_single_object(tmp_path):
    path = tmp_path / "card.json"
    path.write_text(json.dumps(CARD_A), encoding="utf-8")

    assert [d.metadata["source"] for d in load_cards(path)] == ["Card A"]


def test_card_without_title_gets_placeholder_source(tmp_path):
    path = tmp_path / "card.json"
    path.write_text(json.dumps({"factors": []}), encoding="utf-8")

    assert load_cards(path)[0].metadata["source"] == "unknown"


def test_save_and_load_documents_round_trip(tmp_path, documents):
    path = tmp_path / "nested" / "docs.jsonl"

    save_documents(documents, path)
    restored = load_documents(path)

    assert [(d.page_content, d.metadata) for d in restored] == [
        (d.page_content, d.metadata) for d in documents
    ]
