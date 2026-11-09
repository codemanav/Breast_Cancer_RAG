import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.data_loader import load_participants
from src.story_cards import (
    StoryCardError,
    build_story_cards,
    extract_story_card,
    load_story_cards,
)

CARD = {"person_id": "WRONG", "timeline": [], "dimensions": [], "risk_posture": {"overall": "similar"}}


def test_extracts_card_and_sets_participant_id():
    llm = FakeListChatModel(responses=["```json\n" + json.dumps(CARD) + "\n```"])

    card = extract_story_card(llm, "P007", "a life story")

    assert card["person_id"] == "P007"  # never trusted from the model
    assert card["risk_posture"]["overall"] == "similar"


def test_retries_once_after_malformed_output():
    llm = FakeListChatModel(responses=["Sorry, here it is:", json.dumps(CARD)])

    assert extract_story_card(llm, "P1", "story")["person_id"] == "P1"


@pytest.mark.parametrize("bad", ["not json", json.dumps({"timeline": []})], ids=["no-json", "missing-keys"])
def test_raises_after_retries_exhausted(bad):
    llm = FakeListChatModel(responses=[bad, bad])

    with pytest.raises(StoryCardError):
        extract_story_card(llm, "P1", "story")


def test_build_is_resumable(tmp_path):
    participants = load_participants()  # MaryAnne and Sue
    path = tmp_path / "cards.jsonl"

    build_story_cards(FakeListChatModel(responses=[json.dumps(CARD)]), participants[:1], path=path)
    # Second run: only the missing participant should reach the model.
    calls = []
    llm = RunnableLambda(lambda messages: calls.append(messages) or AIMessage(content=json.dumps(CARD)))
    cards = build_story_cards(llm, participants, path=path)

    assert sorted(cards) == ["P001", "P002"]
    assert len(calls) == 1  # one call, for P002 only
    assert sorted(load_story_cards(path)) == ["P001", "P002"]
