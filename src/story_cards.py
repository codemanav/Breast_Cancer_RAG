"""
Turn each participant's life story into a structured story card with the LLM.
"""

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import STORY_CARDS_FILE
from src.prompts import STORY_CARD_SYSTEM, STORY_CARD_USER
from src.utils import parse_json_answer

REQUIRED_KEYS = ("timeline", "dimensions", "risk_posture")


class StoryCardError(ValueError):
    """The model's output could not be turned into a valid story card."""


def _content(message):
    return message.content if hasattr(message, "content") else str(message)


def validate_card(card):
    """Check the minimum structure downstream steps rely on."""
    missing = [k for k in REQUIRED_KEYS if k not in card]
    if missing:
        raise StoryCardError(f"story card missing keys: {missing}")
    if not isinstance(card["dimensions"], list):
        raise StoryCardError("story card 'dimensions' must be a list")
    return card


def extract_story_card(llm, participant_id, story, retries=1):
    """
    Ask the model for a story card, retrying once on malformed output.
    The participant id is always set by the code, never trusted from the model.
    """
    messages = [
        SystemMessage(content=STORY_CARD_SYSTEM),
        HumanMessage(content=STORY_CARD_USER.format(story=story)),
    ]
    last_error = None
    for _ in range(retries + 1):
        raw = _content(llm.invoke(messages))
        card = parse_json_answer(raw)
        if card is None:
            last_error = StoryCardError(f"no JSON in model output: {raw[:200]!r}")
            continue
        try:
            card = validate_card(card)
        except StoryCardError as e:
            last_error = e
            continue
        card["person_id"] = participant_id
        return card
    raise last_error


def build_story_cards(llm, participants, path=STORY_CARDS_FILE, essay_dir=None):
    """
    Extract a card for every participant and save them as JSON Lines.
    Participants that already have a card in `path` are skipped (resumable).
    """
    path = Path(path)
    cards = load_story_cards(path) if path.exists() else {}
    kwargs = {"essay_dir": essay_dir} if essay_dir else {}

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for p in participants:
            if p.participant_id in cards:
                continue
            card = extract_story_card(llm, p.participant_id, p.load_text(**kwargs))
            cards[p.participant_id] = card
            f.write(json.dumps(card, ensure_ascii=False) + "\n")
            f.flush()
    return cards


def load_story_cards(path=STORY_CARDS_FILE):
    """Read story cards into a dict keyed by participant id."""
    cards = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                card = json.loads(line)
                cards[card["person_id"]] = card
    return cards
