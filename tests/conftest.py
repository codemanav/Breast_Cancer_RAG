"""
Shared fixtures. Every test runs offline: OpenAI models are replaced with
deterministic fakes and each vector store lives in a temporary directory.
"""

import json

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.data_loader import load_all_documents

FAKE_ANSWER = {
    "risk_factors": ["alcohol consumption", "chronic stress"],
    "environmental_exposures": ["pesticides near home"],
    "relative_risk": "similar",
    "supporting_sources": ["The Role of Stress in Breast Cancer Incidence"],
}


@pytest.fixture
def fake_embeddings():
    return DeterministicFakeEmbedding(size=64)


@pytest.fixture
def fake_llm():
    # Wrapped in a code fence, as chat models often do, to exercise parsing.
    return FakeListChatModel(responses=["```json\n" + json.dumps(FAKE_ANSWER) + "\n```"])


@pytest.fixture
def documents():
    return load_all_documents()


@pytest.fixture
def chroma_dir(tmp_path):
    return str(tmp_path / "chroma")


@pytest.fixture(autouse=True)
def no_real_api_key(monkeypatch):
    """Guarantee no test can reach OpenAI with a real key."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
