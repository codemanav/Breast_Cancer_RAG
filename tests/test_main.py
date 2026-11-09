"""End-to-end tests of the command-line pipeline with fake models."""

from functools import partial

import pytest

import main
import src.vectorstore as vectorstore
from src.data_loader import load_all_documents
from src.rag_chain import build_rag


@pytest.fixture
def offline_main(monkeypatch, tmp_path, fake_embeddings, fake_llm, chroma_dir):
    """Point main.py at fake models, a temporary vector store, and no paper PDFs."""
    monkeypatch.setattr(main, "load_all_documents", partial(load_all_documents, papers_dir=tmp_path / "none"))
    monkeypatch.setattr(
        main,
        "build_vectorstore",
        partial(vectorstore.build_vectorstore, embeddings=fake_embeddings, persist_directory=chroma_dir),
    )
    monkeypatch.setattr(
        main,
        "load_vectorstore",
        partial(vectorstore.load_vectorstore, embeddings=fake_embeddings, persist_directory=chroma_dir),
    )
    monkeypatch.setattr(main, "vectorstore_exists", partial(vectorstore.vectorstore_exists, chroma_dir))
    monkeypatch.setattr(main, "build_rag", partial(build_rag, llm=fake_llm))

    def run(*args):
        monkeypatch.setattr("sys.argv", ["main.py", *args])
        main.main()

    return run


def test_first_run_builds_index_and_answers(offline_main, capsys):
    offline_main()

    out = capsys.readouterr().out
    assert '"relative_risk": "similar"' in out  # parsed and pretty-printed JSON
    assert "Sources Used:" in out
    assert out.count("Source: ") == 4


def test_second_run_reuses_index(offline_main, monkeypatch, capsys):
    offline_main()

    def fail(*args, **kwargs):
        raise AssertionError("index should be reused, not rebuilt")

    monkeypatch.setattr(main, "build_vectorstore", fail)
    offline_main("--query", "Compare stressors.")

    assert "Sources Used:" in capsys.readouterr().out


def test_rebuild_flag_forces_reindex(offline_main, monkeypatch):
    offline_main()
    calls = []
    original = main.build_vectorstore
    monkeypatch.setattr(main, "build_vectorstore", lambda docs: calls.append(len(docs)) or original(docs))

    offline_main("--rebuild")

    assert calls == [4]


def test_missing_api_key_fails_fast(offline_main, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        offline_main()
