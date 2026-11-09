from langchain_core.documents import Document

from src.rag_chain import build_rag, format_docs
from src.utils import parse_json_answer
from src.vectorstore import build_vectorstore
from tests.conftest import FAKE_ANSWER


def test_format_docs_labels_each_source():
    docs = [
        Document(page_content="essay text", metadata={"source": "Sue", "type": "essay"}),
        Document(page_content="card text", metadata={"source": "Card A", "type": "evidence"}),
    ]

    context = format_docs(docs)

    assert "[essay: Sue]\nessay text" in context
    assert "[evidence: Card A]\ncard text" in context


def test_chain_returns_answer_and_sources(documents, fake_embeddings, fake_llm, chroma_dir):
    vs = build_vectorstore(documents, embeddings=fake_embeddings, persist_directory=chroma_dir)
    qa = build_rag(vs, llm=fake_llm, k=3)

    out = qa.invoke({"query": "Compare environmental exposures."})

    assert out["query"] == "Compare environmental exposures."
    assert parse_json_answer(out["result"]) == FAKE_ANSWER
    assert len(out["source_documents"]) == 3
    assert {d.metadata["type"] for d in out["source_documents"]} <= {"essay", "evidence"}


def test_retrieved_context_reaches_the_prompt(documents, fake_embeddings, chroma_dir):
    """The prompt sent to the model must contain the retrieved documents and the question."""
    seen = []

    class RecordingLLM:
        def invoke(self, prompt_value, *args, **kwargs):
            seen.append(prompt_value.to_string())
            return "{}"

    from langchain_core.runnables import RunnableLambda

    vs = build_vectorstore(documents, embeddings=fake_embeddings, persist_directory=chroma_dir)
    qa = build_rag(vs, llm=RunnableLambda(RecordingLLM().invoke), k=2)

    out = qa.invoke({"query": "What about stress?"})

    prompt = seen[0]
    assert "What about stress?" in prompt
    for doc in out["source_documents"]:
        assert doc.page_content[:200] in prompt
