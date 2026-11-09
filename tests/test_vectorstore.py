import chromadb

from src.config import COLLECTION_NAME
from src.vectorstore import build_vectorstore, load_vectorstore, vectorstore_exists


def _count(chroma_dir):
    return chromadb.PersistentClient(path=chroma_dir).get_collection(COLLECTION_NAME).count()


def test_build_indexes_every_document(documents, fake_embeddings, chroma_dir):
    build_vectorstore(documents, embeddings=fake_embeddings, persist_directory=chroma_dir)

    assert _count(chroma_dir) == len(documents)


def test_rebuild_is_idempotent(documents, fake_embeddings, chroma_dir):
    """Regression: rebuilding used to append a second copy of every document."""
    for _ in range(3):
        build_vectorstore(documents, embeddings=fake_embeddings, persist_directory=chroma_dir)

    assert _count(chroma_dir) == len(documents)


def test_persisted_index_reloads_without_rebuilding(documents, fake_embeddings, chroma_dir):
    build_vectorstore(documents, embeddings=fake_embeddings, persist_directory=chroma_dir)

    vs = load_vectorstore(embeddings=fake_embeddings, persist_directory=chroma_dir)

    assert len(vs.similarity_search("stress", k=2)) == 2


def test_exists_is_false_for_missing_or_empty_store(chroma_dir):
    assert not vectorstore_exists(chroma_dir)

    # A database file without the collection must not count as a usable index.
    chromadb.PersistentClient(path=chroma_dir)
    assert not vectorstore_exists(chroma_dir)


def test_exists_is_true_after_build(documents, fake_embeddings, chroma_dir):
    build_vectorstore(documents, embeddings=fake_embeddings, persist_directory=chroma_dir)

    assert vectorstore_exists(chroma_dir)
