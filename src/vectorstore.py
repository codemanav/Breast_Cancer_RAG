"""
Build and load the local Chroma vector database.
"""

from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL


def _embeddings(embeddings=None):
    return embeddings or OpenAIEmbeddings(model=EMBED_MODEL)


def _open(embeddings=None, persist_directory=CHROMA_DIR, collection_name=COLLECTION_NAME):
    return Chroma(
        collection_name=collection_name,
        embedding_function=_embeddings(embeddings),
        persist_directory=persist_directory,
    )


def build_vectorstore(docs, embeddings=None, persist_directory=CHROMA_DIR, collection_name=COLLECTION_NAME):
    """
    Create (or recreate) the Chroma collection from `docs` and persist it.
    The collection is reset first, so rebuilding never duplicates documents.
    """
    vs = _open(embeddings, persist_directory, collection_name)
    vs.reset_collection()
    vs.add_documents(docs)
    return vs


def load_vectorstore(embeddings=None, persist_directory=CHROMA_DIR, collection_name=COLLECTION_NAME):
    """Open an existing Chroma collection from disk."""
    return _open(embeddings, persist_directory, collection_name)


def vectorstore_exists(persist_directory=CHROMA_DIR, collection_name=COLLECTION_NAME):
    """True when the persisted collection exists and contains documents."""
    if not (Path(persist_directory) / "chroma.sqlite3").exists():
        return False
    client = chromadb.PersistentClient(path=persist_directory)
    try:
        return client.get_collection(collection_name).count() > 0
    except Exception:  # collection missing; error type varies across chromadb versions
        return False
