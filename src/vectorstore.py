"""
Functions for building and loading Chroma vector database locally.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import CHROMA_DIR, EMBED_MODEL

def build_vectorstore(docs):
    """
    Create a new ChromaDB from documents and persist it locally.
    """
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vs = Chroma.from_documents(docs, embedding=embeddings, persist_directory=CHROMA_DIR)
    vs.persist()
    return vs

def load_vectorstore():
    """
    Load an existing ChromaDB from disk.
    """
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
