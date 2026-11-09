"""
RAG pipeline using LangChain.
"""

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from src.config import CHAT_MODEL

def build_rag(vectorstore, prompt=None):
    """
    Build a RetrievalQA chain that uses the vectorstore retriever & GPT-4o.
    """
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":4}),
        chain_type="stuff",  # simplest chain for small docs
        chain_type_kwargs={"prompt": prompt} if prompt else None,
        return_source_documents=True
    )
    return qa
