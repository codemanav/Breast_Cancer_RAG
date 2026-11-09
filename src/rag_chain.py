"""
Retrieval-augmented generation chain built with LangChain runnables.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from src.config import CHAT_MODEL, RETRIEVAL_K
from src.prompts import COMPARE_PROMPT


def format_docs(docs):
    """Join retrieved documents into a single context string, labelled by source."""
    return "\n\n".join(
        f"[{d.metadata.get('type', 'doc')}: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


def build_rag(vectorstore, prompt=COMPARE_PROMPT, llm=None, k=RETRIEVAL_K):
    """
    Build a chain that retrieves the top-k documents and asks the chat model.

    Invoke with {"query": "..."}; returns
    {"query": str, "result": str, "source_documents": list[Document]}.
    """
    llm = llm or ChatOpenAI(model=CHAT_MODEL, temperature=0)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    answer_chain = prompt | llm | StrOutputParser()

    def run(inputs):
        query = inputs["query"]
        docs = retriever.invoke(query)
        result = answer_chain.invoke({"context": format_docs(docs), "question": query})
        return {"query": query, "result": result, "source_documents": docs}

    return RunnableLambda(run)
