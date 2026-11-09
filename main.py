"""
Main script to run RAG pipeline on local essays + evidence.
"""

from src.data_loader import load_essay, load_cards
from src.vectorstore import build_vectorstore, load_vectorstore
from src.rag_chain import build_rag
from src.prompts import COMPARE_PROMPT
from src.utils import pretty_print_sources

def main():
    # Load data
    mary = load_essay("data/essays/maryanne.docx", "MaryAnne")
    sue = load_essay("data/essays/sue.docx", "Sue")
    cards = load_cards("data/evidence/cards.json")

    docs = [mary, sue] + cards

    # Build vectorstore locally
    vs = build_vectorstore(docs)

    # Build RAG chain
    qa = build_rag(vs, prompt=COMPARE_PROMPT)

    # Example query
    query = "Compare MaryAnne and Sue's environmental exposures with scientific evidence."
    result = qa.invoke({"query": query})

    print("Answer:\n", result["result"])
    print("\nSources Used:")
    pretty_print_sources(result["source_documents"])

if __name__ == "__main__":
    main()
