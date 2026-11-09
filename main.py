"""
Run the RAG pipeline on the local essays and evidence cards.

Usage:
    python main.py                 # reuse db/chroma if it exists
    python main.py --rebuild       # re-embed all documents
    python main.py --query "..."   # ask a different question
"""

import argparse
import json

from src.config import require_api_key
from src.data_loader import load_all_documents
from src.rag_chain import build_rag
from src.utils import parse_json_answer, pretty_print_sources
from src.vectorstore import build_vectorstore, load_vectorstore, vectorstore_exists

DEFAULT_QUERY = "Compare MaryAnne and Sue's environmental exposures with scientific evidence."


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--rebuild", action="store_true", help="re-embed all documents")
    args = parser.parse_args()

    require_api_key()

    if args.rebuild or not vectorstore_exists():
        vs = build_vectorstore(load_all_documents())
    else:
        vs = load_vectorstore()

    qa = build_rag(vs)
    result = qa.invoke({"query": args.query})

    parsed = parse_json_answer(result["result"])
    print("Answer:\n", json.dumps(parsed, indent=2) if parsed else result["result"])
    print("\nSources Used:")
    pretty_print_sources(result["source_documents"])


if __name__ == "__main__":
    main()
