"""
Helper functions.
"""

def pretty_print_sources(sources):
    """
    Print metadata of retrieved documents in a readable format.
    """
    for doc in sources:
        print(f"Source: {doc.metadata['source']} | Type: {doc.metadata['type']}")
