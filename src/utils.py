"""
Helper functions.
"""

import json
import re


def pretty_print_sources(sources):
    """Print metadata of retrieved documents in a readable format."""
    for doc in sources:
        print(f"Source: {doc.metadata.get('source')} | Type: {doc.metadata.get('type')}")


def parse_json_answer(text):
    """
    Parse the model's JSON answer. Tolerates surrounding code fences or prose.
    Returns a dict, or None when no JSON object can be parsed.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
