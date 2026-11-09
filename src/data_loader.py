"""
Functions for loading essays (Word documents) and evidence JSON into LangChain Documents.
"""

from langchain.docstore.document import Document
from docx import Document as Docx
import json

def load_essay(path, source_name):
    """
    Load a Word (.docx) essay and return as LangChain Document.
    """
    doc = Docx(path)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return Document(page_content=text, metadata={"source": source_name, "type": "essay"})

def load_cards(path):
    """
    Load evidence from a file that may contain:
    - a JSON array of objects, OR
    - multiple JSON objects (newline or back-to-back).
    Returns a list of LangChain Documents.
    """
    docs = []
    with open(path, "r") as f:
        content = f.read().strip()

        try:
            # Try reading as a JSON array first
            data = json.loads(content)
            if isinstance(data, dict):
                data = [data]
        except json.JSONDecodeError:
            # Fall back: multiple objects (JSON Lines style)
            data = []
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    # If objects are separated without newlines, split manually
                    # This works because each object starts with { and ends with }
                    objects = content.split("}\n{")
                    for i, obj in enumerate(objects):
                        if not obj.startswith("{"):
                            obj = "{" + obj
                        if not obj.endswith("}"):
                            obj = obj + "}"
                        try:
                            data.append(json.loads(obj))
                        except:
                            pass
                    break

    # Wrap each card into a LangChain Document
    for card in data:
        docs.append(
            Document(
                page_content=json.dumps(card, indent=2),
                metadata={"source": card.get("title", "unknown"), "type": "evidence"}
            )
        )

    return docs
