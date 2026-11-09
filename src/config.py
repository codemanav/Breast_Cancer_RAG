"""
Configuration file for project-wide constants.
"""

import os

# Load API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Vector Database settings
CHROMA_DIR = "./db/chroma"

# Models
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o"
