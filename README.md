# Breast Cancer Evidence RAG

A retrieval-augmented generation (RAG) prototype for comparing two personal
stories with structured scientific evidence about breast-cancer risk factors.
The application retrieves relevant source documents and asks an OpenAI model
to produce a structured comparison with supporting sources.

## What is included

- `data/essays/` - the two source essays used by the example pipeline.
- `data/evidence/cards.json` - structured evidence cards.
- `notebooks/` - preprocessing, vector-store, query, and analysis notebooks.
- `src/` - document loading, Chroma vector-store, prompt, and RAG-chain code.
- `main.py` - end-to-end example pipeline.

## Setup

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"
python main.py
```

On Windows, activate the environment with
`.venv\Scripts\activate` and set `OPENAI_API_KEY` in the shell environment.

The first run creates a local Chroma database under `db/chroma/`. The database
is generated from the included essays and evidence cards and is intentionally
ignored by Git.

## How it works

1. Load the two essays and evidence cards as LangChain documents.
2. Create OpenAI embeddings and persist them in a local Chroma store.
3. Retrieve the most relevant documents for the comparison question.
4. Ask GPT-4o to return risk factors, environmental exposures, relative risk,
   and supporting sources in a structured response.

This is a research prototype, not a medical diagnostic tool. Its output must
not be used as medical advice; consult qualified healthcare professionals for
health decisions.

## Privacy and API keys

The OpenAI key is read from `OPENAI_API_KEY`; do not commit keys or generated
vector databases. The included documents may contain sensitive personal or
health-related information, so review repository visibility and access controls
before publishing or sharing this project.

## Notebooks

- `01_preprocess.ipynb` - prepare source material.
- `02_build_vectorstore.ipynb` - build the Chroma index.
- `03_rag_queries.ipynb` - run retrieval and questions.
- `04_analysis.ipynb` - inspect and analyze results.