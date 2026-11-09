# Breast Cancer Evidence RAG

A retrieval-augmented generation (RAG) prototype that compares two personal
life stories with structured scientific evidence about breast-cancer risk
factors. It retrieves the relevant essays and evidence cards from a local
vector store and asks an OpenAI model for a structured comparison with
supporting sources.

**Author:** Manav Shah

## What is included

- `data/essays/` - two first-person life-history essays (MaryAnne and Sue),
  shared with the authors' consent.
- `data/evidence/cards.json` - structured evidence cards (a JSON array)
  summarizing two review articles.
- `data/processed/story_cards.jsonl` - the prepared documents, written by
  `01_preprocess.ipynb`.
- `src/` - document loading, Chroma vector store, prompt, and RAG chain.
- `notebooks/` - the step-by-step pipeline (see below).
- `main.py` - the end-to-end pipeline as a command-line script.

## Setup

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then put your OpenAI key in .env
```

On Windows, activate the environment with `.venv\Scripts\activate`.
Setting `OPENAI_API_KEY` in the shell also works instead of `.env`.

## Run

```bash
python main.py                      # build db/chroma on first run, then reuse it
python main.py --rebuild            # re-embed all documents
python main.py --query "Compare the stressors in MaryAnne and Sue's lives."
```

The Chroma database under `db/chroma/` is generated from the included data
and ignored by Git. Rebuilding resets the collection, so documents are never
duplicated.

## Notebooks

Run them in order from the `notebooks/` folder:

| Notebook | What it does | API key |
| --- | --- | --- |
| `01_preprocess.ipynb` | Loads essays and evidence cards, tabulates the risk factors, saves `story_cards.jsonl` | No |
| `02_build_vectorstore.ipynb` | Embeds the documents into Chroma and spot-checks retrieval | Yes |
| `03_rag_queries.ipynb` | Asks four comparison questions and saves `data/processed/rag_results.json` | Yes |
| `04_analysis.ipynb` | Summarizes the answers, retrieved sources, and coverage of evidence-card factors | No |

## How it works

1. Load the two essays and the evidence cards as LangChain documents.
2. Embed them with `text-embedding-3-large` and store them in local Chroma.
3. Retrieve the top 4 documents for the question.
4. Ask `gpt-4o` (temperature 0) to return JSON with `risk_factors`,
   `environmental_exposures`, `relative_risk`, and `supporting_sources`.

Models can be changed with the `EMBED_MODEL` and `CHAT_MODEL` environment
variables. Rebuild the vector store after changing the embedding model.

## Limitations

- The corpus is small (two essays and two evidence cards), so retrieval
  returns nearly every document; the RAG structure is meant to scale as more
  evidence cards are added.
- Model output is not validated against the sources; check the cited
  evidence before relying on any comparison.

## Disclaimer

This is a research prototype, not a medical diagnostic tool. Its output is not
medical advice; consult qualified healthcare professionals for health
decisions.

## Privacy and API keys

The OpenAI key is read from `OPENAI_API_KEY` (environment or `.env`); never
commit keys or generated vector databases. The essays contain personal health
history and are included with the authors' consent. Do not reuse them outside
this project.
