"""
Custom prompts for guiding GPT-4o in RAG analysis.
"""

from langchain.prompts import PromptTemplate

COMPARE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a breast cancer risk analyst.

Context:
{context}

Question:
{question}

Instructions:
- Use both life stories and scientific evidence.
- Extract exposures, lifestyle choices, and psychosocial stressors.
- Compare them carefully.
- Output in JSON:

{{
  "risk_factors": [],
  "environmental_exposures": [],
  "relative_risk": "",
  "supporting_sources": []
}}
"""
)
