"""
Prompts that guide the chat model's analysis.
"""

from langchain_core.prompts import PromptTemplate

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
- Respond with JSON only, no prose or code fences, in this shape:

{{
  "risk_factors": [],
  "environmental_exposures": [],
  "relative_risk": "",
  "supporting_sources": []
}}
""",
)
