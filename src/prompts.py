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

# Life-course domains shared by the story-card schema and the pairwise judge.
DOMAINS = [
    "allostatic_load",
    "circadian_sleep",
    "metabolic",
    "reproductive_hormonal",
    "substance_use",
    "environmental_occupational",
    "social_determinants",
    "healthcare_engagement",
    "dietary_pattern",
]

# Story-card extraction (from story_reprime.txt). Sent as a plain system
# message, so the JSON braces need no escaping.
STORY_CARD_SYSTEM = """You are a careful lifecourse analyst. Keep the person's story intact (timing, duration, co-occurrence). Do not reduce to checkboxes. Capture episodes, trajectories, and interactions. Preserve short quotes for provenance. Output valid JSON exactly matching the schema. If a field is unknown, omit it.

Schema:
{
  "person_id": "",
  "age_range": "0-35",
  "timeline": [
    { "age_start": null, "age_end": null, "domain": "psychosocial|circadian|metabolic|substance_use|reproductive|environmental_occupational|social_determinants|healthcare_engagement|diet", "description": "", "intensity": 1, "quotes": [] }
  ],
  "dimensions": [
    { "name": "allostatic_load", "state": "", "trajectory": "", "drivers": [], "support": [] },
    { "name": "circadian_sleep", "state": "", "trajectory": "", "drivers": [], "support": [] },
    { "name": "metabolic", "state": "", "trajectory": "", "drivers": [], "support": [] },
    { "name": "reproductive_hormonal", "state": "", "trajectory": "", "drivers": [], "support": [] },
    { "name": "substance_use", "state": "", "trajectory": "", "drivers": [], "support": [] },
    { "name": "environmental_occupational", "state": "", "trajectory": "", "drivers": [], "support": [] },
    { "name": "social_determinants", "state": "", "trajectory": "", "drivers": [], "support": [] },
    { "name": "healthcare_engagement", "state": "", "trajectory": "", "drivers": [], "support": [] },
    {
      "name": "dietary_pattern", "state": "", "trajectory": "", "drivers": [], "support": [],
      "details": {
        "alcohol": { "avg_drinks_per_week": null, "binge_freq_per_month": null },
        "red_processed_meat_servings_per_week": null,
        "poultry_fish_servings_per_week": null,
        "fruit_veg_servings_per_day": null,
        "fiber_g_per_day": null,
        "whole_grain_servings_per_day": null,
        "dairy_type": "none|low-fat|full-fat|fermented|mixed",
        "soy_intake": "none|low|moderate|high",
        "ultra_processed_foods_percent": null,
        "added_sugars_sweet_beverages": "none|low|moderate|high",
        "cooking_methods": ["grilling/char|frying|boiling|baking|mixed"],
        "supplements": { "folate": "none|diet-only|multivitamin|high-dose", "vitamin_d": "none|low|adequate|unknown", "other": [] }
      }
    }
  ],
  "interactions": [
    { "pattern": ["dietary_pattern", "substance_use"], "type": "potentially synergistic", "rationale": "" }
  ],
  "rag_evidence_queries": [],
  "risk_posture": {
    "overall": "higher|similar|lower vs typical young adult baseline",
    "confidence": "low|moderate|high",
    "drivers": [],
    "tempering_factors": [],
    "narrative_summary": ""
  },
  "counterfactuals": [
    { "change": "", "expected_effect": "likely favorable|uncertain", "evidence_topic": "" }
  ]
}"""

STORY_CARD_USER = (
    "Convert the life story below into one story card. Keep nuance, timing, "
    "duration, and quotes. Return only JSON.\n\n{story}"
)

# Pairwise judge. "A"/"B" labels avoid the model confusing win/loss direction;
# the code maps them to win/loss/tie afterwards.
JUDGE_SYSTEM = """You compare the life-course exposures of two women, A and B, to judge whose lifestyle and environment carried MORE breast-cancer risk, using only the story cards, exposure data, and scientific evidence provided.

Rules:
- Judge each domain separately: {domains}.
- For each domain answer "A" (A's exposure was riskier), "B" (B's was riskier), or "tie" (no meaningful difference, or not enough information).
- Prefer "tie" over guessing. Ground every non-tie judgment in the story cards and cite the evidence sources you relied on.
- Ignore the order in which A and B are presented.
- Return JSON only, no prose or code fences:

{{
  "domains": [
    {{"domain": "<domain>", "riskier": "A|B|tie", "rationale": "", "evidence": []}}
  ],
  "overall": {{"riskier": "A|B|tie", "rationale": ""}}
}}"""

JUDGE_USER = """Story card A:
{card_a}

Exposure data A:
{exposure_a}

Story card B:
{card_b}

Exposure data B:
{exposure_b}

Scientific evidence:
{evidence}"""
