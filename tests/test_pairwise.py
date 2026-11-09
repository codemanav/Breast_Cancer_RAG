import json
import re

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.pairwise import (
    OVERALL,
    JudgeError,
    combine,
    compare_pairs,
    consistency_rate,
    evidence_query,
    judge_pair,
    literature_retriever,
    load_comparisons,
    parse_verdicts,
    to_outcomes,
)
from src.prompts import DOMAINS
from src.ranking import LOSS, TIE, WIN
from src.vectorstore import build_vectorstore

CARDS = {
    "P1": {"person_id": "P1", "risk": 3.0, "rag_evidence_queries": ["alcohol and breast cancer"]},
    "P2": {"person_id": "P2", "risk": 1.0, "rag_evidence_queries": ["night shift work"]},
    "P3": {"person_id": "P3", "risk": 2.0},
}


def _verdict(riskier, domains=None):
    domains = domains or {d: riskier for d in DOMAINS}
    return json.dumps(
        {"domains": [{"domain": d, "riskier": v, "rationale": "", "evidence": []} for d, v in domains.items()],
         "overall": {"riskier": riskier, "rationale": ""}}
    )


def _card(messages, label):
    text = messages[-1].content
    return json.loads(re.search(rf"Story card {label}:\n(.*?)\n\nExposure data {label}", text, re.S).group(1))


def oracle_judge():
    """Fake judge that always calls the participant with higher hidden risk riskier."""

    def answer(messages):
        a, b = _card(messages, "A"), _card(messages, "B")
        return AIMessage(content=_verdict("A" if a["risk"] > b["risk"] else "B"))

    return RunnableLambda(answer)


def position_biased_judge():
    """Fake judge that always picks whoever is presented first."""
    return RunnableLambda(lambda messages: AIMessage(content=_verdict("A")))


def test_parse_verdicts_normalises_and_fills_missing_domains():
    raw = json.dumps({"domains": [{"domain": "metabolic", "riskier": "b"},
                                  {"domain": "made_up", "riskier": "A"},
                                  {"domain": "substance_use", "riskier": "maybe"}],
                      "overall": {"riskier": "Tie"}})

    verdicts, _ = parse_verdicts(raw)

    assert verdicts["metabolic"] == "B"
    assert verdicts["substance_use"] == "TIE"  # invalid value -> tie
    assert verdicts["dietary_pattern"] == "TIE"  # missing domain -> tie
    assert verdicts[OVERALL] == "TIE"
    assert "made_up" not in verdicts


def test_parse_verdicts_rejects_garbage():
    with pytest.raises(JudgeError):
        parse_verdicts("I cannot decide.")


def test_combine_keeps_only_order_consistent_verdicts():
    forward = {OVERALL: "A", **{d: "TIE" for d in DOMAINS}, "metabolic": "A", "dietary_pattern": "A"}
    # Reverse order: A is now j. metabolic agrees (i riskier); dietary_pattern flips.
    reverse = {OVERALL: "B", **{d: "TIE" for d in DOMAINS}, "metabolic": "B", "dietary_pattern": "A"}

    outcome, consistent = combine(forward, reverse)

    assert outcome[OVERALL] == "i" and consistent[OVERALL]
    assert outcome["metabolic"] == "i"
    assert outcome["dietary_pattern"] == "tie" and not consistent["dietary_pattern"]
    assert outcome["circadian_sleep"] == "tie" and consistent["circadian_sleep"]


def test_judge_pair_with_consistent_judge():
    record = judge_pair(oracle_judge(), None, "P2", "P1", CARDS, {})

    assert record["outcome"][OVERALL] == "j"  # P1 (j) has higher risk
    assert all(record["consistent"].values())


def test_position_bias_is_neutralised():
    record = judge_pair(position_biased_judge(), None, "P1", "P2", CARDS, {})

    assert set(record["outcome"].values()) == {"tie"}
    assert consistency_rate([record]) == 0.0


def test_evidence_query_uses_card_topics():
    query = evidence_query(CARDS["P1"], CARDS["P2"])

    assert "alcohol and breast cancer" in query and "night shift work" in query
    assert "allostatic load" in evidence_query({}, {})  # falls back to domains


def test_literature_retriever_excludes_essays(documents, fake_embeddings, chroma_dir):
    vs = build_vectorstore(documents, embeddings=fake_embeddings, persist_directory=chroma_dir)

    docs = literature_retriever(vs, k=10).invoke("stress and diet")

    assert docs and {d.metadata["type"] for d in docs} == {"evidence"}


def test_compare_pairs_is_resumable_and_survives_failures(tmp_path):
    path = tmp_path / "comparisons.jsonl"
    pairs = [("P1", "P2"), ("P1", "P3"), ("P2", "P3")]

    def flaky(messages):
        if _card(messages, "A")["person_id"] == "P3" or _card(messages, "B")["person_id"] == "P3":
            raise RuntimeError("rate limited")
        return oracle_judge().invoke(messages)

    records, failures = compare_pairs(RunnableLambda(flaky), None, pairs, CARDS, {}, path=path, max_workers=2)
    assert len(records) == 1 and len(failures) == 2

    calls = []
    counting = RunnableLambda(lambda m: calls.append(1) or oracle_judge().invoke(m))
    records, failures = compare_pairs(counting, None, pairs, CARDS, {}, path=path)

    assert len(records) == 2 and not failures
    assert len(calls) == 4  # only the 2 failed pairs, both orders
    assert len(load_comparisons(path)) == 3


def test_to_outcomes_maps_to_ranking_values():
    records = [
        {"i": "P1", "j": "P2", "outcome": {OVERALL: "i"}},
        {"i": "P2", "j": "P3", "outcome": {OVERALL: "j"}},
        {"i": "P1", "j": "P3", "outcome": {OVERALL: "tie"}},
        {"i": "P1", "j": "UNKNOWN", "outcome": {OVERALL: "i"}},
    ]

    assert to_outcomes(records, {"P1": 0, "P2": 1, "P3": 2}) == [(0, 1, WIN), (1, 2, LOSS), (0, 2, TIE)]
