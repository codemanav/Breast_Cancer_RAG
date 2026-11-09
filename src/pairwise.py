"""
LLM pairwise judging of participants' life-course exposures.

For each pair (i, j) the judge sees both story cards, both PFAS exposure
summaries, and literature retrieved for the pair, and says whose exposures
were riskier in each domain and overall. Every pair is judged twice, with the
presentation order swapped; a domain keeps its verdict only when both orders
agree, otherwise it becomes a tie. This removes position bias, and the
agreement rate is reported as a reliability measure for the judge.

Outcomes are stored per pair as "i" (i riskier, a win for i), "j", or "tie".
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import COMPARISONS_FILE
from src.data_loader import LITERATURE_TYPES
from src.exposure import exposure_text
from src.prompts import DOMAINS, JUDGE_SYSTEM, JUDGE_USER
from src.ranking import LOSS, TIE, WIN
from src.utils import parse_json_answer

OVERALL = "overall"
VALID = {"A", "B", "TIE"}


class JudgeError(ValueError):
    """The judge's output could not be parsed into verdicts."""


def literature_retriever(vectorstore, k=6):
    """Retriever restricted to evidence cards and research papers."""
    return vectorstore.as_retriever(
        search_kwargs={"k": k, "filter": {"type": {"$in": list(LITERATURE_TYPES)}}}
    )


def evidence_query(card_a, card_b, max_queries=6):
    """Build one retrieval query from both cards' suggested evidence topics."""
    topics = []
    for card in (card_a, card_b):
        for q in card.get("rag_evidence_queries", []):
            if q not in topics:
                topics.append(q)
    topics = topics[:max_queries] or [d.replace("_", " ") for d in DOMAINS]
    return "breast cancer risk in young women: " + "; ".join(topics)


def format_evidence(docs):
    blocks = []
    for d in docs:
        page = f", p. {d.metadata['page']}" if d.metadata.get("page") else ""
        blocks.append(f"[{d.metadata.get('source', 'unknown')}{page}]\n{d.page_content}")
    return "\n\n".join(blocks) or "No evidence retrieved."


def _content(message):
    return message.content if hasattr(message, "content") else str(message)


def parse_verdicts(raw):
    """Model JSON -> {"overall": "A|B|TIE", <domain>: "A|B|TIE"}; missing domains are ties."""
    data = parse_json_answer(raw)
    if not data or "overall" not in data:
        raise JudgeError(f"unparseable judge output: {raw[:200]!r}")

    def norm(value):
        value = str(value or "tie").strip().upper()
        return value if value in VALID else "TIE"

    verdicts = {d: "TIE" for d in DOMAINS}
    for item in data.get("domains", []):
        if item.get("domain") in verdicts:
            verdicts[item["domain"]] = norm(item.get("riskier"))
    verdicts[OVERALL] = norm(data["overall"].get("riskier"))
    return verdicts, data


def judge_once(llm, card_a, card_b, exposure_a, exposure_b, evidence, retries=1):
    """One judgment with A/B in the given order."""
    messages = [
        SystemMessage(content=JUDGE_SYSTEM.format(domains=", ".join(DOMAINS))),
        HumanMessage(
            content=JUDGE_USER.format(
                card_a=json.dumps(card_a, ensure_ascii=False),
                card_b=json.dumps(card_b, ensure_ascii=False),
                exposure_a=exposure_text(exposure_a),
                exposure_b=exposure_text(exposure_b),
                evidence=evidence,
            )
        ),
    ]
    error = None
    for _ in range(retries + 1):
        try:
            return parse_verdicts(_content(llm.invoke(messages)))
        except JudgeError as e:
            error = e
    raise error


def combine(forward, reverse):
    """
    Merge verdicts from (i as A, j as B) and (j as A, i as B).
    Returns ({key: "i"|"j"|"tie"}, {key: bool consistent}).
    """
    to_ij_fwd = {"A": "i", "B": "j", "TIE": "tie"}
    to_ij_rev = {"A": "j", "B": "i", "TIE": "tie"}
    outcome, consistent = {}, {}
    for key in [OVERALL, *DOMAINS]:
        a, b = to_ij_fwd[forward[key]], to_ij_rev[reverse[key]]
        consistent[key] = a == b
        outcome[key] = a if a == b else "tie"
    return outcome, consistent


def judge_pair(llm, retriever, pid_i, pid_j, cards, exposures):
    """Judge one pair in both orders and return a comparison record."""
    card_i, card_j = cards[pid_i], cards[pid_j]
    docs = retriever.invoke(evidence_query(card_i, card_j)) if retriever else []
    evidence = format_evidence(docs)
    exp_i, exp_j = exposures.get(pid_i), exposures.get(pid_j)

    forward, raw_fwd = judge_once(llm, card_i, card_j, exp_i, exp_j, evidence)
    reverse, raw_rev = judge_once(llm, card_j, card_i, exp_j, exp_i, evidence)
    outcome, consistent = combine(forward, reverse)
    return {
        "i": pid_i,
        "j": pid_j,
        "outcome": outcome,
        "consistent": consistent,
        "evidence_sources": sorted({d.metadata.get("source", "unknown") for d in docs}),
        "forward": raw_fwd,
        "reverse": raw_rev,
    }


def load_comparisons(path=COMPARISONS_FILE):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compare_pairs(llm, retriever, pairs, cards, exposures, path=COMPARISONS_FILE, max_workers=4, on_error=None):
    """
    Judge every (pid_i, pid_j) pair, appending each record to `path` as soon as
    it is done. Pairs already in the file are skipped, so a long run that is
    interrupted (or rate-limited) resumes where it stopped.
    Returns (records judged now, failures as (pair, error)).
    """
    path = Path(path)
    done = {frozenset((r["i"], r["j"])) for r in load_comparisons(path)}
    todo = [p for p in pairs if frozenset(p) not in done]
    path.parent.mkdir(parents=True, exist_ok=True)

    lock = threading.Lock()
    records, failures = [], []
    with path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(judge_pair, llm, retriever, i, j, cards, exposures): (i, j) for i, j in todo}
        for fut in as_completed(futures):
            pair = futures[fut]
            try:
                record = fut.result()
            except Exception as e:  # keep going; failed pairs are retried on the next run
                failures.append((pair, e))
                if on_error:
                    on_error(pair, e)
                continue
            with lock:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            records.append(record)
    return records, failures


def to_outcomes(records, index_of, key=OVERALL):
    """Comparison records -> (i_index, j_index, y) for the ranking models."""
    value = {"i": WIN, "j": LOSS, "tie": TIE}
    return [
        (index_of[r["i"]], index_of[r["j"]], value[r["outcome"][key]])
        for r in records
        if r["i"] in index_of and r["j"] in index_of
    ]


def consistency_rate(records, key=OVERALL):
    """Share of pairs whose verdict did not change when the order was swapped."""
    if not records:
        return None
    return sum(r["consistent"][key] for r in records) / len(records)
