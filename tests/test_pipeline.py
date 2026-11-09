"""
End-to-end run of pipeline.py on a synthetic study, fully offline.

Eight sister pairs get .docx life stories that state a hidden risk level; the
affected sister is always riskier. A fake LLM extracts that level into the
story card and, as judge, calls the higher-risk woman riskier. The pipeline
should then rank affected sisters above their unaffected sisters with a
significant sister-pair test.
"""

import csv
import json
import re

import openpyxl
import pytest
from docx import Document as Docx
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import pipeline
from src import config
from src.prompts import DOMAINS

N_PAIRS = 8


def _write_study(root):
    essays = root / "essays"
    essays.mkdir()
    participants, residences = [], []
    for k in range(N_PAIRS):
        for member in (0, 1):
            pid = f"P{k:02d}{member}"
            # Families overlap in risk, so cross-family comparisons can separate
            # sisters; the affected sister (member 0) is riskier. Values are distinct.
            risk = k + (4.5 if member == 0 else 0)
            doc = Docx()
            doc.add_paragraph(f"Participant {pid}. Hidden risk level {risk:g}.")
            doc.save(essays / f"{pid}.docx")
            participants.append([pid, pid, f"S{k:02d}", "yes" if member == 0 else "no", f"{pid}.docx"])
            residences.append([pid, "Town", 40.20, -75.09, 1960, "present", ""])

    with (root / "participants.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["participant_id", "name", "pair_id", "affected", "essay_file"], *participants])
    with (root / "residences.csv").open("w", newline="") as f:
        csv.writer(f).writerows(
            [["participant_id", "place", "lat", "lon", "start_year", "end_year", "note"], *residences]
        )

    wb = openpyxl.Workbook()
    ws = wb.create_sheet("USA Contamination Sites")
    ws.append(["Site name", "State", "Other site name(s)", "Industry", "Sample Matrix", "Latittude", "Longitude",
               "Sample Date", "PFOA (ppt)", "PFOS (ppt)", "PFOA+PFOS from one sample (ppt)",
               "Total PFAS from one sample (ppt)"])
    ws.append(["Base A", "PA", None, "Military", "Groundwater", 40.21, -75.07, 2019, 600, 2400, 3000, None])
    wb.save(root / "pfas.xlsx")
    return essays


def fake_llm(require_exposure=False):
    def respond(messages):
        system, user = messages[0].content, messages[-1].content
        if "lifecourse analyst" in system:  # story-card extraction
            pid, risk = re.search(r"Participant (\w+)\. Hidden risk level ([\d.]+)\.", user).groups()
            card = {"timeline": [], "dimensions": [], "risk_posture": {"overall": "similar"}, "risk": float(risk)}
            return AIMessage(content=json.dumps(card))
        # pairwise judge
        a = json.loads(re.search(r"Story card A:\n(.*?)\n\nExposure", user, re.S).group(1))
        b = json.loads(re.search(r"Story card B:\n(.*?)\n\nExposure", user, re.S).group(1))
        if require_exposure:
            assert "PFAS contamination sites" in user  # exposure data reaches the judge
        riskier = "A" if a["risk"] > b["risk"] else "B"
        return AIMessage(content=json.dumps({
            "domains": [{"domain": d, "riskier": riskier} for d in DOMAINS],
            "overall": {"riskier": riskier},
        }))

    return RunnableLambda(respond)


@pytest.fixture
def study(tmp_path, monkeypatch, fake_embeddings):
    essays = _write_study(tmp_path)
    out = tmp_path / "processed"
    out.mkdir()
    for name, value in {
        "ESSAY_DIR": essays,
        "PARTICIPANTS_FILE": tmp_path / "participants.csv",
        "RESIDENCES_FILE": tmp_path / "residences.csv",
        "PFAS_FILE": tmp_path / "pfas.xlsx",
        "PAPERS_DIR": tmp_path / "papers",
        "CHROMA_DIR": str(tmp_path / "chroma"),
        "LITERATURE_FILE": out / "literature.jsonl",
        "STORY_CARDS_FILE": out / "story_cards.jsonl",
        "EXPOSURE_FILE": out / "exposure.json",
        "COMPARISONS_FILE": out / "comparisons.jsonl",
        "SISTER_COMPARISONS_FILE": out / "sister_comparisons.jsonl",
        "RANKING_FILE": out / "ranking.json",
    }.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr(pipeline, "make_llm", fake_llm)
    monkeypatch.setattr(pipeline, "make_embeddings", lambda: fake_embeddings)
    return out


def test_full_pipeline_detects_affected_sisters(study, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "make_llm", lambda: fake_llm(require_exposure=True))
    pipeline.main(["cards"])
    pipeline.main(["exposure"])
    pipeline.main(["compare", "--sisters", "--workers", "4"])
    pipeline.main(["rank", "--n-null", "2000"])

    n = 2 * N_PAIRS
    comparisons = [json.loads(l) for l in (study / "comparisons.jsonl").read_text().splitlines()]
    sister_records = [json.loads(l) for l in (study / "sister_comparisons.jsonl").read_text().splitlines()]
    report = json.loads((study / "ranking.json").read_text())

    assert len(comparisons) == n * (n - 1) // 2 - N_PAIRS  # every non-sister pair once
    assert len(sister_records) == N_PAIRS
    assert all(r["outcome"]["overall"] == "i" for r in sister_records)  # affected listed first
    assert report["judge_order_consistency"]["overall"] == 1.0

    overall = report["results"]["overall"]
    assert overall["sister_test"]["statistic"] > 0.6
    assert overall["sister_test"]["p_value"] < 0.05
    assert set(report["results"]) == {"overall", *DOMAINS}
    assert "Judge order-swap consistency (overall): 100.0%" in capsys.readouterr().out


def test_compare_resumes_without_rejudging(study):
    pipeline.main(["cards"])
    pipeline.main(["compare", "--limit", "5"])
    pipeline.main(["compare"])

    lines = (study / "comparisons.jsonl").read_text().splitlines()
    pairs = [frozenset((json.loads(l)["i"], json.loads(l)["j"])) for l in lines]
    assert len(pairs) == len(set(pairs))


def test_rank_without_labels_skips_sister_test(study, capsys):
    path = config.PARTICIPANTS_FILE
    rows = list(csv.reader(path.open()))
    for row in rows[1:]:
        row[3] = ""
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)

    pipeline.main(["cards"])
    pipeline.main(["compare", "--limit", "20"])
    pipeline.main(["rank", "--n-null", "100"])

    report = json.loads((study / "ranking.json").read_text())
    assert "sister_test" not in report["results"]["overall"]
    assert "sister-pair test skipped" in capsys.readouterr().out


def test_steps_explain_missing_prerequisites(study):
    with pytest.raises(SystemExit, match="No story cards yet"):
        pipeline.main(["compare"])
    with pytest.raises(SystemExit, match="No comparisons yet"):
        pipeline.main(["rank"])


def test_exposure_requires_local_pfas_file(study, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PFAS_FILE", tmp_path / "missing.xlsx")

    with pytest.raises(SystemExit, match="PFAS Project Lab"):
        pipeline.main(["exposure"])


def test_index_includes_paper_chunks(study, tmp_path, monkeypatch, capsys):
    from pypdf import PdfWriter

    papers = tmp_path / "papers"
    papers.mkdir()
    writer = PdfWriter()
    writer.add_blank_page(612, 792)
    writer.write(papers / "blank.pdf")  # no text -> contributes no chunks

    pipeline.main(["index"])

    assert "Indexed: 2 evidence" in capsys.readouterr().out  # literature only, no essays


def test_simulate_prints_power_table(capsys):
    pipeline.main(["simulate", "--pairs", "10", "--effects", "1", "--sims", "5"])

    assert "Power of the sister-pair test" in capsys.readouterr().out
