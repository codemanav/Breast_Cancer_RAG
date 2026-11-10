"""
Sister-pair study pipeline.

    python pipeline.py index        # embed evidence cards and research-paper PDFs
    python pipeline.py cards        # LLM: life story -> structured story card
    python pipeline.py exposure     # PFAS sites near each residence (local data file)
    python pipeline.py compare      # LLM judge: every non-sister pair, both orders
    python pipeline.py compare --sisters   # also judge each sister pair directly
    python pipeline.py rank         # Bradley-Terry / Davidson + sister-pair test
    python pipeline.py simulate     # power analysis of the study design

Steps that call OpenAI need OPENAI_API_KEY; `rank` and `simulate` do not.
Long steps are resumable: rerun the same command to continue.
"""

import argparse
import json
import sys

import numpy as np

from src import config
from src.data_loader import load_literature, load_participants, load_residences, save_documents
from src.pairwise import (
    OVERALL,
    compare_pairs,
    consistency_rate,
    literature_retriever,
    load_comparisons,
    to_outcomes,
)
from src.prompts import DOMAINS
from src.ranking import candidate_pairs, fit_bradley_terry, fit_davidson, sister_pair_test, sister_pairs


# ---- model/vector-store factories (swapped for fakes in tests) -------------

def make_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=config.CHAT_MODEL, temperature=0)


def make_embeddings():
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=config.EMBED_MODEL)


def open_vectorstore(rebuild=False):
    from src.vectorstore import build_vectorstore, load_vectorstore, vectorstore_exists

    where = {"persist_directory": config.CHROMA_DIR, "collection_name": config.LITERATURE_COLLECTION}
    if rebuild or not vectorstore_exists(**where):
        docs = load_literature(config.EVIDENCE_FILE, config.PAPERS_DIR)
        save_documents(docs, config.LITERATURE_FILE)
        return build_vectorstore(docs, embeddings=make_embeddings(), **where)
    return load_vectorstore(embeddings=make_embeddings(), **where)


# ---- steps -----------------------------------------------------------------

def cmd_index(args):
    config.require_api_key()
    vs = open_vectorstore(rebuild=True)
    counts = {}
    for meta in vs.get()["metadatas"]:
        counts[meta["type"]] = counts.get(meta["type"], 0) + 1
    print("Indexed:", ", ".join(f"{n} {t}" for t, n in sorted(counts.items())))


def cmd_cards(args):
    from src.story_cards import build_story_cards

    config.require_api_key()
    participants = load_participants(config.PARTICIPANTS_FILE)
    cards = build_story_cards(make_llm(), participants, path=config.STORY_CARDS_FILE, essay_dir=config.ESSAY_DIR)
    print(f"{len(cards)} story cards in {config.STORY_CARDS_FILE}")


def cmd_exposure(args):
    from src.exposure import load_pfas_sites, participant_exposures, save_exposures

    if not config.PFAS_FILE.exists():
        sys.exit(
            f"PFAS data not found at {config.PFAS_FILE}. Restore data/external/pfas_sites.xlsx "
            "or point PFAS_FILE at a copy of the PFAS Project Lab tracker."
        )
    sites = load_pfas_sites(config.PFAS_FILE)
    exposures = participant_exposures(load_residences(config.RESIDENCES_FILE), sites, radius_km=args.radius_km)
    save_exposures(exposures, config.EXPOSURE_FILE)
    print(f"{len(sites)} PFAS sites; radius {args.radius_km:g} km")
    for pid, e in exposures.items():
        s = e["summary"]
        years = f"{s['known_years_near_sites']}{'+' if s['has_unknown_durations'] else ''}"
        print(f"  {pid}: {s['residences_near_sites']} residences near sites, {years} known years, "
              f"nearest {s['nearest_km']} km, max PFOA+PFOS {s['max_pfoa_pfos_ppt']} ppt")


def _pairs_by_id(participants, sisters_only):
    ids = [p.participant_id for p in participants]
    pair_ids = [p.pair_id for p in participants]
    if sisters_only:
        families = {}
        for pid, fam in zip(ids, pair_ids):
            families.setdefault(fam, []).append(pid)
        return [tuple(m) for m in families.values() if len(m) == 2]
    return [(ids[a], ids[b]) for a, b in candidate_pairs(pair_ids)]


def cmd_compare(args):
    from src.exposure import load_exposures
    from src.story_cards import load_story_cards

    config.require_api_key()
    participants = load_participants(config.PARTICIPANTS_FILE)
    if not config.STORY_CARDS_FILE.exists():
        sys.exit("No story cards yet. Run: python pipeline.py cards")
    cards = load_story_cards(config.STORY_CARDS_FILE)
    exposures = load_exposures(config.EXPOSURE_FILE)
    missing = [p.participant_id for p in participants if p.participant_id not in cards]
    if missing:
        sys.exit(f"Missing story cards for {missing}. Run: python pipeline.py cards")

    retriever = literature_retriever(open_vectorstore(), k=args.k)
    llm = make_llm()
    jobs = [("non-sister", _pairs_by_id(participants, False), config.COMPARISONS_FILE)]
    if args.sisters:
        jobs.append(("sister", _pairs_by_id(participants, True), config.SISTER_COMPARISONS_FILE))

    for label, pairs, path in jobs:
        if args.limit:
            pairs = pairs[: args.limit]
        if not pairs:
            print(f"No {label} pairs to judge ({len(participants)} participants).")
            continue
        records, failures = compare_pairs(
            llm, retriever, pairs, cards, exposures, path=path, max_workers=args.workers,
            on_error=lambda pair, e: print(f"  failed {pair}: {e}", file=sys.stderr),
        )
        total = len(load_comparisons(path))
        print(f"{label}: judged {len(records)} new pairs ({total}/{len(pairs)} done, {len(failures)} failed)")
        if failures:
            print("  Rerun the same command to retry failed pairs.")


def _rank_key(key, n, outcomes, sisters, null_pairs, n_null):
    davidson = fit_davidson(n, outcomes)
    bt = fit_bradley_terry(n, outcomes)
    result = {
        "n_comparisons": len(outcomes),
        "davidson": {"theta": davidson.theta.round(4).tolist(), "nu": round(davidson.nu, 4)},
        "bradley_terry": {"theta": bt.theta.round(4).tolist()},
        "rank_agreement": round(float(np.corrcoef(davidson.theta, bt.theta)[0, 1]), 4)
        if np.std(bt.theta) > 0 else None,
    }
    if sisters:
        t = sister_pair_test(davidson, sisters, null_pairs, n_null=n_null)
        result["sister_test"] = {
            "statistic": round(t.statistic, 4), "null_mean": round(t.null_mean, 4),
            "null_sd": round(t.null_sd, 4), "p_value": round(t.p_value, 5),
            "n_sister_pairs": t.n_sister_pairs,
        }
    return result


def cmd_rank(args):
    participants = load_participants(config.PARTICIPANTS_FILE)
    ids = [p.participant_id for p in participants]
    index_of = {pid: k for k, pid in enumerate(ids)}
    pair_ids = [p.pair_id for p in participants]
    records = load_comparisons(config.COMPARISONS_FILE)
    if not records:
        sys.exit("No comparisons yet. Run: python pipeline.py compare")

    sisters = sister_pairs(pair_ids, [p.affected for p in participants])
    null_pairs = candidate_pairs(pair_ids)
    report = {
        "participants": ids,
        "n_comparisons": len(records),
        "judge_order_consistency": {
            key: round(consistency_rate(records, key), 4) for key in [OVERALL, *DOMAINS]
        },
        "results": {},
    }
    for key in [OVERALL, *DOMAINS]:
        report["results"][key] = _rank_key(
            key, len(ids), to_outcomes(records, index_of, key), sisters, null_pairs, args.n_null
        )

    config.RANKING_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{len(records)} comparisons, {len(ids)} participants, {len(sisters)} labelled sister pairs")
    print(f"Judge order-swap consistency (overall): {report['judge_order_consistency'][OVERALL]:.1%}\n")
    header = f"{'domain':<28}{'nu':>7}{'BT~Dav':>8}"
    if sisters:
        header += f"{'sister P(aff>unaff)':>21}{'p':>9}"
    print(header)
    for key, r in report["results"].items():
        line = f"{key:<28}{r['davidson']['nu']:>7.2f}{(r['rank_agreement'] or 0):>8.2f}"
        if "sister_test" in r:
            line += f"{r['sister_test']['statistic']:>21.3f}{r['sister_test']['p_value']:>9.4f}"
        print(line)
    if not sisters:
        print("\nNo sister pairs with affected/unaffected labels in data/participants.csv; "
              "sister-pair test skipped.")
    print(f"\nSaved {config.RANKING_FILE}")


def cmd_simulate(args):
    from src.simulate import power

    print(f"Power of the sister-pair test (alpha={args.alpha}, {args.sims} simulated studies each)")
    print(f"{'sister pairs':>12}" + "".join(f"{f'effect {e:g} SD':>17}" for e in args.effects))
    for n in args.pairs:
        row = [power(n, e, n_sims=args.sims, alpha=args.alpha) for e in args.effects]
        print(f"{n:>12}" + "".join(f"{p:>17.2f}" for p in row))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("index", help="embed evidence cards and papers").set_defaults(func=cmd_index)
    sub.add_parser("cards", help="extract story cards").set_defaults(func=cmd_cards)

    p = sub.add_parser("exposure", help="PFAS exposure from residence history")
    p.add_argument("--radius-km", type=float, default=config.PFAS_RADIUS_KM)
    p.set_defaults(func=cmd_exposure)

    p = sub.add_parser("compare", help="LLM pairwise judging")
    p.add_argument("--sisters", action="store_true", help="also judge sister pairs directly")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, help="judge at most this many pairs (for a trial run)")
    p.add_argument("--k", type=int, default=6, help="evidence passages per pair")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("rank", help="fit ranking models and test sister pairs")
    p.add_argument("--n-null", type=int, default=10_000)
    p.set_defaults(func=cmd_rank)

    p = sub.add_parser("simulate", help="power analysis")
    p.add_argument("--pairs", type=int, nargs="+", default=[10, 20, 30, 40])
    p.add_argument("--effects", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--sims", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.05)
    p.set_defaults(func=cmd_simulate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
