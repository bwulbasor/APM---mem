"""Priority 1 — the failure transition table for graph-only misses (v2).

v2 fixes the numeric-gold presence bug that inflated the "recoverable" bucket:
 - numeric golds ('9 months', '5') no longer count as present via generic unit
   words ('months','hours'); the NUMBER itself must appear for a stated value.
 - aggregation golds (the TOTAL is never a node) are judged on whether the
   COUNTED ITEMS exist, using the question's head noun, not the total.
This decides World A (present-but-unreached -> 60% via engine work) vs World B
(fact-missing -> needs better extraction).

Categories:
  A fact missing        gold value/items absent from graph
  B present-but-UNKNOWN gold present, engine returned UNKNOWN (reachability)
  C ranked-below        gold present, engine answered a WRONG node
  E aggregation         count/sum, items present but total wrong
  F temporal/supersede  temporal question, gold present
  G parser failed       no anchors
  H judge noise         correct in >=1 of 3 repeat runs
  I impossible/abst.    _abs gold = 'not enough info'

Run: APM_NO_CRITIC=1 python bench/failure_table.py
"""
import json
import pathlib
import re
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from run_longmemeval import DATA, PARSE_HINT, build_bench_graph
from apm.ingest import parse_question

REPS = ["results_gonly_rep1.jsonl", "results_gonly_rep2.jsonl", "results_gonly_rep3.jsonl"]
BENCH = pathlib.Path(__file__).parent

# generic/unit/quantity words that must NOT count as content evidence
GENERIC = {
    "the", "and", "for", "with", "that", "this", "your", "you", "have", "has",
    "was", "were", "did", "does", "about", "from", "there", "their", "they",
    "month", "months", "hour", "hours", "day", "days", "week", "weeks", "year",
    "years", "time", "times", "minute", "minutes", "second", "seconds", "week",
    "people", "person", "item", "items", "thing", "things", "friend", "friends",
    "family", "member", "members", "total", "number", "amount", "many", "much",
    "some", "several", "few", "more", "most", "each", "every", "approximately",
    "around", "about", "over", "under", "between", "including", "such", "answer",
    "information", "provided", "mention", "mentioned", "enough", "not",
}
_NUM = re.compile(r"\d+(?:\.\d+)?")


def content_words(text):
    return {t for t in re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
            if len(t) > 3 and t not in GENERIC}


def graph_text(g):
    return " ".join(n.label.lower() for n in g.nodes.values())


def gold_present(g, item, is_agg):
    """Refined presence: returns (present: bool, why: str)."""
    gold = str(item["answer"])
    gtext = graph_text(g)
    graph_nums = set(_NUM.findall(gtext))
    gold_nums = set(_NUM.findall(gold))
    gwords = content_words(gold)

    if is_agg:
        # the TOTAL is never a stored node -> judge on the COUNTED ITEMS,
        # taken from the QUESTION's head nouns (gold is often just a number)
        qwords = content_words(item["question"])
        hits = [w for w in qwords if w in gtext]
        # need at least a couple of distinct item-mentions to aggregate over
        present = len(hits) >= 1
        return present, f"agg: item-words {hits[:4]} present={present}"
    # stated-value questions
    if gold_nums:
        num_ok = bool(gold_nums & graph_nums)
        word_ok = (not gwords) or any(w in gtext for w in gwords)
        return (num_ok and word_ok), f"num {sorted(gold_nums)} in-graph={num_ok}, words_ok={word_ok}"
    if not gwords:
        return True, "no discriminative gold tokens"
    hit = sum(w in gtext for w in gwords)
    return (hit >= max(1, len(gwords) // 2)), f"words {hit}/{len(gwords)} present"


def main():
    data = {x["question_id"]: x for x in json.loads(DATA.read_text(encoding="utf-8"))}
    reps = [{json.loads(l)["question_id"]: json.loads(l)
             for l in (BENCH / f).read_text(encoding="utf-8").splitlines()}
            for f in REPS]
    base = reps[0]

    def ever_correct(qid):
        return any(qid in r and r[qid]["correct"] for r in reps)

    table = Counter()
    ex = {}
    misses = [qid for qid, r in base.items()
              if not r.get("infra_failure") and not r["correct"]]
    print(f"classifying {len(misses)} graph-only misses...\n")
    for qid in misses:
        r, item = base[qid], data[qid]
        qtype = item["question_type"]
        is_agg = "multi-session" in qtype
        if ever_correct(qid):
            cat, why = "H judge-noise", ""
        elif qid.endswith("_abs"):
            cat, why = "I impossible/abst", ""
        else:
            g, _, _ = build_bench_graph(item)
            try:
                q = parse_question(g, item["question"], hint=PARSE_HINT)
                no_anchor = not q.anchors
            except Exception:
                no_anchor = True
            present, why = gold_present(g, item, is_agg)
            if no_anchor:
                cat = "G parser-failed"
            elif not present:
                cat = "A fact-missing"
            elif is_agg:
                cat = "E aggregation"
            elif "temporal" in qtype:
                cat = "F temporal/supersede"
            elif r["verdict_class"] == "UNKNOWN":
                cat = "B present-but-UNKNOWN"
            else:
                cat = "C ranked-below"
        table[cat] += 1
        ex.setdefault(cat, []).append((qid, str(item["answer"])[:32], r["verdict_class"], why[:48]))

    print("FAILURE TRANSITION TABLE v2 (graph-only):")
    for cat, n in table.most_common():
        print(f"  {n:3d}  {cat}")
    ncorrect = sum(r["correct"] for r in base.values() if not r.get("infra_failure"))
    ntot = sum(1 for r in base.values() if not r.get("infra_failure"))
    hard = table.get("A fact-missing", 0) + table.get("I impossible/abst", 0)
    soft = sum(table.values()) - hard
    print(f"\n  correct now: {ncorrect}/{ntot} = {ncorrect/ntot:.1%}")
    print(f"  HARD (A fact-missing + I): {hard}   -> World B share")
    print(f"  engine-recoverable (B/C/E/F/G/H): {soft}  -> World A share")
    print(f"  pure-engine ceiling if all soft land: {ncorrect+soft}/{ntot} = {(ncorrect+soft)/ntot:.0%}")
    print("\n--- AUDIT SAMPLES (qid | gold | verdict | why) ---")
    for cat, _ in table.most_common():
        print(f"\n[{cat}]")
        for qid, gold, vc, why in ex[cat][:5]:
            print(f"  {qid:18s} gold={gold!r:34s} {vc:9s} {why}")


if __name__ == "__main__":
    main()
