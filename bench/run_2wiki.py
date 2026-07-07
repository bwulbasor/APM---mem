"""2WikiMultiHopQA harness for APM — the multi-hop test.

Does APM's graph reasoning (spreading activation + intersection + fallback)
answer questions that REQUIRE chaining evidence, where LongMemEval (single-fact
recall) did not? Per example: ingest the context paragraphs -> entity/relation
graph -> parse the multi-hop question -> engine -> judge.

Graph-only (no hybrid tier): this tests the ENGINE. Resumable (append; skip
seen ids) because fresh extraction + flaky network.

Usage: python bench/run_2wiki.py [N]
"""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from apm import Engine, EvidenceGraph
from apm.ingest import _ask_json, ingest, parse_question, verbalize

DATA = pathlib.Path(__file__).parent / "data" / "2wiki_slice.json"
OUT = pathlib.Path(__file__).parent / "results_2wiki.jsonl"

INGEST_HINT = ("This is factual encyclopedia text about real-world entities "
               "(people, films, companies, places, dates). Extract the entities "
               "and the relations between them (directed_by, born_in, "
               "spouse/married_to, member_of, located_in, founded, "
               "date_of_birth, etc.) so the chain between entities is traversable.")

PARSE_HINT = ("This is a factual multi-hop question about real-world entities. "
              "anchors = the named entities the question STARTS from (a film, "
              "person, place). unknown_type = what it asks FOR (person, place, "
              "time, org). Include the bridge relation words (mother, director, "
              "founder, birthplace) as constraints where possible.")

JUDGE_SYSTEM = """You judge a model's answer against the gold answer for a
multi-hop question. Output ONLY JSON: {"answered": <bool>, "correct": <bool>}.
- answered: false if the model declines / says unavailable.
- correct: true only if answered AND it names the same entity/value as gold
  (a superset or exact name counts; yes/no must match). Wording may differ."""


def build_graph(ex):
    g = EvidenceGraph()
    n_nodes = 0
    def chunks(text, size=2000):
        lines, buf, out, n = text.split(". "), [], [], 0
        for l in lines:
            if n + len(l) > size and buf:
                out.append(". ".join(buf)); buf, n = [], 0
            buf.append(l); n += len(l) + 2
        if buf:
            out.append(". ".join(buf))
        return out
    for para in ex["context"]:
        doc = f"{para['title']}. {para['text']}"
        for ch in chunks(doc):
            try:
                s = ingest(g, ch, source=f"2wiki:{para['title'][:30]}",
                           hint=INGEST_HINT)
                n_nodes += s["nodes"]
            except Exception as e:
                print(f"    [warn] ingest failed: {str(e)[:70]}", flush=True)
    from apm.canon import canonicalize
    canonicalize(g)
    g.auto_contradict(); g.auto_supersede()
    from apm.taxonomy import ensure_categories
    ensure_categories(g)
    try:
        from apm.enrich import enrich_places
        enrich_places(g)
    except Exception:
        pass
    return g, n_nodes


def run_item(ex):
    g, nn = build_graph(ex)
    if nn == 0:
        return {"id": ex["id"], "type": ex["type"], "question": ex["question"],
                "gold": ex["answer"], "hypothesis": "", "verdict_class": "INFRA_FAILURE",
                "nodes": 0, "correct": False, "invented": False, "infra_failure": True}
    try:
        q = parse_question(g, ex["question"], hint=PARSE_HINT)
        v = Engine(g).answer(q)
        klass = v.klass.value
        hyp = verbalize(v, ex["question"]) if v is not None else ""
    except ValueError:
        klass, hyp, v = "UNKNOWN", "I don't have the information to answer that.", None
    if klass == "UNKNOWN":
        refused, correct = True, False
    else:
        j = _ask_json(JUDGE_SYSTEM, f"Question: {ex['question']}\nGold: {ex['answer']}\n"
                      f"Model answer: {hyp}")
        answered = bool(j.get("answered", True))
        refused = not answered
        correct = answered and bool(j.get("correct", False))
    return {"id": ex["id"], "type": ex["type"], "question": ex["question"],
            "gold": ex["answer"], "hypothesis": hyp, "verdict_class": klass,
            "nodes": nn, "correct": correct, "refused": refused,
            "invented": (not correct) and (not refused) and klass == "CONFIRMED",
            "infra_failure": False}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    data = json.loads(DATA.read_text(encoding="utf-8"))[:n]
    seen, results = set(), []
    if OUT.exists():
        for l in OUT.read_text(encoding="utf-8").splitlines():
            r = json.loads(l); seen.add(r["id"]); results.append(r)
    todo = [x for x in data if x["id"] not in seen]
    print(f"running {len(todo)} of {len(data)} ({len(seen)} done)")
    with OUT.open("a", encoding="utf-8") as f:
        for i, ex in enumerate(todo):
            t0 = time.time()
            try:
                r = run_item(ex)
            except Exception as e:
                print(f"  [{i+1}] {ex['id'][:8]} ERROR {str(e)[:60]}", flush=True); continue
            f.write(json.dumps(r) + "\n"); f.flush(); results.append(r)
            mk = "OK " if r["correct"] else ("REF" if r.get("refused") else "WRG")
            print(f"  [{i+1}/{len(todo)}] {mk} [{r['verdict_class'][:9]}] {r['type'][:12]:12s} "
                  f"{r['hypothesis'][:45]!r} ({time.time()-t0:.0f}s)", flush=True)
    scor = [r for r in results if not r.get("infra_failure")]
    if scor:
        from collections import defaultdict
        bt = defaultdict(lambda: [0, 0])
        for r in scor:
            bt[r["type"]][0 if r["correct"] else 1] += 1
        c = sum(r["correct"] for r in scor)
        print(f"\n2WIKI graph-only: {c}/{len(scor)} = {c/len(scor):.1%}  "
              f"CONFIRMED={sum(1 for r in scor if r['verdict_class']=='CONFIRMED')}  "
              f"inv={sum(r['invented'] for r in scor)}  "
              f"infra={len(results)-len(scor)}")
        for t, (ok, w) in sorted(bt.items()):
            print(f"  {t:20s} {ok}/{ok+w}")


if __name__ == "__main__":
    main()
