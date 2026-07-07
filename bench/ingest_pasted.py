"""Test a PASTED extractor graph (e.g. from GPT-5.5) through APM's engine.

Isolates the "does a stronger EXTRACTOR lift APM" question WITHOUT wiring the
model into the pipeline: you run APM's extraction prompt on GPT-5.5 by hand,
save its JSON, and this ingests THAT graph + runs the same engine + judge.

Usage: python bench/ingest_pasted.py <qid> <path-to-gpt5-graph.json>
  <qid> is a 2wiki_slice.json id (question + gold looked up automatically).
Compares the result to what gemma-4-31b got (results_2wiki.jsonl).
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from apm import Engine, EvidenceGraph
from apm.ingest import (_ask_json, _extract_json, _merge_extraction,
                        parse_question, verbalize)
from run_2wiki import JUDGE_SYSTEM, PARSE_HINT

SLICE = pathlib.Path(__file__).parent / "data" / "2wiki_slice.json"
GEMMA = pathlib.Path(__file__).parent / "results_2wiki.jsonl"


def main(qid, graph_path):
    ex = {x["id"]: x for x in json.loads(SLICE.read_text(encoding="utf-8"))}[qid]
    gem = {json.loads(l)["id"]: json.loads(l)
           for l in GEMMA.read_text(encoding="utf-8").splitlines()}.get(qid, {})

    raw = pathlib.Path(graph_path).read_text(encoding="utf-8")
    data = _extract_json(raw)   # tolerant of prose/fences around the JSON

    g = EvidenceGraph()
    s = _merge_extraction(g, data, source="gpt5:pasted", default_ts=None)
    from apm.canon import canonicalize
    canonicalize(g); g.auto_contradict(); g.auto_supersede()
    from apm.taxonomy import ensure_categories
    ensure_categories(g)
    try:
        from apm.enrich import enrich_places
        enrich_places(g)
    except Exception:
        pass

    print(f"Q: {ex['question']}")
    print(f"gold: {ex['answer']}")
    print(f"GPT-5.5 graph: {s['nodes']} nodes, {s['edges']} edges")
    try:
        q = parse_question(g, ex["question"], hint=PARSE_HINT)
        v = Engine(g).answer(q)
        klass = v.klass.value
        hyp = verbalize(v, ex["question"]) if v else ""
    except ValueError:
        klass, hyp, v = "UNKNOWN", "I don't have the information.", None

    if klass == "UNKNOWN":
        correct = False
    else:
        j = _ask_json(JUDGE_SYSTEM, f"Question: {ex['question']}\nGold: {ex['answer']}\n"
                      f"Model answer: {hyp}")
        correct = bool(j.get("answered", True)) and bool(j.get("correct", False))

    print(f"\nAPM-on-GPT5:  [{klass}] {hyp[:80]!r}  -> {'CORRECT' if correct else 'wrong/refused'}")
    print(f"APM-on-gemma: [{gem.get('verdict_class','?')}] {gem.get('hypothesis','')[:80]!r}"
          f"  -> {'CORRECT' if gem.get('correct') else 'wrong/refused'}")
    print(f"\nVERDICT: extractor swap {'FLIPPED IT (gemma miss -> GPT5 correct)' if correct and not gem.get('correct') else 'no change' if correct==gem.get('correct') else 'REGRESSED'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
