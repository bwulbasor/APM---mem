"""End-to-end demo: raw text -> LLM ingestion -> APM engine -> grounded answer.

This is the full architecture running live:
  [0] memory formation   (LLM extracts, graph stores)
  [1] PARSE              (LLM: question -> anchors + placeholder)
  [3]-[8] REASONING      (deterministic engine — no LLM)
  [9] VERBALIZE          (LLM: chain -> language, grounding-locked)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from datetime import datetime

from apm import Engine, EvidenceGraph
from apm.ingest import ingest, parse_question, verbalize

NOW = datetime.fromisoformat("2026-06-22T10:00:00")

MEMORIES = [
    "Anna, Marco and Priya are all members of IAESTE.",
    "IAESTE plans its General Assembly in late June 2026, around 2026-06-26.",
    "Person X is a member of Organization Y.",
    "Organization Y hosts Meeting X on 2026-06-22 in the evening.",
    "Some documents from Organization Y are still missing and they are needed for Meeting X.",
]

QUESTIONS = [
    "There is something on 2026-06-26 with Anna, Marco and Priya - what is it?",
    "Person X is calling me right now. Why might Person X be calling?",
    "Where does Anna work?",  # nothing in memory supports this -> must refuse
]


def main():
    g = EvidenceGraph()
    print("=" * 74)
    print("MEMORY FORMATION (LLM ingestion)")
    print("=" * 74)
    for text in MEMORIES:
        stats = ingest(g, text)
        print(f"  +{stats['nodes']}n/{stats['edges']}e  <- {text}")
    print(f"\n  graph now: {len(g.nodes)} nodes, {g.g.number_of_edges()} edges")
    for n in g.nodes.values():
        print(f"    [{n.type.value:9s}] {n.label}  attrs={n.attrs}")

    eng = Engine(g)
    for question in QUESTIONS:
        print("\n" + "=" * 74)
        print(f"Q: {question}")
        print("-" * 74)
        try:
            q = parse_question(g, question)
        except ValueError as e:
            print(f"A: I don't have any memory of those entities. ({e})")
            continue
        anchor_labels = [g.node(a).label for a in q.anchors]
        print(f"  [parse] anchors={anchor_labels} unknown={q.placeholder.type.value} "
              f"constraints={q.placeholder.constraints} predictive={q.predictive}")
        v = eng.answer(q, now=NOW)
        print(f"  [engine]\n{v.render()}")
        print(f"\nA: {verbalize(v, question)}")


if __name__ == "__main__":
    main()
