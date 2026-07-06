"""Mass autopsy: for failing questions, rebuild the graph from cache and dump
the full decision anatomy — parse output, candidate table, verdict — so every
miss gets a named cause. Usage: python bench/autopsy.py <qid> [<qid> ...]
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from apm import Engine
from apm.activation import spread
from apm.ingest import parse_question
from apm.scoring import match_constraints, noisy_or
from run_longmemeval import DATA, PARSE_HINT, parse_bench_date
from show_graph import build_graph


def autopsy(qid: str, data):
    item = [x for x in data if x["question_id"] == qid][0]
    print("=" * 78)
    print(f"{qid} ({item['question_type']})")
    print(f"Q: {item['question'][:100]}")
    print(f"gold: {str(item['answer'])[:80]}")
    g = build_graph(item)
    print(f"graph: {len(g.nodes)} nodes")
    try:
        q = parse_question(g, item["question"], hint=PARSE_HINT)
    except ValueError as e:
        print(f"PARSE FAILED: {e}")
        return
    print(f"parse: mode={q.mode} type={q.placeholder.type.value} "
          f"constraints={q.placeholder.constraints} under={q.placeholder.under} "
          f"options={q.options} unresolved={q.unresolved}")
    now = parse_bench_date(item["question_date"])
    # candidate table before gating
    activated = spread(g, q.anchors, now)
    cands = []
    for nid, paths in activated.items():
        n = g.node(nid)
        if n.type != q.placeholder.type or nid in q.anchors:
            continue
        m = match_constraints(n, q.placeholder.constraints,
                              required=q.placeholder.required)
        if m > 0:
            cands.append((noisy_or(paths) * m, n.label))
    cands.sort(reverse=True)
    print("top candidates:")
    for s, label in cands[:6]:
        print(f"  {s:.2f}  {label[:70]}")
    # does the gold answer text appear ANYWHERE in the graph?
    gold = str(item["answer"]).lower()
    gold_tokens = [t for t in gold.replace(",", " ").split() if len(t) > 3][:3]
    hits = [n.label for n in g.nodes.values()
            if any(t in n.label.lower() for t in gold_tokens)]
    print(f"gold-ish nodes in graph: {hits[:4] or 'NONE — extraction miss'}")
    v = Engine(g).answer(q, now=now, write_back=False)
    print(f"verdict: {v.klass.value} -> "
          f"{v.answer.label[:60] if v.answer else None}")


if __name__ == "__main__":
    data = json.loads(DATA.read_text(encoding="utf-8"))
    for qid in sys.argv[1:]:
        try:
            autopsy(qid, data)
        except Exception as e:
            print(f"{qid}: autopsy failed: {e}")
