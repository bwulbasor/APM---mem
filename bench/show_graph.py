"""Rebuild a benchmark question's memory graph from the LLM cache and render it
into an interactive HTML view (reuses the viz template). Cache-served: no new
extraction calls, does not disturb a running benchmark.

Usage: python bench/show_graph.py <question_id>
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from apm import Engine, EvidenceGraph
from apm.ingest import parse_question
from run_longmemeval import (DATA, INGEST_HINT, PARSE_HINT, parse_bench_date,
                             run_item, transcript)
import run_longmemeval as bench
import viz


def build_graph(item):
    """Same ingestion path as the benchmark (cache makes it near-instant).
    Delegates to the bench's own build_bench_graph — a private re-implementation
    here once drifted (chunk size 3000 vs 2200 = different cache keys), so
    autopsies silently diagnosed DIFFERENT graphs than the ones scored."""
    g, _, _ = bench.build_bench_graph(item)
    return g


def main(qid):
    data = json.loads(DATA.read_text(encoding="utf-8"))
    item = [x for x in data if x["question_id"] == qid][0]
    g = build_graph(item)
    print(f"graph: {len(g.nodes)} nodes, {g.g.number_of_edges()} edges")
    q = parse_question(g, item["question"], hint=PARSE_HINT)
    v = Engine(g).answer(q, now=parse_bench_date(item["question_date"]))
    print(v.render()[:400])

    sc = viz.collect(f"LME {qid}", g, q, parse_bench_date(item["question_date"]),
                     item["question"][:110])
    html = viz.TEMPLATE.replace("/*__DATA__*/", json.dumps([sc]))
    out = pathlib.Path(__file__).parent / f"graph_{qid}.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gpt4_2312f94c")
