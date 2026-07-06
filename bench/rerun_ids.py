"""Re-run a specific archived question set under the CURRENT engine (cache-
warm: frozen extraction is served from the LLM cache, so this is a pure
engine ablation). Usage:
  python bench/rerun_ids.py <archive.jsonl>
Reads question ids from the archive, re-runs each through run_item, appends
to bench/results.jsonl (resumable: ids already present are skipped).
"""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from run_longmemeval import DATA, OUT, run_item, summarize


def main(archive: str):
    ids = [json.loads(l)["question_id"]
           for l in open(archive, encoding="utf-8")]
    data = {x["question_id"]: x for x in json.loads(DATA.read_text(encoding="utf-8"))}
    seen, results = set(), []
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            seen.add(r["question_id"]); results.append(r)
    todo = [data[i] for i in ids if i not in seen]
    print(f"re-running {len(todo)} of {len(ids)} ({len(seen)} already done)")
    with OUT.open("a", encoding="utf-8") as f:
        for i, item in enumerate(todo):
            t0 = time.time()
            print(f"[{i+1}/{len(todo)}] {item['question_id']} "
                  f"({item['question_type']})", flush=True)
            r = run_item(item)
            f.write(json.dumps(r) + "\n"); f.flush()
            results.append(r)
            mark = "OK " if r["correct"] else ("REF" if r["refused"] else "WRG")
            print(f"    -> {mark} [{r['verdict_class']}] "
                  f"{r['hypothesis'][:80]} ({time.time()-t0:.0f}s)", flush=True)
    print("\n" + summarize(results))


if __name__ == "__main__":
    main(sys.argv[1])
