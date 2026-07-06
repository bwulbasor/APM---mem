"""Law-firm domain smoke test for APM.

Feeds a synthetic law-firm scenario set (bench/data/law_firm_smoke.json, same
schema as longmemeval_oracle) through the SAME run_item pipeline the LongMemEval
harness uses, under two configs on the identical (cache-warm) graphs:

  GRAPH-ONLY : APM_NO_HYBRID=1 APM_NO_AGG=1  -> pure engine verdict
  HYBRID     : APM_WIDE_SUPPORT=1            -> engine + verified-quote/agg tiers

Both share APM_NO_CRITIC=1 (lean extraction) so the graph is identical between
runs and extraction is paid once. Reports accuracy, refusal, invention
(CONFIRMED-wrong), and tier attribution of correct answers, per config + a diff.

Usage: python bench/run_lawfirm.py
"""
import json
import os
import pathlib
import sys
import time
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

DATA = pathlib.Path(__file__).parent / "data" / "law_firm_smoke.json"

CONFIGS = {
    "graph-only": {"APM_NO_CRITIC": "1", "APM_NO_HYBRID": "1", "APM_NO_AGG": "1"},
    "hybrid":     {"APM_NO_CRITIC": "1", "APM_WIDE_SUPPORT": "1"},
}
FLAGS = ["APM_NO_CRITIC", "APM_NO_HYBRID", "APM_NO_AGG", "APM_WIDE_SUPPORT",
         "APM_CUED"]


def set_env(cfg):
    for f in FLAGS:
        os.environ.pop(f, None)
    for k, v in cfg.items():
        os.environ[k] = v


def summarize(name, results):
    scor = [r for r in results if not r.get("infra_failure")]
    n = len(scor)
    acc = sum(r["correct"] for r in scor)
    inv = sum(r["invented"] for r in scor)
    ref = sum(r["refused"] for r in scor)
    tiers = defaultdict(int)
    for r in scor:
        if r["correct"]:
            tiers[r["verdict_class"]] += 1
    bytype = defaultdict(lambda: [0, 0])
    for r in scor:
        bytype[r["question_type"]][0 if r["correct"] else 1] += 1
    print(f"\n===== {name}: {acc}/{n} = {acc/n:.1%} | refusal {ref} | "
          f"invention(confident-wrong) {inv} =====")
    print("  correct-answer attribution by tier: " +
          ", ".join(f"{k}={v}" for k, v in sorted(tiers.items(), key=lambda x: -x[1])))
    for t, (c, w) in sorted(bytype.items()):
        print(f"    {t:26s} {c}/{c+w}")
    return {"n": n, "acc": acc, "inv": inv, "ref": ref, "tiers": dict(tiers)}


def main():
    from run_longmemeval import run_item
    data = json.loads(DATA.read_text(encoding="utf-8"))
    print(f"loaded {len(data)} law-firm scenarios")
    all_results = {}
    for name, cfg in CONFIGS.items():
        set_env(cfg)
        print(f"\n#### CONFIG: {name}  ({cfg}) ####")
        out = pathlib.Path(__file__).parent / f"lawfirm_{name.replace('-', '')}.jsonl"
        results = []
        with out.open("w", encoding="utf-8") as f:
            for i, item in enumerate(data):
                t0 = time.time()
                try:
                    r = run_item(item)
                except Exception as e:
                    print(f"  [{i+1}] {item['question_id']} ERROR: {str(e)[:80]}",
                          flush=True)
                    continue
                f.write(json.dumps(r) + "\n"); f.flush()
                results.append(r)
                mark = "OK " if r["correct"] else ("REF" if r["refused"] else "WRG")
                print(f"  [{i+1}/{len(data)}] {mark} [{r['verdict_class']:14s}] "
                      f"{item['question_type'][:16]:16s} {r['hypothesis'][:55]} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        all_results[name] = summarize(name, results)

    # head-to-head
    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD (law-firm smoke, n={}):".format(all_results["hybrid"]["n"]))
    g, h = all_results["graph-only"], all_results["hybrid"]
    print(f"  graph-only : {g['acc']}/{g['n']} = {g['acc']/g['n']:.1%}  "
          f"inv={g['inv']}  (proof-chain CONFIRMED={g['tiers'].get('CONFIRMED',0)})")
    print(f"  hybrid     : {h['acc']}/{h['n']} = {h['acc']/h['n']:.1%}  "
          f"inv={h['inv']}  (verified-RAG TEXT={h['tiers'].get('TEXT_SUPPORTED',0)})")


if __name__ == "__main__":
    main()
