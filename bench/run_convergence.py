"""Run the current engine over the hand-built convergence benchmark and report
the metrics that actually test associative pattern matching:

  answer accuracy  — did the right node win (convergence / near-miss handling)?
  tier accuracy    — right confidence class (CONFIRMED/ASSUMED/UNKNOWN)?
  inventions       — CONFIRMED but wrong node (the faithfulness breach)?
  near-miss losses — a distractor beat the true convergent answer.

No LLM, no extraction: graphs are hand-built, so every number is a pure engine
measurement. Usage: python bench/run_convergence.py
"""
import pathlib
import sys
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from apm import Engine, VerdictClass
from scenarios.convergence import NOW, SCENARIOS, build

NOW_DT = datetime.fromisoformat(NOW)


def score(scen, v):
    klass = v.klass.value
    ans = v.answer.id if v.answer else None
    gold = scen["gold"]
    expect = scen["expect"]
    if expect == "UNKNOWN":
        answer_ok = klass == "UNKNOWN"
        tier_ok = klass == "UNKNOWN"
    elif expect == "not_CONFIRMED":
        answer_ok = klass != "CONFIRMED"
        tier_ok = klass != "CONFIRMED"
    else:  # CONFIRMED or ASSUMED, with a specific gold node
        answer_ok = ans == gold
        tier_ok = answer_ok and klass == expect
    invention = klass == "CONFIRMED" and gold is not None and ans != gold
    return klass, ans, answer_ok, tier_ok, invention


def main():
    rows = []
    for scen in SCENARIOS:
        g, q = build(scen)
        v = Engine(g).answer(q, now=NOW_DT, write_back=False)
        klass, ans, answer_ok, tier_ok, inv = score(scen, v)
        rows.append((scen, klass, ans, answer_ok, tier_ok, inv))
        mark = "OK " if answer_ok else "XX "
        inv_mark = "  <-- INVENTION" if inv else ""
        print(f"{mark}[{klass:9s}] {scen['id']:32s} gold={scen['gold']} "
              f"got={ans}{inv_mark}")
        if not answer_ok:
            print(f"       {scen['desc']}")
    n = len(rows)
    ans_acc = sum(r[3] for r in rows)
    tier_acc = sum(r[4] for r in rows)
    inv = sum(r[5] for r in rows)
    near_miss = [r[0]["id"] for r in rows
                 if "nearmiss" in r[0]["id"] and not r[3]]
    print("\n" + "=" * 60)
    print(f"CONVERGENCE BASELINE (current engine, hand-built graphs, n={n})")
    print(f"  answer accuracy : {ans_acc}/{n} = {ans_acc/n:.0%}")
    print(f"  tier accuracy   : {tier_acc}/{n} = {tier_acc/n:.0%}")
    print(f"  inventions      : {inv}  (CONFIRMED-but-wrong)")
    print(f"  near-miss losses: {len(near_miss)}  {near_miss}")


if __name__ == "__main__":
    main()
