"""THE DELTA EXPERIMENT: the same convergence scenarios that score 100% on
hand-built graphs, now pushed through the REAL extraction pipeline (LLM
extract -> canon -> contradict/supersede -> taxonomy/episodes) and queried
GRAPH-ONLY. The gap between 100% (hand-built) and this number IS the
extraction-starvation tax, isolated: identical reasoning engine, identical
questions, the only difference is whether a human or the extractor built the
graph.

Run graph-only so we measure the ENGINE on EXTRACTED structure:
  APM_NO_CRITIC=1 APM_NO_HYBRID=1 APM_NO_AGG=1 python bench/run_convergence_delta.py
"""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from run_longmemeval import run_item

D = "2026/06/22 (Mon) 10:00"


def u(t): return {"role": "user", "content": t}
def a(t): return {"role": "assistant", "content": t}


ITEMS = [
    {  # 1. multi-cue convergence on the General Assembly
        "question_id": "cd_ga_multi_cue", "question_type": "convergence",
        "question": "What meeting on June 26th are Anna, Marco, and Priya attending with me?",
        "answer": "The IAESTE General Assembly",
        "question_date": D,
        "haystack_dates": ["2026/06/01 (Mon) 09:00", "2026/06/10 (Wed) 14:00"],
        "haystack_sessions": [
            [u("Quick note to self — Anna, Marco and Priya are all on the IAESTE committee with me this year."),
             a("Noted. Anyone else on the committee?"),
             u("That's the core three for now.")],
            [u("Busy day on June 26th: IAESTE's General Assembly is that afternoon — it's our big board meeting where the three of us and the others vote. There's also a casual summer party that evening, and I've got a dentist appointment in the morning."),
             a("Got it — General Assembly (board meeting), summer party, dentist, all June 26th.")],
        ],
    },
    {  # 3. near-miss: same org + same date, differ by type
        "question_id": "cd_nearmiss_type", "question_type": "convergence",
        "question": "What MEETING does the chorus society have on July 3rd?",
        "answer": "The board meeting",
        "question_date": D,
        "haystack_dates": ["2026/06/20 (Sat) 11:00"],
        "haystack_sessions": [
            [u("The chorus society has two things on July 3rd — a board meeting at 6pm, and the summer concert at 8pm."),
             a("Logged both for July 3rd: board meeting 6pm, summer concert 8pm.")],
        ],
    },
    {  # 4. parent-climb: reason-for-contact via org -> meeting -> missing docs
        "question_id": "cd_parent_climb", "question_type": "convergence",
        "question": "Person X from Organization Y is trying to reach me. What outstanding item is most likely the reason?",
        "answer": "The signed budget documents that are missing and needed for Organization Y's board meeting",
        "question_date": D,
        "haystack_dates": ["2026/06/12 (Fri) 09:00", "2026/06/20 (Sat) 09:00"],
        "haystack_sessions": [
            [u("Person X is a member of Organization Y. Y has a board meeting coming up on June 23rd."),
             a("Noted: Person X in Org Y; board meeting June 23rd.")],
            [u("Reminder: the signed budget documents for Organization Y are still missing, and they're needed for that board meeting."),
             a("Logged: signed budget docs missing, needed for the Org Y board meeting.")],
        ],
    },
    {  # 5. near-miss: same people, wrong time
        "question_id": "cd_nearmiss_time", "question_type": "convergence",
        "question": "What did Sam, Jo and I do together in June?",
        "answer": "The lake trip",
        "question_date": D,
        "haystack_dates": ["2026/01/10 (Sat) 08:00", "2026/06/15 (Mon) 08:00"],
        "haystack_sessions": [
            [u("Went on a ski trip with Sam and Jo back in January, it was freezing.")],
            [u("Just got back from a lake trip with Sam and Jo this month — much warmer.")],
        ],
    },
    {  # 8. knowledge-update: current role
        "question_id": "cd_supersede_role", "question_type": "convergence",
        "question": "Where does Dana currently work?",
        "answer": "NewCorp",
        "question_date": D,
        "haystack_dates": ["2024/01/05 (Fri) 09:00", "2026/05/02 (Sat) 09:00"],
        "haystack_sessions": [
            [u("Dana works at OldCorp as an analyst.")],
            [u("Dana just started a new job — she's at NewCorp now.")],
        ],
    },
    {  # 10. near-miss: same category, agenda disambiguates
        "question_id": "cd_nearmiss_agenda", "question_type": "convergence",
        "question": "Which IAESTE meeting has the budget vote on its agenda?",
        "answer": "The General Assembly",
        "question_date": D,
        "haystack_dates": ["2026/06/18 (Thu) 10:00"],
        "haystack_sessions": [
            [u("IAESTE has two meetings coming up: a committee sync on the 24th about scheduling, and the General Assembly on the 26th where we hold the budget vote."),
             a("Noted: committee sync (24th, scheduling) and General Assembly (26th, budget vote).")],
        ],
    },
]


def main():
    results = []
    for i, item in enumerate(ITEMS):
        t0 = time.time()
        print(f"[{i+1}/{len(ITEMS)}] {item['question_id']}", flush=True)
        try:
            r = run_item(item)
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            continue
        results.append(r)
        mark = "OK " if r["correct"] else ("REF" if r["refused"] else "WRG")
        print(f"    -> {mark} [{r['verdict_class']}] {r['hypothesis'][:70]} "
              f"(graph {r['graph']['nodes']}n/{r['graph']['edges']}e, {time.time()-t0:.0f}s)",
              flush=True)
    scor = [r for r in results if not r.get("infra_failure")]
    n = len(scor)
    acc = sum(r["correct"] for r in scor)
    inv = sum(r["invented"] for r in scor)
    conf = sum(1 for r in scor if r["verdict_class"] == "CONFIRMED")
    # the 4 scenarios that were CONFIRMED on hand-built graphs
    PROOF = {"cd_ga_multi_cue", "cd_nearmiss_type", "cd_nearmiss_time",
             "cd_nearmiss_agenda"}
    survived = sorted(r["question_id"] for r in scor
                      if r["question_id"] in PROOF and r["verdict_class"] == "CONFIRMED")
    print("\n" + "=" * 60)
    print(f"DELTA EXPERIMENT (extracted graphs, graph-only, n={n})")
    print(f"  accuracy      : {acc}/{n} = {acc/n:.0%}   inventions={inv}")
    print(f"  CONFIRMED     : {conf}/{n}")
    print(f"  PROOF SURVIVAL: {len(survived)}/4  {survived}  (hand-built: 4/4)")


if __name__ == "__main__":
    main()
