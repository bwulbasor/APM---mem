"""Judge A/B: re-judge archived answers with a second judge model and report
agreement — settles empirically whether judge choice is load-bearing.
Usage: python bench/judge_ab.py bench/results_heldout_v07.jsonl
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import apm.llm  # noqa: F401  loads .env so the VERBALIZE key exists below
os.environ["APM_LLM_MODEL_JUDGE2"] = "gemini-3.1-flash-lite"
os.environ["APM_LLM_BASE_URL_JUDGE2"] = \
    "https://generativelanguage.googleapis.com/v1beta/openai"
os.environ["APM_LLM_API_KEY_JUDGE2"] = \
    os.environ.get("APM_LLM_API_KEY_VERBALIZE", "")

from apm.ingest import _ask_json          # noqa: E402
from run_longmemeval import JUDGE_SYSTEM  # noqa: E402


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "bench/results_heldout_v07.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    agree = total = 0
    for r in rows:
        if r.get("infra_failure") or r["verdict_class"] == "UNKNOWN":
            continue
        if not r.get("hypothesis", "").strip():
            continue
        try:
            j = _ask_json(JUDGE_SYSTEM,
                          "Question: " + r["question"] +
                          "\nGold answer: " + r["gold"] +
                          "\nModel answer: " + r["hypothesis"],
                          role="judge2", cacheable=False)
        except Exception as e:
            print(f"skip {r['question_id']}: {str(e)[:70]}")
            continue
        answered = bool(j.get("answered", True))
        correct2 = (not answered) if r["abstention_q"] else \
            (answered and bool(j.get("correct", False)))
        total += 1
        if correct2 == r["correct"]:
            agree += 1
        else:
            print(f"DISAGREE {r['question_id']}: mimo={r['correct']} "
                  f"gemini={correct2}")
    if total:
        print(f"\nagreement: {agree}/{total} = {agree / total:.0%}")
    else:
        print("nothing to judge")


if __name__ == "__main__":
    main()
