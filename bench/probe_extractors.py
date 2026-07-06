"""Extractor bake-off: run one standardized bench chunk through every model
reachable on our keys; score stated-fact recall, density, JSON validity, speed.
Usage: python bench/probe_extractors.py
"""
import json
import pathlib
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from apm.llm import endpoint_keys_for
from apm.ingest import INGEST_SYSTEM

HINT = ("This is one session of a dialogue between 'User' and 'Assistant'. "
        "Represent the human as a single person node labeled exactly 'User'. "
        "Attach facts stated in first person ('I bought X') to User. "
        "Only skip greetings and filler; never skip concrete details.")

# the coupon session (51a45a95) — a 7-run extraction nemesis, all facts STATED
CHUNK = """(session date: 2023/05/27)
User: I'm trying to organize my coupons and receipts, do you have any tips on the best way to do that?
User: I like the idea of using a binder with labeled sections. I've been using the Cartwheel app from Target and it's been really helpful for saving money on household items.
User: I actually redeemed a $5 coupon on coffee creamer last Sunday, which was a nice surprise since I didn't know I had it in my email inbox.
Assistant: That's a great feeling! Many retailers, like Target, send exclusive coupons and promotions to their email subscribers. How often do you typically shop at Target?
User: I shop at Target pretty frequently, maybe every other week. I like buying household items, toiletries, and sometimes clothes for my kids. I've saved about $20 total with the Cartwheel app so far."""

# stated facts and the tokens that witness them in nodes/edges JSON
CHECKS = {
    "coupon $5 creamer":   ["coupon", "creamer"],
    "redeemed last Sunday/date": ["2023-05-21"],
    "coupon from email":   ["email"],
    "Cartwheel is Target's": ["cartwheel", "target"],
    "shops every other week": ["every other week"],
    "$20 saved so far":    ["20"],
}

CANDIDATES = [
    ("extract", "gemma-4-31b"),          # current baseline (Cerebras)
    ("extract", "gpt-oss-120b"),         # Cerebras reasoning
    ("extract", "zai-glm-4.7"),          # Cerebras reasoning
    ("parse", "gemini-3.1-flash-lite"),  # old extractor baseline (AI Studio)
    ("parse", "gemini-3-flash-preview"),
    ("parse", "gemini-3.5-flash"),
    ("parse", "gemini-3.1-pro-preview"),
    ("parse", "gemma-4-31b-it"),         # same weights, Google quota
    ("parse", "gemini-2.0-flash"),
]


def call(role, model, json_mode=True, max_tokens=6000):
    base, keys = endpoint_keys_for(role)
    payload = {"model": model,
               "messages": [{"role": "system", "content": INGEST_SYSTEM},
                            {"role": "user", "content": HINT + "\n\n" + CHUNK}],
               "max_tokens": max_tokens, "temperature": 0.0}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    last = None
    for key in keys:
        req = urllib.request.Request(
            f"{base}/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}", "User-Agent": "apm/0.1"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code in (400, 422) and json_mode:
                return call(role, model, json_mode=False, max_tokens=max_tokens)
            continue
        except Exception as e:
            last = str(e)[:60]
            continue
    return {"error": last}


def main():
    print(f"{'model':28s} {'time':>5s} {'nodes':>5s} {'edges':>5s} "
          f"{'recall':>6s}  missing")
    for role, model in CANDIDATES:
        t0 = time.time()
        data = call(role, model)
        dt = time.time() - t0
        if "error" in data:
            print(f"{model:28s} {dt:4.0f}s  ERROR {data['error']}")
            continue
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
        if not content:
            fr = (data.get("choices") or [{}])[0].get("finish_reason")
            print(f"{model:28s} {dt:4.0f}s  EMPTY (finish={fr})")
            continue
        try:
            start = content.find("{")
            depth, end = 0, -1
            for i, ch in enumerate(content[start:], start):
                depth += ch == "{"
                depth -= ch == "}"
                if depth == 0:
                    end = i
                    break
            g = json.loads(content[start:end + 1])
        except Exception as e:
            print(f"{model:28s} {dt:4.0f}s  BAD JSON ({str(e)[:40]})")
            continue
        blob = json.dumps(g).lower()
        hits = {name: all(t.lower() in blob for t in toks)
                for name, toks in CHECKS.items()}
        missing = [n for n, ok in hits.items() if not ok]
        print(f"{model:28s} {dt:4.0f}s {len(g.get('nodes', [])):5d} "
              f"{len(g.get('edges', [])):5d} "
              f"{sum(hits.values())}/{len(CHECKS):d}     "
              f"{', '.join(missing) or '-'}")


if __name__ == "__main__":
    main()
