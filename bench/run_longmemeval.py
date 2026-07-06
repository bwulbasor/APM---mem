"""LongMemEval harness for APM (oracle variant: evidence sessions only).

Per question: fresh graph -> ingest each evidence session (LLM extraction) ->
parse the question -> APM engine -> verbalize -> judge against the gold answer.

Tracks the three-way outcome that IS the architecture's thesis:
  correct / wrong-but-refused (honest) / wrong-and-invented (hallucination).
Abstention questions (id ends in _abs) are scored: refusal = correct.

Usage:
  python bench/run_longmemeval.py --n 5            # first N questions
  python bench/run_longmemeval.py --n 5 --offset 20
Results append to bench/results.jsonl (resumable: seen ids are skipped).
"""
import argparse
import json
import pathlib
import sys
import time
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from datetime import datetime

from apm import Engine, EvidenceGraph, VerdictClass
from apm.ingest import (_ask_json, ingest, parse_question, relevance_veto,
                        verbalize)

DATA = pathlib.Path(__file__).parent / "data" / "longmemeval_oracle.json"
OUT = pathlib.Path(__file__).parent / "results.jsonl"

INGEST_HINT = ("This is one session of a dialogue between 'User' and 'Assistant'. "
               "Represent the human as a single person node labeled exactly 'User'. "
               "Attach facts stated in first person ('I bought X') to User. "
               "Facts, recommendations, schedules, and answers PROVIDED BY the "
               "Assistant are also durable memory: record them as assertion nodes "
               "with their concrete details in the label and attrs (e.g. "
               "'quarterly report deadline: March 3rd'), linked to their subject. "
               "Only skip greetings and filler; never skip concrete details.")

PARSE_HINT = ("The question is asked by 'User' about their own history. "
              "Words like I/me/my refer to the entity 'User' — include 'User' "
              "in anchors.")

JUDGE_SYSTEM = """You judge a model's answer against the gold answer for a question.
Output ONLY JSON: {"answered": <bool>, "correct": <bool>}.
- answered: false if the model declines, says the info is unavailable/not in
  evidence, or gives no substantive answer to what was asked — even if phrased
  affirmatively ("X was discussed, but the specific detail is not available").
- correct: true only if answered AND it semantically matches the gold answer
  (wording may differ).
- For "how many" questions, judge by the NUMERIC QUANTITY: if the stated
  number equals the gold number, it is correct even when the two sides
  describe the counted items differently."""


def transcript(session, date, max_user=2600, max_asst=1400, cap=18000):
    """Flatten a session for LLM extraction. Markdown tables are pulled out
    FIRST (flattening destroys their structure) and returned separately for
    deterministic ingestion."""
    from apm.tables import find_tables, strip_tables
    lines = [f"(session date: {date})"]
    tables = []
    for turn in session:
        raw = turn["content"]
        tables.extend(find_tables(raw))
        text = strip_tables(raw).strip().replace("\n", " ")
        limit = max_user if turn["role"] == "user" else max_asst
        if len(text) > limit:
            text = text[:limit] + "..."
        lines.append(f"{turn['role'].capitalize()}: {text}")
    out = "\n".join(lines)
    return out[:cap], tables


def parse_bench_date(d):  # "2023/04/10 (Mon) 17:50"
    return datetime.strptime(d.split(" (")[0] + " " + d.rsplit(" ", 1)[-1],
                             "%Y/%m/%d %H:%M")


def build_bench_graph(item):
    """The benchmark's exact ingestion + post-ingest chain, factored out so
    autopsy tools diagnose the SAME graph the bench scored (cache-warm:
    identical prompts -> identical cache keys -> no new extraction calls).
    Returns (graph, n_nodes, n_edges)."""
    g = EvidenceGraph()
    n_nodes = n_edges = 0
    def chunks(text, size=2200):
        """Split on line boundaries. Long inputs send free reasoning models
        into runaway thinking loops (finish=length, zero content) — short
        chunks are the model-agnostic fix, and entity resolution stitches
        entities that recur across chunks."""
        lines, buf, out = text.splitlines(), [], []
        n = 0
        for l in lines:
            if n + len(l) > size and buf:
                out.append("\n".join(buf)); buf, n = [], 0
            buf.append(l); n += len(l) + 1
        if buf:
            out.append("\n".join(buf))
        return out

    def ingest_text(text, date):
        # v1.10: pass 2 is the DELTA-CRITIC (sees transcript + what pass 1
        # captured, extracts only the missing stated facts) — replaces the
        # blind "second pass" at the same call cost. Probe: recovered the
        # asides every extractor dropped. APM_NO_CRITIC=1 for ablations.
        import os as _os
        from apm.ingest import critic_ingest
        total = {"nodes": 0, "edges": 0}
        chunk_list = chunks(text)
        for ci, chunk in enumerate(chunk_list):
            prefix = f"(part {ci+1}/{len(chunk_list)} of one session)\n" \
                if len(chunk_list) > 1 else ""
            try:
                s = ingest(g, prefix + chunk, source=f"lme:{date}",
                           hint=INGEST_HINT,
                           default_ts=parse_bench_date(date).isoformat())
                total = {"nodes": total["nodes"] + s["nodes"],
                         "edges": total["edges"] + s["edges"]}
            except Exception as e:
                print(f"    [warn] chunk {ci+1} extract failed: "
                      f"{str(e)[:100]}", flush=True)
                continue
            if _os.environ.get("APM_NO_CRITIC"):
                continue
            try:
                c = critic_ingest(g, prefix + chunk,
                                  s.get("summary", []),
                                  source=f"lme:{date}",
                                  default_ts=parse_bench_date(date).isoformat())
                total = {"nodes": total["nodes"] + c["nodes"],
                         "edges": total["edges"] + c["edges"]}
            except Exception as e:
                print(f"    [warn] chunk {ci+1} critic failed: "
                      f"{str(e)[:100]}", flush=True)
        return total

    ASSISTANT_HINT = ("These are ANSWERS AND RECOMMENDATIONS the Assistant gave "
                      "the User. Extract every concrete fact, name, schedule, "
                      "value and recommendation as assertion nodes with details "
                      "in labels and attrs, linked to 'User' and their subjects. "
                      "This content was PROVIDED TO the user — it is their memory.")

    from apm.tables import ingest_table, strip_tables
    for session, date in zip(item["haystack_sessions"], item["haystack_dates"]):
        text, tables = transcript(session, date)
        stats = ingest_text(text, date)   # chunked; failures handled per-chunk
        n_nodes += stats["nodes"]; n_edges += stats["edges"]
        # assistant-recall pass (0/3 category in every run): the main pass
        # truncates assistant turns hard; re-read the substantial ones alone
        asst = " ".join(t["content"].strip().replace("\n", " ")[:1800]
                        for t in session
                        if t["role"] == "assistant" and len(t["content"]) > 300)
        if asst:
            for ci, chunk in enumerate(chunks(strip_tables(asst)[:6000])):
                try:
                    s = ingest(g, chunk, source=f"lme:{date}:assistant",
                               hint=ASSISTANT_HINT,
                               default_ts=parse_bench_date(date).isoformat())
                    n_nodes += s["nodes"]; n_edges += s["edges"]
                except Exception as e:
                    print(f"    [warn] assistant pass chunk failed: "
                          f"{str(e)[:80]}", flush=True)
        # tables ingest AFTER prose so cells can link to freshly-made entities;
        # anchored to the User node so they are never a disconnected island
        from apm.entities import resolve as _res
        for t in tables:
            n_nodes += ingest_table(g, t, context_label=f"session {date}",
                                    default_ts=parse_bench_date(date).isoformat(),
                                    link_to=_res(g, "User", None))

    # v0.6+ post-ingest passes: deterministic knowledge-update linking, taxonomy,
    # containment enrichment, episode clustering, consolidation.
    # structural enrichment (APM_ENRICH=1): recover typed structure the proof
    # chain needs (dates, event_type, agenda, participants) from what the
    # extractor already produced — BEFORE normalization. Faithful: only
    # surfaces what is already in each node's label/edges/session date.
    import os as _oe
    if _oe.environ.get("APM_ENRICH"):
        from apm.enrich_structure import enrich_structure
        enrich_structure(g)
    # canonicalize FIRST: every downstream rule keys on canonical relations
    from apm.canon import canonicalize
    canonicalize(g)
    g.auto_contradict()
    g.auto_supersede()
    from apm.taxonomy import ensure_categories
    ensure_categories(g)
    try:
        from apm.enrich import enrich_places
        enrich_places(g)
    except Exception as e:
        print(f"    [warn] enrichment failed: {str(e)[:80]}", flush=True)
    from apm.episodes import build_episodes
    build_episodes(g)
    g.consolidate()
    return g, n_nodes, n_edges


def raw_sessions(item):
    """Untruncated per-session transcripts for cued recall — ingest-time
    truncation is part of why low-salience facts were lost."""
    out = []
    for session, date in zip(item["haystack_sessions"], item["haystack_dates"]):
        text = "\n".join(f"{t['role'].capitalize()}: {t['content']}"
                         for t in session)
        out.append({"id": date, "text": f"(session date: {date})\n" + text,
                    "ts": parse_bench_date(date).isoformat()})
    return out


def run_item(item):
    g, n_nodes, n_edges = build_bench_graph(item)
    eng = Engine(g)

    now = parse_bench_date(item["question_date"])
    if n_nodes == 0:
        # ingestion produced NOTHING (rate limits / API failure): scoring a
        # refusal over an empty graph as "honest" would poison the metrics.
        return {"question_id": item["question_id"],
                "question_type": item["question_type"],
                "question": item["question"], "gold": str(item["answer"]),
                "hypothesis": "", "verdict_class": "INFRA_FAILURE",
                "graph": {"nodes": 0, "edges": 0},
                "abstention_q": item["question_id"].endswith("_abs"),
                "refused": False, "class_text_mismatch": False,
                "correct": False, "invented": False, "infra_failure": True}
    cued_meta = {"triggered": False}
    first_klass = None
    try:
        q = parse_question(g, item["question"], hint=PARSE_HINT)
        # APM_CONTRACT=1: canonicalize parsed constraints into the shared
        # schema (keys + date values) so query fields align with node fields
        # (parser 'type'/'July 3rd' -> 'kind'/'2026-07-03'). Representation
        # alignment only — no new claims.
        import os as _oc
        if _oc.environ.get("APM_CONTRACT"):
            from apm.contract import canon_constraints
            q.placeholder.constraints = canon_constraints(
                q.placeholder.constraints, now)
        v = eng.answer(q, now=now)
        first_klass = v.klass.value
        # v1.9 CUED RECALL: first pass failed -> re-read the most relevant
        # raw sessions with the question as the extraction focus, re-run
        # the engine on a query overlay. Adoption only upgrades verdicts.
        # OFF BY DEFAULT (APM_CUED=1 enables): the 84-question ablation
        # measured cued recall NET NEGATIVE as built — conversions were
        # cancelled by displacements of correct answers and lost abstentions;
        # adoption rules need rework before it earns its place
        import os as _os2
        from apm.recall import adopt, cued_recall, needs_recall
        if _os2.environ.get("APM_CUED") and needs_recall(v):
            try:
                v2, cued_meta = cued_recall(
                    g, q, item["question"], raw_sessions(item), now=now,
                    parse_fn=lambda g2, qq: parse_question(g2, qq,
                                                           hint=PARSE_HINT))
                v, adopted = adopt(v, v2)
                cued_meta["adopted"] = adopted
            except Exception as e:
                print(f"    [warn] cued recall failed: {str(e)[:80]}",
                      flush=True)
        if v.answer is not None and (v.answer.attrs or {}).get("members"):
            try:
                v = relevance_veto(v, item["question"], g)
            except Exception as e:
                print(f"    [warn] relevance veto failed: {e}", flush=True)
        klass = v.klass.value
        hypothesis = verbalize(v, item["question"]) if v is not None else ""
        # v1.11 HYBRID TIER: trigger on the FINAL ANSWER STATE, not the
        # engine class — refusals hide inside weak-ASSUMED hedges. Answers
        # require a mechanically VERIFIED transcript quote (TEXT_SUPPORTED);
        # an unverifiable quote is discarded. CONFIRMED still means proof.
        from apm.hybrid import looks_like_refusal, text_supported_answer
        from apm.scoring import THETA_CONFIRM
        weak = (v.klass.value == "UNKNOWN"
                or (v.klass.value == "ASSUMED"
                    and (v.score < THETA_CONFIRM
                         or looks_like_refusal(hypothesis))))
        # v1.15 WIDE SUPPORT (experiment): also let the verified-quote /
        # aggregation route attempt STRONG ASSUMED answers (score>=theta),
        # adopting only when a mechanically-verified quote/table is found —
        # guess -> verified-quote is an upgrade. TEXT_SUPPORTED never CONFIRMs,
        # so this CANNOT raise the invention count. Pooled slices: ASSUMED tier
        # 44% precise vs TEXT_SUPPORTED 81%. Enable with APM_WIDE_SUPPORT=1.
        if _os2.environ.get("APM_WIDE_SUPPORT") and v.klass.value == "ASSUMED":
            weak = True
        # RESEARCH graph-only mode: APM_NO_HYBRID disables the quote tier;
        # APM_NO_AGG disables the raw-text aggregation route too, so the
        # verdict is PURE ENGINE (CONFIRMED/ASSUMED/UNKNOWN). This is the
        # research KPI config — measures what the graph reasons on its own.
        if _os2.environ.get("APM_NO_HYBRID") and _os2.environ.get("APM_NO_AGG"):
            weak = False
        if weak:
            # v1.12 AGGREGATION ROUTE first: counts/sums need a COMPLETE
            # quoted member table, not a single supporting quote
            from apm.aggregate import aggregate_answer, is_aggregation
            handled = False
            if is_aggregation(q, item["question"]):
                try:
                    agg = aggregate_answer(g, q, item["question"],
                                           raw_sessions(item))
                    if agg:
                        n = agg["value"]
                        n_txt = f"{n:g}"
                        prefix = "at least " if agg["partial"] else ""
                        names = ", ".join(str(i["item"]) for i in
                                          agg["items"][:8])
                        klass = "TEXT_SUPPORTED"
                        hypothesis = (f"{prefix}{n_txt} — text-supported "
                                      f"aggregation over quoted items: "
                                      f"{names}")
                        handled = True
                except Exception as e:
                    print(f"    [warn] aggregation route failed: "
                          f"{str(e)[:80]}", flush=True)
            if not handled:
                try:
                    ts = text_supported_answer(g, q, item["question"],
                                               raw_sessions(item))
                    if ts:
                        klass = "TEXT_SUPPORTED"
                        hypothesis = (f"{ts['answer']} (Text-supported, not "
                                      f"graph-verified — source: "
                                      f"\"{ts['quote']}\" [{ts['session']}])")
                except Exception as e:
                    print(f"    [warn] hybrid tier failed: {str(e)[:80]}",
                          flush=True)
    except ValueError:  # no anchors resolvable -> honest refusal
        klass, hypothesis, v = "UNKNOWN", "I don't have any memory about that.", None

    # the verbalizer is grounding-locked and may veto in text (refuse) even when
    # the engine emitted a verdict. What the user RECEIVES is the deliverable —
    # the judge classifies answered-vs-refused semantically (string matching on
    # refusal phrasings proved too brittle: see 7161e7e2 in results_v03).
    is_abstention = item["question_id"].endswith("_abs")
    if klass == "UNKNOWN":
        refused, correct = True, is_abstention
    else:
        j = _ask_json(JUDGE_SYSTEM,
                      f"Question: {item['question']}\nGold answer: {item['answer']}\n"
                      f"Model answer: {hypothesis}")
        answered = bool(j.get("answered", True))
        refused = not answered
        correct = (refused if is_abstention
                   else answered and bool(j.get("correct", False)))
    class_text_mismatch = refused and klass != "UNKNOWN"

    return {
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "question": item["question"],
        "gold": str(item["answer"]),
        "hypothesis": hypothesis,
        "verdict_class": klass,
        "verdict_score": round(getattr(v, "score", 0.0), 3) if v else 0.0,
        "graph": {"nodes": n_nodes, "edges": n_edges},
        "abstention_q": is_abstention,
        "refused": refused,
        "class_text_mismatch": class_text_mismatch,  # engine said CONFIRMED, text refused
        "first_klass": first_klass,      # verdict class BEFORE cued recall
        "cued": cued_meta,               # cued-recall trigger/adoption info
        "correct": correct,
        # the headline metric: a wrong claim CONFIDENTLY DELIVERED = invention
        "invented": (not correct) and (not refused) and klass == "CONFIRMED",
    }


def summarize(results):
    skipped = sum(1 for r in results if r.get("infra_failure"))
    results = [r for r in results if not r.get("infra_failure")]
    n = len(results)
    if not n:
        return f"no scoreable results ({skipped} infra failures)"
    acc = sum(r["correct"] for r in results) / n
    refusals = sum(r["refused"] for r in results) / n
    inventions = sum(r["invented"] for r in results) / n
    by_type = {}
    for r in results:
        by_type.setdefault(r["question_type"], []).append(r["correct"])
    lines = [f"n={n}  accuracy={acc:.1%}  refusal={refusals:.1%}  "
             f"invention(confident-wrong)={inventions:.1%}"
             + (f"  [{skipped} infra failures excluded]" if skipped else "")]
    for t, xs in sorted(by_type.items()):
        lines.append(f"  {t:28s} {sum(xs)}/{len(xs)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stratified", type=int, default=0,
                    help="take K questions per question_type instead of a slice")
    ap.add_argument("--heldout", type=int, default=0,
                    help="take K NEVER-SEEN questions per type (skips ids present "
                         "in any results_v*.jsonl dev archive) — the honest number")
    args = ap.parse_args()

    data = json.loads(DATA.read_text(encoding="utf-8"))
    seen = set()
    results = []
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            seen.add(r["question_id"])
            results.append(r)

    if args.heldout:
        # dev-set ids = anything ever iterated on. results_v*.jsonl are the
        # original dev archives; results_heldout_*.jsonl are the 21-question
        # slice that has since driven ~15 fix cycles — contaminated too.
        # Matching results_*.jsonl retires ALL of it; each --heldout run on a
        # frozen config then draws a never-touched slice (2026-07-04).
        dev_ids = set(seen)
        for f in OUT.parent.glob("results_*.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                dev_ids.add(json.loads(line)["question_id"])
        buckets = {}
        for x in data:
            if x["question_id"] in dev_ids:
                continue
            key = ("abstention" if x["question_id"].endswith("_abs")
                   else x["question_type"])
            buckets.setdefault(key, []).append(x)
        todo = [x for xs in buckets.values() for x in xs[:args.heldout]]
        print(f"HELD-OUT mode: {len(dev_ids)} dev ids excluded")
    elif args.stratified:
        buckets = {}
        for x in data:
            key = ("abstention" if x["question_id"].endswith("_abs")
                   else x["question_type"])
            buckets.setdefault(key, []).append(x)
        todo = [x for xs in buckets.values() for x in xs[:args.stratified]
                if x["question_id"] not in seen]
    else:
        todo = [x for x in data[args.offset:args.offset + args.n]
                if x["question_id"] not in seen]
    print(f"running {len(todo)} questions ({len(seen)} already done)")

    with OUT.open("a", encoding="utf-8") as f:
        for i, item in enumerate(todo):
            t0 = time.time()
            print(f"[{i+1}/{len(todo)}] {item['question_id']} "
                  f"({item['question_type']}): {item['question'][:70]}", flush=True)
            try:
                r = run_item(item)
            except Exception:
                traceback.print_exc()
                continue
            f.write(json.dumps(r) + "\n"); f.flush()
            results.append(r)
            mark = "OK " if r["correct"] else ("REF" if r["refused"] else "WRG")
            print(f"    -> {mark} [{r['verdict_class']}] {r['hypothesis'][:90]} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print("\n" + summarize(results))


if __name__ == "__main__":
    main()
