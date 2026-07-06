# APM — Session Transfer (2026-07-06)

> **2026-07-06 (LATER) STATE — 70% ON UNSEEN DATA + A DELIBERATE TWO-TRACK
> SPLIT. (v1.15 "wide-support", 137 tests green.) Full detail: HANDOFF.md top
> entry. Read that before doing anything.
>
> PRODUCT NUMBER (ship config `APM_NO_CRITIC=1 APM_WIDE_SUPPORT=1`, Cerebras
> gemma-4-31b free):
>   dev-42 76.2/0 · valid-41 63.4/0 · confirm-38 68.4/1 · fresh-29 82.8/0
>   POOLED UNSEEN (n=108) = 70.4% / 1 invention (0.9%).  <- the honest headline
>   (fresh-29 lacked preference cat; pooled 70.4% is the figure to quote.)
> Archives: results_*_widesupport*.jsonl.
>
> THE FINDING THAT SET THE DIRECTION: attribution of correct answers is now
> ~81% verified-RAG (TEXT_SUPPORTED), ~10% graph-heuristic (ASSUMED), ~5%
> graph-PROOF (CONFIRMED). CONFIRMED has been FLAT at 4/121 (~3%) all campaign;
> on the fresh slice the graph contributed 0% of correct answers. HONEST
> VERDICT: this is verified RAG with a graph-based HONESTY GATE (the gate =
> the real innovation, holds inventions ~0 w/ provenance; but it is a gate,
> not a reasoner). The "graph reasons" thesis is currently NOT supported.
>
> DECISION (Abiram): ONE LEVER — IMPROVE THE GRAPH ONLY. Both the research
> claim AND the shippable product are pursued via the same effort: grow the
> graph's real reasoning share. As the graph reclaims answers, each migrates
> TEXT_SUPPORTED->CONFIRMED (better provenance = better product) AND pure-graph
> accuracy rises toward the product number (thesis proven). Verified-RAG is the
> BASELINE THE GRAPH MUST CLIMB TOWARD — do NOT optimize the RAG fallback.
>
> TWO CONFIGS, MEASURE BOTH EVERY CHANGE:
>   PRODUCT: `APM_NO_CRITIC=1 APM_WIDE_SUPPORT=1`            -> 70.4% unseen.
>   RESEARCH KPI: `APM_NO_CRITIC=1 APM_NO_HYBRID=1 APM_NO_AGG=1` -> PURE ENGINE
>     verdict. Graph-only baseline (n=108, results_graphonly_baseline.jsonl):
>     30.6% acc / 4.6% CONFIRMED coverage; true proof-chain coverage only
>     3/108 (2.8%) — the rest of graph-only's 30.6% is ASSUMED heuristic. The
>     gap to 70% (~40pts RAG-carried) IS the research runway. New guard in
>     run_longmemeval.py enables this mode.
>
> NEXT (graph track): (1) lock the graph-only baseline number. (2) DURATION +
> ORDERING temporal: relative-time phrases ("3 months ago") ARE in the graph,
> session-dated — resolve them vs session date and subtract -> CONFIRMED points
> (5+ instances). (3) multi-session count via GRAPH-scoped membership (not
> raw-text tables). (4) verify each graph win DISPLACES a TEXT_SUPPORTED
> product answer with CONFIRMED, product acc not dropping. LoCoMo harness is
> DEFERRED behind the graph work.**

> **2026-07-06 STATE — THE TIER ARCHITECTURE, CONFIRMED (v1.14.2, 137 tests).
> HEADLINE NUMBERS (all Cerebras gemma-4-31b, free, APM_NO_CRITIC=1):
>   dev-42            64.3% / 0 inventions  (results_dev64_v113)
>   validation-41     56.1% / 0 inventions  (results_revalidation_v1142)
>   CONFIRMATION-38   57.9% / 1 invention   (results_confirm_v1142_cerebras) <- THE untouched paper number
> Pooled unseen (validation+confirmation, n=79): ~57%, 1 confident-wrong.
> vs Zep 63.8 / Mem0 49 — competitive on accuracy, uncontested on faithfulness.
> Answer contract: CONFIRMED(graph proof) / ASSUMED(named rule) /
> TEXT_SUPPORTED(mechanically verified transcript quote) / refusal.
> Winning config = LEAN single-pass extraction (critic OFF = +12pts;
> densification ALWAYS lost), engine+guards, hybrid quote tier, aggregation
> table route. Key lesson (Abiram's call): "the graph needs a judge, not a
> bigger memory." 10 invention classes found+closed across the campaign.
>
> CONFIRMATION-SLICE INVENTION (do NOT fix from this slice — it is the paper
> number; fixing spends it): gpt4_8c8961ae, an ordering question ("which trip
> first, Europe or Thailand solo") CONFIRMED Europe off a date when gold was
> Thailand — likely a MISEXTRACTED date (garbage-in boundary: the gate
> verifies chains, not facts) OR an ordering edge case with one
> mis-dated endpoint. Autopsy on a FUTURE dev slice, not this one.
>
> NEXT SESSION (priority order):
> (1) LoCoMo harness + 2WikiMultiHopQA — friendlier benchmarks (clean-text
>     extraction, multi-hop = our strength; LoCoMo has an unanswerable
>     category = our home turf). Same harness shape as bench/run_longmemeval.py.
> (2) PUBLISH THE METRIC: refuses-vs-invents 2x2 (correct / honest-refusal /
>     TEXT_SUPPORTED / confident-wrong) run over Mem0+Zep+raw-LLM baselines.
> (3) MCP server + 3-method API (remember / recall -> tiered verdict / why ->
>     chain) — the standard-memory-architecture artifact; dogfood on
>     [[portable-brain]].
> (4) One more untouched slice (n>=40) ONLY if a claim needs re-confirming
>     after any engine change; ~300 LongMemEval qs remain unburned.
> (5) OPTIONAL: frontier/DeepSeek extractor as an ablation (attacks Type-B
>     dropped-fact misses; expect the density<->ranking trade to erode gains).
> Details + full invention ledger in HANDOFF.md (top entry).**

> **NEXT SESSION PLAN (agreed with Abiram 2026-07-04 night). Goal restated:
> APM as a STANDARD memory architecture AND remarkable/competitive (beat
> Zep's 63.8% while holding invention=0). Order of work:
> (1) LLM TIE-ARBITER (SPEC §3.5 role 2: LLM picks among ENGINE-produced
>     tied candidates + their graph context; stays ASSUMED; ~3 misses/25).
> (2) JUDGE AUDIT (gpt4_f2262a51 scored "3 doctors" wrong vs gold "three
>     doctors" — check ~10 judged-wrong rows for donated points).
> (3) THIRD CLEAN SLICE — honest v1.5/v1.6 number (v1.5 = past-value guard +
>     convergent activation + JSON-mode never measured together on untouched
>     data). --heldout now auto-excludes all archives.
> (4) DEEPSEEK RUN (~$2 top-up) — THE decisive experiment for the ≥64% claim.
> (5) If (4) lands: MCP server + 3-method API (remember/recall/why) for
>     adoption, and Phase 7 trace factory (pure engine) toward the SPEC §9
>     fine-tune — the "remarkable" flagship claim.
> Current state: v1.5, 109 tests green, Cerebras pool 6 keys, dev2 archived
> (results_dev2_v15_cerebras.jsonl). Details in HANDOFF.md.**

> **2026-07-04 UPDATE 2 — v1.4 + FRESH-SLICE RESULT: 47.6% / 0 inventions on
> 21 NEVER-SEEN questions (results_freshslice_v14_cerebras.jsonl) — zero
> generalization gap vs the dev 21 (also 47.6%). Temporal 4/4. Bucket-1
> engine fixes shipped (107 tests): echo cap, span dedupe, sum-over-durations
> + lexical anchors, count oversize guard, contradiction propagation.
> --heldout now excludes ALL archives. DeepSeek held for the endgame per
> Abiram. Remaining walls = extraction density (assistant 0/3, sum addend
> completeness). Details in HANDOFF.md.**

> **2026-07-04 UPDATE — canonicalization DONE, regression recovered.**
> v1.3 = 42.9% / 0 inventions on the held-out 21 (Cerebras, was 33.3%).
> apm/canon.py + tightened INGEST_SYSTEM shipped, 98 tests green. Two-stage
> ablation proved the recovery came from PROMPT-side shape control (stage 1,
> canon alone on frozen graphs: zero changes; stage 2, tightened prompt:
> +4/-2). Fragility is extractor SHAPE (echo nodes, duplicate assertions,
> reversed containment), not verb spelling. Autopsy tooling fidelity bug
> fixed (build_bench_graph). Next: queued engine fixes in HANDOFF.md (echo
> penalty, contradiction propagation, contains-flip) + the fresh-slice
> generalization number. Details in HANDOFF.md v1.3 entry.

Self-contained handoff for a fresh chat. Read this + [SPEC.md](SPEC.md) + [HANDOFF.md](HANDOFF.md).

## What APM is (one paragraph)
**Associative Pattern Matching** — a reasoning architecture where a knowledge graph does
the inference and the LLM only translates. Input → LLM extracts entities/relations into an
evidence graph → deterministic engine (spreading activation, fallback ladder, intersection,
contradiction, ordering/count/sum/span/suggest modes) → **three-class gate**: CONFIRMED
(evidence chain exists) / ASSUMED (tagged guess) / UNKNOWN (refuse). Thesis: **structural,
not statistical, anti-hallucination** — a confident answer without a chain is unrepresentable.
Endgame (Phases 7-9): fine-tune a small model so graph-ops are its native output.

## Where the code is
`D:\Claude\apm-architecture\` — `apm/` (engine: graph, activation, scoring, fallback, gate,
engine, entities, taxonomy, episodes, embed, enrich, ingest, llm, tables), `scenarios/`,
`tests/` (89 green), `bench/` (LongMemEval harness + result archives), `demo*.py`, `viz.py`.

## Status: WORKING, thoroughly evaluated, at a clear decision point
- Engine complete: 6 query modes, contradiction/supersede, decay/Hebbian, entity resolution
  (numbers-are-identity), taxonomy+episodes+consolidation, embedding tiebreak. **89 tests green.**
- Benchmark: LongMemEval oracle, held-out slice of 21 questions (id-excluded from dev set).
- **Best held-out: v0.7 = 48%, 0 inventions.** Trend swings 30-48% BY EXTRACTOR.
- **Invention rate = 0 across essentially every run** (one guard: unverified-update).

## THE key finding (drives everything next)
Accuracy plateaued ~40% and is **fragile to extractor choice** because the engine's ranking
rules were overfit to gemini's output conventions. A denser/faster extractor (Cerebras) made
accuracy DROP (30%) because its different relation/attr vocabulary made engine rules no-op →
refusals. **Stable metrics: 0 inventions, ~67% precision-on-attempts, honest refusal.**
Conclusion: raw accuracy is the wrong target; the defensible claim is verifiable/never-lies.

## IMMEDIATE NEXT TASK (user approved, build first)
**Extractor-agnostic canonicalization layer** (pure engine, no API needed):
1. Tighten `INGEST_SYSTEM` (apm/ingest.py) to a controlled relation vocabulary + attr keys.
2. New `apm/canon.py`: deterministic relation-synonym → canonical map + attr-key normalize;
   wire as the FIRST post-ingest pass in `bench/run_longmemeval.py run_item()` (before
   auto_contradict). Canonical set must cover what engine rules key on: functional rels
   (lives_in/works_at — see graph.FUNCTIONAL_RELS), parent rels (member_of/part_of),
   containment (located_in — see taxonomy.CONTAINMENT_RELS), attends, about/needed_for.
3. Add tests (gemma-style relations → canonical → engine rules fire). Then re-run v1.2
   Cerebras (fast: ~15 min) and check the flips recover (Glass Menagerie, mortgage).

## THEN (after canonicalization, per user's real goal)
- User's stated target was 80% on the 21-set, BUT: that set is now dev-contaminated (drove
  many fixes) AND accuracy is extractor-fragile. The HONEST deliverable is a **fresh untouched
  slice** (use `--heldout` — it auto-excludes all ids in results_v*.jsonl archives; may need
  to also exclude the current held-out 21) run once for a generalization number, + writeup.
- Two open engine-only failure classes worth fixing (no API): multi-session aggregation
  (0/3 every run — count/sum routing + scoped membership) and span-ranking on dense graphs.

## Config / keys (in .env, gitignored)
- Per-role models+endpoints+KEY POOLS (comma-sep, rotate on 429/403). Roles: EXTRACT, PARSE,
  VERBALIZE, VETO, JUDGE, ENRICH, SUMMARIZE, EMBED.
- CURRENT extractor: Cerebras gemma-4-31b (3 keys pooled, ~1.2s, 1M tok/day each, 8k ctx).
  Cerebras base https://api.cerebras.ai/v1. Models on key: gemma-4-31b (instruct, use this),
  gpt-oss-120b + zai-glm-4.7 (reasoning, avoid).
- Others: PARSE/VERBALIZE/VETO/JUDGE = gemini-3.1-flash-lite (3-key gemini pool, AI Studio
  https://generativelanguage.googleapis.com/v1beta/openai). EMBED = gemini-embedding-001.
- DeepSeek key exists but 402 zero-balance (needs ~$2 top-up → deepseek-chat = best extractor).
- FREE-TIER LESSON (proven 8 ways): no free tier gives dense+fast+sustainable except Cerebras
  (dense+fast+sustainable but engine overfit to gemini shape → canonicalization fixes this).
- LLM cache: .llm_cache.sqlite (content-keyed; frozen config re-runs are instant/free).

## How to run
`python -m pytest tests/ -q` · `python bench/run_longmemeval.py --heldout 3` (21 Qs) ·
`python bench/autopsy.py <qid>` (rebuild+diagnose from cache) ·
`python bench/show_graph.py <qid>` (render graph HTML) · `python viz.py`.
Monitor runs by tailing bench/results.jsonl; results append (resumable). INFRA_FAILURE =
0-node ingest (quota), excluded from scoring.

## Meta-guidance carried from this session
- User values honesty over rosy framing; flag overfitting/sprawl proactively.
- Don't chase the accuracy number blindly — it's the wrong, fragile target; consolidate the
  robust claim. Don't test more free keys (exhausted). Prefer engine/pure fixes over API spend.
- Every fix = derived from a named failure via autopsy, tested, then re-run. Archive results.
