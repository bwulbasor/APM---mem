# HANDOFF — Associative Pattern Matching (APM)

**Last updated:** 2026-07-02 (Phase 1 milestone reached)
**Owner:** Abiram

## What this project is
A new reasoning architecture: LLM + evidence graph where the **graph performs inference**
(spreading activation, fallback ladder, constraint intersection) and the LLM only parses input
and verbalizes output. Every answer is gated: CONFIRMED (evidence chain) / ASSUMED (tagged
default) / UNKNOWN (refuse). Goal: structural hallucination prevention; endgame is a
memory-native LLM (fine-tuned so graph ops are its only path to facts — SPEC.md §9);
SOTA target: LongMemEval/LoCoMo vs Zep (63.8%) and Mem0 (49%).

**Naming:** "Associative Pattern Matching" = the mechanism (like *attention*). Architecture
name TBD (candidate: Graph Link Models). Acronym APM collides with App Performance Monitoring.

## Current state — Phases 1-6 IN FLIGHT (40 tests green)

**======================================================================**
**2026-07-07 (GPT-5.5 EXTRACTOR TEST) — THE KEY FINDING: BOTH extraction AND an**
**ENGINE-CONSUMPTION BUG were blocking; fixing the engine (free) + better**
**extraction = the answer flips. This nuances 'the wall is extraction'.**

METHOD: manual extractor test (bench/ingest_pasted.py) — ran APM's extraction
prompt on GPT-5.5 by hand (prompts in bench/gpt5_extract/*.txt), fed GPT-5.5's
graph JSON into APM's engine. Isolates 'does a stronger EXTRACTOR lift APM'
with zero API/network. First question (2Wiki comparison 'which film came out
first, Blind Shaft or The Mask of Fu Manchu?', gold Mask):
 - GPT-5.5 EXTRACTED BOTH release years as assertions (2003, 1932) — gemma had
   NOT (gemma refused 'lack release dates'). So better extraction DID supply
   the missing facts.
 - BUT APM on GPT-5.5's graph STILL REFUSED (identical UNKNOWN) — because
   `_order`/`_span` only read a date ATTR on the option node; the years lived
   in LINKED assertion nodes ('2003 release year for X' --about--> film). An
   ENGINE-CONSUMPTION bug, not extraction. (Same class as the temporal-NLP
   null.)
FIX (engine._linked_date, DEFAULT ON): when an ordering/span candidate has no
own date, surface a date/year from a linked ASSERTION neighbour. Yields ASSUMED
(not CONFIRMED — the date isn't a stated attr on the node), so faithfulness-
safe. 141 tests green; dev-42 graph-only UNCHANGED 17/42, 0 inv (only helps
where linked dates exist, e.g. 2Wiki comparisons).
RESULT: re-ran GPT-5.5's graph -> APM answers 'The Mask of Fu Manchu' CORRECT
(ASSUMED). **THE MISS FLIPPED.** Neither piece alone works: gemma+fix has no
years; GPT-5.5+no-fix can't read them. BOTH needed.
IMPLICATION: 'the wall is extraction' is HALF the story. For date/comparison
questions there was ALSO an engine-consumption bug (now fixed) that would have
masked ANY better extractor's gains. So: (1) the _linked_date fix is a real
free win that unblocks consuming extracted dates; (2) with it in place, a
BETTER EXTRACTOR (DeepSeek/GPT-5.5-quality) can NOW actually lift date/
comparison questions — the DeepSeek experiment is more promising than before.
TODO: user is running the other 3 gpt5_extract prompts (q2 death-date, q3
director-birthplace, q4 grandfather-chain) on GPT-5.5; feed each via
`python bench/ingest_pasted.py <qid> <json>` to see how many flip. There may
be MORE engine-consumption bugs like this one hiding behind extraction.

**======================================================================**
**2026-07-07 (2WikiMultiHopQA TEST) — 'WRONG EXAM' HYPOTHESIS REFUTED; WALL IS**
**EXTRACTION ON BOTH BENCHMARKS. But the ENGINE'S MULTI-HOP CHAINING WORKS.**

Built bench/run_2wiki.py (datasets lib installed; voidful/2WikiMultihopQA
validation; 32 examples balanced across compositional/comparison/inference/
bridge_comparison saved to bench/data/2wiki_slice.json). Ingest context
paragraphs -> entity/relation graph -> parse multi-hop Q -> engine -> judge.
Graph-only, gemma-4-31b extractor. RESULT (n=32):
  **5/32 = 16%** | refused 17 (53%) | CONFIRMED 1 | invention 1 | infra 0
  compositional 3/9 (33%) · comparison 1/8 · inference 1/6 · bridge_comparison 0/9
FINDINGS:
 - APM's MULTI-HOP CHAINING WORKS: compositional 'mother of the director of
   film X' / 'award the director won' answered correctly via spreading
   activation (film->director->mother/award). The convergence engine DOES do
   relation-composition — the open architectural question is answered YES.
 - BUT 2Wiki is HARDER than LongMemEval (16% vs 40%), not easier. Why: multi-
   hop MULTIPLIES extraction dependencies (need the chain relation AND the
   endpoint fact); comparison/bridge need DATES (extraction miss); bridge_
   comparison = two chains + compare + dates = 0/9 (every dependency stacks).
 - The 1 invention: bridge_comparison 'which director died first' CONFIRMED off
   a bad/incomplete date — same garbage-in-date ordering class as gpt4_8c8961ae.
   Faithfulness otherwise held (53% honest refusals).
CONCLUSION: the wall is IDENTICAL on both benchmarks = EXTRACTION QUALITY.
LongMemEval exposed it via missing dates/facts (single recall); 2Wiki exposes
it harder via compounded multi-hop dependencies. The ENGINE is NOT the ceiling
on either — validated: it chains, refuses honestly, ~0 inventions. This is the
3rd independent confirmation (LongMemEval buckets + delta + 2Wiki) that the
lever is extraction. Reinforces: the decisive remaining experiment is a
stronger (paid) extractor; free options exhausted (gemma best, GLM-4.7 worse).

**======================================================================**
**2026-07-06 (TYPED-EVENT EXTRACTION) — PROMPT-ONLY VERSION FAILED TO TRANSFER;**
**DIAGNOSIS MANDATES A DETERMINISTIC TEMPORAL-NLP LAYER. (APM_EVENTS gated off.)**

Greenlit extraction pivot: augmented INGEST_SYSTEM (EVENTS_ADDENDUM, gated
APM_EVENTS) to emit STRUCTURED events — kind/date/participants/topic/org on ONE
node (not scattered), + auto attends-edges from participants. Also apm/ingest.py
_merge_extraction wires participant edges. Mapping stores arbitrary attrs so
fields land automatically.
 - DELTA (6 hand-built scenarios, EVENTS+CONTRACT, fresh extract): 6/6=100%,
   CONFIRMED 1->3, PROOF SURVIVAL 1/4 -> 3/4, 0 inv. cd_nearmiss_agenda (the
   'missing relational edge' case) CONFIRMED — agenda now ON the event node.
   STRONG on the toy set.
 - REAL DATA (dev-42, fresh re-extract, graph-only EVENTS+CONTRACT): **13/42 =
   31% vs 36% baseline — REGRESSED -2, CONFIRMED still 0.** DID NOT TRANSFER
   (clean-benchmark-wins-don't-transfer, again). (Network flaked mid-run twice;
   resumed; final n=42, 0 infra.)
 - DIAGNOSIS (checked the real graphs): NOT non-compliance — the 31B model DID
   emit kind/topic/participants. But FIELD QUALITY on messy real text is poor:
   kind='other' (uncategorized), topic='photography' (too generic),
   participants=['User'] (ubiquitous/non-discriminative), and **date=None (the
   critical field, still empty)**. Structure present but uninformative -> no
   proof + slight ranking perturbation. Also 'EVENTS' cat node pollutes the
   event set.
 KEY FINDING: the LLM emits event STRUCTURE but will not reliably fill DATES
 even when told to. Dates are the highest-value + most-deterministic field ->
 this is exactly a DETERMINISTIC TEMPORAL-NLP job, not a prompt problem.

DIRECTION (evidence-mandated, was Abiram's 'NLP pipeline?' question): split the
work — LLM for SEMANTIC event extraction (it complies), a DETERMINISTIC NLP
layer for NORMALIZATION the LLM is unreliable at, above all TEMPORAL
(dateparser/HeidelTime/SUTime over raw text + session date -> absolute
dates/ranges on events). This could un-bind the F bucket (which is 'extraction-
bound' ONLY because the LLM's dates are bad). Secondary: cleaner kind
classifier, coref for discriminative participants. Cons: deps (spaCy/Java),
open-IE/coref brittle on dialogue (keep LLM for semantics), ranking-noise risk.
TEMPORAL-NLP LAYER — BUILT (apm/temporal_nlp.py, dateparser installed,
APM_TEMPORAL gated off), MEASURED NULL, not promoted. Source-local date
attachment (dateparser RELATIVE_BASE=session date; a date attaches to an
event/assertion only when in the SAME SENTENCE, unambiguously; month/week ->
ranges; never overrides explicit dates; stated_at[timestamp] kept separate
from event_time[attrs]). Controlled dev-42 A/B/C (cache-warm): reach-only
17/42, +temporal 17/42, +temporal+contract 17/42 — **+0 marginal**. WHY NULL:
the layer dates ASSERTION nodes mentioned near a date, but _span/_order read
specific ENDPOINT option nodes (activity-start, the event) — the dates are
correct + present but don't land on the nodes the operators consume. Would
need ENDPOINT-LEVEL dating (identify span/order endpoints, date THOSE; the
'activity start' is usually implied, not a clean node) — deeper integration,
uncertain ROI. NOTE: real reach-only dev-42 baseline is 17/42 (~40.5%), not
15 (that was judge-noise in an older run).

GLM-4.7 EXTRACTOR TEST (2026-07-07, APM_LLM_MODEL_EXTRACT=zai-glm-4.7, free
Cerebras key) — NEGATIVE. Full dev-42 fresh extraction was UN-COMPLETABLE in
this network (180s/call timeouts stack on flaky DNS + GLM reasoning = minutes/
question; auto-resume loop couldn't progress past ~5/42). Completable delta
test instead: GLM plain 3/5=60% (1 scenario FAILED extraction:
finish_reason=length — reasoning model burned token budget on <think>, empty
content — the exact 'avoid' failure the campaign flagged), proof-survival 1/4,
CONFIRMED 1. vs gemma plain 5/6=83% proof 1/4, and gemma+events 6/6 proof 3/4.
**GLM-4.7 is WORSE than gemma, not better: same proof-survival, lower accuracy,
reasoning-overflow failures, 5-10x slower.** Ruled out. FREE-EXTRACTOR LEVER
EXHAUSTED: gemma-4-31b is the best free extractor; a genuinely stronger one
now requires PAID (DeepSeek deepseek-chat, ~$2 top-up, noted best all campaign;
or frontier) = a SPEND DECISION for Abiram. temporal_nlp.py REMOVED (null).

CONSOLIDATED CONCLUSION (4 consecutive nulls after Module 1: F, C, Module-3,
typed-events, temporal-NLP): **graph + extraction-layer levers on the current
Cerebras extractor are TAPPED at ~40% graph-only.** Each remaining lever needs
the LLM to emit structure it won't (dates), a deeper integration, or fights a
faithfulness guard. The engine is NOT the bottleneck. TO GET PAST ~40% the real
options are: (a) a genuinely STRONGER EXTRACTOR (DeepSeek/frontier — held in
reserve all campaign; would attack dropped facts + dates + typed events at the
source), or (b) endpoint-level temporal integration (narrow, uncertain).
Recommend: this is the point to (1) take the full trajectory to the expert, and
(2) seriously weigh the frontier-extractor experiment — the diagnosis now says
the ceiling is EXTRACTION QUALITY, which a better extractor directly addresses.
Product remains ~70% (hybrid), unaffected. REPO PUSHED to
github.com/bwulbasor/APM---mem @ 74e869b (before this temporal work; local is
ahead — offer to push the session's engine/extraction changes).

**======================================================================**
**2026-07-06 (C AUDIT + MODULE 3 NULL) — SAFE PURE-ENGINE CEILING = ~40%.**
**60% NEEDS EXTRACTION. C is not safely reachable (would fight the guards).**
**(Earlier ~44-46% estimate assumed a safe C squeeze; it measured 0 — see below.)**

Audited all 24 bucket-C misses (present-but-wrong-node). Split:
 - ENGINE-RECOVERABLE ~7-9: exact gold present, engine hedged/picked stale.
   Two kinds: (i) reachability — SAME pattern as Module 1 but landing in
   hedged-ASSUMED not UNKNOWN (ad7109d1 'Internet speed: 500 Mbps' present ->
   'I do not know'; 8464fc84 'Roscioli' present -> answered St Peter's); (ii)
   SUPERSESSION — stale value present, current also present (5a4f22c0 TechCorp
   vs stale Edtech; 2698e78f weekly vs stale biweekly; 6071bd76 water ratio).
 - EXTRACTION-BOUND / not-really-present ~8-10 (World B): 488d3006 GR-90 absent;
   25e5aa4f graph has Berkeley, gold UCLA; 8550ddae rosemary!=lavender; count
   cases 45dc21b6/6a27ffc2.
 - CORRECTLY-GUARDED ~2-3: c8c3f81d 'favorite shoes'->Nike IS invention-class-#8
   the guard demotes ON PURPOSE; 3249768e fifth-bottle ambiguous.
MODULE 3 (attribute-reachability, APM_REACH2) — BUILT, MEASURED NULL, REVERTED.
Extended Module-1 reachability to read DIRECTLY-STATED entity attributes
('Internet speed: 500 Mbps', 'Rachel works at TechCorp'), triggered on the
genuinely-WEAK path only (score<THETA_CONFIRM) to avoid overriding faithfulness
guards. RESULT: graph-only 40.0%->40.0% (gained 0, the -1 was judge-noise
a89d7624). It could NOT reach the C targets: they win at HIGH score with a
WRONG node, then get guard-demoted to ASSUMED — so they never reach the weak
path, and catching them REQUIRES overriding the silent-condition/selection
guards (unsafe: re-opens closed invention classes). REVERTED to keep the hot
path clean (only keep what measurably helps). CONCLUSION: the C recoverable
slice is NOT safely reachable — pure-engine is DONE at 40%.

CONFIRMED TRAJECTORY: 33.3% (baseline) -> 40.0% (Module 1 reachability, clean)
-> F null (extraction-bound) -> C audit says ~44-46% pure-engine ceiling.
**The remaining ~15pts to 60% are in EXTRACTION (typed/dated event structure),
not engine modules.** This is the consistent finding across F + C + the delta
experiment: the engine reaches/near-reaches the answer; it lacks the typed,
dated, linked structure to prove/select it. NEXT MAJOR EFFORT (needs Abiram's
go — it's a strategy shift the expert flagged as delicate): typed-event
EXTRACTION — conversation -> dated events w/ kind/participants/agenda/role
edges, NOT 'more facts'. Optionally first: the safe supersession/attribute-
reachability sub-slice (~+4-6) to squeeze pure-engine toward its ceiling.

**======================================================================**
**2026-07-06 (MODULE 2 ATTEMPT: F TEMPORAL) — NULL RESULT; F IS EXTRACTION-**
**BOUND. Refines pure-engine ceiling DOWN to ~44-48%. 60% needs extraction.**

Tried bucket F (temporal ordering/duration/most-recent). KEY FINDING: the
temporal OPERATORS the plan wanted ALREADY EXIST and are already correct —
`_span` does duration arithmetic (ASSUMED over dated endpoints), `_order` does
first/last. The F misses are NOT missing engine logic:
 - ordering (Crown/GoT, road-trip/lens): option nodes have NO dates -> engine
   correctly can't order / refuses. Extraction-bound.
 - duration (cc6d1ec1): `_span` DID subtract -> got '30 days' from MENTION-time
   stamps because true event dates weren't extracted. Extraction-bound.
 - most-recent (camera lens 41698283): needs acquisition-DATES + candidate
   precision the extractor didn't produce.
Attempted a candidate-hygiene operator (exclude Recommendation/hypothetical
nodes from ANSWER candidates for first-person factual Qs; APM_HYGIENE=1,
_drop_advice/_is_advice in engine.py). MEASURED graph-only: 40.0% -> 40.0%
(NULL; the +1/-2 deltas are the known judge-noise flip-floppers a89d7624/
09ba9854_abs/0862e8bf_abs, not real effects). Camera lens did NOT convert
(removing the recommendation just let a different wrong non-advice node win).
NOT PROMOTED — APM_HYGIENE stays gated OFF, documented measured-neutral
(like APM_CUED). 141 tests green, default engine unchanged.

IMPLICATION (honest, revises earlier optimism): F was the 2nd-biggest
'recoverable' bucket in the failure table, but it's actually EXTRACTION-BOUND
(World B). So the failure table's question_type routing OVER-counted engine-
recoverable. Pure-engine ceiling is now estimated ~44-48% (not 57-64%), and
**60% graph-only almost certainly REQUIRES typed/dated event extraction, not
more engine modules.** Module 1 (reachability, +6.7 -> 40%) may be close to
the pure-engine plateau. NEXT: either (a) C candidate-selection (check if any
real non-temporal near-miss points remain), or (b) pivot to the real lever —
typed-event/dated EXTRACTION (World B) — which is where the remaining ~20pts
live. Recommend confirming (a) is small, then committing to (b).

**======================================================================**
**2026-07-06 (MODULE 1 LANDED: BUCKET-B REACHABILITY) — GRAPH-ONLY 34% -> 40%,**
**PRODUCT FLAT, 0 NEW INVENTIONS. First point on the road to 60%. (141 tests.)**

From the failure table, bucket B (present-but-UNKNOWN) was the cleanest ROI:
the answer is a STATED value sitting in an ASSERTION node that the type/mode
filter excluded (parser sets placeholder type=time/person, or mode=count tries
to enumerate members instead of reading a stated total). FIX
(apm/engine.py `_stated_value_fallback`, DEFAULT ON, ablate w/ APM_NO_REACH=1):
on the UNKNOWN path ONLY (so it can never displace an existing answer), read
the value from the best-supported, non-recommendation, non-superseded,
topically-relevant assertion; return ASSUMED (so it can NEVER make an
invention). Supersession- and recency-aware (picks 9 months over stale 6;
6x over 4x; the user's '4 hours' over LACK/MALM recommendation rows).
GUARD: does NOT fire on time-scoped/snapshot questions ('how many pages had I
read WHEN I started my job') — those must still refuse (test_v14 031748ae).
RESULTS (all 150, cache-warm):
  graph-only  33.3% -> 40.0%  (+10 net, 0 lost, inv 1->1)
  product     71.3% -> 71.3%  (flat, inv 1->1) — doesn't displace hybrid quotes
Converted 11 graph-only misses (Fitbit 9mo, IKEA 4h, bass 12, Converse 6x,
comedians 10, + adjacent). Archives results_gonly_reach.jsonl,
results_product_reach.jsonl. NEW GRAPH-ONLY BASELINE = 40%.
NEXT (per evidence-ranked order): F temporal ordering/duration + supersession
(~11 misses, e.g. GoT/Crown, 'how long had I been X' duration arithmetic),
then C candidate-selection/stale-value, then E range-aware aggregation.
Point math: 40% now; need +20 for 60%; F+C+E realistically ~+12-18 -> 60% is
at the edge, consistent with the ~57-64% pure-engine ceiling.

**======================================================================**
**2026-07-06 (VERIFY 34% + FAILURE TABLE) — GRAPH-ONLY ~34% IS REAL; 60% IS**
**TIGHT-BUT-PLAUSIBLE. Contract did NOT transfer to real data.**

CONTRACT REGRESSION (APM_CONTRACT on real LongMemEval, cache-warm):
  product+contract  107->108/150 (flat), inv 1->1  -> promotable but NOT a win
  graph-only+contract (n=108 real-baseline slices): acc 33->36 (+3) BUT
  CONFIRMED coverage 5->3 (DOWN). **The delta's proof-survival win (1/4->2/4)
  did NOT replicate on real data** — date-norm helped a couple ASSUMED land
  right but demoted CONFIRMEDs (exact-date -> range stopped strict-matching).
  Lesson: clean-hand-built wins don't auto-transfer to messy real graphs.
  (Also: the earlier 'dev-42 0->15' was a comparison ARTIFACT — the graph-only
  baseline file never contained dev-42 ids. Expert caught it.)

PRIORITY 0 — VERIFIED graph-only number (3 repeats, all 150, cache-warm,
APM_NO_CRITIC=1 APM_NO_HYBRID=1 APM_NO_AGG=1): 33.3 / 34.7 / 34.0% —
**stable ~34%, +-0.7% (pure judge noise; engine deterministic), CONFIRMED=3,
inv=1 every run.** Archives results_gonly_rep{1,2,3}.jsonl. THE honest
graph-only baseline is 34%.

PRIORITY 1 — FAILURE TRANSITION TABLE (bench/failure_table.py, 100 misses):
  26 C ranked-below-near-miss | 21 E aggregation | 20 A fact-missing |
  15 F temporal/supersession | 12 B present-but-UNKNOWN | 4 I abstention |
  2 H judge-noise. Raw 'recoverable'=76, naive ceiling 84% -- **BUT AUDITED
  AND INFLATED**: gold_in_graph is a loose token match and treats NUMERIC
  golds as always-present, so fact-missing aggregation misses hide in E and A
  is undercounted (spot-check: 'babies born=5' filed E but gold-ish=NONE).
  CALIBRATED read from 5 autopsies: true hard-loss ~30-40, genuinely
  engine-recoverable ~35-45, REALISTIC PURE-ENGINE CEILING ~57-64%.
  ENCOURAGING: several misses have the EXACT gold node present, lost only to
  ranking / supersession / reachability (Fitbit '9 months' present->UNKNOWN;
  therapist 'weekly' present->ignored; lens present->wrong pick).

  CORRECTED TABLE v2 (numeric-gold presence FIXED: number must appear for a
  stated value; aggregation judged on counted-ITEMS via question head-nouns,
  not the total; unit words excluded). 100 misses:
    26 A fact-missing | 24 C ranked-below | 22 E aggregation |
    11 F temporal/supersede | 11 B present-but-UNKNOWN | 4 I abst | 2 H noise.
  AUDITED (5+/bucket) -> WORLD A vs B verdict = MOSTLY WORLD A, hard floor caps
  pure-engine ~60%:
   - B (11) SOLID World-A: value literally in graph, engine returned UNKNOWN
     (Fitbit 9mo, 4 hours, 12, 10). Zero extraction needed = reachability. Free.
   - C (24) SOLID World-A: gold present, engine answered WRONG node
     (lens/Roscioli/Borges/weekly). Selection/supersession.
   - F (11) SOLID World-A: both facts present, needs temporal ops (GoT/Crown,
     durations).  => B+C+F ~= 46 genuinely present-but-mishandled.
   - E (22) SOFT/CONTAMINATED: '>=1 item-word' != complete set (babies=5 hits
     'born' but not all 5 births extracted). ~10 real-recoverable, ~12 hidden
     World-B (aggregation-incomplete = extraction loss).
   - A (26) + ~12 hidden-in-E = TRUE WORLD-B ~38 (~25% of all Qs): hard floor.
  CALIBRATED BUDGET: ~46 clean + ~10 real-E + 4 refuse-abstention ~= 60
  addressable, imperfect conversion -> PURE-ENGINE CEILING ~= 60% (50 + ~40).
  VERDICT: 60% graph-only achievable by ENGINE work but it IS the ceiling;
  >60% strictly needs typed-event EXTRACTION (World B). Not a mirage now.
  EVIDENCE-RANKED MODULE ORDER: (1) B reachability [cleanest ROI, present->
  UNKNOWN], (2) F temporal ordering/duration + supersession, (3) C candidate
  selection / stale-value rejection, (4) E range-aware aggregation (~10 real),
  (5) typed-event extraction for the World-B floor. failure_table.py v2 is the
  standing point-budget instrument; refine E by member-counting for exactness.

FEASIBILITY (honest): product 70% ~85-90% (already there). graph-only 60%
~45% — tight, needs temporal+supersession+reachability+range-aggregation ALL
landing, near the pure-engine ceiling. graph-only 65%+ needs extraction to
emit TYPED EVENT structure (conversation->event graph, not loose facts). A
verifier/reranker alone won't do it: the table shows misses are 'present but
unproven/mis-picked', not 'mis-ranked among good candidates'. NEXT MODULE
ORDER (evidence-based, disagrees w/ expert's verifier-first): (1) temporal +
supersession first-class, (2) schema-reachability (present-but-UNKNOWN),
(3) range-aware aggregation, THEN (4) verifier, (5) typed-event extraction.

**======================================================================**
**2026-07-06 (CONVERGENCE + DELTA + ENRICHMENT) — THE THESIS IS SOUND; THE**
**PROBLEM IS PROOF-SURVIVAL, AND IT'S A FULL-LOOP ALIGNMENT ISSUE. (141 tests.)**

Built a HAND-BUILT convergence benchmark (scenarios/convergence.py,
bench/run_convergence.py) to test associative pattern matching in ISOLATION
from extraction. RESULT: current engine = **10/10, 0 inventions, 0 near-miss
losses.** Multi-cue convergence, weak-cues-beat-strong, near-miss (wrong
type/time/agenda), parent-climb, supersede — all handled ALREADY. So the
ranker/near-miss machinery the expert flagged is NOT the bottleneck (caveat:
I authored it knowing the engine; near-miss traps pass because I supplied
event_type/required attrs — structure real extraction lacks).

DELTA EXPERIMENT (bench/run_convergence_delta.py) — the key result: same 6
realistic scenarios, HAND-BUILT vs REAL-EXTRACTION, graph-only:
  hand-built 6/6 = 100%  |  extracted 5/6 = 83%, 0 inventions.
Accuracy tax only ~17pts BUT the tier story is the point: of 4 hand-built
CONFIRMED, extraction preserved **1**; the rest collapsed CONFIRMED->ASSUMED/
UNKNOWN. **Extraction preserves the ANSWER (engine still ranks it right 83%)
but destroys the PROOF.** This is finally WHY CONFIRMED is ~3% on real data.

STRUCTURAL ENRICHMENT (apm/enrich_structure.py, APM_ENRICH=1, gated OFF):
targets = relative-date normalization, event-typing, topic/agenda hoist,
participant linking. Faithfulness invariant: only surfaces what is already in
a node's label/edges/session-date, never imports outside facts. RESULT after
debugging: **proof-NEUTRAL — recovered 0 additional CONFIRMED.** Two honest
root causes, both diagnosed precisely:
 (a) PARSE<->ENRICH MISALIGNMENT — the recovered fields are keyed/formatted
     differently than the parser queries them: parser emits constraint key
     'type', enrich wrote 'event_type'; parser emits date 'July 3rd' (raw)
     vs node '2026-07-03'; parser 'budget vote' vs enrich 'budget'. Aligning
     the key alone was insufficient (date-format gap remained). Fixable, but
     it is a CO-DESIGN problem across parse+enrich, not a one-pass fix.
 (b) CROSS-NODE EXTRACTION LOSS — the agenda 'budget vote' landed in a
     DIFFERENT node than 'General Assembly'; the attendee links (Anna/Marco/
     Priya -> GA) were never extracted at all. Label-hoisting CANNOT rejoin
     structure extraction split/dropped. This is extraction FIDELITY (of
     relational edges), the one thing the expert said not to just 'densify'.
 ALSO measured: naive participant-hoisting HURT first (added 'User' to every
 event -> 3-way tie -> CONFIRMED lost). Fixed to require DISCRIMINATIVE
 (non-ubiquitous) people. Lesson: enrichment must add DISTINGUISHING fields,
 never uniform ones (densification-hurts-ranking, via enrichment).

CONCLUSION (for the research track): proof-survival is NOT a hoisting pass.
It is a FULL-LOOP problem — (1) extractor must preserve relational EDGES
(attendance/roles/agenda-links), (2) parser must emit NORMALIZED TYPED
constraints (type/date-range/agenda) that align with node fields, (3)
enrichment surfaces label-fields, key/format-aligned with (2). No single leg
moves the number; the engine reasoning itself is already good (delta proves
it). enrich_structure.py stays gated OFF (neutral, safe) as scaffolding for
the co-design work.

ALIGNMENT PROBE RESULT (apm/contract.py = canonical schema; A/B/C on delta):
  A baseline                    acc 5/6  CONFIRMED 1/6  proof-survival 1/4
  B contract (query-side canon) acc 6/6  CONFIRMED 2/6  proof-survival 2/4
  C contract + enrichment       acc 6/6  CONFIRMED 2/6  proof-survival 2/4
0 inventions throughout; near-miss stability held (every ASSUMED still picked
the RIGHT node). **PROOF SURVIVAL 1/4 -> 2/4 from PURE REPRESENTATION ALIGNMENT
(date normalization recovered cd_nearmiss_time), accuracy 5/6->6/6. HYPOTHESIS
CONFIRMED: part of the proof gap is representation misalignment, not reasoning.**
C added nothing over B. FAILURE CLASSIFICATION of the collapse:
  - cd_nearmiss_time  = ALIGNMENT (date-norm) -> RECOVERED by contract.
  - cd_nearmiss_type  = residual DATE-NORM: 'kind' aligned fine (meeting vs
    social both typed), but the parser dropped the month ('3rd' -> anchor-month
    fallback guessed 2026-06-03 vs node 2026-07-03). Fixable, NOT reasoning.
  - cd_nearmiss_agenda = MISSING RELATIONAL EDGE: 'budget vote' topic lives in
    a SEPARATE assertion node from the GA event; hoisting from the event label
    cannot rejoin. Needs extraction fidelity (the (b) class).
NET: proof gap ~= 2/3 representation-alignment (date normalization) + 1/3
missing relational edges; reasoner NOT implicated. apm/contract.py +
enrich_structure.py gated OFF (APM_CONTRACT / APM_ENRICH). BEFORE promoting
APM_CONTRACT: run LongMemEval dev+unseen regression (canon_constraints rewrites
EVERY query's constraints -> confirm no accuracy drop / 0 new inventions).
Instruments standing: convergence (hand-built) + delta (extracted) + law-firm
OOD gate. EXPERT-READY FRAMING: "Engine 10/10 on structured graphs, 6/6 on
extracted after representation alignment; proof survival 1/4->2/4 from a
canonical parse/graph schema ALONE, 0 inventions. Remaining gap = parser date
fidelity + genuinely missing relational edges — a representation+extraction
problem, not a reasoning one."

**======================================================================**
**2026-07-06 (LAW-FIRM SMOKE TEST) — THE FAITHFULNESS CONTRACT BROKE OOD.**
**2 CONFIRMED-tier inventions on 20 fresh-domain scenarios. FIX IN PROGRESS.**

First out-of-distribution test: 20 synthetic law-firm memory scenarios
(Sonnet-authored, LongMemEval schema, bench/data/law_firm_smoke.json; runner
bench/run_lawfirm.py; archives lawfirm_{graphonly,hybrid}.jsonl). Results:
```
  graph-only  3/20 = 15%   inventions 2   (proof-chain CONFIRMED=1)
  hybrid     12/20 = 60%   inventions 2   (verified-RAG TEXT=11)
```
Accuracy re-confirms the LongMemEval pattern (RAG carries hybrid; graph
reasons on ~1). **THE FINDING: 2/20 (10%) CONFIRMED-WRONG on fresh data vs ~0
on LongMemEval — the "structural, never confidently wrong" claim is
INSTANCE-FIT to LongMemEval, not a general property.** The fallback cannot
catch these because CONFIRMED is never overridden, so a falsely-confident
graph answer is unprotected. Both are the two already-known-fragile classes,
recurring in a new domain because the 10 "closed" classes were patches vs
specific examples, not general invariants:
 - lawfirm_0014 (ORDERING): "which happened first, the deposition or the
   FIRST RESCHEDULING of the hearing?" CONFIRMED "deposition first" by
   comparing the deposition's event date (May 16) to the hearing's REFERENT
   date (scheduled May 19) instead of the RESCHEDULING EVENT date (Apr 28,
   the session). Compared timestamps of different semantic kinds.
 - lawfirm_0015 (KNOWLEDGE-UPDATE): address changed 118 Maple Ct (Mar) ->
   4471 Birchwood (May, eff June). CONFIRMED the OLD address — supersede
   never fired; the stale value had a strong evidenced anchor (EEOC filing).

REPRIORITIZATION (Abiram: "fix it"): the graph track's #1 job is now MAKE
CONFIRMED TRUSTWORTHY OOD, before growing coverage. Two GENERAL invariants:
 (I1) Ordering CONFIRMED must compare timestamps of the SAME semantic kind
      (event-occurrence vs event-occurrence). Different kinds, or an
      inferred/referent date, degrade to ASSUMED.
 (I2) CONFIRMED must be BLOCKED on any value a later update supersedes;
      knowledge-update supersede must fire structurally on "X is now Y",
      not by matching known phrasings.
The law-firm smoke test is now the OOD REGRESSION GATE (re-run after each
change; target 2->0 inventions WITHOUT hardcoding these two scenarios), run
alongside the LongMemEval dev/pooled regression (invention must stay 0,
product accuracy must not drop).

**STATUS: FIXED + VERIFIED (v1.16, 141 tests). Three general invariants shipped:**
 - I1 (engine._order): ordering degrades CONFIRMED->ASSUMED when the question
   names a modification event (reschedul/postpon/cancel/amend/continuance/
   adjourn...) that NO candidate node's label reflects. Positive control test
   ensures legit dated ordering still CONFIRMs.
 - I2a (graph.auto_supersede): functional-edge supersede now orders by
   edge.ts -> DST-node.ts -> SRC-node.ts (extractors leave edge ts null; the
   VALUE's own mention time is the reliable key). Fixed the backwards
   supersede that demoted the NEW address.
 - I2b (engine resolve): structural, phrasing-independent recency-conflict
   guard — when a strictly more-recent conflicting value exists for the SAME
   subject (both ASSERTION nodes sharing a subject; entities/events excluded
   so intersection & sequence queries are untouched), never CONFIRM the older
   value; answer the newest, tagged ASSUMED. Uses _same_value (Jaccard on
   distinctive tokens, domain-agnostic).
 Tests: tests/test_ood_invariants.py (4). 
 RESULTS after fix:
   OOD gate (law-firm): inventions 2->0 BOTH configs; ACCURACY ROSE
     graph-only 15->20%, hybrid 60->70% (demoting the false-CONFIRMED let the
     hybrid fallback engage and quote the right answer — faithfulness fix that
     also lifts accuracy; 0015 now ASSUMED-correct on the NEW address).
   LongMemEval regression (cache-warm, hybrid): valid-41 63->63/0, fresh-29
     83->83/0, confirm-38 68->66 — the single "loss" is 09ba9854_abs, an
     abstention whose post hypothesis is BYTE-IDENTICAL to pre (judge
     nondeterminism, NOT a code regression). Net: ZERO real regression.
   Archives: lawfirm_{graphonly,hybrid}.jsonl (post-fix),
     results_postfix_regression.jsonl.
 REMAINING: confirm-38 still shows 1 invention = gpt4_8c8961ae (ordering
 "Europe first" off a misextracted date; no modification-event word so I1
 doesn't fire, and it's the confirmation-slice garbage-in case we do NOT fix
 from that slice). The GENERAL ordering-date-trust question (CONFIRM only when
 dates are verified occurrences) remains open for the graph track.

**======================================================================**
**2026-07-06 (LATER) — 70% CROSSED ON UNSEEN DATA, AND THE PROJECT SPLIT**
**INTO TWO TRACKS. (v1.15 "wide-support", 137 tests green.) READ THIS FIRST.**

**WHAT HAPPENED.** Abiram asked to push accuracy toward 70% before starting
LoCoMo. Diagnostic on pooled slices (n=121) showed the bleeding tier:
ASSUMED = 25 correct / 35 wrong (44% precise) while TEXT_SUPPORTED = 39/9
(81% precise) AND abstention-safe (you cannot quote what was never said).
FIX (`APM_WIDE_SUPPORT=1`, one guard in bench/run_longmemeval.py): let the
verified-quote / aggregation route ALSO arbitrate STRONG ASSUMED answers
(score>=theta), adopting only when a quote/table mechanically verifies —
guess -> verified-quote is a strict upgrade. It is parameter-free (adopt ANY
verified quote, no fitted threshold) and structurally invention-safe
(TEXT_SUPPORTED never CONFIRMs, so it CANNOT raise the invention count).

**NUMBERS (all Cerebras gemma-4-31b free, APM_NO_CRITIC=1 APM_WIDE_SUPPORT=1):**
```
                 before(v1.14.2)   after(v1.15)   inventions
  dev-42            64.3%            76.2%          0
  valid-41 (unseen) 56.1%            63.4%          0
  confirm-38 (unseen)57.9%           68.4%          1*  (*same pre-existing
  fresh-29 (NEVER seen) —            82.8%          0    CONFIRMED ordering
  POOLED UNSEEN(108) ~57%            70.4%          1    case gpt4_8c8961ae)
```
Archives: results_{dev,valid,confirm}_widesupport.jsonl,
results_freshslice_widesupport_complete.jsonl. Fresh-29 = a genuinely
untouched --heldout draw (network flaked mid-run, 12 DNS failures re-run to
completion; resumable via rerun_ids after stripping infra rows). **CAVEAT:
fresh-29 had NO single-session-preference (our worst cat ~37%, exhausted from
prior archives); fold it back and the all-cat number is low-70s. The honest
figure to quote is the POOLED UNSEEN 70.4% / 0.9% invention.**

**THE HONEST FINDING (drives the split). Attribution of CORRECT answers:**
```
                       graph-PROOF   graph-HEUR   verified-RAG
                       (CONFIRMED)   (ASSUMED)    (TEXT_SUPPORTED)
  BEFORE (59.5%)         4  (6%)      25 (35%)     39 (54%)
  AFTER  (69.4%)         4  (5%)       8 (10%)     68 (81%)
  FRESH slice (82.8%)    0             0           24 (100%)
```
The graph-PROOF tier has been FLAT AT 4 the entire campaign (~3% of all
questions). Today's +10pts came ENTIRELY by moving work OFF the graph:
converting graph-heuristic guesses (ASSUMED 25->8) into verified quotes
(TEXT 39->68). On genuinely fresh data the graph contributed 0% of correct
answers. CONCLUSION, stated plainly: **what we have built is verified RAG
with a graph-based honesty gate. The gate is the real innovation (it is what
holds inventions at ~0 with per-answer provenance), but it is a GATE, not a
reasoner.** The "graph does the inference" thesis is currently unsupported by
evidence — graph-exclusive contribution ceiling is 12/121 (~10%) and 0/29 on
fresh data.

**THE DECISION (Abiram, 2026-07-06). ONE LEVER: IMPROVE THE GRAPH ONLY.**
Both goals — (A) the research claim "structural reasoning" and (B) shipping a
final product — are pursued through the SAME effort: grow the graph's genuine
reasoning contribution. The unifying insight: as the graph gets stronger,
every product answer it reclaims migrates from TEXT_SUPPORTED ("verified
quote") to CONFIRMED ("graph proof") = strictly better provenance = better
PRODUCT; and the pure-graph accuracy rises toward the product number =
research thesis PROVEN. When pure-graph ≈ product number, the RAG crutch is
removable and the thesis stands. Verified-RAG is now the BASELINE THE GRAPH
MUST CLIMB TOWARD, not a thing to optimize. DO NOT spend future effort making
the RAG fallback better — that deepens the same trench.

**MEASUREMENT DISCIPLINE (two configs, both on the same slices):**
 - PRODUCT number (ship): `APM_NO_CRITIC=1 APM_WIDE_SUPPORT=1` -> 70.4% unseen
   / 0.9% invention. This is the current shippable artifact.
 - RESEARCH KPI (the number to grow): `APM_NO_CRITIC=1 APM_NO_HYBRID=1
   APM_NO_AGG=1` -> PURE ENGINE verdict (CONFIRMED/ASSUMED/UNKNOWN, no
   raw-text tiers). New guard added in run_longmemeval.py. GRAPH-ONLY
   BASELINE (pooled unseen n=108, results_graphonly_baseline.jsonl):
   **30.6% accuracy · 4.6% CONFIRMED coverage (5/108) · 1 invention.** And
   the sharper floor: only 3/108 (2.8%) correct answers came from an actual
   PROOF CHAIN — the other 30 correct are ASSUMED heuristic guesses that
   happened to land (76 ASSUMED @ ~40% precise). So structural-reasoning
   coverage today ≈ 3%. THE RESEARCH GAP: product 70.4% vs graph-only 30.6%
   (~40pts RAG-carried) and proof-chain 2.8%. Track BOTH every change; the
   research win is these gaps closing.

**NEXT (graph-only track, priority order):**
 (1) Record the graph-only baseline (pure-engine acc + CONFIRMED coverage %)
     as the research starting line. Re-run per change.
 (2) Attack the classes where the graph SHOULD win but doesn't — start with
     temporal DURATION ("how long had I been X when Y") and ORDERING ("which
     first"): autopsy shows the relative-time phrases ARE in the graph ("3
     months ago", session-dated) but the engine emits placeholders instead of
     resolving relative expr against session date and subtracting. This is a
     deterministic engine capability = pure-graph CONFIRMED points, the RIGHT
     kind. 5+ instances across slices (e4e14d04, cc6d1ec1, c9f37c46, 993da5e2,
     gpt4_483dd43c, gpt4_d9af6064).
 (3) Multi-session count/sum: move aggregation from raw-text member tables to
     GRAPH-scoped membership counting so counts become CONFIRMED, not
     TEXT_SUPPORTED.
 (4) Every graph win should DISPLACE a TEXT_SUPPORTED product answer with a
     CONFIRMED one — verify the migration each time (product acc must not drop
     while research acc rises).
LoCoMo/2WikiMultiHop harness (the old task 1) is DEFERRED behind this.

**======================================================================**

**Phase 5 (temporal/event) done:** `date_range` constraints, `Engine.on_event()` API
(event-triggered pre-emptive inference), ordering-mode queries (`Query.mode="first"/"last"`
with named `options` — "which did I do first, A or B?"; deterministic over dated nodes →
CONFIRMED, undated candidates degrade to ASSUMED with disclosure). Parser emits mode/options.

**Phase 6 v0.3 RESULT (bench/results_v03.jsonl, same stratified set, n=14):**
**accuracy 57.1% · refusal 42.9% · invention 7.1%** — trend 13.3% → 28.6% → 57.1%.
Per category: temporal-reasoning 4/4 (PERFECT — ordering+premise fixes), preference 2/2
(suggest mode), knowledge-update 1/2, assistant 1/2, multi-session 0/2 (count semantics),
single-session-user 0/2 (extraction misses). The 1 "invention" is a scoring artifact
(Admon rotation: text DISCLOSED the missing detail, string-based refusal detection missed
the phrasing — v0.4 judge-based scoring reclassifies; v0.4 table ingestion fixes the
underlying extraction miss). Entity module converted the workshop-vs-webinar refusal
exactly as designed; false-premise feature reconverted the Google regression.

**Phase 6 v0.4 RESULT (bench/results_v04.jsonl, dev set, n=14):**
accuracy 57.1% (held vs v0.3 under STRICTER judge-based scoring) · refusal 35.7% ·
**invention 0.0%**. First-ever conversions: clothing count via relevance veto
("probably 3" = gold), commute question. Lost GPS to same-timestamp coin-flip
(fixed in v0.5 mention-order stamping). 4-point dev trend: 13.3 → 28.6 → 57.1 → 57.1%,
inventions 0→1→1*→0 (*scoring artifact). Discount 7161e7e2 in all runs (hint leak).

**v0.5 FROZEN (62 tests green):** mention-order timestamp offsets, relation-aware
relevance veto, dual-pass extraction, table-to-User anchoring, judge-based scoring,
hint leak removed, --heldout mode. HELD-OUT RUN IN PROGRESS (~21 never-seen questions,
frozen code) — its number is the honest headline. DO NOT iterate on held-out results.

**CONFIRMATION SLICE (2026-07-06): 57.9% / 1 invention on a FULLY UNTOUCHED
n=38 slice (results_confirm_v1142_cerebras.jsonl, frozen v1.14.2,
APM_NO_CRITIC=1) — THE paper number.** Per-cat: multi-session 7/10 (!),
single-session-user 5/6, assistant 4/6, knowledge-update 3/6, temporal 1/6
(this draw's temporal skewed to undated/mis-dated orderings). The 1
invention = gpt4_8c8961ae (ordering: CONFIRMED "Europe trip first" off a
date; gold was the Thailand solo trip) — a misextracted/mis-dated endpoint,
i.e. the GARBAGE-IN boundary (gate verifies chains not facts), NOT fixed
(would spend the confirmation slice; autopsy on a future dev slice). Pooled
unseen validation+confirmation (n=79): ~57% acc, 1 confident-wrong.
STANDING NUMBERS: dev-42 64.3%/0, validation-41 56.1%/0, confirmation-38
57.9%/1. Story to tell: competitive with Zep(63.8)/Mem0(49) on accuracy,
alone at ~0 confident-wrong with per-answer provenance.

**VALIDATION + v1.14.x (2026-07-05 evening): 56.1% / 0 INVENTIONS on the
never-seen validation slice (results_revalidation_v1142_cerebras.jsonl,
n=41); dev-42 = 64.3% / 0 (results_dev64_v113). 137 tests.** The v1.13
fresh-slice validation initially scored 56.1% with 3 NEW inventions —
autopsied and closed as classes #8-#10 (ledger now 10/10):
 (#8) ASSOCIATIVE CONFIRM OVER UNEVIDENCED CONDITIONS ('favorite shoes'
      confirmed by association strength): CONFIRMED now requires every
      asked constraint EVIDENCED (attr, date-window, or label mention;
      participants evidenced by anchor convergence — S1 preserved);
      scoring.unevidenced_keys.
 (#9) SELF-QUANTIFIED COUNT ('counted 5 assertions when one SAID Caught 12
      largemouth bass' — also counted TROUT advice for a BASS question):
      count members must mention what is counted (ALL string constraints
      feed the lexical filter now, not just item-keys; rstrip('s') bug
      fixed — 'bass'->'ba'), and members stating their own numbers cap the
      count at ASSUMED. RESULT: bass question CONVERTED (TEXT_SUPPORTED
      '12' with verified quote).
 (#10) PREMISE SUBSTITUTION ('UNCLE's birthday party' fuzzy-resolved onto
      the NIECE's party node): scope resolution rejects matches missing a
      distinguishing content token -> premise machinery engages.
REVALIDATION: 56.1% / 0 inventions (bass +1, preference churn -2). NOTE:
the validation slice is now dev-spent (guards derived from it); the FINAL
paper number needs one untouched confirmation slice under frozen v1.14.2.
Two-day arc: 33% -> 64.3 dev / 56.1 unseen-class, inventions 10 classes
found/closed, invention rate 0 on every post-guard run. Network flaked
repeatedly this evening (runs are resumable; 4 restarts needed).

**THE TIER BREAKTHROUGH (2026-07-05 afternoon): 35.7% -> 50.0% / 0 inventions
(results_hybrid2_v111_cerebras.jsonl) — the first real accuracy jump, from
Abiram's directive: "the graph needs a judge, not a bigger memory."**
ROAD THERE (all measured, all archived):
 - delta-critic ingest pass (Abiram's reasoning-over-extraction idea):
   probe recovered every universally-dropped fact BUT benchmark-neutral
   (33.3% vs 35.7 — density<->ranking trade eats recall gains; results_
   criticablation_v110). KEPT (cost-neutral, principled). APM_NO_CRITIC=1.
 - cued recall v1 (query-time re-read): NET NEGATIVE (-3) — same-class
   adoption displaced correct answers; abstentions lost. OFF by default
   (APM_CUED=1). Adoption now upgrade-only. Two invention classes found
   DURING the build: #5a question-echo contamination (cued nodes carry the
   question's constraint attrs -> strict-match false CONFIRM; cued evidence
   now NEVER confirms) and #5b WRITE-BACK SELF-CONFIRMATION (generic,
   pre-existing: repeat query strict-matched its OWN stored ASSUMED guess;
   ASSUMED-status nodes can no longer direct-match or confirm).
 - HYBRID TIER v2 = the winner (apm/hybrid.py): trigger on FINAL answer
   state (UNKNOWN + weak/refusal-sounding ASSUMED), read top-20 sessions,
   and REQUIRE a verbatim source quote that the harness MECHANICALLY
   verifies against the named session — unverifiable quote kills the
   answer. New tier label TEXT_SUPPORTED (never CONFIRMED). Result: fired
   15x, 10 correct (67% precision), abstentions 5/6 PROTECTED (you cannot
   quote what was never said — the quote requirement is an abstention
   shield), knowledge-update 8/10 (best-ever), net +11/-5.
 - AGGREGATION ROUTE (apm/aggregate.py, v1.12.x): count/sum questions build
   a quoted member TABLE (each row mechanically verified, deduped) ->
   arithmetic -> "at least N" when incomplete. v1 overreached (regex stole
   single-stated-value "how many" questions from the quote tier: 45.2%);
   now count/sum modes only + tables that dedupe <2 rows fall through.
   Converted the baking count. v1.12.2 run in flight (~52% expected).
CONTRACT NOW: CONFIRMED(graph proof) / ASSUMED(named rule) /
TEXT_SUPPORTED(verified quote) / refusal. Inventions still CONFIRMED-wrong
only; ledger 7 classes found / 7 closed. 134 tests. GARBAGE-IN BOUNDARY
documented (hamster 'Luna': extraction asserted a falsehood as observed ->
gate confirmed it; gates verify CHAINS not FACTS — quote-verification/NLI
is the mitigation and TEXT_SUPPORTED already implements it for the hybrid
tier). REMAINING to 60: multi-session still 1/6, preference 2/6, user 3/8;
candidate verifier/reranker (judge over evidence packets) is the next
engine-side move; critic on/off ablation owed (~5 churn losses suspected).

**MEASUREMENT SLICE n=42 (2026-07-05, results_measure42_v18_cerebras.jsonl,
frozen v1.8, never-seen, 104 prior ids excluded): accuracy 35.7% · refusal
57.1% · INVENTION 0.0%.** The citable baseline. Read:
 - INVENTIONS 0/42 with all four guard classes live — the never-lies claim
   measured at 2x slice size right after class #4's discovery would have
   broken it. Unseen-data ledger: slice2 0/21, slice3 2/21 (pre-guard),
   measure42 0/42 (post-guard).
 - knowledge-update 5/10 — BEST EVER for the category (superseded-value
   machinery scored: the "4 mummies" past-value question; snapshot guard
   held). New machinery scored on clean data: LLM arbiter (hamster name),
   superseded preference, oversize guard (138/84-candidate counts refused
   instead of garbage).
 - temporal 0/6 — mention-stamp ordering/undated spans now hedge or refuse
   instead of falsely confirming; without real date attrs the category is
   extraction-capped. multi-session 2/6, assistant 2/6, refusal 57% —
   extraction recall remains THE binding constraint.
 - Accuracy 35.7% (pooled unseen now ~37% n=84) vs Zep 63.8: NOT yet
   competitive on raw accuracy with free extraction. NEXT MOVE = the
   extraction card (DeepSeek ~$0.30/run or frontier ~$5-15) against this
   frozen baseline; engine work has plateaued honestly.

**v1.8 (2026-07-05): TAGGED-INFERENCE EXTRACTION (Abiram's idea) — the
Type-A unlock. 117 tests green.** INGEST_SYSTEM now LICENSES inference but
requires labeling: strongly-implied connections may be emitted as edges with
"implied": true, connecting ONLY already-extracted entities. ingest() caps
implied weight at 0.55 + provenance-marks source ':implied'; _gate caps ANY
chain containing an implied edge at ASSUMED (deterministic backstop). Net:
implied knowledge guides retrieval and yields hedged answers, can never mint
a CONFIRMED — wrong inferences cost hedged-wrong, never inventions. PROBED
LIVE: gemma-4-31b emitted the 7-run-unreachable coupon->Target bridge
('$5 coupon on coffee creamer --located_in--> Target w=0.5 implied') on the
51a45a95 chunk. EXTRACTOR BAKE-OFF (bench/probe_extractors.py, coupon chunk,
6 stated facts): gemma-4-31b 4/6 recall 10n/1s BEST on Cerebras; gpt-oss-120b
and zai-glm-4.7 thinner (7n/6n) — no upgrade on that key; all three drop the
same 2 low-salience facts. UNPROBED: the AI-Studio key holds gemini-3.5-flash,
gemini-3-flash-preview, gemini-3.1-pro-preview, gemma-4-31b-it (separate
quota) — candidate free frontier extractors, one call each to test.

**SLICE 3 + v1.7 (2026-07-05): THE HUMBLING SLICE — 28.6% / 2 INVENTIONS
(results_slice3_v16_cerebras.jsonl, n=21 never-seen, frozen v1.6; slice now
DEV-SPENT because the inventions were autopsied+fixed).** Two claims revised:
 (a) ZERO-GAP DOESN'T REPLICATE AS A CONSTANT: fresh slices scored 47.6%
     then 28.6% — +-20pt draw variance at n=21. Pooled unseen: 16/42 = 38.1%.
     CITABLE NUMBERS NEED n>=40 SLICES. This slice drew aggregation-heavy
     (4 sums, multi-session 0/4 — extraction addend-completeness again).
 (b) INVENTION CLASS #4 FOUND ON CLEAN DATA — STALE NUMERIC SNAPSHOT:
     031748ae_abs ("how many engineers WHEN I JUST STARTED" — confirmed the
     CURRENT team size; abstention gold) and 184da446 ("pages read SO FAR"
     — confirmed page-200 snapshot; state had advanced to 220, newer value
     unextracted). v1.7 FIX (116 tests): engine._snapshot_guard — a numeric
     answer to a current-state ("so far/currently/as of") or time-scoped
     (constraints timing/when/period) question caps at ASSUMED; if the
     question pins a time the evidence can't match (no date attr, label
     doesn't satisfy the scope) -> UNKNOWN (makes the abstention correct).
     ALSO CLOSED: direct-lookup path bypassed ALL confirm-caps (strict attr
     match -> straight CONFIRMED) — now runs the shared guard; full
     guard-consolidation into _gate remains TECH DEBT.
 Assistant category 2/3 again (JSON-mode extraction recovery holding).
 INVENTION LEDGER now 4 classes found / 4 closed structurally.

**v1.6 (2026-07-05): LLM TIE-ARBITER + RANKING FIXES — dev2 48.0%, inventions
2 -> 0 (results_dev2_v16_cerebras.jsonl). 114 tests green. THIRD CLEAN SLICE
RUNNING (frozen v1.6, never-seen questions — the honest number; measurement
only, DO NOT iterate on it).** Shipped:
 (1) apm/arbiter.py — LLM tie-arbiter (SPEC §3.5 role 2): picks among
     ENGINE-produced tied candidates + their graph-neighbor context; CLOSED
     list (out-of-list pick -> None -> embedding fallback); ASSUMED-only;
     role=veto (gemini); APM_NO_ARBITER=1 for ablations.
 (2) scoring._label_satisfies — string constraint satisfied by the node's
     own LABEL text counts as a match, not a missing attr (extractors put
     discriminators in labels, not attr keys).
 (3) parse anchor-promotion guards — constraint values no longer promote to
     anchors when the resolved node is of the unknown's own type (it may BE
     the answer: gpt4_5dcc0aab's gold was recruited as an anchor and thereby
     excluded from candidacy!) or when the value is short fuzzy junk (<6 ch).
 (4) judge: 'how many' questions judged on NUMERIC quantity (audit found
     exactly 1 flip — gpt4_f2262a51 '3 doctors' — genuinely-wrong counts
     stayed wrong; archived numbers NOT retro-edited).
 (5) bench/rerun_ids.py — re-run an archived id set under the current
     engine (cache-warm pure-engine ablation).
LIVE VALIDATION: shoe question converted via the full chain (guard ->
candidate, label-match -> tie band, arbiter -> correct pick). Dev2 v1.5->v1.6
diff: +shoes +coworker-pref / -2 suggest-mode churn (bundle composition
noise, subjectively judged category), inventions 2->0 (ordering + past-value
guards both verified). KNOWN LEFTOVERS: relevance veto shrank doctors count
3->2 (a +1 under the new judge left on the table); bedtime wrong-topic
CONFIRM class/text mismatch class; suggest-bundle sensitivity to activation
changes.

**DEV2 DIAGNOSTIC + v1.5 (2026-07-04 late, results_dev2_v15_cerebras.jsonl,
n=25 declared DEV/iteration-fuel, 48.0% / 2 inventions):** targeted
failure-mining run per Abiram. SHIPPED v1.5 (109 tests green):
 (1) STRUCTURED JSON OUTPUT (llm.py json_mode + response_format, auto-
     fallback on 400/422; verified Cerebras+Gemini) — ZERO chunk-failure
     warnings all run (previously constant) — extraction recall recovered;
 (2) ORDERING MENTION-STAMP GUARD (invention gpt4_0b2f1d21 "coffee maker
     first" CONFIRMED off mention-times): order pick keeps working via
     stamps but caps at ASSUMED; CONFIRMED needs attrs.date on all compared;
 (3) PAST-VALUE GUARD + SUPERSEDED PREFERENCE (invention 07741c44 "initially
     keep sneakers" -> CONFIRMED the CURRENT closet, gold = superseded
     under-bed): auto_supersede now RECORDS superseded edges
     (graph.superseded); "initially/previously/used to" questions answer
     with the best superseded/contradicted-by-newer candidate (ASSUMED), and
     absent such evidence the current value cannot CONFIRM. NOTE: superseded
     values sit below THETA_FLOOR by construction — the guard searches the
     PRE-floor candidate list;
 (4) CONVERGENT ACTIVATION (SPEC §3.1 'a(target)+=' was silently best-path-
     only Dijkstra): spread() keeps up to 4 route-distinct paths per
     (anchor,node) with a cycle/revisit guard; parallel corroborating facts
     now accumulate via noisy-OR (S2 single-cue safety test held after the
     cycle guard).
CEREBRAS POOL NOW 6 KEYS (~6M tok/day, Abiram added 3).
DEV2 FAILURE TABLE (13 misses): 3 fixed-tonight classes (2 inventions + 1),
3 ranking-tie/wrong-topic (LLM TIE-ARBITER is the named candidate fix —
SPEC §3.5 role 2, LLM picks among ENGINE-produced tied candidates + their
graph context, stays ASSUMED; Abiram inclined but wants LLM calls minimal),
3 aggregation-extraction (members missing from graph), 2 extraction-absent,
1 judge-strictness artifact (gpt4_f2262a51: answered "3 doctors" w/ names,
gold "three doctors" — scored wrong; judge calibration worth a look),
1 preference nuance. 5.5-EXTRACTION OFFLINE EXPERIMENT (scratchpad
test_55_graph.py): even frontier extraction + convergence can't convert
implied-fact questions (coupon->Target) — the bridge edge is never stated;
1-hop theme topics dominate. Some LongMemEval golds are themselves
inferences = principled ceiling for a never-invent architecture (paper
paragraph, not a bug).

**FRESH-SLICE GENERALIZATION RESULT (2026-07-04, results_freshslice_v14_
cerebras.jsonl, n=21 NEVER-SEEN, frozen v1.4, Cerebras): accuracy 47.6% ·
refusal 33.3% · invention 0.0% — IDENTICAL to the contaminated dev-21's
47.6%. ZERO generalization gap** (v0.5's gap was ~12pt). THE headline
result: the engine fixes were principled, not overfit, and invention=0
holds on unseen data. Per category: temporal 4/4 (PERFECT — span/order
machinery), single-session-user 2/3, preference 2/3, knowledge-update 1/3,
multi-session 1/5 (sculpture-hours SUM converted — the new sum machinery
works when extraction captures all addends; other sums missed on addend
completeness), assistant 0/3 (semantic-tiebreak guesses on starved graphs —
the known extraction gap, DeepSeek bucket). Both abstentions correctly
refused; count-oversize guard fired correctly on unseen data (egg-tarts,
180 candidates -> refuse). Refusal rate DROPPED 52->33% (more attempts,
same precision). PROTOCOL: this slice is now spent for iteration — any fix
derived from it requires a third slice for the next honest number
(500-question pool; 58 ids burned so far).

**v1.4 BUCKET-1 ENGINE FIXES (2026-07-04): 47.6% / 0 inventions on the old 21
(results_heldout_v14_cerebras.jsonl) — ties project best. 107 tests green
(tests/test_v14.py). FRESH-SLICE RUN LAUNCHED (the honest generalization
number): --heldout glob widened results_v*.jsonl -> results_*.jsonl so ALL
archives (incl. the ~15-cycle-contaminated heldout 21) are excluded; new 21
never-seen questions, frozen v1.4 config.** Fixes, each from an autopsied
v1.3 miss:
 (1) question-echo cap (58bf7951): top candidate whose label near-restates
     the question (no novel non-numeric tokens) cannot CONFIRM -> ASSUMED.
     Killed the class/text mismatch; NOT a conversion — autopsy showed 'The
     Glass Menagerie' node is graph-DISCONNECTED from the play event
     (extraction miss, DeepSeek bucket).
 (2) span endpoint dedupe (e831120c): both options resolving to the SAME
     node -> UNKNOWN (was manufacturing '0 weeks').
 (3) sum over duration assertions + parse routing (e831120c CONVERTED:
     3.5 weeks exact): ASSERTION nodes always sum-eligible (type gate was
     emptying every sum since ph.type=time), PARSE_SYSTEM routes 'how many
     weeks did it take' -> sum; + lexical anchor on activity terms (camping
     b5ef892d briefly summed noise '40 days' -> now honest refusal; gold
     absent from graph).
 (4) count lexical anchor + oversize guard (gpt4_59c863d7: '173 model kits'
     -> honest refusal): counted members must mention the item head-noun;
     parsed predicate matching >25 candidates refuses.
 (5) contradiction propagation across value-duplicates (852ce960 UPGRADED
     ASSUMED->CONFIRMED $400k): if A contradicts B, A contradicts every
     assertion sharing B's numbers+subject (not sharing A's).
DECISION (Abiram): DeepSeek held in reserve until the very end; extraction-
ceiling misses (Rachel, coupon, Plesiosaur, Miami, Roscioli, playlist,
camping) stay unfixed until then. v1.4 remaining misses on old 21: those 7
extraction + workshop-span (mention-stamp, gold date absent) + Orlando +
model-kits/camping honest refusals + Glass Menagerie.

**v1.3 CANONICALIZATION RESULT (2026-07-04): 42.9% / 0 inventions — CEREBRAS
REGRESSION RECOVERED (33.3% -> 42.9%, results_heldout_v13_cerebras.jsonl), and
the mechanism is now precisely understood.** Built per plan: apm/canon.py
(~90 relation synonyms -> canonical, type-aware disambiguation (based_in:
person->lives_in / org->located_in; went_to: event->attends / place->
traveled_to; never guesses on unknown types), attr-key normalization
copy-if-absent, idempotent, wired per-edge in ingest() AND as the first
post-ingest pass in run_item) + tightened INGEST_SYSTEM (controlled relation
vocabulary, attr-key conventions, metric-naming rule, "X located_in Y never
Y contains X"). 98 tests green (9 new in tests/test_canon.py).
TWO-STAGE ABLATION (the important part):
 - STAGE 1 (canon layer alone, frozen v1.2 gemma graphs, cache-warm): 33.3%,
   ZERO per-question changes (results_heldout_v13_canon_frozengraphs.jsonl).
   The synonym layer was NOT the binding constraint.
 - STAGE 2 (tightened prompt + canon, fresh extraction): 42.9%. +4 conversions
   (church-mass span 08f4fc43 — the 7-run nemesis; mortgage 852ce960 ->
   $400,000; yoga 945e3d21; cultural-events 35a27287) / -2 losses (Roscioli
   4c36ccef, Spotify playlist 1e043500 — re-extraction shape variance cuts
   both ways). knowledge-update 0/3 -> 2/3, temporal 5/6.
REFINED DIAGNOSIS (autopsy on EXACT bench graphs): extractor fragility is
SHAPE, not verb spelling — (a) duplicate same-value assertions with
idiosyncratic metric attrs ('currency' vs 'mortgage_pre_approval') dodge
auto_contradict (mortgage: the $400k update killed ONE $350k phrasing, the
duplicate survived and won the tiebreak); (b) question-echo nodes ("User
visited local community theater play", 0.79) outrank the real answer (Glass
Menagerie IS in the graph — engine confirms the echo, verbalizer refuses);
(c) relation census over all 1292 cached extractions: 901 DISTINCT verbs;
the uncovered mass is reversed containment (includes/contains ~650 edges,
invisible to taxonomy's inward located_in walk) + a ~1200-edge recommendation
family (recommended_to/for/by, provided_to) with no engine rules at all.
CONCLUSION: prompt-side shape control works where post-hoc renaming can't;
canon layer stays as cheap insurance for uninstructable extractors.
TOOLING FIX (important): bench/show_graph.py rebuilt graphs with chunk size
3000 vs the bench's 2200 — different cache keys, so ALL autopsies silently
diagnosed DIFFERENT graphs than the ones scored. run_item's ingestion is now
factored into build_bench_graph(); show_graph/autopsy delegate to it.
QUEUED ENGINE FIXES (each from a named failure, awaiting Abiram's go):
 (1) contradiction propagation — X contradicts Y => X contradicts every
     assertion sharing Y's numbers+subject ("the update contradicts the
     VALUE, not one phrasing of it"); insurance for the mortgage class;
 (2) question-echo penalty at ranking — candidate whose label near-restates
     the question is uninformative (converts Glass Menagerie; echoes
     persist even under the tightened prompt because the vague sentence is
     genuinely in the transcript);
 (3) direction-flipping canon rule: "Y contains/includes X" -> "X located_in Y";
 (4) NEW pathology to autopsy: model-kits count now guesses 173 (scoped
     membership wildly overbroad under denser extraction) — was a refusal.

**v1.2 CEREBRAS RESULT + THE KEY DIAGNOSIS (2026-07-03, results_heldout_v12_cerebras.jsonl,
21/21): 33% (7 correct / 0 inventions) — REGRESSED from v1.0's 42.9%.** Cerebras gemma-4-31b: FAST (1.2s/call, 30-60x NIM) + dense (60-258 node graphs)
+ sustainable (1M tok/day × 3 keys pooled) — the free trifecta we'd sought. YET accuracy
DROPPED. Root cause = **EXTRACTOR-SHAPE OVERFITTING**: every engine rule (scope resolution,
auto_contradict/supersede, count/sum routing, constraint matching, scope-neighbors) was
tuned to gemini's node-label/edge-relation/attr-key conventions. gemma emits the same
FACTS with different relation verbs + attr keys → rules silently no-op → engine can't
rank the answer it HAS → refuses. Proof: Glass Menagerie CORRECT on lite → REFUSED on
gemma; mortgage CORRECT on gemini → WRONG on gemma (same Qs, denser graphs, worse).
STABLE across all 4 extractors: 0 inventions, ~67% precision-on-attempts. FRAGILE: raw
accuracy (swings 30-48% by extractor). CONCLUSION: accuracy is the wrong target — it's
extractor-fragile; the robust/defensible claim is invention=0 + honest refusal + precision.

**NEXT ACTION DECIDED (building when transferred): EXTRACTOR-AGNOSTIC CANONICALIZATION
LAYER** — the root-cause fix for the overfitting, pure-engine (no API/quota), testable:
 (1) tighten INGEST_SYSTEM to a CONTROLLED relation vocabulary + attr-key conventions
     (bias every extractor toward one shape);
 (2) NEW apm/canon.py: deterministic post-ingest pass mapping relation synonyms →
     canonical (resides_in/based_in/moved_to→lives_in; went_to/participated_in→attends;
     belongs_to/affiliated_with→member_of; etc.) + normalize attr keys. Run FIRST in the
     post-ingest chain (before auto_contradict/supersede/taxonomy/enrich/episodes).
 This makes the engine's rules fire regardless of extractor → fixes the fragility AND is
 the missing robustness claim ("architecture robust to commodity extractor swap").
 Validate by re-running v1.2 Cerebras (fast/cheap now) and checking the flips recover.

**FREE-TIER EXTRACTOR RESEARCH (2026-07-03, published limits):** a dense run needs
~1.5M tokens/day (21 histories re-ingested, dual-pass). Confirmed ceilings: SambaNova
20 req/day, Groq 100K TPD, Cloudflare 10K neurons/day, OpenRouter 50 req/day, GitHub
Models 50 RPD, Gemini token-throttled. NONE fit — all sized for prototyping. NIM is
the ONLY no-token-cap free option (40 RPM) but latency-bound: nemotron-120B dense but
176s+truncates, llama-3.3-70B thin(18n)+37s. Testing NIM mid-size MoE/instruct models
(llama-4-maverick, gemma-3-12b, ministral-14b, phi-4-mini) for a fast+dense sweet spot.
KEY STRATEGIC OPTION found: FROZEN-CACHE PLAY — first run of any config seeds the
sqlite cache; identical re-runs are instant/free. So ONE slow dense NIM run → free
fast engine iteration forever after. NIM key: nvapi-odOC...GLoI (works).
Nemotron reasoning disableable via system 'detailed thinking off'.

**v1.1 (gemini-2.5-flash extractor) ABORTED — quota-starved (results_heldout_v11_
partial_2.5flash.jsonl, 5 attempted / 3 scored / 2 INFRA-FAIL).** Proved the density
hypothesis: 2.5-flash graphs 2-3x lite (293/216/159 vs ~90) and CONVERTED the 7-run
church-mass span miss (08f4fc43 — first ever). BUT: (a) density flipped a span C->W
(more candidate dates → ranking precision problem, not extraction); (b) ~8 min/question;
(c) heavy token load exhausted the 3-key Gemini pool by Q5 → INFRA-FAILs. Net: strong
extractor breaks the extraction ceiling but is NOT completable on free quota AND trades
extraction-misses for ranking-misses. CONCLUSION recorded for Abiram: free tier can't
finish a strong-extractor run; the real unblock is the DeepSeek key.
DEEPSEEK KEY (sk-87d6...c003): valid/authenticates but 402 Insufficient Balance — trial
grant not landed/expired; needs ~$2 top-up. Once funded: deepseek-chat = fast (2-5s) +
dense, ~$0.30/run, unblocks everything. Extractor reverted to gemini-3.1-flash-lite.

**v1.0 = 42.9% / 1 invention (results_heldout_v10.jsonl).** Frozen-graph ablation PROVED
the ceiling is extraction, not engine: identical graphs, engine fixes moved ~1-2 Qs.
~6 misses are facts ABSENT from the graph (Rachel's suburbs, church mass, model kits,
coupon store, yoga freq) — unreachable by any traversal rule. auto_supersede couldn't
fix Rachel (graph had only the stale Chicago, one destination). So: (a) shipped
UNVERIFIED-UPDATE GUARD (89 tests) — a question implying a change ("moved/now/after")
with no supersede evidence in the chain cannot CONFIRM → kills that invention class
structurally; (b) EXTRACTOR UPGRADE gemini-3.1-flash-lite → gemini-2.5-flash @16k
(validated 30-42 nodes/chunk vs lite ~20) — directly attacks the extraction ceiling.
v1.1 run (2.5-flash, same 21, controlled extractor ablation) in progress.
STRATEGIC NOTE surfaced to Abiram: 80% on this contaminated set needs either a paid
frontier extractor or overfitting; the defensible claim is invention=0 + fresh-slice
generalization number. Fresh untouched slice still owed.

**CAMPAIGN CYCLE 3 (v0.9 → v1.0):** v0.9 = 38.1% (results_heldout_v09.jsonl), variance
dominated: core 8 held, 2 flips lost (945e3d21 parse routed "how often"→count;
1e043500 type-echo constraint pushed 0.54→0.49 under premise bar), 1 NEW invention
(830ce83f "Rachel moved" — stale lives_in CONFIRMED). Fixes shipped (88 tests):
(1) auto_supersede — functional relations (lives_in/works_at...) newer edge
downweights older ×0.35 (closes relational knowledge-update inventions);
(2) span/order endpoints prefer EVENT-type resolution (7-run miss 08f4fc43: was
spanning the CHURCH's mention-stamp, not the MASS);
(3) "how often" → resolve not count; (4) type-echo keys (type/kind/category) no
longer penalize attr absence; (5) premise refusal exemption when the candidate's
edge relation matches the question verb (principled fix — threshold fudge reverted).
KEY POOL: 3 gemini keys rotate on 429/403 in llm.py (~3k req/day headroom).
STRATEGY SHIFT: ingestion FROZEN (prompts+caps untouched) → extraction cache serves
identical graphs → v1.0 is a pure engine ablation. Expected v1.0: 12-14/21.
Verified on frozen graphs pre-run: Summer Vibes✓, yoga→resolve✓, Rachel class✓.

**80% CAMPAIGN (Abiram directive: iterate until ≥17/21 on dev-set-2; final claim
still needs a FRESH slice).** Cycle log:
- v0.8/v0.8.1: 38.1% — scope-refusal regression + 3 ranking pathologies autopsied.
- v0.9 fixes (in flight): scope=subtree∪neighbors (Glass Menagerie→CONFIRMED),
  relation tie-filter (question verb ↔ edge rel; Summer Vibes converted — names beat
  embeddings), anchor promotion (constraint values that resolve become anchors),
  exact-type dominance, echo penalty in semantic_pick, verbalizer must-deliver-answer
  clause, truncation caps 1400/900→2600/1400 (multi-session enumeration recall).
- Parked: 51a45a95 (coupon — extraction shape), e831120c (MCU weeks — needs sum
  over durations), sum mode for camping-days (BUILDING NOW).
- Quota watch: raised caps = cold extraction cache; AI Studio RPD ~1000-1500/day.
- SUM MODE shipped (85 tests): mode="sum" totals quantities across members
  ("camping days"), unit-aware regex, ASSUMED with all addends in chain. Debugging
  war story: an invisible \\x08 (backspace — heredoc-mangled \\b) inside the regex
  silently killed all matches; purged. LESSON: never write regexes through bash
  heredocs; use Write/Edit tools.
- QUOTA PAUSE: AI Studio RPD exhausted mid-v0.9 (429 storm); run stopped cleanly,
  auto-resume probe armed (30-min cycle) — v0.9 (all campaign fixes + sum mode)
  launches itself on quota reset. Groq also daily-dead. Expected v0.9 result:
  12-14/21; remaining gap to 17: assistant trio, preference pair, coupon question,
  MCU-weeks (needs sum-over-durations wiring in parse).

**v0.8 RESULT + REGRESSION (results_heldout_v08.jsonl, n=21): 38.1% / 0 inventions —
REGRESSED from v0.7's 47.6%.** Root cause found in minutes via v0.7/v0.8 diff on
58bf7951 ("engine knew The Glass Menagerie and refused anyway"): the new `under` scope
wiring appended unresolvable scope labels to Query.unresolved → false-premise refusals
on legitimate questions (scope containers like "the local theater" are descriptive,
not premises). One-line fix: unresolvable scope degrades to unscoped. v0.8.1 rerun
in progress (cache-warm). Mortgage + knowledge-update conversions HELD in v0.8;
multi-session 0/3 and assistant 0/3 unconverted by scoping/assistant-pass — need
autopsy (parser may not emit scopes for these question shapes at all).

**v0.8 SHIPPED (84 tests green, run in progress):** (1) parser emits `under=` taxonomy
scopes (multi-session counter); (2) containment enrichment (apm/enrich.py — cached LLM
call per parentless place, builds Arusha→Tanzania→Africa tier; add-only, can't touch
facts); (3) assistant-focused extraction pass in bench (targets 0/3 assistant category);
(4) embedding semantic tiebreak (apm/embed.py, gemini-embedding-001 — REORDERS tied
guesses only, never upgrades class; validated on the live 'degree' failure);
(5) episode clustering (apm/episodes.py — Louvain communities = memory episodes,
ep_* nodes obey hub rule, deterministic seed/labels); (6) consolidation (graph.
consolidate — merges ≥95-sim assertion rephrasings sharing a subject, drops observed
orphans; numbers-are-identity guards); (7) verdict_score logged for calibration;
(8) JUDGE A/B: mimo vs gemini agreement 13/13 = 100% → judge switched to gemini
(pipeline now 100% AI-Studio + engine; zen retired).

**v0.7 HELD-OUT — PROJECT BEST (2026-07-03, results_heldout_v07.jsonl, n=21):**
**accuracy 47.6% · refusal 52.4% · invention 0.0%** · ~1 min/question. Beat mimo's
45.5% at 15x the speed with ZERO inventions. Config: gemini-3.1-flash-lite all
transducers, FORCED dual-pass (APM_BENCH_PASS_MIN=999), v0.7 rules (numbers-are-
identity + auto_contradict v2). Conversions: mortgage 852ce960 CORRECT (3-run nemesis
— the full autopsy chain validated live), knowledge-update 0/3→2/3, single-session-
user →2/3, temporal 5/6 held. Remaining walls: multi-session 0/3 (needs taxonomy-
scoped counting wired into parser — subtree scoping EXISTS in engine, parser doesn't
emit `under` yet), assistant-recall 0/3 (assistant-heavy content starves extraction;
one graph = 6 nodes — needs an assistant-focused extraction pass or turn-level
chunking). Held-out trend: 45.5 → 33.3 → 47.6; inventions 9.1% → 4.8% → 0.0%.
REMINDER: this slice is contaminated (drove fixes) — paper numbers need a fresh slice.

**GEMINI v0.6 HELD-OUT (2026-07-03, results_heldout_gemini_v06.jsonl, n=21):**
acc 33.3% · refusal 57.1% · invention 4.8% · ~1 min/question (vs mimo's 15).
Temporal-reasoning 5/6 (BEST of any run — span mode converted its target; abstentions
3/3). Recall categories starved by LEAN extraction (Gemini-lite avg ~35 nodes vs mimo
72): knowledge-update 0/3, multi-session 0/3, assistant 0/3. The 1 invention (852ce960
mortgage AGAIN) autopsied: ENTITY RESOLUTION MERGED $350k and $400k assertions (alias
proves it) so auto-contradict had nothing to compare. → **v0.7 SHIPPED (79 tests):
numbers-are-identity rule in entities.similarity (disjoint number sets can never merge)
+ auto_contradict v2 (metric = label-with-numbers-stripped similarity ≥85 + any
value-ish attr; no longer needs 'metric' key).** Verified against the exact Gemini node
shapes. NOTE: held-out set now contaminated (drove fixes twice) — final paper number
needs a THIRD untouched slice (460+ questions remain).
NEXT RUN CONFIG: Gemini triple/forced-dual-pass extraction (speed buys richness) + v0.7.
Transducers: AI Studio gemini-3.1-flash-lite (key in .env; gemini-2.0-flash retired,
gemini-2.5-flash returns EMPTY = thinking model). Groq exhausted daily (~150k tok);
NIM identified as best uncapped free option (40 RPM, no token cap) — not yet keyed.

**DEEPSEEK SAGA + HYBRID CONFIG (2026-07-03):** pure-deepseek held-out run was
INVALIDATED — 13/21 questions had EMPTY graphs (rate-limit storm + a deeper issue:
ALL free models on this endpoint are hidden-reasoning models; on long/dense inputs
thinking consumes the whole token budget → finish_reason=length, content empty).
The archived results_heldout_deepseek.jsonl and its 14.3% are infra artifacts — DO NOT
cite as a model comparison. Fixes shipped: (1) integrity guard — 0-node ingestion →
INFRA_FAILURE, excluded from stats, never scored as honest refusal; (2) 429/5xx
exponential backoff in llm.py; (3) chunked ingestion (3k chars, per-chunk failures
contained); (4) APM_LLM_MAXTOK_<ROLE> — extract needs 24576 (verified: deepseek 24k
on one chunk = 32 nodes OK; 8k = empty). CURRENT CONFIG (hybrid): EXTRACT=mimo@24k
(best full-transcript extractor, 72-node graphs), PARSE/VERBALIZE/VETO=deepseek-flash
(fast, short prompts), JUDGE=mimo (cross-family; judge A/B still pending). Hybrid
held-out run in progress. Judge-A/B + preference-generalization diagnosis queued.

**HELD-OUT MIMO RECORD (2026-07-03, bench/results_heldout_mimo.jsonl, n=11, capped
by Abiram):** accuracy 45.5% · refusal 63.6% · invention 9.1%. Dev 57.1% → held-out
45.5% = ~12pt generalization gap (n=11 noise ±15pt; mimo API degradation asterisk —
repeated truncated-JSON ingest failures). Abstentions on unseen data: 3/3 CORRECT.
The 1 TRUE invention (852ce960): "$350,000 CONFIRMED" vs gold $400,000 — a
knowledge-update miss; both value assertions existed but no contradicts edge (the
extractor doesn't emit them across sessions). → **v0.6 rule (queued, post-protocol):
same subject + same metric attr + different value → AUTO-contradict, newer suppresses
older. Deterministic, at ingest/write time.** New unseen failure class also queued:
date-arithmetic span mode. DEEPSEEK HELD-OUT RUN IN PROGRESS (same frozen v0.5, full
21 questions) → model-swap ablation vs the mimo 11.

**MODEL SWITCH DECISION (2026-07-02, Abiram):** after the mimo held-out run completes:
archive as results_heldout_mimo.jsonl (mimo performance recorded for the report, with
API-degradation asterisk — repeated truncated-JSON failures visible in graph sizes),
then OFFICIALLY switch default transducer model to deepseek-v4-flash-free (~6-8x faster,
validated: e2e demo 1m46s vs 10min+; judge stays mimo-v2.5-free for cross-family grading)
and re-run the SAME held-out questions on deepseek → clean model-swap ablation.
Per-role config in .env (APM_LLM_MODEL_EXTRACT/PARSE/VERBALIZE/VETO/JUDGE).
LLM cache live (.llm_cache.sqlite, content-keyed, judge excluded — hygiene + exact
reproducibility). Held-out has ALSO surfaced a new unseen failure class: date-arithmetic
("how many days between X and Y") — no span mode exists; queued for v0.6 AFTER the
held-out protocol completes.

**OVERFITTING PROTOCOL (2026-07-02, prompted by Abiram):** the 14 stratified questions
are a DEV SET — all v0.2-v0.5 fixes were derived from and measured on them, so the
13→29→57% trend is a development trend. One outright leak found & removed: INGEST_HINT
contained "Admon Sunday shift" (copied from question 7161e7e2 — discount that question's
results). Remedy: `--heldout K` mode excludes all ids in results_v*.jsonl archives.
PROTOCOL: freeze v0.5 → run `--heldout 5` (~33 never-seen questions) → THAT number is
the honest headline. Never iterate on the held-out set.

**v0.4 SHIPPED, UNMEASURED (61 tests green):** judge-based answered/refused scoring
(replaces brittle string matching), deterministic markdown-table ingestion
(apm/tables.py — cells become assertions linked to resolvable entities; tables never
touch the LLM), relevance veto over enumerations (apm/ingest.py relevance_veto —
LLM may only SHRINK a member set; converts the clothing-count near-miss).
NEXT: v0.4 run for the 4-point trend, then n=50 slice for a citable number.

**Phase 6 v0.2 RESULT (bench/results_v02.jsonl, same stratified set):**
accuracy 28.6% (2.2x v0.1) · refusal 35.7% · invention 7.1% (1: count question "21 items"
CONFIRMED — drove the v0.3 structural rule: parsed-predicate counts cap at ASSUMED).
Converted to correct: abstention-via-unresolvable-option, value question (5K time),
assistant-recall (Bandung), GPS ordering. Regression: Google-job abstention (drove
false-premise feature). One crash (bike-vs-car; split-retry ingestion added).

**v0.3 SHIPPED, UNMEASURED (57 tests green):** entity resolution w/ alias memory
(apm/entities.py — prerequisite for knowledge-update), count caps, false-premise
refusal, mode="suggest" (grounding-bundle for preference questions), split-retry
ingestion, flushed bench logging. NEXT: v0.3 stratified run → 3-point trend; then n=50.
Known remaining gap: semantic constraint matching ("degree" vs node labels) → v0.4
embedding linking.

**Phase 6 v0.1 BASELINE (bench/results_v01.jsonl, n=15 stratified):**
accuracy 13.3% · refusal 86.7% · **invention 0.0%** — the gate held perfectly; every miss
was an honest refusal. Failure taxonomy from the run (each drove a v0.2 fix):
1. Ordering questions ("which first, A or B?") → `mode="first"/"last"` (shipped)
2. Counting questions ("how many X?") → `mode="count"`, observed-facts-only (shipped)
3. Value questions ("what was my 5K time?") → ingest reifies measured values as
   assertion nodes; parse maps value questions to assertions (shipped)
4. Assistant-recall questions → INGEST_HINT now keeps assistant-provided facts;
   assistant turn budget 300→900 chars (shipped)
5. Preference questions → open-ended generation, fundamentally hard for node
   resolution; UNADDRESSED (acceptable: refusals, not inventions)
v0.2 re-run of the same stratified set in progress (results.jsonl).

**Phase 6 (LongMemEval) LIVE:** real dataset downloaded (`bench/data/longmemeval_oracle.json`,
500 questions, from HF `xiaowu0162/longmemeval-cleaned`). Harness: `bench/run_longmemeval.py`
(resumable JSONL, `--stratified K` per type, LLM judge, abstention scored as refusal-correct).
First 3-question smoke run: 1 correct / 2 honest refusals / **0 inventions**. The two refusals
were ordering questions — fixed by ordering mode (added after; re-run those ids for current
numbers by deleting their lines from bench/results.jsonl). Stratified run in progress.
Metrics tracked: accuracy / refusal rate / **invention rate (wrong-and-CONFIRMED — the
number the gate must hold at zero)**.

## Earlier — Phases 1-4 CORE COMPLETE ✅ (33 tests green)

**Phase 2 (gate hardening) done:**
- `contradicts` edges with temporal precedence (newer attacks older, never reverse) — engine [6d]
- Edge decay (`graph.apply_decay`, forgetting curve) + Hebbian reinforcement on recall
  (`graph.reinforce`, wired into write-back — rehearsal protects memories)
- `Placeholder.required`: DEFINING constraints (candidate missing the attr is excluded,
  not soft-penalized) — discovered via B3, the word-"bank" disambiguation
- Adversarial suite: near-miss dates, unconnected-but-famous candidates, empty graph → UNKNOWN

**Brain scenarios (scenarios/brain.py), all tested:** B1 interference/updating,
B2 episodic sequence, B3 semantic priming, B4 forgetting+rehearsal, B5 source-confusion
guard (confabulation → ASSUMED never CONFIRMED), B6 salience/flashbulb.

**Visualization:** `python viz.py` → `apm_viz.html` — self-contained interactive
force-directed explorer, 11 scenarios, chain highlighted green, contradicts edges red-dashed,
anchors ringed, answer glowing, verdict panel. Verified rendering in headless Chromium
(screenshots viz_s1.png, viz_b1.png; zero console errors).

**Phase 3/4 (LLM transducer) working end-to-end** (`apm/ingest.py`, `demo_e2e.py`):
- `ingest(graph, text)` — LLM extraction → validated nodes/edges, entity resolution v0
  (normalized exact + containment ≥4 chars)
- `parse_question(graph, q)` — NL → anchors + placeholder (why/intent → assertion, predictive)
- `verbalize(verdict, q)` — grounding-locked NL answer
- Live demo verified: CONFIRMED with chain, ASSUMED hedged ("probably the missing docs"),
  UNKNOWN refused ("I don't know where Anna works")
- NOTE: free reasoning model (mimo) burns tokens thinking → llm.py raises on empty content,
  max_tokens=8192, _ask_json retries 3x. Extraction varies run to run — ingestion
  determinism is an open problem for Phase 6 benchmarking.

## Earlier state — Phase 1 ✅
- ✅ Phase 0: [SPEC.md](SPEC.md) — architecture v0.1 (data model, 10-step loop, LLM-in-traversal
  rules §3.5, confidence calculus, gate contract, comparison vs GraphRAG/Graphiti/Mem0/MemGPT,
  memory-native endgame §9).
- ✅ Phase 1: reasoning engine built, **all 11 milestone tests pass** (`python -m pytest tests/`):
  - `apm/graph.py` — typed nodes/edges, parent ladder, write path + retraction
  - `apm/activation.py` — spreading activation, per-anchor path tracking, temporal boost
  - `apm/scoring.py` — noisy-OR over independent paths, soft/strict constraint matching
  - `apm/fallback.py` — parent climb → constraint projection → recency default
  - `apm/gate.py` — Verdict (CONFIRMED/ASSUMED/UNKNOWN) + printable chains
  - `apm/engine.py` — the loop, steps [3]–[8] + [10] write-back
  - `scenarios/fixtures.py` — S1–S4 founding scenarios; `demo.py` prints all traces
  - Key proven behaviors: agenda-deletion flips S1 CONFIRMED→UNKNOWN; S4 corrupted signal
    → ASSUMED via recency, retracted by later observed fact; predictive queries capped at
    ASSUMED; direct lookup requires UNIQUE strict match; write-back makes repeat queries direct.
- ✅ LLM transducer client (`apm/llm.py`) wired to opencode zen (model mimo-v2.5-free),
  smoke-tested OK. Key in `.env` (gitignored; burner key, no payment attached).
  NOTE: endpoint 403s without a User-Agent header.

## Engine design decisions (beyond SPEC v0.1)
1. **Direct lookup requires uniqueness** — ambiguous strict matches fall through to activation.
2. **Candidate merge keeps max score** — projection route can override a weak activation route
   (the 3 two-hop paths to a candidate through one hub correctly collapse to 1 under the
   independence rule; the projection carries the hub's full convergence score instead).
3. **Predictive queries capped at ASSUMED** — intents/futures can never be CONFIRMED.
4. Thresholds: θ_confirm=0.6, θ_floor=0.2, tie ε=0.05, decay=0.6, max 4 hops.

## Next: Phase 2 — gate hardening
- Adversarial suite where UNKNOWN is the right answer (near-miss constraints, contradicting
  evidence, plausible-but-unsupported candidates).
- Contradiction handling in scoring (`score × (1 − p_contra)` per SPEC §6) — not implemented yet.
- Edge decay + Hebbian reinforcement — not implemented yet.
- Then Phase 3 (LLM ingestion via `apm/llm.py`), Phase 4 (NL parse/verbalize), Phase 5
  (temporal/event mode), Phase 6 (LongMemEval/LoCoMo benchmarks).

## How to run
```
cd D:\Claude\apm-architecture
python -m pytest tests/ -v     # 11 tests, all green
python demo.py                 # prints all six inference traces
python -m apm.llm              # LLM smoke test
```
