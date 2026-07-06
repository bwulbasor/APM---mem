# Associative Pattern Matching (APM) — Architecture Specification v0.1

**Working name:** Associative Pattern Matching (APM) — the *mechanism*.
**Architecture name:** TBD (candidate: Graph Link Models). APM : architecture-name :: attention : Transformer.
**Author:** Abiram
**Status:** Founding spec — every design decision below is v0.1 and revisable.

---

## 1. Thesis

Current LLMs answer by predicting the next token over retrieved or in-context text. Even
retrieval-augmented systems (RAG, GraphRAG) ultimately let the LLM *freely synthesize* over
whatever was retrieved — there is no structural guarantee that an answer is backed by evidence.

APM inverts this: **the graph does the reasoning; the LLM only translates.**

> Retrieval fetches what is *similar*. Context memory remembers what was *said*.
> APM infers what must be *true* — and shows the chain that proves it.

The architectural contract: **generation is gated by an evidence state.** The system may only
assert what binds to a chain in a linked evidence graph. Everything else is emitted as
`assumed` (with the default rule that produced it) or `unknown`. Hallucination becomes
structurally impossible rather than statistically unlikely.

---

## 2. The Evidence Graph (substrate)

### 2.1 Nodes
| Type | Examples | Notes |
|---|---|---|
| Person | Person X, Abiram | |
| Org | IAESTE, Organization Y | Acts as *parent* via `member_of` |
| Event | Meeting X, General Assembly | Has time span |
| Place | China, Greece, Japan | |
| Time | 2026-06-26 (Friday) | First-class nodes so constraints can attach |
| Topic/Thing | movie, documents, agenda item | |
| Assertion | "docs missing from Org Y" | Reified facts — statements about other nodes |
| **Placeholder** | "unknown event on the 26th" | A typed unknown with constraints; the query target |

Every node carries: `id`, `type`, `timestamp` (created/observed), `source` (provenance),
`status` ∈ {observed, inferred, assumed}, `confidence` ∈ [0,1].

### 2.2 Edges
Typed, directed, weighted, timestamped:
`member_of`, `part_of`, `attends`, `scheduled_at`, `traveled_to`, `needs`, `acted_in`,
`related_to`, `precedes`, `contradicts`, `supports`, …

Every edge carries: `weight` ∈ (0,1] (association strength), `timestamp`, `source`.

### 2.3 Hierarchy
`member_of` / `part_of` edges induce the **parent ladder** — the structure the fallback
mechanism climbs (people → IAESTE; Person X → Org Y). Parents aggregate what is known about
their children (an org has an agenda, a travel history hangs off a person).

---

## 3. The APM Mechanism (the reasoning loop)

```
INPUT (query or event)
   │
   ▼
[1] PARSE (LLM)          → anchors (known entities) + placeholders (typed unknowns + constraints)
   │
   ▼
[2] LINK                 → resolve anchors to graph node ids (entity resolution)
   │
   ▼
[3] DIRECT LOOKUP        → node satisfying ALL placeholder constraints?
   │        └─ hit → chain = direct edges → go to [8] as CONFIRMED
   ▼ miss
[4] SPREAD ACTIVATION    → from each anchor independently; decay per hop; threshold cutoff; ≤4 hops
   │
   ▼
[5] INTERSECT            → candidates activated from MULTIPLE independent anchors score highest
   │
   ▼
[6] FALLBACK LADDER (when intersection is empty or weak)
   │     a. climb parent edges from anchors (people → IAESTE)
   │     b. project placeholder constraints onto the parent's known facts (agenda × time window)
   │     c. still nothing → apply DEFAULTS (recency, frequency) → result tagged ASSUMED
   ▼
[7] RANK                 → direct > multi-path intersection > temporal proximity > recency default
   │
   ▼
[8] GATE
   │     chain score ≥ θ_confirm            → CONFIRMED + chain
   │     only a default fired               → ASSUMED  + default rule + partial chain
   │     nothing                            → UNKNOWN  + "what evidence would resolve this"
   ▼
[9] VERBALIZE (LLM)      → natural-language answer that cites the chain
   │
   ▼
[10] WRITE-BACK          → inference stored as node (status: inferred, provenance: chain);
                           contradiction with existing node → retraction/update
```

The LLM appears exactly twice — [1] and [9] — as a language transducer. Steps [2]–[8] and [10]
are deterministic graph operations. **This is the line between APM and every GraphRAG variant.**

### 3.1 Spreading activation (step 4)
- Seed each anchor node with activation 1.0.
- Propagate: `a(target) += a(source) × edge.weight × DECAY` per hop (DECAY ≈ 0.5–0.7).
- Cut off below `ACTIVATION_THRESHOLD`; hard cap at 4 hops (combinatorial + cognitive limit).
- Track **which anchor** each unit of activation came from (needed for intersection).

### 3.2 Intersection scoring (step 5)
A candidate reached from k independent anchors with path scores p₁…p_k combines via noisy-OR:

```
score(c) = 1 − Π (1 − pᵢ)        where pᵢ = Π(edge weights along path) × DECAY^hops
```

Independence matters: two paths sharing most edges are ONE path. v0.1 rule: paths are
independent iff they share no intermediate node.

### 3.3 The fallback ladder (step 6) — the novel part
No existing system does this deliberately:
1. **Direct** — the fact itself exists.
2. **Sibling/parent generalization** — the fact doesn't exist, but the anchors share a parent;
   the parent's knowledge (agenda, patterns, history) is projected through the placeholder's
   constraints (type, time window) to *generate a candidate*.
3. **Defaults** — no structural evidence at all; fall back to recency ("latest travel = China")
   or frequency ("he usually calls about invoices"). Output is ASSUMED, never CONFIRMED.

### 3.4 The gate (step 8) — the contract
Three output classes. This is the architecture's public API:

| Class | Meaning | Emission rule |
|---|---|---|
| **CONFIRMED** | evidence chain exists, score ≥ θ | answer + full chain |
| **ASSUMED** | only a default fired | answer + default rule + partial chain, explicitly tagged |
| **UNKNOWN** | no chain, no applicable default | refuse + state what evidence would resolve it |

A bare guess presented as fact is unrepresentable in the output space. This is the
anti-hallucination guarantee, and it is *structural*, not statistical.

### 3.5 LLM participation in traversal (decided 2026-07-02)
**Rule: the LLM may influence WHERE we look, never WHAT we conclude.** Proposer may be neural;
verifier stays symbolic (AlphaGo split). Safety test for any LLM role: its mistakes may only
cost *coverage* (→ UNKNOWN), never *correctness* (→ wrong CONFIRMED).

Allowed roles:
1. **Semantic edge weighting** — embedding similarity between query and edge/node labels
   re-weights activation spread (deterministic at query time; cf. arXiv:2512.15922). Do first.
2. **Constraint projection filter (step 6b)** — LLM scores graph-produced candidates against
   fuzzy constraints ("late June" ≈ 26th). It filters; it cannot add candidates.
3. **Frontier pruning** — when activation exceeds budget, LLM picks which frontier edges to
   expand. Bad pruning → missed evidence → UNKNOWN, never a wrong answer.

Forbidden forever: LLM inventing nodes/edges mid-traversal; LLM judging final chain validity.

Why not full LLM-guided walking (StepChain/HopRAG style): (a) motivated reasoning — the model
steers toward its priors, breaking the independence that makes the gate meaningful;
(b) cost/latency — calls per hop × beam × depth; (c) non-determinism kills reproducibility AND
the synthetic-trace teacher (§9). Note the endgame model (§9) DOES traverse — it emits ops as
native actions — but the symbolic engine executes and validates them.

### 3.6 Query modes (added Phase 5/6)
- **resolve** (default): what IS the unknown.
- **first / last**: order a candidate set by time ("which did I do first, A or B?").
  Deterministic over dated nodes → CONFIRMED with all dates in the chain; any undated
  candidate degrades to ASSUMED with the exclusion disclosed; <2 dated → UNKNOWN.
- **event-triggered** (`Engine.on_event`): no question — an event happened; infer the
  best explanation. Always predictive → capped at ASSUMED.
- Constraints support `date` (exact/window) and `date_range` ((from,to) — a dated node
  inside the range is a strict match).
- `Placeholder.required`: DEFINING constraint keys — a candidate silent on one is not a
  weaker candidate, it is not a candidate (lexical disambiguation needs this: B3).

### 3.7 Write path (step 10)
- Every CONFIRMED/ASSUMED inference becomes a node (`status: inferred/assumed`, provenance = chain).
- New observed facts that contradict an assumed node **retract** it (defeasible reasoning —
  this is why ASSUMED must be tagged: it's the retractable class).
- Edges decay over time unless reinforced (touching a chain reinforces its edges — Hebbian).

---

## 4. The four founding scenarios, traced

### S1 — "Something on Friday the 26th" (fallback ladder)
Anchors: {date: 26th}, people mentioned. Placeholder: Event(date=26th, type=?).
[3] Calendar lookup → miss. [4] Activation from people-nodes converges on IAESTE (all
`member_of` IAESTE). [6b] Climb to IAESTE → agenda node → project constraint date≈26th →
"General Assembly" in window. [8] CONFIRMED (multi-path intersection + constraint fit):
*"Likely the IAESTE General Assembly — all attendees are IAESTE members and the annual agenda
places the GA in this window."*

### S2 — Movie recall (intersection search)
Anchors: actor, maybe co-actors, music, setting. Placeholder: Movie(?).
[4] Activate each anchor's neighborhood. [5] The single movie node reached from ≥3 independent
anchors dominates via noisy-OR. [8] CONFIRMED with the converging paths as chain.
(= tip-of-the-tongue resolution; spreading activation à la Collins & Loftus 1975.)

### S3 — Person X is calling (pre-emptive, event-driven inference)
Input is an EVENT, not a question. Anchors: Person X. Placeholder: CallIntent(?).
[4] Person X → `member_of` Org Y; Org Y → Meeting X (`scheduled_at` +12h — temporal proximity
boost); Assertion node "docs missing, needed for Meeting X" `supports` the candidate.
[8] CONFIRMED (multi-hop chain): *"Probably calling about the missing documents for tomorrow's
meeting."* Demonstrates: inference can be *triggered*, not just queried.

### S4 — "He goes to ^£*#>" (corrupted signal → default)
Anchors: "he" → Person X (coreference). Placeholder: Place(?) — constraint unrecoverable.
[3]–[5] no discriminating evidence. [6c] travel node → {China, Greece, Japan} → recency
default → China. [8] **ASSUMED**: *"Probably China — his most recent destination (assumed
from recency; no direct evidence)."* Later fact "he went to Greece" retracts it cleanly.

---

## 5. How APM differs from everything adjacent

| Dimension | Vector RAG | GraphRAG (MSR) | StepChain-style graph QA | Graphiti / Zep | Mem0 | MemGPT / context memory | **APM** |
|---|---|---|---|---|---|---|---|
| Retrieval principle | similarity | community summaries | LLM-guided BFS | temporal graph search | fact store + graph | paging text in/out | **activation + intersection** |
| Who reasons | LLM | LLM | LLM (graph assists) | LLM | LLM | LLM | **graph engine; LLM translates** |
| Unknowns | n/a | n/a | sub-questions | n/a | n/a | n/a | **first-class typed placeholders** |
| Fallback when lookup fails | more chunks | bigger community | more hops | none | none | none | **deliberate ladder: parent → constraint projection → defaults** |
| Output guarantee | none | none | evidence paths shown, not enforced | none | none | none | **gate: confirmed / assumed / unknown; unbacked assertion unrepresentable** |
| Hallucination control | statistical | statistical | statistical | statistical | statistical | statistical | **structural** |
| Defaults | — | — | — | — | — | — | **defeasible, tagged, retractable** |
| Trigger | query | query | query | query | query | query | **query OR event (pre-emptive)** |
| Write path | none | batch re-index | none | live episodes | live facts | live text | **inferences become nodes; contradictions retract** |
| Temporal | none | none | none | strong (bi-temporal) | weak | none | strong (proximity boosts, decay, recency defaults) |

**vs GraphRAG in one paragraph:** GraphRAG uses a graph to *retrieve better context*, then the
LLM free-generates over it — the answer is still next-token prediction, communities are used for
summarization, indexing is batch, and there is no notion of unknowns, fallback, defaults, or
output classes. APM uses the graph to *perform the inference itself*; the LLM never gets to
assert anything the traversal didn't license. GraphRAG improves what the LLM sees; APM
constrains what the LLM may say.

**vs Graphiti/Zep (closest substrate):** Graphiti solves storage — temporal knowledge graph,
live updates, fact invalidation. It does not do fallback reasoning, intersection scoring,
evidence gating, or pre-emptive inference; retrieval feeds an LLM that free-generates. APM's
reasoning layer can (and in Phase 1 will) sit **on top of** Graphiti — same graph, same LLM,
only the reasoning policy differs, which makes every benchmark point attributable to APM.

---

## 6. Confidence calculus (v0.1, to be tuned)

- Edge weight w ∈ (0,1]: observed facts start ~0.9; inferred ~score of their chain; assumed ~0.4.
- Path score: product of edge weights × DECAY^hops.
- Multi-path combination: noisy-OR over independent paths.
- Temporal proximity boost: multiply relevant edges by `1 + β·exp(−Δt/τ)`.
- Recency default: among tied candidates, argmax(timestamp) — but result is capped at ASSUMED.
- Thresholds (initial): θ_confirm = 0.6, θ_floor = 0.2 (below → UNKNOWN). Tune on eval set.
- Open question: Bayesian network vs Dempster-Shafer for conflict (an edge that *contradicts*).
  v0.1: contradicting evidence subtracts via `score × (1 − p_contra)`.

## 7. Known hard problems (acknowledged, not solved)

1. **Entity resolution** — is "Person X" in message 40 the node from message 2? This is where
   graph-memory projects die. Phase 3 problem; start with exact/alias matching.
2. **Independence detection** for noisy-OR — shared-node heuristic is crude.
3. **Traversal budget** — 4-hop cap + decay should bound it; verify on large graphs.
4. **Graph rot** — decay + reinforcement is the v0.1 answer; needs empirical tuning.
5. **Constraint projection** (fallback 6b) is the least-specified step: how exactly does
   "IAESTE's agenda" get filtered by "date≈26th"? v0.1: constraint attributes are matched
   against candidate node attributes; anything fancier needs the LLM (acceptable — it's a
   *filter*, the LLM still can't inject new assertions).

## 8. Build roadmap

- **Phase 0 — this spec.** ✅
- **Phase 1 — reasoning engine on a hardcoded graph.** Python + NetworkX. ~50 nodes encoding
  S1–S4. Implement steps [3]–[8]. **Milestone: all four scenarios answered with printed chains,
  and S4 correctly comes out ASSUMED, and deleting the agenda node flips S1 to UNKNOWN.**
- **Phase 2 — confidence calculus + gate hardening.** Adversarial cases where the right answer
  is UNKNOWN.
- **Phase 3 — ingestion.** LLM extraction (Claude API) → nodes/edges; entity linking v0.
  Consider swapping substrate to Graphiti here.
- **Phase 4 — NL in/out.** Query parsing to anchors/placeholders; chain verbalization.
- **Phase 5 — temporal machinery.** Proximity boosts, decay, event-triggered (pre-emptive) mode.
- **Phase 6 — evaluation.** LongMemEval / LoCoMo vs Zep & Mem0 baselines; multi-hop QA
  (MuSiQue, 2Wiki, HotpotQA); **novel hallucination eval: refuses-vs-invents rate** — likely a
  definable new metric.
  STATUS: harness LIVE (`bench/run_longmemeval.py`, oracle variant, resumable, stratified
  sampling, LLM judge + abstention scoring). Tracks correct / wrong-but-refused (honest) /
  wrong-and-CONFIRMED (invention) — the last one is the metric that must stay at zero.
- **Phase 7 — synthetic trace factory.** Engine-generated gold traces at scale (see §9).
- **Phase 8 — SFT.** QLoRA a small open model into the memory-native transducer.
- **Phase 9 — RLVR.** Chain-validity rewards; re-run Phase 6 benchmarks with the trained model.
- **Then:** architecture paper (APM as mechanism; memory-native LLM as architecture) +
  open-source reference implementation + model weights.

## 9. Endgame: the memory-native LLM (decided 2026-07-02)

APM is not a bolt-on layer — the target is a **memory architecture in an LLM**: a model hard
fine-tuned so lookup-then-speak is its native behavior.

**The weights/graph split (critical design rule):**
- The graph holds the *memories* (external: editable, retractable, provenance-carrying).
  Facts are NEVER fine-tuned into weights — parametric memory cannot retract or prove.
- The weights hold the *skill*: emitting graph ops (`LOOKUP`, `ACTIVATE`, `CLIMB`,
  `INTERSECT`) as first-class actions — the model's ONLY path to factual content — plus
  chain-bound verbalization and native `ASSUMED`/`UNKNOWN` emission (trained refusal).

**Training recipe (the flywheel):**
1. **Synthetic trace factory:** the Phase 1 symbolic engine generates unlimited gold traces —
   sample graph → sample query/event → run engine → (anchors, placeholders, traversal ops,
   chain, gated answer). Provably correct by construction; no annotation.
2. **SFT:** QLoRA on a small open model (Qwen 7–8B class) to reproduce traces.
3. **RLVR:** evidence chains are machine-checkable → reward = chain exists in graph ∧ chain
   entails answer ∧ gate class correct. Hallucination gets negative reward with a proof.
   (Few domains have verifiable rewards natively; APM has them by design.)
4. **Decode-time verifier stays on** (belt and suspenders): external re-validation of chains
   even after training.

**Positioning ladder, revised:**
- **Depth 1:** symbolic engine + frozen LLM — now serves as prototype, *teacher*, and
  *reward function* for the trained model.
- **Depth 2:** the fine-tuned memory-native model (SFT → RLVR) + hard decode gate. ← target
- **Depth 3 (labs):** same interface learned at pretraining scale.

Headline claim to aim for: a small memory-native model beating frontier models on
LongMemEval/LoCoMo + near-zero invention rate on the refuses-vs-invents eval.
