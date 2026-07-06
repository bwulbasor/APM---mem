# Associative Pattern Matching (APM)

**A reasoning architecture where the graph performs inference and the LLM only translates.**

> Retrieval fetches what is *similar*. Context memory remembers what was *said*.
> APM infers what must be *true* — and shows the chain that proves it.

Every answer is gated into one of three classes — this is the architecture's contract:

| Class | Meaning |
|---|---|
| **CONFIRMED** | an evidence chain exists; it is printed with the answer |
| **ASSUMED** | only a default fired (recency, weak evidence, predicted intent); tagged and retractable |
| **UNKNOWN** | no chain, no default — the system refuses and names the missing evidence |

A confident answer without evidence is *unrepresentable*. Hallucination is prevented
structurally, not statistically.

## Quickstart

```bash
pip install networkx pytest
python -m pytest tests/ -v     # 33 tests: 4 founding + 6 brain scenarios + adversarial suite
python demo.py                 # printed inference traces for the founding scenarios
python viz.py                  # generates apm_viz.html — interactive graph explorer
python demo_e2e.py             # full pipeline: raw text -> LLM ingest -> engine -> answer
```

`demo_e2e.py` needs LLM credentials in `.env` (`APM_LLM_BASE_URL`, `APM_LLM_MODEL`,
`APM_LLM_API_KEY`, OpenAI-compatible endpoint). Everything else runs offline.

## How it works

```
input ─▶ [1] PARSE (LLM)      anchors + typed unknowns
         [3] DIRECT LOOKUP    unique strict match? done
         [4] ACTIVATE         spreading activation, decay per hop, ≤4 hops
         [5] INTERSECT        noisy-OR over independent anchor paths
         [6] FALLBACK         parent climb -> constraint projection -> tagged defaults
         [7] RANK             direct > intersection > temporal > recency
         [8] GATE             CONFIRMED / ASSUMED / UNKNOWN
         [9] VERBALIZE (LLM)  answer citing the chain
        [10] WRITE-BACK       inference stored; contradictions retract; chain reinforced
```

The LLM appears at [1] and [9] only. Steps [3]–[8], [10] are deterministic graph
operations — reproducible, auditable, and usable as a training-data factory for the
memory-native model (SPEC.md §9).

## Brain-inspired behaviors (implemented + tested)

| Human phenomenon | APM mechanism | Scenario |
|---|---|---|
| Tip-of-the-tongue recall | intersection of partial cues | S2 |
| Pre-emptive intent guessing | event-triggered predictive query, capped at ASSUMED | S3 |
| Retroactive interference ("he moved") | `contradicts` edges, newer suppresses older | B1 |
| Episodic sequence recall | `precedes` links + shared time context | B2 |
| Semantic priming ("the bank") | context anchors + defining constraints | B3 |
| The forgetting curve | edge decay -> honest UNKNOWN | B4 |
| Rehearsal protects memories | Hebbian reinforcement on recall | B4 |
| False memory / confabulation | source confusion capped at ASSUMED | B5 |
| Flashbulb memories | salience = edge weight | B6 |

## Benchmark (LongMemEval)

```bash
# dataset: bench/data/longmemeval_oracle.json (downloaded from HF longmemeval-cleaned)
python bench/run_longmemeval.py --n 10             # slice
python bench/run_longmemeval.py --stratified 2     # 2 per question type
```

Results append to `bench/results.jsonl` (resumable). The summary reports the three-way
outcome that is the architecture's thesis:

- **accuracy** — judged against gold answers (LLM judge; abstention questions score
  refusal as correct)
- **refusal rate** — wrong-but-honest (UNKNOWN instead of a guess)
- **invention rate** — wrong AND emitted as CONFIRMED. *This is the number the gate
  exists to hold at zero.*

## Layout

```
apm/            the engine: graph, activation, scoring, fallback, gate, engine, ingest, llm
scenarios/      founding scenarios (fixtures.py) + brain scenarios (brain.py)
tests/          33 tests incl. the adversarial "UNKNOWN is correct" suite
viz.py          -> apm_viz.html (self-contained interactive explorer)
demo.py         offline traces · demo_e2e.py  live text-to-answer pipeline
SPEC.md         the architecture specification (v0.1)
HANDOFF.md      session continuity + current state
```
