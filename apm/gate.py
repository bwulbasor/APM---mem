"""The gate: CONFIRMED / ASSUMED / UNKNOWN + printable evidence chains (SPEC.md §3.4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .activation import Path
from .graph import EvidenceGraph, Node


class VerdictClass(str, Enum):
    CONFIRMED = "CONFIRMED"
    ASSUMED = "ASSUMED"
    UNKNOWN = "UNKNOWN"


@dataclass
class ChainStep:
    kind: str          # "anchor-path" | "projection" | "default" | "direct" | "note"
    text: str


@dataclass
class Chain:
    steps: list[ChainStep] = field(default_factory=list)

    def add(self, kind: str, text: str):
        self.steps.append(ChainStep(kind, text))

    def render(self) -> str:
        return "\n".join(f"  [{s.kind}] {s.text}" for s in self.steps)


@dataclass
class Verdict:
    klass: VerdictClass
    answer: Node | None
    score: float
    chain: Chain
    default_rule: str | None = None      # set on ASSUMED via defaults
    missing_evidence: str | None = None  # set on UNKNOWN
    note: str | None = None
    evidence_edges: list = field(default_factory=list)  # (src, dst, rel) triples

    def render(self) -> str:
        head = f"{self.klass.value}"
        if self.answer is not None:
            head += f": {self.answer.label} (score {self.score:.2f})"
        if self.default_rule:
            head += f" — assumed via {self.default_rule}"
        if self.note:
            head += f" [{self.note}]"
        lines = [head, self.chain.render()]
        if self.missing_evidence:
            lines.append(f"  [missing] {self.missing_evidence}")
        return "\n".join(l for l in lines if l.strip())


def path_text(graph: EvidenceGraph, p: Path) -> str:
    parts = [graph.node(p.anchor).label]
    cur = p.anchor
    for e in p.edges:
        nxt = e.dst if e.src == cur else e.src
        parts.append(f"--{e.rel}({e.weight:.2f})--> {graph.node(nxt).label}")
        cur = nxt
    return " ".join(parts) + f"  [activation {p.score:.2f}]"
