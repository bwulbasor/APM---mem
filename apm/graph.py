"""Evidence graph substrate — typed nodes/edges with provenance (SPEC.md §2)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import networkx as nx


class NodeType(str, Enum):
    PERSON = "person"
    ORG = "org"
    EVENT = "event"
    PLACE = "place"
    TIME = "time"
    TOPIC = "topic"
    ASSERTION = "assertion"


class Status(str, Enum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    ASSUMED = "assumed"
    RETRACTED = "retracted"


PARENT_RELS = {"member_of", "part_of"}


@dataclass
class Node:
    id: str
    type: NodeType
    label: str
    attrs: dict = field(default_factory=dict)
    timestamp: str | None = None  # ISO 8601
    source: str = "fixture"
    status: Status = Status.OBSERVED
    confidence: float = 0.9
    # set on ASSUMED/INFERRED nodes: the query signature they answer,
    # so a later observed fact can retract them (SPEC.md §3.6)
    resolves: str | None = None


@dataclass
class Edge:
    src: str
    dst: str
    rel: str
    weight: float = 0.9
    timestamp: str | None = None
    source: str = "fixture"


class EvidenceGraph:
    def __init__(self):
        self.g = nx.MultiDiGraph()
        self.nodes: dict[str, Node] = {}

    def add_node(self, node: Node) -> Node:
        self.nodes[node.id] = node
        self.g.add_node(node.id)
        return node

    def add_edge(self, edge: Edge) -> Edge:
        if edge.src not in self.nodes or edge.dst not in self.nodes:
            raise KeyError(f"edge endpoints must exist: {edge.src} -> {edge.dst}")
        self.g.add_edge(edge.src, edge.dst, rel=edge.rel, obj=edge)
        return edge

    def node(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def remove_node(self, node_id: str):
        self.g.remove_node(node_id)
        del self.nodes[node_id]

    def neighbors(self, node_id: str):
        """Association is bidirectional: yield (other_id, Edge) over both directions."""
        for _, dst, data in self.g.out_edges(node_id, data=True):
            yield dst, data["obj"]
        for src, _, data in self.g.in_edges(node_id, data=True):
            yield src, data["obj"]

    def parents(self, node_id: str):
        """Parent ladder: nodes reached via member_of / part_of (SPEC.md §2.3)."""
        for _, dst, data in self.g.out_edges(node_id, data=True):
            if data["obj"].rel in PARENT_RELS:
                yield dst, data["obj"]

    def by_type(self, node_type: NodeType):
        return [n for n in self.nodes.values() if n.type == node_type and n.status != Status.RETRACTED]

    # -- forgetting & reinforcement (SPEC.md §3.6) --------------------------
    def apply_decay(self, now: datetime, half_life_days: float = 365.0,
                    floor: float = 0.05):
        """Unreinforced associations weaken over time — the forgetting curve.
        Edges with no timestamp are treated as timeless (semantic) and spared."""
        for _, _, data in self.g.edges(data=True):
            e = data["obj"]
            if not e.timestamp:
                continue
            age_days = (now - parse_ts(e.timestamp)).total_seconds() / 86400.0
            if age_days > 0:
                e.weight = max(floor, e.weight * 0.5 ** (age_days / half_life_days))

    def reinforce(self, edges, factor: float = 1.08, cap: float = 0.95,
                  now: datetime | None = None):
        """Hebbian: edges that carry a successful inference get stronger,
        and their recency resets — recall protects a memory from decay."""
        for e in edges:
            e.weight = min(cap, e.weight * factor)
            if now is not None:
                e.timestamp = now.isoformat()

    def auto_contradict(self) -> int:
        """[v0.6, from held-out invention 852ce960: '$350,000 CONFIRMED' when
        the pre-approval had been updated to $400,000] Same subject + same
        metric + different value = the newer assertion contradicts the older.
        Extractors don't reliably emit contradicts edges across sessions, so
        this MUST be a deterministic rule, not an LLM behavior. Idempotent."""
        import re

        def numbers(n: Node) -> set[str]:
            out = set(re.findall(r"\d[\d,.:]*", n.label))
            for k in ("value", "amount", "price", "cost", "score", "quantity",
                      "total", "metric_value"):
                v = n.attrs.get(k)
                if v is not None:
                    out |= set(re.findall(r"\d[\d,.:]*", str(v)))
            return out

        def metric_sig(n: Node) -> str:
            """The metric is the label with its numbers removed — '$___
            pre-approval' — plus any explicit metric attr."""
            stripped = re.sub(r"[\d$,.:]+", " ", n.label.lower())
            stripped = " ".join(stripped.split())
            return f"{n.attrs.get('metric', '')}|{stripped}"

        cands = [n for n in self.nodes.values()
                 if n.type == NodeType.ASSERTION and numbers(n)]
        existing = {(e.src, e.dst) for _, _, d in self.g.edges(data=True)
                    for e in [d["obj"]] if e.rel == "contradicts"}
        added = 0
        for i, a in enumerate(cands):
            for b in cands[i + 1:]:
                if numbers(a) & numbers(b):        # shared value: no conflict
                    continue
                from .entities import similarity as _sim
                sig_a, sig_b = metric_sig(a), metric_sig(b)
                same_metric = (a.attrs.get("metric") and a.attrs.get("metric")
                               == b.attrs.get("metric")) or \
                    _sim(sig_a.split("|", 1)[1] or a.label,
                         sig_b.split("|", 1)[1] or b.label) >= 85
                if not same_metric:
                    continue
                subj_a = {o for o, _ in self.neighbors(a.id)}
                subj_b = {o for o, _ in self.neighbors(b.id)}
                if not (subj_a & subj_b):          # different subjects: no clash
                    continue
                newer, older = ((a, b) if parse_ts(a.timestamp)
                                >= parse_ts(b.timestamp) else (b, a))
                if (newer.id, older.id) in existing:
                    continue
                self.add_edge(Edge(newer.id, older.id, "contradicts", 0.9,
                                   timestamp=newer.timestamp,
                                   source="auto_contradict"))
                existing.add((newer.id, older.id))
                added += 1

        # value-duplicate propagation (v1.4, mortgage 852ce960): extractors
        # emit the same fact in several phrasings ('Pre-approval of $350,000' /
        # 'Pre-approved amount: $350,000') whose metric attrs differ, so the
        # pairwise rule kills ONE phrasing and the duplicate wins the tie.
        # The update contradicts the VALUE, not one phrasing of it: if A
        # contradicts B, A contradicts every assertion sharing B's numbers
        # and a subject (and not sharing A's numbers).
        for a_id, b_id in list(existing):
            a, b = self.nodes.get(a_id), self.nodes.get(b_id)
            if a is None or b is None:
                continue
            subj_b = {o for o, _ in self.neighbors(b.id)}
            for c in cands:
                if c.id in (a.id, b.id) or (a.id, c.id) in existing:
                    continue
                if not (numbers(c) & numbers(b)) or (numbers(c) & numbers(a)):
                    continue
                if parse_ts(a.timestamp) < parse_ts(c.timestamp):
                    continue                    # older never attacks newer
                if not (subj_b & {o for o, _ in self.neighbors(c.id)}):
                    continue
                self.add_edge(Edge(a.id, c.id, "contradicts", 0.9,
                                   timestamp=a.timestamp,
                                   source="auto_contradict:dup"))
                existing.add((a.id, c.id))
                added += 1
        return added

    # relations where one CURRENT object is expected: a newer edge supersedes
    # older ones (from invention 830ce83f: "Rachel moved" — we CONFIRMED her
    # OLD city because auto_contradict only covered value assertions, not edges)
    FUNCTIONAL_RELS = {"lives_in", "moved_to", "works_at", "employed_by",
                       "married_to", "dating", "resides_in", "based_in",
                       "current_role", "drives", "named"}

    def auto_supersede(self) -> int:
        """Same subject + same functional relation + different object:
        the newest edge keeps its weight; older ones are downweighted so the
        stale fact can no longer outrank (or CONFIRM over) the current one.
        Deterministic, idempotent. Superseded edges are REMEMBERED
        (self.superseded) — past-value questions ("where did I initially...")
        ask for exactly those."""
        if not hasattr(self, "superseded"):
            self.superseded = set()          # {(src, rel, dst)}
        groups: dict[tuple, list] = {}
        for _, _, data in self.g.edges(data=True):
            e = data["obj"]
            if e.rel in self.FUNCTIONAL_RELS:
                groups.setdefault((e.src, e.rel), []).append(e)
        def _edge_time(e):
            # extractors routinely leave edge timestamps null (lawfirm_0015:
            # both 'lives_in' edges had ts=None, so the sort kept insertion
            # order and superseded the NEW address). Fall back to the object
            # node's mention time, then the subject's — the value's own
            # timestamp is the reliable ordering key.
            for ts in (e.timestamp,
                       self.node(e.dst).timestamp if e.dst in self.nodes else None,
                       self.node(e.src).timestamp if e.src in self.nodes else None):
                if ts:
                    return parse_ts(ts)
            return datetime.min

        changed = 0
        for (_, _), edges in groups.items():
            dsts = {e.dst for e in edges}
            if len(dsts) < 2:
                continue
            edges.sort(key=_edge_time, reverse=True)
            newest_dst = edges[0].dst
            for e in edges[1:]:
                if e.dst != newest_dst:
                    self.superseded.add((e.src, e.rel, e.dst))
                    if e.weight > 0.35:
                        e.weight = round(e.weight * 0.35, 3)
                        changed += 1
        return changed

    def consolidate(self) -> dict:
        """[v0.8] The sleep pass: merge near-duplicate assertions (dual-pass
        extraction produces rephrasings of the same fact) and drop observed
        orphans (nodes whose edges all failed to materialize — unreachable by
        activation, pure noise). Numbers-are-identity already guards merges.
        Inferred/assumed nodes are never touched: they carry provenance."""
        from .entities import merge_nodes, similarity
        merged = dropped = 0
        # near-duplicate assertions sharing a neighbor -> merge into the older
        assertions = [n for n in self.nodes.values()
                      if n.type == NodeType.ASSERTION
                      and n.status == Status.OBSERVED]
        gone: set[str] = set()
        for i, a in enumerate(assertions):
            if a.id in gone:
                continue
            for b in assertions[i + 1:]:
                if b.id in gone or a.id in gone:
                    continue
                if similarity(a.label, b.label) < 95:
                    continue
                na = {o for o, _ in self.neighbors(a.id)}
                nb = {o for o, _ in self.neighbors(b.id)}
                if not (na & nb):
                    continue
                keep, drop = ((a, b) if parse_ts(a.timestamp)
                              <= parse_ts(b.timestamp) else (b, a))
                merge_nodes(self, keep.id, drop.id)
                gone.add(drop.id)
                merged += 1
        # observed orphans: no edges at all -> unreachable, noise
        for n in list(self.nodes.values()):
            if (n.status == Status.OBSERVED and not n.id.startswith(("cat_", "ep_"))
                    and self.g.degree(n.id) == 0):
                self.remove_node(n.id)
                dropped += 1
        return {"merged": merged, "dropped": dropped}

    def copy(self) -> "EvidenceGraph":
        """Deep-enough copy for a QUERY OVERLAY (v1.9 cued recall): cued
        edges merge into the copy and the engine re-runs there, so a bad
        second-chance extraction can never pollute the base graph."""
        import dataclasses
        g2 = EvidenceGraph()
        for n in self.nodes.values():
            g2.add_node(dataclasses.replace(n, attrs=dict(n.attrs)))
        for _, _, data in self.g.edges(data=True):
            e = data["obj"]
            g2.add_edge(Edge(e.src, e.dst, e.rel, e.weight,
                             timestamp=e.timestamp, source=e.source))
        g2.superseded = set(getattr(self, "superseded", set()))
        return g2

    # -- write path -------------------------------------------------------
    def add_observation(self, node: Node, edges: list[Edge] = ()):
        """Add an observed fact; retract any ASSUMED node answering the same query."""
        node.status = Status.OBSERVED
        self.add_node(node)
        for e in edges:
            self.add_edge(e)
        if node.resolves:
            for other in self.nodes.values():
                if (other.id != node.id and other.resolves == node.resolves
                        and other.status == Status.ASSUMED):
                    other.status = Status.RETRACTED
        return node


def parse_ts(ts: str | None) -> datetime:
    return datetime.fromisoformat(ts) if ts else datetime.min
