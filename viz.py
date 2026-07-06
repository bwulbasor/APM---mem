"""Generate apm_viz.html — interactive force-directed view of every scenario.

Runs each scenario through the engine first, so the visualization shows the
POST-inference graph: inferred/assumed nodes, reinforced weights, and the
evidence chain highlighted. Zero dependencies; open the file in any browser.
"""
import json
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

from apm import Engine
from scenarios import brain, fixtures


def collect(name: str, g, q, now, description: str):
    v = Engine(g).answer(q, now=now)
    chain_set = set()
    for s, d, r in v.evidence_edges:
        chain_set.add((s, d, r))
        chain_set.add((d, s, r))
    nodes = [{
        "id": n.id, "label": n.label, "type": n.type.value,
        "status": n.status.value, "conf": round(n.confidence, 2),
        "ts": n.timestamp or "",
        "anchor": n.id in q.anchors,
        "answer": bool(v.answer and n.id == v.answer.id),
    } for n in g.nodes.values()]
    edges = [{
        "src": e.src, "dst": e.dst, "rel": e.rel, "w": round(e.weight, 2),
        "chain": (e.src, e.dst, e.rel) in chain_set,
    } for _, _, data in g.g.edges(data=True) for e in [data["obj"]]]
    return {"name": name, "description": description,
            "verdict": v.render(), "klass": v.klass.value,
            "nodes": nodes, "edges": edges}


def build():
    cases = [
        ("S1 Friday the 26th", *fixtures.s1_iaeste(), fixtures.NOW,
         "Fallback ladder: people converge on IAESTE, agenda projects the GA."),
        ("S1b agenda deleted", *fixtures.s1_iaeste(with_agenda=False), fixtures.NOW,
         "Same query, evidence removed: the gate must refuse (UNKNOWN)."),
        ("S2 movie recall", *fixtures.s2_movie(), fixtures.NOW,
         "Tip-of-the-tongue: three partial cues intersect on one movie."),
        ("S3 incoming call", *fixtures.s3_call(), fixtures.NOW,
         "Pre-emptive inference: why is Person X calling? (capped at ASSUMED)"),
        ("S4 corrupted signal", *fixtures.s4_travel(), fixtures.NOW,
         "Garbled destination: recency default, tagged ASSUMED, retractable."),
        ("B1 interference", *brain.b1_interference(), brain.NOW,
         "Bob moved: the newer fact suppresses the older via `contradicts`."),
        ("B2 episodic chain", *brain.b2_episodic_sequence(), brain.NOW,
         "What came after the gym? Sequence links + shared time context."),
        ("B3 priming: 'bank'", *brain.b3_semantic_priming(), brain.NOW,
         "Context (river, fishing buddy) disambiguates an ambiguous word."),
        ("B4 forgetting", *brain.b4_forgetting(), brain.NOW,
         "A 2015 memory, still recallable — decay would erase it honestly."),
        ("B5 source confusion", *brain.b5_source_confusion(), brain.NOW,
         "Two similar dinners: humans confabulate; APM tags ASSUMED."),
        ("B6 flashbulb memory", *brain.b6_salience(), brain.NOW,
         "Salient (high-weight) memories dominate recall of the trip."),
    ]
    data = [collect(name, g, q, now, desc) for name, g, q, now, desc in cases]
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data))
    with open("apm_viz.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote apm_viz.html ({len(data)} scenarios)")


TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>APM — evidence graph explorer</title>
<style>
  * { box-sizing: border-box; margin: 0; }
  body { background: #14161a; color: #d7dae0; font: 14px/1.5 "Segoe UI", system-ui, sans-serif; }
  #bar { display: flex; gap: 10px; align-items: center; padding: 10px 14px; flex-wrap: wrap;
         border-bottom: 1px solid #262a31; }
  #bar h1 { font-size: 15px; font-weight: 600; color: #eef0f4; margin-right: 8px; }
  select { background: #1d2026; color: #d7dae0; border: 1px solid #333842; border-radius: 6px;
           padding: 6px 10px; font-size: 13px; }
  #desc { color: #8b93a1; font-size: 12.5px; }
  #main { display: flex; height: calc(100vh - 51px); }
  #canvasWrap { flex: 1; position: relative; }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; }
  #side { width: 380px; border-left: 1px solid #262a31; padding: 14px; overflow-y: auto; }
  #side h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: #8b93a1;
             margin: 14px 0 6px; }
  #verdict { white-space: pre-wrap; font: 12px/1.6 Consolas, monospace; background: #1a1d22;
             border: 1px solid #262a31; border-radius: 8px; padding: 10px; }
  .k-CONFIRMED { border-left: 3px solid #47c78f !important; }
  .k-ASSUMED   { border-left: 3px solid #f7b64f !important; }
  .k-UNKNOWN   { border-left: 3px solid #e35d6a !important; }
  #legend div { display: flex; align-items: center; gap: 8px; margin: 3px 0; font-size: 12.5px; }
  .sw { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
  #tip { position: absolute; pointer-events: none; background: #1d2026; border: 1px solid #333842;
         border-radius: 6px; padding: 6px 9px; font-size: 12px; display: none; max-width: 260px; }
</style></head><body>
<div id="bar">
  <h1>APM evidence graph</h1>
  <select id="sel"></select>
  <span id="desc"></span>
</div>
<div id="main">
  <div id="canvasWrap"><canvas id="cv"></canvas><div id="tip"></div></div>
  <div id="side">
    <h2>Verdict (gated output)</h2>
    <div id="verdict"></div>
    <h2>Legend</h2>
    <div id="legend"></div>
    <h2>Reading the graph</h2>
    <div style="font-size:12.5px;color:#8b93a1">
      Bold ring = query anchor. Green glow = the answer node.
      Bright green edges = the evidence chain that licensed the verdict.
      Dashed outline = assumed node; dotted = inferred (written back by the engine).
      Edge thickness = association weight. Drag nodes; hover for details.
    </div>
  </div>
</div>
<script>
const DATA = /*__DATA__*/;
const COLORS = { person:"#4f8ef7", org:"#f7b64f", event:"#e35d6a", place:"#47c78f",
                 time:"#9b7ff7", topic:"#47b8c7", assertion:"#d966b8" };
const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
const tip = document.getElementById("tip");
let nodes = [], edges = [], drag = null, hover = null;

function load(i) {
  const sc = DATA[i];
  document.getElementById("desc").textContent = sc.description;
  const vd = document.getElementById("verdict");
  vd.textContent = sc.verdict; vd.className = "k-" + sc.klass;
  const W = cv.clientWidth, H = cv.clientHeight;
  nodes = sc.nodes.map((n, j) => ({ ...n,
    x: W/2 + Math.cos(j*2.399) * (90 + 24*j), y: H/2 + Math.sin(j*2.399) * (70 + 14*j),
    vx: 0, vy: 0 }));
  const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
  edges = sc.edges.map(e => ({ ...e, a: byId[e.src], b: byId[e.dst] }));
  const lg = document.getElementById("legend");
  const types = [...new Set(sc.nodes.map(n => n.type))];
  lg.innerHTML = types.map(t =>
    `<div><span class="sw" style="background:${COLORS[t]}"></span>${t}</div>`).join("");
}

function resize() {
  cv.width = cv.clientWidth * devicePixelRatio; cv.height = cv.clientHeight * devicePixelRatio;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
}
window.addEventListener("resize", resize);

function step() {
  const W = cv.clientWidth, H = cv.clientHeight;
  for (let i = 0; i < nodes.length; i++) for (let j = i+1; j < nodes.length; j++) {
    const a = nodes[i], b = nodes[j];
    let dx = b.x-a.x, dy = b.y-a.y, d2 = dx*dx+dy*dy || 1, d = Math.sqrt(d2);
    const f = Math.min(2200 / d2, 8);
    dx /= d; dy /= d;
    a.vx -= dx*f; a.vy -= dy*f; b.vx += dx*f; b.vy += dy*f;
  }
  for (const e of edges) {
    let dx = e.b.x-e.a.x, dy = e.b.y-e.a.y, d = Math.sqrt(dx*dx+dy*dy) || 1;
    const want = 130, f = (d-want) * 0.004 * (0.5 + e.w/2);
    dx /= d; dy /= d;
    e.a.vx += dx*f*d*0.02; e.a.vy += dy*f*d*0.02;
    e.b.vx -= dx*f*d*0.02; e.b.vy -= dy*f*d*0.02;
  }
  for (const n of nodes) {
    n.vx += (W/2-n.x)*0.0012; n.vy += (H/2-n.y)*0.0012;
    if (drag !== n) { n.x += n.vx *= 0.86; n.y += n.vy *= 0.86; }
    n.x = Math.max(30, Math.min(W-30, n.x)); n.y = Math.max(30, Math.min(H-40, n.y));
  }
}

function draw() {
  const W = cv.clientWidth, H = cv.clientHeight;
  ctx.clearRect(0, 0, W, H);
  for (const e of edges) {
    ctx.beginPath(); ctx.moveTo(e.a.x, e.a.y); ctx.lineTo(e.b.x, e.b.y);
    if (e.chain) { ctx.strokeStyle = "#38e08c"; ctx.lineWidth = 1.2 + e.w*2.4; ctx.globalAlpha = .95; }
    else { ctx.strokeStyle = "#3a4150"; ctx.lineWidth = .6 + e.w*1.8; ctx.globalAlpha = .8; }
    if (e.rel === "contradicts") { ctx.setLineDash([5,4]); ctx.strokeStyle = "#e35d6a"; }
    ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
    const mx = (e.a.x+e.b.x)/2, my = (e.a.y+e.b.y)/2;
    ctx.fillStyle = e.chain ? "#7fe8b6" : "#6b7382"; ctx.font = "10px Consolas, monospace";
    ctx.textAlign = "center"; ctx.fillText(e.rel + " " + e.w.toFixed(2), mx, my - 4);
  }
  for (const n of nodes) {
    const r = n.answer ? 15 : n.anchor ? 12 : 9;
    if (n.answer) {
      ctx.beginPath(); ctx.arc(n.x, n.y, r+7, 0, 7);
      ctx.fillStyle = "rgba(56,224,140,.18)"; ctx.fill();
    }
    ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, 7);
    ctx.fillStyle = COLORS[n.type] || "#888";
    ctx.globalAlpha = n.status === "retracted" ? .3 : 1; ctx.fill();
    if (n.status === "assumed") { ctx.setLineDash([3,3]); }
    if (n.status === "inferred") { ctx.setLineDash([1.5,2.5]); }
    ctx.lineWidth = n.anchor ? 3 : 1.2;
    ctx.strokeStyle = n.anchor ? "#eef0f4" : "#14161a";
    ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
    ctx.fillStyle = n.status === "retracted" ? "#565d69" : "#c6cbd4";
    ctx.font = (n.answer || n.anchor ? "600 " : "") + "11px system-ui";
    ctx.textAlign = "center";
    const label = n.label.length > 26 ? n.label.slice(0, 25) + "…" : n.label;
    ctx.fillText(label, n.x, n.y + r + 13);
    if (n.status === "retracted") ctx.fillText("(retracted)", n.x, n.y + r + 25);
    if (n.status === "inferred") ctx.fillText("(inferred — written back)", n.x, n.y + r + 25);
    if (n.status === "assumed") ctx.fillText("(assumed)", n.x, n.y + r + 25);
  }
}

function loop() { step(); draw(); requestAnimationFrame(loop); }

cv.addEventListener("mousedown", ev => {
  const p = pos(ev); drag = pick(p.x, p.y); if (drag) cv.style.cursor = "grabbing";
});
window.addEventListener("mousemove", ev => {
  const p = pos(ev);
  if (drag) { drag.x = p.x; drag.y = p.y; drag.vx = drag.vy = 0; return; }
  hover = pick(p.x, p.y);
  if (hover) {
    tip.style.display = "block";
    tip.style.left = (p.x + 16) + "px"; tip.style.top = (p.y + 8) + "px";
    tip.innerHTML = `<b>${hover.label}</b><br>type: ${hover.type} · status: ${hover.status}` +
      `<br>confidence: ${hover.conf}` + (hover.ts ? `<br>ts: ${hover.ts}` : "");
  } else tip.style.display = "none";
});
window.addEventListener("mouseup", () => { drag = null; cv.style.cursor = "grab"; });
function pos(ev) { const r = cv.getBoundingClientRect(); return { x: ev.clientX-r.left, y: ev.clientY-r.top }; }
function pick(x, y) { return nodes.find(n => (n.x-x)**2 + (n.y-y)**2 < 260); }

const sel = document.getElementById("sel");
DATA.forEach((s, i) => sel.add(new Option(s.name, i)));
sel.addEventListener("change", () => load(+sel.value));
resize(); load(0); loop();
</script></body></html>
"""

if __name__ == "__main__":
    build()
