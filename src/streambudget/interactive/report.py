"""Offline, dependency-free report. Reports claims, not invented game completion scores."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

from ..store import EvidenceStore


def make_report(out: Path):
    meta = json.loads((out / "run.json").read_text())
    store = EvidenceStore(out / "memory.sqlite", meta["config"]["source"])
    records = []
    for row in store.db.execute("SELECT * FROM game_actions ORDER BY rowid"):
        r = dict(row)
        image = store.get(r["source_frame"]).payload.get("media_key")
        records.append({"image": "media/"+image, "action": json.loads(r["action"]),
                        "status": r["status"], "evidence": r["source_frame"]})
    current = store.get(meta["current_frame_id"])
    records.append({"image": "media/"+current.payload["media_key"], "action": {"id": "END"},
                    "status": meta["status"], "evidence": current.id})
    store.close()
    data = json.dumps(records, ensure_ascii=False).replace("<", "\\u003c")
    summary = html.escape(json.dumps({k: meta.get(k) for k in
        ("status", "initialization", "completed_steps", "elapsed_wall_s", "accounting", "game_success", "fixture_model_calls")}, indent=2))
    title = "SOFTWARE FIXTURE — not a model or Pokémon result" if meta["fixture"] else "PIXELS-ONLY GAME RUN — ungraded"
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>StreamBudget Pixel Agent</title><style>
body{font:16px system-ui,sans-serif;max-width:1100px;margin:3rem auto;padding:0 1.2rem;background:#f5f6f8;color:#17212b}
h1{font-size:2.2rem} .tag{padding:1rem;background:#fff2cf;border-left:4px solid #c08718}
main{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);gap:1.5rem;margin:2rem 0}
section{min-width:0}img{width:480px;max-width:100%;image-rendering:pixelated;
border:1px solid #ddd}pre{white-space:pre-wrap;word-break:break-word;background:white;padding:1rem}
input{width:100%}button{padding:.5rem 1rem}small{display:block;margin:1rem 0}
@media(max-width:850px){main{grid-template-columns:1fr}}
</style><h1>StreamBudget · Pixel Agent</h1><div class="tag">TITLE</div>
<p>Rendered pixels → generic observations → persistent beliefs → bounded categorical action.</p>
<small>Stepped emulator: paused during inference. Playback below is a decision slideshow, not real-time video.
No badge count or success score is inferred from model assertions.</small>
<main><section><img id="frame" alt="Recorded frame"><input id="scrub" type="range" min="0" value="0">
<button id="prev">Previous</button> <button id="next">Next</button><pre id="step"></pre></section>
<section><h2>Run receipts</h2><pre>SUMMARY</pre></section></main>
<h2>Final world beliefs</h2><pre>WORLD</pre>
<script>const records=DATA;const slider=document.getElementById('scrub');slider.max=records.length-1;
function show(){const r=records[Number(slider.value)];document.getElementById('frame').src=r.image;
const {image,...details}=r;document.getElementById('step').textContent=JSON.stringify({decision:Number(slider.value),...details},null,2);}
slider.oninput=show;document.getElementById('next').onclick=()=>{slider.value=Math.min(Number(slider.max),Number(slider.value)+1);show()};
document.getElementById('prev').onclick=()=>{slider.value=Math.max(0,Number(slider.value)-1);show()};show();</script></html>"""
    replacements = {"TITLE": html.escape(title), "SUMMARY": summary,
                    "WORLD": html.escape(json.dumps(meta.get("final_world"), indent=2, ensure_ascii=False)),
                    "DATA": data}
    # One template pass: untrusted inserted text must not become a second placeholder.
    page = re.sub(r"\b(TITLE|SUMMARY|WORLD|DATA)\b", lambda m: replacements[m[0]], page)
    (out / "report.html").write_text(page, encoding="utf-8")
    return out / "report.html"


def export_transitions(out: Path, destination: Path):
    meta = json.loads((out / "run.json").read_text())
    store = EvidenceStore(out / "memory.sqlite", meta["config"]["source"])
    lines = [json.loads(x) for x in (out / "trace.jsonl").read_text().splitlines()]
    with destination.open("x", encoding="utf-8") as f:
        for r in lines:
            if r["kind"] != "action":
                continue
            before, after = store.get(r["before"]), store.get(r["after"])
            row = {"before": before.payload["media_key"], "after": after.payload["media_key"],
                "action": r["action"], "intent": r["intent"], "receipt": r["receipt"],
                "before_evidence": before.id, "after_evidence": after.id,
                "fixture": meta["fixture"], "label_status": "unreviewed", "task_success": None}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    store.close()
