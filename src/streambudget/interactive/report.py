"""Offline, dependency-free report. Reports claims, not invented game completion scores."""
from __future__ import annotations

import html
import json
import re
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from statistics import median

from ..store import EvidenceStore
from ..types import ContractError


def run_metrics(out: Path):
    """Descriptive offline receipts, never semantic success or a controller input."""
    meta = json.loads((out / "run.json").read_text())
    with closing(sqlite3.connect((out / "memory.sqlite").resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        actions = list(db.execute("SELECT status,receipt FROM game_actions ORDER BY rowid"))
        calls = list(db.execute("SELECT role,status,elapsed FROM model_calls ORDER BY rowid"))
        frames = list(db.execute("SELECT payload FROM evidence WHERE kind='frame' ORDER BY seq"))
    payloads = [json.loads(r['payload']) for r in frames]
    receipts = [json.loads(r['receipt']) for r in actions if r['receipt']]
    by_role = {}
    for role in sorted({r['role'] for r in calls}):
        rows = [r for r in calls if r['role'] == role]
        latencies = [r['elapsed'] for r in rows if r['elapsed'] is not None]
        by_role[role] = {'attempts': len(rows), 'failed_or_pending': sum(r['status'] != 'completed' for r in rows),
                         'wall_s': sum(latencies), 'median_wall_s': median(latencies) if latencies else None}
    return {'run_id': meta['run_id'], 'condition': meta['config']['baseline'], 'fixture': meta['fixture'],
        'status': meta['status'], 'completed_steps': meta['completed_steps'],
        'action_attempts': len(actions), 'action_statuses': dict(Counter(r['status'] for r in actions)),
        'advanced_frames_from_receipts': sum(r['end_frame'] - r['start_frame'] for r in receipts),
        'retained_frames': len(payloads),
        'initial_screen_sha256': payloads[0].get('sha256') if payloads else None,
        'unchanged_pixel_transitions': sum(a.get('sha256') == b.get('sha256') for a, b in zip(payloads, payloads[1:])),
        'model_attempts_by_role': by_role, 'elapsed_wall_s': meta.get('elapsed_wall_s'),
        'accounting': meta.get('accounting'), 'fixture_model_calls': meta.get('fixture_model_calls'),
        'game_success': None,
        'note': 'Pixels unchanged is not a semantic stall. Milestones/errors need independent source review; '
                'fixture calls are not measured inference. No success rate is computed.'}


def compare_runs(paths: list[Path]):
    if len(paths) != 2 or paths[0].resolve() == paths[1].resolve():
        raise ContractError('Compare two distinct recent/world runs')
    metas = [json.loads((p / 'run.json').read_text()) for p in paths]
    if {m['config']['baseline'] for m in metas} != {'recent', 'world'}:
        raise ContractError('Comparison requires one recent and one world condition')
    if any(m['status'] == 'running' for m in metas):
        raise ContractError('Stop both runs before comparison')
    configs = [{k: v for k, v in m['config'].items() if k != 'baseline'} for m in metas]
    differences = ['config.' + k for k in sorted(set(configs[0]) | set(configs[1]))
                   if configs[0].get(k) != configs[1].get(k)]
    for key in ('fixture', 'initialization', 'initial_state_sha256', 'rom_sha256', 'environment',
                'resume_configurations'):
        # Resume histories differ only in baseline for a legitimately matched pair.
        values = [m.get(key) for m in metas]
        if key == 'resume_configurations':
            values = [[{k: v for k, v in c.items() if k != 'baseline'} for c in (x or [])] for x in values]
        if values[0] != values[1]:
            differences.append(key)
    if differences:
        raise ContractError('Unmatched conditions: ' + ', '.join(differences))
    if not metas[0]['fixture'] and (not metas[0].get('rom_sha256') or not metas[0].get('environment')):
        raise ContractError('Native comparison lacks ROM/emulator provenance')
    metrics = [run_metrics(p) for p in paths]
    if (not metrics[0]['initial_screen_sha256'] or
            metrics[0]['initial_screen_sha256'] != metrics[1]['initial_screen_sha256']):
        raise ContractError('Initial retained screenshots differ or lack hashes')
    return {'settings_matched': True, 'fixture': metas[0]['fixture'], 'runs': metrics,
        'success_rate': None, 'note': 'Matched configuration is not statistical evidence of benefit. '
        'Failures remain included. Recent retains text/history and lexical retrieval; it is not memoryless. '
        'Source grounding, start-screen equivalence and milestone reviews remain required.'}


def make_report(out: Path):
    meta = json.loads((out / "run.json").read_text())
    store = EvidenceStore(out / "memory.sqlite", meta["config"]["source"])
    records = []
    trace = [json.loads(line) for line in (out / "trace.jsonl").read_text().splitlines()]
    transitions = {r['execution_id']: r for r in trace if r['kind'] == 'action'}
    calls_by_frame = {}
    for call in store.db.execute("SELECT c.*,i.context FROM model_calls c LEFT JOIN model_inputs i ON c.id=i.call_id ORDER BY c.rowid"):
        context = json.loads(call['context']) if call['context'] else {}
        detail = {k: call[k] for k in ('role', 'model', 'status', 'elapsed', 'error')}
        detail['usage'] = json.loads(call['usage']) if call['usage'] else None
        response = json.loads(call['response']) if call['response'] else None
        detail['reported_service_tier'] = response.get('service_tier') if isinstance(response, dict) else None
        calls_by_frame.setdefault(context.get('current_frame_id'), []).append(detail)
    for row in store.db.execute("SELECT * FROM game_actions ORDER BY rowid"):
        r = dict(row)
        source = store.get(r["source_frame"])
        transition = transitions.get(r['id'])
        after = store.get(transition['after']) if transition else None
        records.append({"image": "media/"+source.payload['media_key'],
            "after_image": "media/"+after.payload['media_key'] if after else None,
            "action": json.loads(r["action"]), "receipt": json.loads(r['receipt']) if r['receipt'] else None,
            "intent": transition['intent'] if transition else None,
            "wall_s": transition['wall_s'] if transition else None,
            "status": r["status"], "evidence": source.id,
            "after_evidence": after.id if after else None, 'source_frame': source.payload.get('frame_number'),
            'model_calls_for_source': calls_by_frame.get(source.id, [])})
    current = store.get(meta["current_frame_id"])
    records.append({"image": "media/"+current.payload["media_key"], "after_image": None, "action": {"id": "END"},
                    "status": meta["status"], "evidence": current.id})
    store.close()
    data = json.dumps(records, ensure_ascii=False).replace("<", "\\u003c")
    summary = html.escape(json.dumps(run_metrics(out), indent=2))
    title = "SOFTWARE FIXTURE — not a model or Pokémon result" if meta["fixture"] else "PIXELS-ONLY GAME RUN — ungraded"
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>StreamBudget Pixel Agent</title><style>
body{font:16px system-ui,sans-serif;max-width:1100px;margin:3rem auto;padding:0 1.2rem;background:#f5f6f8;color:#17212b}
h1{font-size:1.6rem} .tag{padding:1rem;background:#fff2cf;border-left:4px solid #c08718}
main{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:1.5rem;margin:2rem 0}
section{min-width:0}img{width:480px;max-width:100%;aspect-ratio:10/9;object-fit:contain;image-rendering:pixelated;
border:1px solid #ddd}pre{white-space:pre-wrap;word-break:break-word;background:white;padding:1rem}
input{width:100%;min-width:0}button{width:40px;height:40px;font-size:20px}small{display:block;margin:1rem 0}
.toolbar{display:flex;gap:8px;align-items:center} .toolbar input{flex:1}
@media(max-width:850px){main{grid-template-columns:1fr}}
</style><h1>StreamBudget · Pixel Agent</h1><div class="tag">TITLE</div>
<p>Rendered pixels → generic observations → persistent beliefs → bounded categorical action.</p>
<small>Stepped emulator: paused during inference. Playback below is a decision slideshow, not real-time video.
No badge count or success score is inferred from model assertions.</small>
<div class="toolbar"><button id="prev" title="Previous decision" aria-label="Previous decision">&larr;</button>
<input id="scrub" aria-label="Decision" type="range" min="0" value="0">
<button id="next" title="Next decision" aria-label="Next decision">&rarr;</button></div>
<main><section><h2>Before</h2><img id="frame" alt="Recorded before frame"></section>
<section><h2>After</h2><img id="after" alt="Recorded after frame"><p id="after-status"></p></section></main>
<h2>Decision receipts</h2><pre id="step"></pre><h2>Run receipts</h2><pre>SUMMARY</pre>
<h2>Final world beliefs</h2><pre>WORLD</pre>
<script>const records=DATA;const slider=document.getElementById('scrub');slider.max=records.length-1;
function show(){const r=records[Number(slider.value)];document.getElementById('frame').src=r.image;
const after=document.getElementById('after');after.hidden=!r.after_image;
if(r.after_image)after.src=r.after_image;else after.removeAttribute('src');
document.getElementById('after-status').textContent=r.after_image?'':r.action.id==='END'?'Final retained frame.':'No retained after frame; outcome unknown.';
const {image,after_image,...details}=r;document.getElementById('step').textContent=JSON.stringify({decision:Number(slider.value),...details},null,2);}
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
