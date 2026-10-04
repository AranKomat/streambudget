"""Read-only, self-contained HTML diagnostics. No game-success inference or external assets."""

from __future__ import annotations

import html
import json
from pathlib import Path
import sqlite3


def summary(run):
    root = Path(run).resolve()
    meta = json.loads((root / "run.json").read_text())
    db = sqlite3.connect((root / "memory.sqlite").as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        jobs = [dict(r) for r in db.execute("SELECT * FROM async_jobs ORDER BY queued_at")]
        actions = [dict(r) for r in db.execute("SELECT * FROM async_actions ORDER BY wall_s")]
        acceptance = {r["job_id"]: dict(r) for r in db.execute("SELECT * FROM async_acceptance")}
        memory = {
            key: db.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
            for key, table in {
                "entities": "tm_nodes",
                "interpretations": "tm_patches",
                "events": "tm_events",
                "messages": "tm_messages",
                "route_observations": "tm_routes",
                "ocr_packets": "tm_ocr_packets",
            }.items()
        }
        usage = []
        for r in db.execute("SELECT role,elapsed,usage FROM model_calls"):
            counts = json.loads(r["usage"]) if r["usage"] else None
            usage.append({"role": r["role"], "elapsed_s": r["elapsed"], "usage": counts})
        return {
            "meta": meta,
            "jobs": jobs,
            "actions": actions,
            "acceptance": acceptance,
            "memory": memory,
            "usage": usage,
            "task_success": None,
            "note": "Raw counts are not semantic accuracy. Intervals may overlap.",
        }
    finally:
        db.close()


def make_report(run):
    root = Path(run).resolve()
    data = summary(root)
    meta = data["meta"]

    def esc(value):
        return html.escape(str(value))

    cards = "".join(
        f'<div class="card"><small>{esc(k)}</small><b>{esc(v)}</b></div>'
        for k, v in [
            ("Actions", meta.get("completed_steps", 0)),
            ("Memory nodes", data["memory"]["entities"]),
            ("Jobs offered", len(data["jobs"])),
            ("Wall seconds", round(meta.get("elapsed_wall_s", 0), 3)),
        ]
    )
    rows = []
    for j in data["jobs"]:
        outcome = data["acceptance"].get(j["id"], {}).get("status", "not consumed")
        duration = round(j["response_wall_s"], 4) if j["response_wall_s"] is not None else "—"
        queue = round(j["started_at"] - j["queued_at"], 4) if j["started_at"] is not None else "—"
        frame = json.loads(j["basis"])["frame_number"]
        rows.append(
            "<tr>"
            + "".join(f"<td>{esc(x)}</td>" for x in (j["role"], frame, j["status"], outcome, queue, duration))
            + "</tr>"
        )
    db = sqlite3.connect((root / "memory.sqlite").as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    frames = []
    try:
        for a in data["actions"][-24:]:
            r = db.execute("SELECT payload FROM evidence WHERE id=?", (a["after_frame"],)).fetchone()
            if r is None:
                continue
            payload = json.loads(r["payload"])
            key = payload.get("media_key", "")
            path = (root / "media" / key).resolve()
            if path.parent != (root / "media").resolve() or not path.is_file():
                continue
            label = json.loads(a["action"])["id"]
            frames.append(
                f'<figure><img src="media/{esc(key)}" alt="Retained source frame"><figcaption>'
                f"{esc(label)} · source frame {esc(payload.get('frame_number'))}</figcaption></figure>"
            )
    finally:
        db.close()
    fixture = (
        "SOFTWARE FIXTURE — injected delays and pixel rules, not a learned model or GPU benchmark."
        if meta.get("fixture")
        else "NATIVE DIAGNOSTICS — task success still requires independent review."
    )
    text = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>StreamBudget · Background memory</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#18262f;font:15px/1.55 system-ui,sans-serif}
main{max-width:1150px;margin:40px auto;padding:0 24px}h1{font-size:36px;letter-spacing:-1px;margin-bottom:6px}
h2{margin-top:32px;font-size:22px}.eyebrow{font-size:12px;letter-spacing:2px;font-weight:750}.notice{padding:14px 18px;border-left:4px solid #9a600d;background:#fff2d8;margin:22px 0}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}.card{background:white;padding:16px;border:1px solid #d9e1e6;border-radius:9px}.card small{display:block;color:#56666d}.card b{font-size:27px}
.scroll{overflow:auto;background:white;border:1px solid #d9e1e6;border-radius:9px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #e4e8eb;white-space:nowrap}th{background:#eaf0f3}
.frames{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}figure{margin:0;padding:10px;background:white;border-radius:8px}img{width:100%;image-rendering:pixelated}figcaption{font-size:12px;color:#56666d}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eaf0f3;padding:18px;border-radius:8px;font-size:12px}
@media(max-width:700px){.cards,.frames{grid-template-columns:repeat(2,1fr)}h1{font-size:28px}}</style><main>
<div class="eyebrow">STREAMBUDGET / PIXELS → STATE → ACTION</div><h1>Background memory, foreground action</h1>
<p>One source/evidence store. Independent bounded inference requests. Historical interpretations retain their original source time.</p>"""
    text += f'<div class="notice">{esc(fixture)}<br>Emulation is stepped. This report does not establish real-time control, OCR quality, task success or inference savings.</div>'
    text += f'<div class="cards">{cards}</div><h2>Request timeline</h2><p>Status: <b>{esc(meta.get("status"))}</b>. Model wall times overlap and must not be added to estimate episode time.</p>'
    text += (
        '<div class="scroll"><table><thead><tr><th>Role</th><th>Source frame</th><th>Transport</th><th>Application acceptance</th><th>Queue s</th><th>Response s</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )
    text += '<h2>Retained action boundaries</h2><div class="frames">' + "".join(frames) + "</div>"
    text += "<h2>Memory bookkeeping</h2><pre>" + esc(json.dumps(data["memory"], indent=2)) + "</pre>"
    text += "<p>Full source images and interpretations remain in this run directory. Similar appearance is not an identity proof. Text and model interpretations remain untrusted evidence.</p></main></html>"
    out = root / "background-report.html"
    out.write_text(text, encoding="utf-8")
    return out
