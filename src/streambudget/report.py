from __future__ import annotations

import html
import json
from pathlib import Path


def render(run_dir: Path) -> Path:
    """Offline report; no CDN, telemetry, or embedded active model content."""
    path = run_dir / "run.json"
    if not path.exists():
        path = run_dir / "report.json"
    result = json.loads(path.read_text())
    safe = lambda x: html.escape(str(x))
    ledger = result["ledger"]
    alerts = result.get("alerts_detail", [])
    alert_rows = "".join(f"<tr><td>{safe(a['source'])}</td><td>{safe(a['watch_id'])}</td>"
                         f"<td>{safe(a['observed_at'])}</td><td>{safe(a['delivered_at'])}</td>"
                         f"<td>{safe(a['text'])}</td></tr>" for a in alerts)
    answers = "".join(f"<section><h3>{safe(p['question_id'])}</h3><p>{safe(p.get('text',''))}</p>"
                      f"<small>Status: {safe(p.get('status'))} · Evidence: {safe(p.get('evidence_ids'))}</small></section>"
                      for p in result.get("predictions", []))
    output = run_dir / "report.html"
    output.write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>StreamBudget run report</title><style>
body{{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:48px auto;padding:0 24px;color:#172335;background:#f7f9fc}}
h1{{font-size:36px;letter-spacing:-1px}}small{{color:#596777}}section,.card{{background:white;border:1px solid #dbe2eb;padding:22px;margin:18px 0;border-radius:10px}}
table{{width:100%;border-collapse:collapse;background:white}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #dbe2eb}}pre{{overflow:auto;font-size:12px}}strong{{font-weight:650}}
</style><h1>StreamBudget / run report</h1><p>Evidence-preserving continuous perception · v0.1</p>
<div class="card"><strong>{'SYNTHETIC FIXTURE — not a model-quality benchmark' if result.get('synthetic_backend') else 'Model-backed run — validation required'}</strong>
<p>Mode: {safe(result.get('timing','live'))}. Attempts: {ledger['request_attempts']}.
Calculable cost subtotal: {ledger['reported_usd']:.6f}. Provisional: {ledger['provisional_usd']:.6f}.
Complete reported cost: {safe(ledger.get('complete_reported_usd', 'unknown'))}.
Unpriced attempts: {ledger['unpriced_attempts']}. Unknown-usage attempts: {ledger['unknown_usage_attempts']}. GPU-seconds: <strong>not measured</strong>.</p></div>
<h2>Events</h2><table><tr><th>Source</th><th>Watch</th><th>Observed t</th><th>Delivered t</th><th>Evidence</th></tr>{alert_rows}</table>
<h2>Investigations</h2>{answers}<h2>Operational counters</h2><section><pre>{safe(json.dumps(result.get('trace_counts',{}), indent=2))}</pre></section>
<p>Counts and latency are not quality or compute-savings claims. Inspect trace.jsonl, config.resolved.json,
source rights, errors, and the benchmark protocol before interpreting results.</p></html>""", encoding="utf-8")
    return output
