"""Publish only provider-schema metrics and query completion statuses."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(root):
    providers = json.loads((root / 'providers/receipt.json').read_text())
    manifest = json.loads((root / 'providers/manifest.json').read_text())
    gemini = json.loads((root / 'gemini/receipt.json').read_text())
    if providers['status'] != 'complete' or gemini['status'] != 'complete':
        raise ValueError('Diagnostic phases incomplete')
    ledgers = [providers['ledger']] + [q['ledger'] for q in gemini['queries']]
    totals = {key: sum(row[key] for row in ledgers) for key in
              ('request_attempts', 'reported_usd', 'provisional_usd', 'unknown_usage_attempts')}
    totals['complete_reported_usd'] = (totals['reported_usd'] if
        all(row['complete_reported_usd'] is not None for row in ledgers) else None)
    assert totals['request_attempts'] <= 60
    rows = []
    for tag, name in manifest['providers']:
        items = [r for r in providers['rows'] if r['tag'] == tag]
        rows.append({'provider': name, 'tag': tag, 'attempted': len(items),
            'valid': sum(r['status'] == 'valid' for r in items),
            'failures': [{'case': r['case'], 'error_type': r['error_type']} for r in items if r['status'] != 'valid']})
    queries = []
    for q in gemini['queries']:
        answer = q.get('answer', {})
        queries.append({'question_id': q['question_id'], 'execution_status': q['status'],
            'answer_status': answer.get('status'), 'model_calls': q['ledger']['request_attempts'],
            'tool_steps': [s['tool'] for s in answer.get('tool_steps', [])],
            'elapsed_s': answer.get('elapsed_s'), 'imported': q.get('imported'),
            'reported_usd': q['ledger']['reported_usd']})
    trace = [json.loads(s) for s in (root / 'providers/trace.jsonl').read_text().splitlines()]
    errors = Counter(issue['type'] for r in trace if r['kind'] == 'schema_failure' for issue in r['issues'])
    return {'protocol': 'Synthetic pinned-provider format probes plus post-hoc frozen-state Gemini queries',
            'fresh_benchmark': False, 'admission_ceiling': {'calls': 60, 'usd': 1},
            'totals': totals, 'providers': rows, 'gemini_queries': queries,
            'schema_issue_types': dict(errors), 'provider_prompt_hashes': manifest['prompt_sha256']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.run)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report['totals'], indent=2))
