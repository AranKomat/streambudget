"""Post-hoc query diagnostics using only evidence retained before each original query."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3

from qualify_glm_providers import QUOTAS, save
from streambudget.config import BudgetConfig, Config, load_config
from streambudget.replay import TaskEvent, read_jsonl
from streambudget.runtime import Runtime


def query_boundary(source, qid):
    run = json.loads((source / 'run.json').read_text())
    prediction = next(p for p in run['predictions'] if p['question_id'] == qid)
    trace = read_jsonl(source / 'trace.jsonl')
    action_index = next(i for i, r in enumerate(trace)
                        if r['kind'] == 'agent_action' and r['question_id'] == qid)
    timing = next(r for r in reversed(trace[:action_index])
                  if r['kind'] == 'model_timing' and r['operation'] == 'plan')
    return prediction['as_of'], timing['wall_time'] - timing['wall_s'], timing['request_hash']


def eligible_rows(db_path, source, as_of, wall_cutoff):
    with sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute('''SELECT * FROM evidence WHERE source=? AND input_end<=?
            AND input_available_at<=? AND created_wall<=? ORDER BY seq''',
            (source, as_of, as_of, wall_cutoff)).fetchall()
    return [dict(r) for r in rows]


async def import_evidence(runtime, rows, media_root, as_of):
    ids, counts = {}, {'raw': 0, 'derived': 0}
    runtime.advance(as_of)
    for row in rows:
        parents, payload = json.loads(row['parents']), json.loads(row['payload'])
        if parents:
            if any(p not in ids for p in parents):
                raise ValueError('Missing causal parent in frozen import')
            item = runtime.store.derive(source=row['source'], kind=row['kind'], text=row['text'],
                parents=[ids[p] for p in parents], snapshot=runtime.snapshot(as_of), payload=payload)
            counts['derived'] += 1
        elif row['kind'] == 'frame':
            key = payload['media_key']
            path = (media_root / key).resolve()
            if path.parent != media_root.resolve():
                raise ValueError('Invalid original media path')
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != path.stem:
                raise ValueError('Original media hash mismatch')
            item = await runtime.ingest_frame(row['source'], row['end'], data,
                                              available_at=row['available_at'], schedule=False)
            counts['raw'] += 1
        else:
            item = runtime.store.add_raw(source=row['source'], kind=row['kind'], start=row['start'],
                end=row['end'], available_at=row['available_at'], text=row['text'], payload=payload)
            counts['raw'] += 1
        ids[row['id']] = item.id
    return counts


async def run(pilot, out, allow_network):
    if not allow_network:
        raise ValueError('Explicit --allow-network required')
    provider_receipt = json.loads((out / 'providers/receipt.json').read_text())
    if provider_receipt['status'] != 'complete' or provider_receipt['ledger']['request_attempts'] > 36:
        raise ValueError('Reconcile provider phase before continuing')
    manifest = json.loads((pilot / 'manifest.json').read_text())
    job = next(t for t in manifest['trials'] if t['id'].startswith('gemini-'))
    source = pilot / 'trials' / job['id']
    tasks = [TaskEvent.model_validate(row) for row in read_jsonl(
        Path(manifest['dataset_root']) / job['video'] / 'tasks.jsonl') if row['type'] == 'ask']
    if {task.id for task in tasks} != {'3', '5'}:
        raise ValueError('This diagnostic is restricted to the two approved retained queries')
    folder = out / 'gemini'
    folder.mkdir(exist_ok=False)
    base = load_config('configs/streamarena-followup.yaml')

    async def query(task):
        root = folder / task.id
        config = Config(namespace='gemini-followup-' + task.id, models={'perception': base.models['perception']},
                        budget=BudgetConfig(**QUOTAS['gemini-q' + task.id]))
        config.scheduler.deadline_s = 120
        config.policy.escalation = False
        runtime = Runtime(config, root, allow_network=True)
        receipt = {'question_id': task.id, 'status': 'preparing', 'baseline': 'retained frozen query state',
                   'evaluation': 'post-hoc development diagnostic, not a new held-out benchmark'}
        try:
            cutoff, wall_cutoff, request_hash = query_boundary(source, task.id)
            if cutoff != task.at:
                raise ValueError('Task and original query cutoff differ')
            rows = eligible_rows(source / 'memory.sqlite', task.source, cutoff, wall_cutoff)
            receipt.update(as_of=cutoff, created_wall_cutoff=wall_cutoff, original_request_hash=request_hash,
                imported=await import_evidence(runtime, rows, source / 'media', cutoff))
            runtime.start()
            answer = await runtime.ask(task.question, task.source, as_of=cutoff, question_id=task.id)
            receipt.update(status='complete', answer=asdict(answer))
        except Exception as exc:
            receipt.update(status='failed', error_type=type(exc).__name__)
        finally:
            await runtime.close()
            receipt['ledger'] = runtime.ledger.summary()
            save(root / 'diagnostic.json', receipt)
        print(json.dumps({k: receipt[k] for k in ('question_id', 'status', 'ledger')}), flush=True)
        return receipt

    receipts = await asyncio.gather(*(query(task) for task in tasks))
    save(folder / 'receipt.json', {'status': 'complete', 'queries': receipts,
                                 'successful_queries': sum(r['status'] == 'complete' for r in receipts)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--allow-network', action='store_true')
    args = parser.parse_args()
    asyncio.run(run(args.pilot, args.out, args.allow_network))
