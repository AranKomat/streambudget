"""Small synthetic transport/schema diagnostic, not a video-understanding benchmark."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import httpx

from streambudget.agent import Action, AnswerArgs, InspectArgs, PLAN_SYSTEM, SearchArgs, SensorArgs, StateArgs
from streambudget.backend import ImageInput, ModelPool, Request
from streambudget.budget import Ledger
from streambudget.config import BudgetConfig, Config, Prices, load_config
from streambudget.runtime import SYSTEM_PERCEPTION
from streambudget.screening import frame
from streambudget.trace import Trace
from streambudget.types import Perception
from streambudget.validation import validate_model, validate_response

PROVIDERS = [('z-ai/fp8', 'Z.AI'), ('fireworks', 'Fireworks'), ('novita/fp8', 'Novita'),
             ('sail-research/fp8', 'Sail Research'), ('together', 'Together'), ('modal/fp8', 'Modal')]
QUOTAS = {'providers': {'max_requests': 36, 'max_usd': .5},
          'gemini-q3': {'max_requests': 12, 'max_usd': .25},
          'gemini-q5': {'max_requests': 12, 'max_usd': .25}}
ARGUMENTS = {'answer': AnswerArgs, 'search': SearchArgs, 'inspect': InspectArgs, 'ocr': InspectArgs,
             'query_sensor': SensorArgs, 'read_state': StateArgs}


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def cases():
    images = [ImageInput('fixture-0', 0, frame([])),
              ImageInput('fixture-1', 1, frame([('red', 'A', 100, 150)]))]
    result = []
    for name, selected, watches in (
        ('empty-no-watch', images[:1], []),
        ('red-no-watch', images[1:], []),
        ('red-transition-watch', images, [{'id': 'w1', 'goal': 'A red rectangle is visible.'}]),
    ):
        context = {'source': 'fixture', 'as_of': selected[-1].timestamp, 'watches': watches, 'signals': []}
        result.append((name, Request('perceive', SYSTEM_PERCEPTION,
            'Observe these frames and conditions. JSON context:\n' + json.dumps(context), selected, context)))
    evidence = {'id': 'caption-1', 'source': 'fixture', 'kind': 'caption', 'start': 1, 'end': 1,
                'text': 'A red rectangle is visible.', 'parents': ['fixture-1']}
    for name, question, steps, last, ids in (
        ('plan-start', 'What color is the rectangle?', [], {}, []),
        ('plan-evidence', 'What color is the rectangle?',
         [{'tool': 'inspect', 'arguments': {'source': 'fixture', 'start': 1, 'end': 1},
           'result': {'evidence': [evidence]}}], {'evidence': [evidence]}, ['caption-1']),
        ('plan-missing-audio', 'What color was the rectangle when the horn sounded?',
         [{'tool': 'query_sensor', 'arguments': {'source': 'fixture'}, 'result': {'evidence': []}}],
         {'evidence': []}, []),
    ):
        ctx = {'question': question, 'source': 'fixture', 'as_of': 1, 'steps': steps,
               'last_result': last, 'allowed_evidence_ids': ids}
        result.append((name, Request('plan', PLAN_SYSTEM, json.dumps(ctx), context=ctx)))
    return result


async def run(out, allow_network):
    if not allow_network:
        raise ValueError('Explicit --allow-network required')
    out.mkdir(parents=True, exist_ok=False)
    base = load_config('configs/streamarena-followup.yaml')
    response = httpx.get('https://openrouter.ai/api/v1/models/z-ai/glm-5.3-flash/endpoints', timeout=30)
    response.raise_for_status()
    endpoints = {e['tag']: e for e in response.json()['data']['endpoints']}
    models, catalog = {}, []
    for tag, name in PROVIDERS:
        endpoint = endpoints[tag]
        if endpoint['provider_name'] != name or 'response_format' not in endpoint['supported_parameters']:
            raise ValueError('Provider identity/parameter support changed')
        model = base.models['glm'].model_copy(deep=True)
        model.expected_provider_names = [name]
        model.extra_body['provider'] = {'only': [tag], 'allow_fallbacks': False, 'require_parameters': True}
        prices = endpoint['pricing']
        model.prices = Prices(input_per_million=float(prices['prompt']) * 1e6,
            output_per_million=float(prices['completion']) * 1e6,
            cached_input_per_million=float(prices.get('input_cache_read', prices['prompt'])) * 1e6)
        models[tag] = model
        catalog.append(endpoint)
    cfg = Config(models={'perception': models[PROVIDERS[0][0]], **models}, budget=BudgetConfig(**QUOTAS['providers']))
    requests = cases()
    save(out / 'manifest.json', {'quotas': QUOTAS, 'providers': PROVIDERS, 'catalog': catalog,
        'config': cfg.public_dict(), 'purpose': 'Synthetic schema/route probes, no video-quality claims',
        'prompt_sha256': [hashlib.sha256((r.system + r.text).encode()).hexdigest() for _, r in requests]})
    trace = Trace(out / 'trace.jsonl')
    ledger = Ledger(cfg.budget, trace)
    pool = ModelPool(cfg, ledger, trace, allow_network=True)
    rows = []
    try:
        # Priority order is explicit; cases within one pinned provider run concurrently.
        for tag, name in PROVIDERS:
            async def work(case, request):
                row = {'provider': name, 'tag': tag, 'case': case, 'operation': request.operation}
                try:
                    result = await pool.call(tag, request)
                    parsed = validate_response(Perception if request.operation == 'perceive' else Action,
                                               result, trace, request.operation)
                    if request.operation == 'plan':
                        args = validate_model(ARGUMENTS[parsed.tool], parsed.arguments, trace, 'tool_arguments')
                        if parsed.tool == 'answer' and not set(args.evidence_ids).issubset(request.context['allowed_evidence_ids']):
                            raise ValueError('Unexposed evidence citation')
                        row['tool'] = parsed.tool
                    else:
                        expected = {w['id'] for w in request.context['watches']}
                        if {c.watch_id for c in parsed.checks} != expected or len(parsed.checks) != len(expected):
                            raise ValueError('Missing/duplicate/invented watch check')
                    row.update(status='valid', latency_s=result.latency_s, route=result.routing)
                except Exception as exc:
                    row.update(status='failed', error_type=type(exc).__name__)
                rows.append(row)
                save(out / 'receipt.json', {'status': 'running', 'ledger': ledger.summary(), 'rows': rows})
            await asyncio.gather(*(work(name, request) for name, request in requests))
            print(json.dumps({'provider': name, 'valid': sum(r['status'] == 'valid' for r in rows if r['provider'] == name),
                              'ledger': ledger.summary()}), flush=True)
    finally:
        await pool.close()
        save(out / 'receipt.json', {'status': 'stopped', 'ledger': ledger.summary(), 'rows': rows})
    save(out / 'receipt.json', {'status': 'complete', 'ledger': ledger.summary(), 'rows': rows})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--allow-network', action='store_true')
    args = parser.parse_args()
    asyncio.run(run(args.out, args.allow_network))
