import json
import sqlite3
import httpx
import pytest
from pydantic import ValidationError
from streambudget.interactive.contracts import ActionChoice, ActionSpec, Endpoint, GameConfig
from streambudget.interactive.models import ChatBackend, Ledger, ModelError, BudgetExhausted
from streambudget.types import ContractError

def ledger(tmp_path, max_calls=10, max_usd=10):
    db = sqlite3.connect(tmp_path / 'ledger.sqlite')
    db.row_factory = sqlite3.Row
    return Ledger(db, max_calls, max_usd)

def config(**kwargs):
    return GameConfig(endpoints={'main': Endpoint(model='test-model', **kwargs)})

def good_response(content='{"action_id":"A"}', usage=None, reason='stop'):
    return {'choices': [{'finish_reason': reason, 'message': {'content': content}}], 'usage': usage}

def test_network_and_paid_opt_in(tmp_path):
    book = ledger(tmp_path)
    with pytest.raises(ContractError):
        ChatBackend(config(), book)
    cfg = config(base_url='https://api.example.test/v1', billing='metered', input_per_million=1, output_per_million=1)
    with pytest.raises(ContractError):
        ChatBackend(cfg, book, allow_network=True)

@pytest.mark.parametrize('url', ['file:///etc/passwd', 'http://key:pass@localhost/v1', 'https://x/v1?key=secret', 'https://x/v1#fragment'])
def test_endpoint_rejects_credential_routes(url):
    with pytest.raises(ValidationError):
        Endpoint(base_url=url)

def test_remote_cannot_silently_claim_local_free():
    with pytest.raises(ValidationError):
        Endpoint(base_url='https://api.example.test/v1', billing='local')

@pytest.mark.parametrize('extra', [{'messages': []}, {'model': 'other'}, {'tools': []}, {'base_url': 'https://evil'}])
def test_protected_request_fields(extra):
    with pytest.raises(ValidationError):
        Endpoint(extra_body=extra)

def test_same_checkpoint_multiple_roles_and_compact_action(tmp_path):
    book = ledger(tmp_path)
    seen = []

    def respond(request):
        b = json.loads(request.content)
        seen.append(b)
        assert b['model'] == 'test-model'
        assert b['messages'][0]['role'] == 'system'
        return httpx.Response(200, json=good_response())
    b = ChatBackend(config(), book, allow_network=True, transport=httpx.MockTransport(respond))
    assert b.complete('act', 'system', {}, [], ActionChoice).action_id == 'A'
    assert book.summary()['calls'] == 1 and book.summary()['known_provider_usd'] == 0
    assert book.summary()['local_compute_cost'] is None
    assert seen[0]['response_format'] == {'type': 'json_object'}
    b.close()

@pytest.mark.parametrize('bad', ['not json', '{"action_id":"A","reason":"unwanted"}', '{"action_id":"../../x"}'])
def test_invalid_response_is_accounted_without_retry(tmp_path, bad):
    book = ledger(tmp_path)
    calls = []

    def handle(r):
        calls.append(1)
        return httpx.Response(200, json=good_response(bad))
    b = ChatBackend(config(), book, allow_network=True, transport=httpx.MockTransport(handle))
    with pytest.raises(ModelError):
        b.complete('act', 'system', {}, [], ActionChoice)
    assert len(calls) == 1 and book.summary()['failed_calls'] == 1
    b.close()

def test_timeout_keeps_charge_hold(tmp_path):
    book = ledger(tmp_path)
    cfg = config(base_url='https://api.example.test/v1', billing='metered', input_per_million=1, output_per_million=2)

    def fail(req):
        raise httpx.ReadTimeout('do not log secret', request=req)
    b = ChatBackend(cfg, book, allow_network=True, allow_paid=True, transport=httpx.MockTransport(fail))
    with pytest.raises(ModelError):
        b.complete('act', '', {}, [], ActionChoice)
    row = book.db.execute('SELECT * FROM model_calls').fetchone()
    assert row['cost'] is None and row['error'] == 'ReadTimeout'
    assert book.summary()['outstanding_reserved_usd'] == 0.1
    assert 'secret' not in json.dumps(dict(row))
    b.close()

def test_cached_input_billing_and_parse_failure_charge(tmp_path):
    book = ledger(tmp_path)
    cfg = config(base_url='https://api.example.test/v1', billing='metered', input_per_million=1, output_per_million=2, cached_input_per_million=0.5)
    usage = {'prompt_tokens': 1000, 'completion_tokens': 10, 'prompt_tokens_details': {'cached_tokens': 500}}
    b = ChatBackend(cfg, book, allow_network=True, allow_paid=True, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=good_response('bad json', usage))))
    with pytest.raises(ModelError):
        b.complete('act', '', {}, [], ActionChoice)
    assert book.summary()['known_provider_usd'] == pytest.approx(0.00077)
    b.close()

def test_unpriced_cache_is_unknown_not_free(tmp_path):
    book = ledger(tmp_path)
    ep = Endpoint(base_url='https://example.test/v1', model='m', billing='metered', input_per_million=1, output_per_million=2)
    id = book.reserve('act', ep, 'x', 0, 0)
    book.finish(id, endpoint=ep, response={'usage': {'prompt_tokens': 100, 'completion_tokens': 2, 'prompt_tokens_details': {'cached_tokens': 50}}})
    assert book.summary()['unknown_charge_calls'] == 1

def test_call_budget_persists(tmp_path):
    book = ledger(tmp_path, max_calls=1)
    ep = Endpoint(model='test')
    id = book.reserve('act', ep, 'x', 0, 0)
    book.finish(id, endpoint=ep)
    l2 = Ledger(book.db, 1, 10)
    with pytest.raises(BudgetExhausted):
        l2.reserve('act', ep, 'y', 0, 0)

def test_truncation_refused(tmp_path):
    book = ledger(tmp_path)
    b = ChatBackend(config(), book, allow_network=True, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=good_response(reason='length'))))
    with pytest.raises(ModelError):
        b.complete('act', '', {}, [], ActionChoice)
    b.close()

@pytest.mark.parametrize('n', [0, 25, -1, True, '4'])
def test_duration_is_bounded_integer(n):
    with pytest.raises(ValidationError):
        ActionSpec(id='A', button='a', press_frames=n)

def test_unknown_action_extra_fields_rejected():
    with pytest.raises(ValidationError):
        ActionChoice(action_id='A', stop_if='whatever')


def test_flex_and_medium_are_explicit_request_fields(tmp_path):
    book = ledger(tmp_path)
    cfg = config(base_url='https://api.example.test/v1', billing='metered',
                 input_per_million=1, output_per_million=2, service_tier='flex',
                 reasoning_effort='medium', token_parameter='max_completion_tokens', timeout_s=900)
    cfg.role_output_limits['act'] = 2048
    seen = []

    def handle(request):
        seen.append(json.loads(request.content))
        result = good_response(usage={'prompt_tokens': 100, 'completion_tokens': 200,
            'completion_tokens_details': {'reasoning_tokens': 180}})
        result['service_tier'] = 'flex'
        return httpx.Response(200, json=result)

    b = ChatBackend(cfg, book, allow_network=True, allow_paid=True,
                    transport=httpx.MockTransport(handle))
    try:
        b.complete('act', '', {}, [], ActionChoice)
    finally:
        b.close()
    assert seen[0]['service_tier'] == 'flex' and seen[0]['reasoning_effort'] == 'medium'
    assert seen[0]['max_completion_tokens'] == 2048 and 'max_tokens' not in seen[0]
    assert book.summary()['known_provider_usd'] == pytest.approx(0.0005)
    assert book.summary()['reported_tokens']['reasoning_tokens'] == 180
    assert book.summary()['reported_service_tiers'] == {'flex': 1}


@pytest.mark.parametrize('reported', [None, 'default'])
def test_unconfirmed_flex_tier_keeps_hold(tmp_path, reported):
    book = ledger(tmp_path)
    ep = Endpoint(base_url='https://example.test/v1', billing='metered', service_tier='flex',
                  input_per_million=1, output_per_million=2)
    call = book.reserve('act', ep, 'x', 0, 0)
    book.finish(call, endpoint=ep, response={'service_tier': reported,
        'usage': {'prompt_tokens': 100, 'completion_tokens': 10}})
    assert book.summary()['known_provider_usd'] == 0
    assert book.summary()['unknown_charge_calls'] == 1
    assert book.summary()['outstanding_reserved_usd'] == ep.reservation_usd


def test_flex_capacity_failure_has_no_tier_fallback(tmp_path):
    book = ledger(tmp_path)
    cfg = config(base_url='https://example.test/v1', billing='metered', service_tier='flex',
                 input_per_million=1, output_per_million=2)
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(429, json={'error': {'code': 'resource_unavailable'}})

    b = ChatBackend(cfg, book, allow_network=True, allow_paid=True,
                    transport=httpx.MockTransport(handle))
    try:
        with pytest.raises(ModelError, match='HTTP_429'):
            b.complete('act', '', {}, [], ActionChoice)
    finally:
        b.close()
    assert len(requests) == 1 and requests[0]['service_tier'] == 'flex'
    assert book.summary()['unknown_charge_calls'] == 1


@pytest.mark.parametrize('kwargs', [
    {'service_tier': 'auto'}, {'service_tier': 'priority'},
    {'service_tier': 'flex'}, {'reasoning_effort': 'invalid'},
    {'reasoning_effort': 'medium', 'extra_body': {'reasoning_effort': 'low'}},
    {'extra_body': {'service_tier': 'default'}}])
def test_tier_and_reasoning_configuration_fails_closed(kwargs):
    with pytest.raises(ValidationError):
        Endpoint(**kwargs)


def test_missing_token_usage_is_not_reported_as_measured_zero(tmp_path):
    book = ledger(tmp_path)
    ep = Endpoint(model='local-test')
    call = book.reserve('act', ep, 'x', 0, 0)
    book.finish(call, endpoint=ep, response=good_response())
    assert book.summary()['calls_without_complete_token_usage'] == 1
    assert book.summary()['reported_service_tiers'] == {'unreported': 1}
