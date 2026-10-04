import pytest
from streambudget.store import EvidenceStore
from streambudget.types import ContractError, NotAvailable
from streambudget.interactive.contracts import Fact, Mention, ObservationPatch, Region, Relation, SchemaPatch, Property, Utterance
from streambudget.interactive.world import World

@pytest.fixture
def setup(tmp_path):
    s = EvidenceStore(tmp_path / 'memory.sqlite')
    w = World(s)
    yield (s, w)
    s.close()

def frame(s, t):
    return s.add_raw(source='game', kind='frame', start=t, end=t, available_at=t, payload={'n': t})

def apply(s, w, f, patch, offered=None):
    return w.apply(patch, s.snapshot(f.end), offered_frames={f.id}, offered_entities=offered or set(), consumed_evidence=[f.id])

def test_evidence_source_hash_is_not_temporal_identity(setup):
    s, w = setup
    a, b = (frame(s, 1), frame(s, 2))
    assert a.id != b.id
    with pytest.raises(NotAvailable):
        s.get(b.id, s.snapshot(1))

def test_instances_do_not_change_schema(setup):
    s, w = setup
    f = frame(s, 1)
    r = apply(s, w, f, ObservationPatch(frame_id=f.id, summary='A room and two similar workers.', mentions=[Mention(ref='new:r', kind='place', label='room'), Mention(ref='new:a', kind='entity', label='same uniform'), Mention(ref='new:b', kind='entity', label='same uniform')]))
    assert len(set(r.ids.values())) == 3
    assert w.ontology.current['version'] == 0

def test_patch_idempotency(setup):
    s, w = setup
    f = frame(s, 1)
    p = ObservationPatch(frame_id=f.id, summary='Thing', mentions=[Mention(ref='new:x', kind='entity', label='x')])
    a = apply(s, w, f, p)
    b = apply(s, w, f, p)
    assert a.ids == b.ids and (not b.changed)
    with pytest.raises(ContractError):
        apply(s, w, f, p.model_copy(update={'summary': 'changed'}))

@pytest.mark.parametrize('kind', ['bad_frame', 'bad_fact', 'bad_relation', 'bad_reference', 'bad_region'])
def test_invalid_patch_has_no_partial_entities(setup, kind):
    s, w = setup
    f = frame(s, 1)
    p = ObservationPatch(frame_id=f.id, summary='test', mentions=[Mention(ref='new:x', kind='entity', label='x')])
    if kind == 'bad_frame':
        p.frame_id = 'not_offered'
    if kind == 'bad_fact':
        p.facts = [Fact(subject='new:x', key='nonexistent', value='x')]
    if kind == 'bad_relation':
        p.relations = [Relation(subject='new:x', target='new:x', predicate='invented')]
    if kind == 'bad_reference':
        p.facts = [Fact(subject='new:y', key='role', value='x')]
    if kind == 'bad_region':
        p.mentions[0].region = Region(frame_id='unknown', box=[0, 0, 1, 1])
    with pytest.raises(ContractError):
        apply(s, w, f, p)
    assert not w.entity_ids()

def test_low_confidence_association_never_merges(setup):
    s, w = setup
    f = frame(s, 1)
    a = apply(s, w, f, ObservationPatch(frame_id=f.id, summary='a', mentions=[Mention(ref='new:a', kind='entity', label='person')]))
    id = a.ids['new:a']
    g = frame(s, 2)
    p = ObservationPatch(frame_id=g.id, summary='similar', mentions=[Mention(ref=id, kind='entity', label='person', association='reidentified', confidence=0.4)])
    with pytest.raises(ContractError):
        apply(s, w, g, p, {id})
    assert len(w.entity_ids()) == 1

def test_snapshot_unchanged_after_future_world_update(setup):
    s, w = setup
    f = frame(s, 1)
    r = apply(s, w, f, ObservationPatch(frame_id=f.id, summary='red', mentions=[Mention(ref='new:a', kind='entity', label='a')], facts=[Fact(subject='new:a', key='status', value='red', confidence=1)]))
    snap = s.snapshot(1)
    old = w.view(snap)
    id = r.ids['new:a']
    g = frame(s, 2)
    apply(s, w, g, ObservationPatch(frame_id=g.id, summary='blue', mentions=[Mention(ref=id, kind='entity', label='a', association='continuity', confidence=1)], facts=[Fact(subject=id, key='status', value='blue', confidence=1)]), {id})
    assert w.view(snap) == old
    assert w.view(s.snapshot(2))['entities'][0]['facts']['status']['value'] == 'blue'

def test_same_sprite_entities_keep_separate_conversations(setup):
    s, w = setup
    f = frame(s, 1)
    p = ObservationPatch(frame_id=f.id, summary='two people', mentions=[Mention(ref='new:a', kind='entity', label='same sprite'), Mention(ref='new:b', kind='entity', label='same sprite')], utterances=[Utterance(speaker='new:a', text='A', occurrence='one'), Utterance(speaker='new:b', text='B', occurrence='two')])
    r = apply(s, w, f, p)
    rows = w.view(s.snapshot(1))['recent_conversations']
    assert len(rows) == 2 and rows[0]['thread'] != rows[1]['thread']
    assert {r['speaker'] for r in rows} == set(r.ids.values())

def test_repeated_visible_dialogue_dedup_but_new_occurrence_survives(setup):
    s, w = setup
    f = frame(s, 1)
    p = ObservationPatch(frame_id=f.id, summary='text', utterances=[Utterance(text='Hello', occurrence='one')])
    apply(s, w, f, p)
    g = frame(s, 2)
    apply(s, w, g, p.model_copy(update={'frame_id': g.id}))
    assert w.db.execute('SELECT COUNT(*) FROM world_messages').fetchone()[0] == 1
    h = frame(s, 3)
    apply(s, w, h, ObservationPatch(frame_id=h.id, summary='no text'))
    j = frame(s, 4)
    apply(s, w, j, p.model_copy(update={'frame_id': j.id}))
    assert w.db.execute('SELECT COUNT(*) FROM world_messages').fetchone()[0] == 2

def test_schema_reuse_gate_and_versioning(setup):
    s, w = setup
    o = w.ontology
    p = SchemaPatch(parent_version=0, properties=[Property(name='condition', definition='Observed condition')], reason='needed', evidence_ids=[])
    id = o.propose(p)
    with pytest.raises(ContractError):
        o.approve(id, review='No evidence')
    o.approve(id, review='User-approved startup prior', initial_prior=True)
    assert o.current['version'] == 1
    with pytest.raises(ContractError):
        o.propose(p)
    p.parent_version = 1
    with pytest.raises(ContractError):
        o.propose(p)

@pytest.mark.parametrize('key,value', [('visible', 'yes'), ('value', True), ('affordances', [1]), ('role', {}), ('status', 'x' * 4001)])
def test_fact_types(setup, key, value):
    _, w = setup
    with pytest.raises(ContractError):
        w.ontology.validate_value(key, value)

def test_low_confidence_kept_as_evidence_not_current_fact(setup):
    s, w = setup
    f = frame(s, 1)
    r = apply(s, w, f, ObservationPatch(frame_id=f.id, summary='uncertain', mentions=[Mention(ref='new:x', kind='entity', label='x')], facts=[Fact(subject='new:x', key='status', value='maybe', confidence=0.1)]))
    assert not w.view(s.snapshot(1))['entities'][0]['facts']
    assert s.get(r.evidence_id).payload['interpretation']['facts'][0]['value'] == 'maybe'

def test_current_place_requires_place_kind(setup):
    s, w = setup
    f = frame(s, 1)
    with pytest.raises(ContractError):
        apply(s, w, f, ObservationPatch(frame_id=f.id, summary='a', mentions=[Mention(ref='new:a', kind='entity', label='a')], current_place='new:a'))

def test_current_location_relation_is_functional_but_history_kept(setup):
    s, w = setup
    f = frame(s, 1)
    r = apply(s, w, f, ObservationPatch(frame_id=f.id, summary='at A', mentions=[Mention(ref='new:x', kind='entity', label='x'), Mention(ref='new:a', kind='place', label='a'), Mention(ref='new:b', kind='place', label='b')], relations=[Relation(subject='new:x', predicate='located_in', target='new:a', confidence=1)]))
    ids = r.ids
    g = frame(s, 2)
    apply(s, w, g, ObservationPatch(frame_id=g.id, summary='at B', relations=[Relation(subject=ids['new:x'], target=ids['new:b'], predicate='located_in', confidence=1)]), set(ids.values()))
    rs = w.view(s.snapshot(2))['relations']
    assert len(rs) == 1 and rs[0]['target'] == ids['new:b']
    assert w.db.execute('SELECT COUNT(*) FROM world_relations').fetchone()[0] == 2

def test_explicit_new_text_occurrence_is_not_deduplicated(setup):
    s, w = setup
    f = frame(s, 1)
    apply(s, w, f, ObservationPatch(frame_id=f.id, summary='one', utterances=[Utterance(text='Hit!', occurrence='event1')]))
    g = frame(s, 2)
    apply(s, w, g, ObservationPatch(frame_id=g.id, summary='two', utterances=[Utterance(text='Hit!', occurrence='event2')]))
    assert w.db.execute('SELECT COUNT(*) FROM world_messages').fetchone()[0] == 2

def test_retrieval_exposes_persistent_ids_without_inventing_them(setup):
    s, w = setup
    f = frame(s, 1)
    r = apply(s, w, f, ObservationPatch(frame_id=f.id, summary='Met a helper', mentions=[Mention(ref='new:x', kind='entity', label='helper')]))
    hits = s.search(r.ids['new:x'], s.snapshot(1))
    enriched = w.retrieval_record(hits[0], s.snapshot(1))
    assert enriched['entities'][0]['id'] == r.ids['new:x']
    with pytest.raises(ContractError):
        w.retrieval_record(hits[0], s.snapshot(0))

def test_current_place_is_pinned_before_unrelated_new_entities(setup):
    s, w = setup
    f = frame(s, 1)
    r = apply(s, w, f, ObservationPatch(frame_id=f.id, summary='room', mentions=[Mention(ref='new:p', kind='place', label='room')], current_place='new:p'))
    g = frame(s, 2)
    apply(s, w, g, ObservationPatch(frame_id=g.id, summary='thing', mentions=[Mention(ref='new:x', kind='entity', label='thing')]))
    view = w.view(s.snapshot(2), max_entities=1)
    assert view['entities'][0]['id'] == r.ids['new:p']
    assert view['current_place_observed_at'] == 1
    assert view['omitted_entities'] == 1


def test_late_interpretation_enriches_history_without_rewinding_live_state(setup):
    s, w = setup
    first = frame(s, 1)
    r = apply(s, w, first, ObservationPatch(frame_id=first.id, summary='at A', mentions=[
        Mention(ref='new:x', kind='entity', label='x'), Mention(ref='new:a', kind='place', label='A'),
        Mention(ref='new:b', kind='place', label='B')], current_place='new:a'))
    ids = r.ids
    old, new = frame(s, 2), frame(s, 3)
    apply(s, w, new, ObservationPatch(frame_id=new.id, summary='new', facts=[
        Fact(subject=ids['new:x'], key='status', value='new', confidence=1)], relations=[
        Relation(subject=ids['new:x'], predicate='located_in', target=ids['new:b'], confidence=1)],
        current_place=ids['new:b'], utterances=[Utterance(text='New dialogue', occurrence='new')]), set(ids.values()))
    late = apply(s, w, old, ObservationPatch(frame_id=old.id, summary='old', facts=[
        Fact(subject=ids['new:x'], key='status', value='old', confidence=1),
        Fact(subject=ids['new:x'], key='role', value='helper', confidence=1)], relations=[
        Relation(subject=ids['new:x'], predicate='located_in', target=ids['new:a'], confidence=1)],
        current_place=ids['new:a'], needs_planning=True,
        utterances=[Utterance(text='Old dialogue', occurrence='old')]), set(ids.values()))
    view = w.view(s.snapshot(3))
    x = next(e for e in view['entities'] if e['id'] == ids['new:x'])
    assert x['facts']['status']['value'] == 'new'
    assert x['facts']['role']['observed_at'] == 2
    assert view['relations'][0]['target'] == ids['new:b']
    assert view['current_place'] == ids['new:b']
    assert [m['text'] for m in view['recent_conversations']] == ['Old dialogue', 'New dialogue']
    assert not late.needs_planning
    assert s.get(late.evidence_id).payload['observed_frame_id'] == old.id
    assert w.db.execute('SELECT text FROM world_dialogue').fetchone()[0] == 'New dialogue'


def test_late_dialogue_does_not_duplicate_preceding_occurrence(setup):
    s, w = setup
    a, b, c = frame(s, 1), frame(s, 2), frame(s, 3)
    apply(s, w, a, ObservationPatch(frame_id=a.id, summary='text', utterances=[Utterance(text='Hello', occurrence='one')]))
    apply(s, w, c, ObservationPatch(frame_id=c.id, summary='gone'))
    apply(s, w, b, ObservationPatch(frame_id=b.id, summary='same', utterances=[Utterance(text='Hello', occurrence='one')]))
    assert w.db.execute('SELECT COUNT(*) FROM world_messages').fetchone()[0] == 1
    assert w.db.execute('SELECT COUNT(*) FROM world_dialogue').fetchone()[0] == 0
