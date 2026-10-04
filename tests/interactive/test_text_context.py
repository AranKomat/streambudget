import pytest
from PIL import Image, ImageDraw
from streambudget.interactive.text import TextObservation, TemporalTextTracker, TextRegionCache, estimate_vertical_scroll
from streambudget.interactive.context import bounded_context
from streambudget.types import ContractError

def obs(text, box=(0.1, 0.1, 0.8, 0.3), confidence=0.9):
    return TextObservation('surface', box, text, confidence, 'frame')

def test_stationary_text_recognized_once():
    t = TemporalTextTracker()
    assert t.update([obs('Hello')], 0)[0]['kind'] == 'text_appeared'
    assert t.update([obs('Hello')], 0.1) == []
    assert len(t.tracks) == 1

def test_identical_words_on_two_objects_not_merged():
    t = TemporalTextTracker()
    events = t.update([obs('OPEN', (0.1, 0.1, 0.3, 0.2)), obs('OPEN', (0.7, 0.1, 0.9, 0.2))], 0)
    assert len(events) == 2 and len(t.tracks) == 2
    assert t.update([obs('OPEN', (0.1, 0.1, 0.3, 0.2)), obs('OPEN', (0.7, 0.1, 0.9, 0.2))], 0.1) == []

def test_typewriter_revision():
    t = TemporalTextTracker()
    t.update([obs('Hel')], 0)
    events = t.update([obs('Hello')], 0.1)
    assert events[0]['kind'] == 'text_revised' and len(t.tracks) == 1

def test_same_words_after_disappearance_are_new_event():
    t = TemporalTextTracker(max_gap=1)
    a = t.update([obs('Hello')], 0)[0]['track']
    t.update([], 2)
    b = t.update([obs('Hello')], 3)[0]['track']
    assert a != b

def test_scroll_preserves_track():
    t = TemporalTextTracker()
    a = t.update([obs('A', (0.1, 0.5, 0.8, 0.7))], 0)[0]['track']
    events = t.update([obs('A', (0.1, 0.4, 0.8, 0.6))], 0.1, scroll={'surface': (0, -0.1)})
    assert events == [] and a in t.tracks

def test_low_confidence_text_does_not_replace_better_hypothesis():
    t = TemporalTextTracker()
    t.update([obs('HELLO', confidence=0.99)], 0)
    t.update([obs('HELL0', confidence=0.7)], 0.1)
    assert next(iter(t.tracks.values())).text == 'HELLO'

def test_out_of_order_text_rejected():
    t = TemporalTextTracker()
    t.update([], 1)
    with pytest.raises(ValueError):
        t.update([], 0)

def test_cache_refresh_and_change():
    c = TextRegionCache(max_age=2)
    im = Image.new('RGB', (100, 30), 'white')
    assert c.needs_read('a', im, 0)
    c.record('a', im, 0)
    assert not c.needs_read('a', im, 0.1)
    assert c.needs_read('a', im, 2)
    assert c.needs_read('a', Image.new('RGB', (100, 30), 'black'), 0.2)

def test_scroll_estimate_and_texture_abstention():
    a = Image.new('L', (100, 100), 255)
    d = ImageDraw.Draw(a)
    d.rectangle((5, 40, 70, 46), fill=0)
    d.rectangle((20, 60, 85, 66), fill=0)
    b = Image.new('L', a.size, 255)
    b.paste(a, (0, -10))
    shift, residual = estimate_vertical_scroll(a, b, 15)
    assert shift == -10 and residual == 0
    assert estimate_vertical_scroll(Image.new('L', (20, 20)), Image.new('L', (20, 20)))[0] is None

def test_context_budget_explicit_omissions():
    c = bounded_context(goal='goal', intent='intent', schema={}, actions=[], world={'entities': [{'id': 'a', 'fact': 'x' * 1000} for _ in range(10)]}, recent=[], current_frame_id='f', max_chars=1500)
    assert c['omissions']['entities'] > 0 and c['goal'] == 'goal'

def test_essential_context_not_silently_cut():
    with pytest.raises(ContractError):
        bounded_context(goal='x' * 5000, intent='i', schema={}, actions=[], world={}, recent=[], current_frame_id='f', max_chars=1000)


def test_ocr_crop_cache_reuses_read_without_merging_two_regions(tmp_path):
    from types import SimpleNamespace
    import numpy as np
    import yaml
    from streambudget.interactive.text import RapidTextReader
    params = {}
    for role in ('Det', 'Rec', 'Cls'):
        path = tmp_path / (role + '.onnx')
        path.write_bytes(b'test-not-real-weights')
        params[role + '.model_path'] = str(path)
    cfg = tmp_path / 'ocr.yaml'
    cfg.write_text(yaml.safe_dump(params))
    class Engine:
        def __call__(self, pixels, *, use_det, use_rec, use_cls):
            if use_det:
                return SimpleNamespace(boxes=np.array([[[0, 0], [10, 0], [10, 10], [0, 10]],
                    [[20, 0], [30, 0], [30, 10], [20, 10]]]))
            return SimpleNamespace(txts=['OPEN'], scores=[0.99])
    reader = RapidTextReader(cfg, scale=1, engine=Engine())
    observations = reader.read(Image.new('RGB', (40, 20), 'white'), 'f')
    assert len(observations) == 2 and observations[0].box != observations[1].box
    assert reader.stats['recognizer_calls'] == 1 and reader.stats['recognition_cache_hits'] == 1
    tracker = TemporalTextTracker()
    assert len(tracker.update(observations, 0)) == 2
    assert len(tracker.tracks) == 2


def test_ocr_requires_preprovisioned_weights(tmp_path):
    from streambudget.interactive.text import RapidTextReader
    cfg = tmp_path / 'ocr.yaml'
    cfg.write_text('{}')
    with pytest.raises(ContractError, match='explicit local ONNX'):
        RapidTextReader(cfg)


def test_hot_text_context_is_bounded_with_explicit_omissions():
    c = bounded_context(goal='g', intent='i', schema={}, actions=[], world={}, recent=[],
        current_frame_id='f', max_chars=1000,
        hot_text={'source_frame': 'old', 'tracks': [{'text': 'x'*500} for _ in range(8)]})
    assert c['hot_text']['source_frame'] == 'old'
    assert c['omissions']['text_tracks'] > 0
