"""Synthetic regressions: no provider credentials, media capture, or user data."""
import asyncio
import copy
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pipeline import store
from pipeline.prompts import prune_knowledge
from pipeline.providers import parse_json_response, filter_low_confidence
import app


@pytest.mark.parametrize('text', [None, '', 'not json', '[1,2]'])
def test_invalid_model_output_is_explicit_error(text):
    assert 'error' in parse_json_response(text)


@pytest.mark.parametrize('text', ['```json\n{"label":"calm"}\n```', 'Answer: {"label":"calm"} End'])
def test_wrapped_json_is_accepted(text):
    assert parse_json_response(text) == {'label': 'calm'}


def test_confidence_filter_keeps_boundary_and_drops_invalid():
    result = filter_low_confidence({'candidates': [
        {'label': 'keep', 'confidence': .5},
        {'label': 'low', 'confidence': .49},
        {'label': 'invalid', 'confidence': 'unknown'},
    ]})
    assert [c['label'] for c in result['candidates']] == ['keep']


def test_no_supported_candidate_marks_insufficient():
    result = filter_low_confidence({'candidates': [{'confidence': .1}]})
    assert result['candidates'] == []
    assert result['insufficient_evidence'] is True


@pytest.mark.parametrize('modality,removed', [
    ('image', {'acoustic', 'temporal_signals'}),
    ('audio', {'signals', 'temporal_signals'}),
    ('text', {'temporal_signals'}), ('video', set()), ('unknown', set()),
])
def test_pruning_preserves_shared_knowledge_and_constraints(modality, removed):
    source = {k: {'example': True} for k in (
        'meta', 'output_constraints', 'conflict_rules', 'context_modifiers',
        'behavior_patterns', 'signals', 'acoustic', 'temporal_signals')}
    source['pain_indicators'] = {'hard_boundary': 'not diagnosis', 'facial_units': ['synthetic']}
    before = copy.deepcopy(source)
    result = prune_knowledge(source, modality)
    assert source == before
    assert removed.isdisjoint(result)
    assert result['output_constraints'] == source['output_constraints']
    if modality == 'audio':
        assert result['pain_indicators'] == {'hard_boundary': 'not diagnosis'}


def test_history_filters_device_and_empty_ids(tmp_path):
    db = str(tmp_path / 'test.sqlite')
    store.init_db(db)
    ids = {}
    for client in ('a', 'b', ''):
        ids[client] = store.insert_record(
            species='dog', mode='text', context='synthetic', media_path='',
            result={}, predicted_label=None, confidence=None, tokens_in=0,
            tokens_out=0, latency_s=0, provider='mock', model='mock',
            client_id=client, db_path=db)
    assert [r['id'] for r in store.list_records('a', db_path=db)] == [ids['a']]
    assert store.list_records('', db_path=db) == []
    assert store.list_records('missing', db_path=db) == []
    with pytest.raises(ValueError):
        store.insert_feedback(999, 'synthetic', db_path=db)


@pytest.mark.parametrize('failure', [None, 'model', 'exception', 'database'])
def test_background_job_finishes_and_cleans_media_on_every_exit(tmp_path, monkeypatch, failure):
    media = tmp_path / 'synthetic.mp4'
    audio = tmp_path / 'synthetic_16k.wav'
    media.write_bytes(b'synthetic fixture; not playable')
    audio.write_bytes(b'synthetic fixture; not playable')
    def analyze(*args, **kwargs):
        if failure == 'exception':
            raise RuntimeError('synthetic failure')
        return {'result': {'error': 'synthetic failure'} if failure == 'model' else {'candidates': []}}
    def insert(**kwargs):
        if failure == 'database':
            raise RuntimeError('synthetic database failure')
        return 42
    monkeypatch.setattr(app.analyze_mod, 'analyze', analyze)
    monkeypatch.setattr(app.store, 'insert_record', insert)
    job = {'events': [], 'done': False, 'finished_at': None}
    asyncio.run(app._run_analysis_job(job, media_path=str(media), size=10,
        species='dog', context='', mode='native', provider=SimpleNamespace(),
        knowledge={}, max_frames=2, text='', client_id='synthetic'))
    assert job['done'] and job['finished_at']
    assert job['events'][-1][0] == ('error' if failure else 'done')
    assert not media.exists() and not audio.exists()
    if not failure:
        assert job['events'][-1][1]['record_id'] == 42


@pytest.mark.parametrize('expected,key,status', [('', '', 503), ('test-only-password', 'wrong', 403)])
def test_export_is_closed_without_correct_secret(monkeypatch, expected, key, status):
    monkeypatch.setenv('EXPORT_KEY', expected)
    with pytest.raises(HTTPException) as exc:
        app.export_csv(key=key)
    assert exc.value.status_code == status
