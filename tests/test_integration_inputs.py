"""Offline input boundaries: remove storage metadata without summarizing facts."""
import copy
import json
import socket

import pytest

from pipeline.event_integration import chapter_input, fragments_for, integration_prompt, repair_integration_result
from pipeline.integration_material import compact_json, model_material, FRAGMENT_COLUMNS
from pipeline.story_bible import digest
from pipeline.story_model import StoryModel, ResponseTruncated
from pipeline.event_integration import integrate_chapter
from test_event_integration import analysis, merged_response
from test_engagement_flow import story


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*a, **k):
        raise AssertionError('Input checks must never call an API')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)
    import config
    monkeypatch.setattr(config, 'STORY_INTEGRATION_MAX_OUTPUT_TOKENS', '')


def table(material):
    return [dict(zip(material['fragment_columns'], r)) for r in material['fragments']]


def test_storage_metadata_and_identity_attempt_history_stay_local(analysis):
    analysis['panels'][0].update(original_path='LOCAL_PATH_ONLY', snapshot_path='LOCAL_SNAPSHOT_ONLY',
        source_page='LOCAL_SOURCE_PAGE_ONLY', sha256='LOCAL_HASH_ONLY', bbox={'x': 831987},
        derivation={'retained_rows': [[1234567, 2345678]], 'original_sha256': 'LOCAL_DERIVATION_ONLY'},
        extracted_text='UNVERIFIED_OCR_ONLY')
    analysis['identity_review'] = {'status': 'needs_review', 'error': 'LOCAL_REVIEW_STACK_ONLY'}
    analysis['boundary_review'] = {'panels': [], 'debug': 'LOCAL_BOUNDARY_DEBUG_ONLY'}
    analysis['characters'] = [{'character_id': 'C1', 'name': None, 'name_status': 'unknown',
        'description': 'CURRENT_RECOGNITION', 'source_panel_ids': ['1'],
        'identity_evidence': [{'detail': 'LOCAL_IDENTITY_ATTEMPTS_ONLY'}],
        'pending_names': [{'name': 'PENDING_NAME', 'status': 'unknown', 'source_panel_id': '1'}],
        'name_claims': [], 'label_claims': [{'label': 'CURRENT_LABEL', 'reveal_panel_id': '1', 'reveal_order': 0}]}]
    before = copy.deepcopy(analysis)
    prompt = integration_prompt(analysis, fragments_for(analysis))
    for token in ('LOCAL_PATH_ONLY', 'LOCAL_SNAPSHOT_ONLY', 'LOCAL_SOURCE_PAGE_ONLY', 'LOCAL_HASH_ONLY',
                  'LOCAL_DERIVATION_ONLY', '1234567', '831987', 'LOCAL_IDENTITY_ATTEMPTS_ONLY', 'UNVERIFIED_OCR_ONLY',
                  'LOCAL_REVIEW_STACK_ONLY', 'LOCAL_BOUNDARY_DEBUG_ONLY'):
        assert token not in prompt
    assert all(token in prompt for token in ('CURRENT_RECOGNITION', 'PENDING_NAME', 'CURRENT_LABEL', 'unknown', 'reveal_order'))
    assert analysis == before


def test_all_semantic_fragments_are_lossless_ordered_and_keep_exact_sources(analysis):
    analysis['panels'][0]['world_rules'] = ['ONE_PERSON_LIMIT']
    analysis['panels'][1]['environment_anomalies'] = ['REVERSE_FLOW']
    analysis['panels'][2]['identity_clues'] = [{'detail': 'SPEAKER_UNKNOWN', 'needs_review': True}]
    analysis['identity_review'] = {'status': 'needs_review', 'unresolved': [{'detail': 'RELATION_UNKNOWN'}]}
    analysis['relationships'] = [{'relation': 'RELATION_AS_RECORDED', 'source_panel_id': '3', 'reveal_order': 2}]
    fragments = fragments_for(analysis)
    material = model_material(analysis, fragments)
    cells = table(material)
    assert [r['id'] for r in cells] == [f['fragment_id'] for f in fragments]
    for cell, fragment in zip(cells, fragments):
        assert cell['source'] == (fragment['source_event_id'] or fragment['source_panel_ids'][0])
        assert cell['status'] == fragment['status']
        if cell['kind'] == 'event':
            restored = {**cell['content'], 'event_id': cell['source']}
            assert restored == fragment['value']
        else:
            assert cell['content'] == fragment['value']
    assert material['identity_review'] == analysis['identity_review']
    assert material['relationships'] == analysis['relationships']
    assert material['panels'][2]['identity_clues'] == analysis['panels'][2]['identity_clues']
    assert 'ONE_PERSON_LIMIT' in compact_json(material) and 'REVERSE_FLOW' in compact_json(material)


def test_exact_shared_dialogue_occurs_once_but_all_event_links_remain(analysis):
    line = {'speaker_id': None, 'text': 'SHARED_LITERAL_DIALOGUE'}
    analysis['panels'][0]['dialogue'] = [line]
    for e in analysis['events'][:2]:
        e['source_panel_ids'] = ['1']
        e['dialogue'] = [copy.deepcopy(line)]
    before = copy.deepcopy(analysis)
    cells = table(model_material(analysis, fragments_for(analysis)))
    assert compact_json(cells).count('SHARED_LITERAL_DIALOGUE') == 1
    assert all('dialogue' not in r['content'] for r in cells if r['kind'] == 'event' and r['source'] in {'E1', 'E2'})
    assert all(r['content']['source_panel_ids'] == ['1'] for r in cells if r['kind'] == 'event' and r['source'] in {'E1', 'E2'})
    assert analysis == before


@pytest.mark.parametrize('difference', ['speaker', 'text', 'source'])
def test_dialogue_disagreement_is_retained_without_keyword_guessing(analysis, difference):
    line = {'speaker_id': None, 'text': 'EXACT_SAME_WORDS'}
    analysis['panels'][0]['dialogue'] = [line]
    event = analysis['events'][0]
    event['dialogue'] = [copy.deepcopy(line)]
    if difference == 'speaker':
        event['dialogue'][0]['speaker_id'] = 'C9'
    elif difference == 'text':
        event['dialogue'][0]['text'] = 'DISTINCT_ORIGINAL_WORDS'
    else:
        event['source_panel_ids'] = ['2']
    cells = table(model_material(analysis, fragments_for(analysis)))
    stored = next(r['content'] for r in cells if r['kind'] == 'event' and r['source'] == 'E1')
    assert stored['dialogue'] == event['dialogue']
    assert next(r['content'] for r in cells if r['kind'] == 'dialogue' and r['source'] == '1') == line


def test_same_text_on_different_panels_keeps_separate_fragment_ids(analysis):
    line = {'speaker_id': None, 'text': 'IDENTICAL_TEXT_DISTINCT_SOURCES'}
    for p in analysis['panels'][:2]:
        p['dialogue'] = [copy.deepcopy(line)]
    cells = table(model_material(analysis, fragments_for(analysis)))
    lines = [r for r in cells if r['kind'] == 'dialogue']
    assert [r['source'] for r in lines] == ['1', '2']
    assert len({r['id'] for r in lines}) == 2 and [r['content'] for r in lines] == [line, line]


def test_repair_uses_same_lean_view_and_includes_only_exact_dialogue_dependencies(analysis):
    analysis['events'][0]['dialogue'] = copy.deepcopy(analysis['panels'][0]['dialogue'])
    fragments = fragments_for(analysis)
    eid = next(f['fragment_id'] for f in fragments if f['source_event_id'] == 'E1')
    cells = table(model_material(analysis, fragments, {eid}))
    assert {r['kind'] for r in cells} == {'event', 'dialogue'}
    assert all(r['source'] in {'E1', '1'} for r in cells)
    assert next(r['content'] for r in cells if r['kind'] == 'dialogue') == analysis['panels'][0]['dialogue'][0]
    assert 'dialogue' not in next(r['content'] for r in cells if r['kind'] == 'event')


def test_actual_repair_prompt_has_no_raw_paths_or_long_source_index(analysis):
    analysis['panels'][0]['derivation'] = {'retained_rows': 'REPAIR_BACKEND_ONLY'}
    fragments = fragments_for(analysis)
    correct = merged_response(analysis)
    broken = copy.deepcopy(correct)
    broken['events'][0]['summary'] = ''
    calls = []
    class Repair:
        def ask(self, stage, prompt, images=None, fresh=False):
            calls.append(prompt)
            payload = json.JSONDecoder().raw_decode(prompt.split('资料：', 1)[1])[0]
            assert payload['fragment_columns'] == FRAGMENT_COLUMNS
            assert 'REPAIR_BACKEND_ONLY' not in prompt and '"material"' not in prompt
            assert payload['issues'][0]['path'] == '/events/0/summary'
            return {'patches': [{'issue_id': payload['issues'][0]['issue_id'],
                                 'value': correct['events'][0]['summary']}], 'unresolved': []}
    repair_integration_result(Repair(), broken, analysis, fragments)
    assert len(calls) == 1


def test_backend_hash_remains_a_cache_dependency_even_though_prompt_omits_it(analysis):
    changed = copy.deepcopy(analysis)
    changed['panels'][0]['sha256'] = 'CHANGED_LOCAL_EVIDENCE_HASH'
    assert integration_prompt(analysis, fragments_for(analysis)) == integration_prompt(changed, fragments_for(changed))
    assert digest(analysis) != digest(changed)


def test_input_audit_only_reads_the_source_project(story, monkeypatch):
    from tools.audit_integration_inputs import audit
    from pipeline.story_model import StoryModel
    store, state, _, _ = story
    before = {str(p.relative_to(store.root)): p.read_bytes() for p in store.root.rglob('*') if p.is_file()}
    def forbidden(*a, **k):
        raise AssertionError('Input auditing must not create a model')
    monkeypatch.setattr(StoryModel, '__init__', forbidden)
    result = audit(store.root)
    assert result['report']['fragments'] == 10 and result['report']['events'] == 5
    assert result['report']['model_calls'] == result['report']['project_writes'] == 0
    assert result['report']['new_prompt_tokens'] is None
    assert {str(p.relative_to(store.root)): p.read_bytes() for p in store.root.rglob('*') if p.is_file()} == before


def test_audit_compares_only_the_identical_source_version(story, tmp_path):
    from tools.audit_integration_inputs import audit
    store, state, _, _ = story
    a = chapter_input(state['bible'], state['revision'])
    trace = tmp_path/'baseline.json'
    record = {'request': {'stage': 'event_integration', 'prompt': '资料：'+json.dumps({'analysis': a}, ensure_ascii=False)},
              'response_metadata': {'usage': {'prompt_tokens': 1234}}}
    trace.write_text(json.dumps(record), encoding='utf-8')
    result = audit(store.root, trace)
    assert result['report']['baseline_provider_reported_usage'] == {'prompt_tokens': 1234}
    assert result['report']['new_prompt_tokens'] is None
    record['request']['prompt'] = '资料：'+json.dumps({'analysis': {**a, 'source_revision': 'OTHER_VERSION'}})
    trace.write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(ValueError, match='exact same'):
        audit(store.root, trace)


def test_provider_truncation_does_not_retry_or_save_a_partial_integration(story):
    store, state, _, _ = story
    class Transport:
        kind = 'test'
        host = 'offline.invalid'
        last_response_metadata = {'finish_reason': 'length', 'usage': {'completion_tokens': 5}}
        def __init__(self):
            self.calls = 0
        def chat(self, **request):
            assert request['max_tokens'] is None
            self.calls += 1
            return '{"events":[]}'
    transport = Transport()
    with pytest.raises(ResponseTruncated, match='截断'):
        integrate_chapter(store, StoryModel(store, transport), state['bible'], state['revision'])
    assert transport.calls == 1 and store.read() == state
    assert not list((store.root/'cache').glob('valid-*.json'))
    trace = json.loads(next((store.root/'calls').glob('*.json')).read_text(encoding='utf-8'))
    assert trace['response'] == '{"events":[]}' and trace['response_metadata']['finish_reason'] == 'length'


def test_provider_truncation_in_targeted_repair_stops_without_identical_retry(analysis):
    fragments = fragments_for(analysis)
    broken = merged_response(analysis)
    broken['events'][0]['summary'] = ''
    calls = []
    class Repair:
        def ask(self, stage, prompt, images=None, fresh=False):
            calls.append(stage)
            raise ResponseTruncated('截断')
    with pytest.raises(ResponseTruncated):
        repair_integration_result(Repair(), broken, analysis, fragments)
    assert calls == ['event_integration']
