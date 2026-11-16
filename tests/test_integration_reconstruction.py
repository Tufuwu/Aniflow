"""Hand-authored reconstruction through real requests/validators, never a model eval."""
import copy
import json
from pathlib import Path
import socket

import pytest

import pipeline.event_integration as integration
import pipeline.integration_repair as repair
import pipeline.story_model as story_model
from pipeline.event_integration import (chapter_input, fragments_for, consume, integrate_chapter,
                                        integration_prompt, repair_integration_result, resolve_fragment)
from pipeline.story_bible import StoryBible, StoryError
from pipeline.story_model import StoryModel


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('Reconstruction acceptance must never access a network')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)


@pytest.fixture
def case():
    summaries = ['孩子推门未开', '有人说“太重了”，壮汉转头', '壮汉手抵门板，孩子在他身后',
                 '门已打开，孩子松口气', '回忆里去年这扇门曾夹伤人', '进入屋内寻找药盒', '平台片尾标志']
    panels = [{'source_panel_id': f'P{i}', 'order': i - 1, 'dialogue': [],
               'visible_text': [], 'uncertainties': [], 'scene': ''} for i in range(1, 8)]
    panels[0]['scene'] = '门口；左下角水印；人物衣服印有ORIGINAL字样'
    panels[1]['dialogue'] = [{'speaker_id': None, 'text': '太重了，推不动。', 'status': 'unknown'}]
    panels[1]['uncertainties'] = ['单格无法确认喊话者，无法确认壮汉是否准备帮忙']
    panels[3]['dialogue'] = [{'speaker_id': 'C1', 'text': '听说里面藏着宝物。'}]
    panels[5]['dialogue'] = [{'speaker_id': 'C1', 'text': '我来照明，你找药。'}]
    panels[6]['panel_type'] = 'non_story'
    events = [{'event_id': f'E{i}', 'source_panel_ids': [f'P{i}'], 'order': i - 1,
               'summary': summary, 'character_ids': ['C1', 'C2'] if i < 7 else [],
               'actions': [summary], 'evidence': [], 'causes': [], 'status': 'supported'}
              for i, summary in enumerate(summaries, 1)]
    events[1]['status'] = 'uncertain'
    events[-1]['narrative_eligible'] = False
    bible = {'panels': panels, 'events': events, 'characters': [], 'relationships': []}
    analysis = chapter_input(bible, 'fixture-source')
    fragments = fragments_for(analysis)

    def ids(*pids):
        return [f['fragment_id'] for f in fragments if set(f['source_panel_ids']) <= set(pids)]

    process = ids('P1', 'P2', 'P3', 'P4')
    summary = '孩子推不开沉重的门，抱怨太重；壮汉听见后过来帮忙，把门推开，孩子松了口气，又说听说里面藏着宝物。'
    facets = [
        {'role': 'obstacle', 'text': '门太重，孩子推不开。', 'fragment_ids': ids('P1', 'P2'), 'status': 'uncertain'},
        {'role': 'action', 'text': '壮汉听到孩子抱怨后过来帮忙推门。', 'fragment_ids': process, 'status': 'uncertain'},
        {'role': 'result', 'text': '门打开，孩子松了口气。', 'fragment_ids': ids('P4'), 'status': 'supported'},
        {'role': 'reaction', 'text': '孩子说听说里面藏着宝物。', 'fragment_ids': ids('P4'), 'status': 'supported'},
    ]
    rows = [{'summary': summary, 'fragment_ids': process, 'kind': 'current_action',
             'time': {'original_time': '本话当前', 'status': 'supported'}, 'facets': facets}]
    for pid, kind, when, role, text in [
        ('P5', 'recollection', '去年', 'premise', '回忆中，这扇门去年曾夹伤人。'),
        ('P6', 'current_action', '本话当前', 'action', '两人进入屋内找药盒，孩子负责照明，壮汉寻找。'),
        ('P7', 'non_story', '不适用', 'setting', '平台片尾标志，非剧情。'),
    ]:
        rows.append({'summary': text, 'fragment_ids': ids(pid), 'kind': kind,
                     'time': {'original_time': when, 'status': 'supported'},
                     'facets': [{'role': role, 'text': text, 'fragment_ids': ids(pid),
                                 'status': 'uncertain' if pid == 'P6' else 'supported'}]})
    return bible, analysis, fragments, {'events': rows}


class Transport:
    kind, host = 'test', 'offline.invalid'

    def __init__(self, response):
        self.response, self.calls = response, []

    def chat(self, **request):
        self.calls.append(copy.deepcopy(request))
        return json.dumps(self.response, ensure_ascii=False)


def payload(prompt):
    return json.JSONDecoder().raw_decode(prompt.split('资料：', 1)[1])[0]


def test_context_interpretation_merges_process_through_transport_without_rewriting_sources(tmp_path, case):
    bible, analysis, fragments, raw = case
    before = copy.deepcopy(bible)
    store = StoryBible(tmp_path)
    transport = Transport(raw)
    model = StoryModel(store, transport)
    result = integrate_chapter(store, model, bible, 'fixture-source')
    assert len(transport.calls) == 1
    request = transport.calls[0]
    assert request['images'] == [] and request['system'] == story_model.INTEGRATION_SYSTEM
    assert request['prompt'] == integration_prompt(analysis, fragments)
    material = payload(request['prompt'])
    assert [r[0] for r in material['fragments']] == [f['fragment_id'] for f in fragments]
    assert [r[3] for r in material['fragments']] == [f['status'] for f in fragments]
    assert '单格无法确认' in request['prompt']  # Raw observation remains available.
    assert result['analysis_events'] == before['events'] and bible == before
    events = result['events']
    assert len(events) == 4 < len(bible['events'])
    assert events[0]['source_event_ids'] == ['E1', 'E2', 'E3', 'E4']
    assert events[0]['summary'] == raw['events'][0]['summary']
    assert events[0]['facets'] == raw['events'][0]['facets']
    assert events[0]['status'] == 'uncertain'
    assert events[0]['dialogue'][0]['speaker_id'] is None  # Interpretation doesn't rewrite attribution.
    assert events[0]['dialogue'][0]['status'] == 'unknown'
    assert '听说' in events[0]['facets'][-1]['text']
    # Coverage is exact even though watermark, clothing and old doubts aren't narrated.
    assert {f for e in events for f in e['fragment_ids']} == {f['fragment_id'] for f in fragments}
    prose = json.dumps([{'summary': e['summary'], 'facets': e['facets']} for e in events], ensure_ascii=False)
    assert all(word not in prose for word in ('水印', 'ORIGINAL', '单格无法确认', '无法确认壮汉'))
    assert events[1]['kind'] == 'recollection' and events[1]['time']['original_time'] == '去年'
    assert events[1]['reveal_panel_id'] == 'P5' and events[2]['source_panel_ids'] == ['P6']
    assert '照明' in events[2]['summary']  # Supporting character action survives.
    assert events[-1]['narrative_eligible'] is False
    index = result['event_integration']['fragment_index']
    assert [f['status'] for f in index] == [f['status'] for f in fragments]
    for f in index:
        assert resolve_fragment(result, f) == resolve_fragment(bible, f)
    assert not result['event_integration']['field_repairs']
    assert not result['event_integration']['local_repairs']


def test_event_coverage_does_not_require_a_facet_per_annotation(case):
    _, analysis, fragments, raw = case
    # Summary still carries the complete process; facets needn't repeat every source/detail.
    raw['events'][0]['facets'] = raw['events'][0]['facets'][2:]
    result = consume(raw, analysis, fragments)
    cited = {fid for facet in result[0]['facets'] for fid in facet['fragment_ids']}
    assert cited < set(result[0]['fragment_ids'])
    assert '过来帮忙' in result[0]['summary'] and result[0]['status'] == 'uncertain'


def test_partial_event_references_pass_transport_and_keep_complete_original_index(tmp_path, case):
    bible, _, fragments, raw = case
    raw['events'].pop(0)
    before = copy.deepcopy(bible)
    store, transport = StoryBible(tmp_path), Transport(raw)
    model = StoryModel(store, transport)
    result = integrate_chapter(store, model, bible, 'fixture-source')
    assert len(transport.calls) == 1 and len(result['events']) == 3
    cited = {fid for e in result['events'] for fid in e['fragment_ids']}
    assert cited < {f['fragment_id'] for f in fragments}
    material = payload(transport.calls[0]['prompt'])
    assert [r[0] for r in material['fragments']] == [f['fragment_id'] for f in fragments]
    assert result['analysis_events'] == before['events'] and bible == before
    for fragment in result['event_integration']['fragment_index']:
        assert resolve_fragment(result, fragment) == resolve_fragment(bible, fragment)
    assert [f['status'] for f in result['event_integration']['fragment_index']] == [f['status'] for f in fragments]
    assert result['event_integration']['local_repairs'] == result['event_integration']['field_repairs'] == []
    assert integrate_chapter(store, model, result, 'fixture-source') == result and len(transport.calls) == 1


@pytest.mark.parametrize('mark', ['cover', 'blank', 'advertisement', 'boundary_review', 'event_flag'])
def test_explicit_non_story_can_be_omitted_and_all_original_fragments_remain_traceable(tmp_path, case, mark):
    bible, _, _, raw = case
    bible['panels'][-1].pop('panel_type')
    bible['events'][-1].pop('narrative_eligible')
    if mark == 'boundary_review':
        bible['boundary_review'] = {'panels': [{'source_panel_id': 'P7', 'classification': 'non_story'}]}
    elif mark == 'event_flag':
        bible['events'][-1]['narrative_eligible'] = False
    else:
        bible['panels'][-1]['panel_type'] = mark
    raw['events'].pop()
    before = copy.deepcopy(bible)
    transport = Transport(raw)
    store = StoryBible(tmp_path)
    result = integrate_chapter(store, StoryModel(store, transport), bible, 'fixture-source')
    assert len(transport.calls) == 1
    assert len(result['events']) == 3 and all(e['narrative_eligible'] for e in result['events'])
    assert result['analysis_events'] == before['events'] and bible == before
    omitted = [f for f in result['event_integration']['fragment_index'] if f['source_panel_ids'] == ['P7']]
    assert omitted and all(resolve_fragment(result, f) == resolve_fragment(bible, f) for f in omitted)
    material = payload(transport.calls[0]['prompt'])
    assert set(f['fragment_id'] for f in omitted) <= {r[0] for r in material['fragments']}
    assert result['event_integration']['local_repairs'] == result['event_integration']['field_repairs'] == []


@pytest.mark.parametrize('content', ['版权说明', '未完待续', '空白页', '水印', '无法确认'])
def test_uncited_fragments_are_not_classified_or_deleted_by_content_keywords(case, content):
    bible, _, _, raw = case
    bible['panels'][-1]['panel_type'] = 'story'
    bible['panels'][-1]['scene'] = content
    bible['events'][-1].pop('narrative_eligible')
    bible['events'][-1]['summary'] = content
    analysis = chapter_input(bible, 'fixture-source')
    fragments = fragments_for(analysis)
    raw['events'].pop()
    assert len(consume(raw, analysis, fragments)) == len(raw['events'])
    assert analysis['panels'][-1]['scene'] == content
    assert analysis['events'][-1]['summary'] == content


def test_uncited_mixed_source_fragment_stays_in_original_analysis(case):
    bible, _, _, _ = case
    # Omitting a reference doesn't reclassify or remove the original fragment.
    bible['events'][-1]['source_panel_ids'] = ['P6', 'P7']
    bible['events'][-1].pop('narrative_eligible')
    analysis = chapter_input(bible, 'fixture-source')
    fragments = fragments_for(analysis)
    from integration_fixture import integration_response
    raw = integration_response(integration_prompt(analysis, fragments))
    mixed = next(f['fragment_id'] for f in fragments if f['source_event_id'] == 'E7')
    raw['events'] = [e for e in raw['events'] if mixed not in e['fragment_ids']]
    assert len(consume(raw, analysis, fragments)) == len(raw['events'])
    assert next(f for f in fragments if f['fragment_id'] == mixed)['source_panel_ids'] == ['P6', 'P7']


@pytest.mark.parametrize('path', ['summary', 'facet_text', 'facet_role'])
def test_contextual_field_repairs_preserve_explanations_siblings_and_exact_scope(case, path):
    _, analysis, fragments, raw = case
    original = copy.deepcopy(raw)
    if path == 'summary':
        raw['events'][0]['summary'] = ''
        expected_path, value = '/events/0/summary', original['events'][0]['summary']
    else:
        key = 'text' if path == 'facet_text' else 'role'
        raw['events'][0]['facets'][1][key] = ''
        expected_path, value = '/events/0/facets/1/' + key, original['events'][0]['facets'][1][key]
    broken, journal, requests = copy.deepcopy(raw), [], []

    class Model:
        def ask(self, stage, prompt, images=None, fresh=False):
            assert stage == 'event_integration' and not images and fresh
            requests.append(prompt)
            data = payload(prompt)
            assert len(data['issues']) == 1
            issue = data['issues'][0]
            assert issue['path'] == expected_path
            assert set(issue['fragment_ids']) == set(original['events'][0]['fragment_ids'])
            assert {r[0] for r in data['fragments']} == set(issue['fragment_ids'])
            assert not {'events', 'analysis', 'material'} & data.keys()
            assert '去年这扇门' not in prompt and '我来照明' not in prompt
            assert '无法确认壮汉是否准备帮忙' in prompt
            return {'patches': [{'issue_id': issue['issue_id'], 'value': value}], 'unresolved': []}

    result = repair_integration_result(Model(), raw, analysis, fragments, field_repairs=journal)
    assert result == consume(original, analysis, fragments) and raw == broken
    assert len(journal) == len(requests) == 1 and journal[0]['path'] == expected_path
    assert result[0]['facets'][1]['text'] == '壮汉听到孩子抱怨后过来帮忙推门。'


@pytest.mark.parametrize('attack', ['invented_id', 'foreign_facet', 'reveal_order',
                                   'non_story_merge', 'promoted', 'invented_cause'])
def test_reconstruction_does_not_weaken_existing_validation(case, attack):
    _, analysis, fragments, raw = case
    if attack == 'invented_id': raw['events'][0]['fragment_ids'].append('F99999')
    elif attack == 'foreign_facet': raw['events'][0]['facets'][0]['fragment_ids'] = raw['events'][1]['fragment_ids']
    elif attack == 'reveal_order': raw['events'][0:2] = reversed(raw['events'][0:2])
    elif attack == 'non_story_merge':
        raw['events'][-2]['fragment_ids'] += raw['events'].pop()['fragment_ids']
    elif attack == 'promoted': raw['events'][0]['facets'][1]['status'] = 'supported'
    else: raw['events'][0]['causes'] = [{'event_id': 'E7'}]
    with pytest.raises(StoryError):
        consume(raw, analysis, fragments)


def test_v3_and_v4_requests_checkpoints_traces_and_reviewed_revisions_are_isolated(tmp_path, monkeypatch, case):
    bible, _, _, raw = case
    legacy = json.loads((Path(__file__).parent / 'fixtures/integration_v3_contract.json').read_text(encoding='utf-8'))
    store, transport = StoryBible(tmp_path), Transport(raw)
    model = StoryModel(store, transport)
    # Run the exact previous prompt/system/contract through the same transport and cache paths.
    with monkeypatch.context() as old:
        old.setattr(integration, 'VERSION', legacy['version'])
        old.setattr(integration, 'INSTRUCTIONS', legacy['instructions'])
        old.setattr(integration, 'OUTPUT_CONTRACT', legacy['output_contract'])
        old.setattr(repair, 'REPAIR_VERSION', legacy['repair_version'])
        old.setattr(repair, 'REPAIR_INSTRUCTIONS', legacy['repair_instructions'])
        old.setattr(story_model, 'INTEGRATION_SYSTEM', legacy['system'])
        previous = integrate_chapter(store, model, bible, 'fixture-source')
    saved = store.save({'bible': previous, 'script': {'segments': [{'text': '人工审核原文', 'review_status': 'approved'}]}}, 'fixture_v3')
    old_files = {p: p.read_bytes() for p in store.root.rglob('*') if p.is_file()}
    current = integrate_chapter(store, model, previous, 'fixture-source')
    assert len(transport.calls) == 2
    assert previous['event_integration']['version'] == 3 and current['event_integration']['version'] == integration.VERSION
    assert previous['event_integration']['repair_version'] == 1 and current['event_integration']['repair_version'] == repair.REPAIR_VERSION
    for key in ('prompt_hash', 'repair_prompt_hash', 'output_contract_hash', 'model'):
        assert previous['event_integration'][key] != current['event_integration'][key]
    assert previous['analysis_events'] == current['analysis_events'] == bible['events']
    assert transport.calls[0]['system'] == legacy['system']
    assert transport.calls[1]['system'] == story_model.INTEGRATION_SYSTEM
    # Each generation has both its own raw response cache and validated checkpoint.
    assert len(list((store.root / 'cache').glob('valid-*.json'))) == 2
    assert len(list((store.root / 'cache').glob('*.json'))) == 4
    assert integrate_chapter(store, model, current, 'fixture-source') == current
    assert len(transport.calls) == 2
    assert model.ask('event_integration', transport.calls[1]['prompt']) == raw
    with monkeypatch.context() as old:
        old.setattr(story_model, 'INTEGRATION_SYSTEM', legacy['system'])
        assert model.ask('event_integration', transport.calls[0]['prompt']) == raw
    assert len(transport.calls) == 2
    assert all(p.read_bytes() == data for p, data in old_files.items())
    assert store.read() == store.read(saved['revision']) == saved
    assert saved['script']['segments'][0]['review_status'] == 'approved'
    for f in previous['event_integration']['fragment_index']:
        assert resolve_fragment(previous, f) == resolve_fragment(current, f)


def test_contract_structure_changes_only_versions_coverage_and_source_order_granularity():
    legacy = json.loads((Path(__file__).parent / 'fixtures/integration_v3_contract.json').read_text(encoding='utf-8'))
    assert integration.OUTPUT_CONTRACT == {
        **legacy['output_contract'], 'version': integration.VERSION, 'coverage': 'not_required',
        'reveal_order': 'source_panel_order_allow_intra_panel_interleaving',
        'repair': {**legacy['output_contract']['repair'], 'version': repair.REPAIR_VERSION}}
