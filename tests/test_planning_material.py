"""Offline checks for best-effort director input compaction and fact retention."""
import copy
import json
import socket

import pytest

from pipeline.character_identity import identity_context
from pipeline.narrative_director import NarrativeDirector, PLANNING_CONTRACT_VERSION
from pipeline.planning_material import planning_material, planning_json
from pipeline.story_model import validated_result
from test_story_reuse import grounding_model, counts
from test_engagement_flow import story


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('Planning material tests must not use real APIs')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)


@pytest.fixture
def chapter():
    events = []
    for i in range(1, 4):
        events.append({
            'event_id': f'E{i}', 'order': i-1, 'summary': '看到约定信号后推开门。',
            'character_ids': ['C1'], 'source_panel_ids': [str(i)],
            'kind': 'current_action', 'status': 'uncertain',
            'time': {'original_time': '未知', 'status': 'unknown'},
            'reveal_panel_id': str(i), 'reveal_order': i-1,
            'facets': [{'role': 'action', 'text': '推开门。', 'status': 'supported',
                        'fragment_ids': [f'F{i}']},
                       {'role': 'unknown', 'text': '无法确认门外人的身份。',
                        'status': 'uncertain', 'fragment_ids': [f'F{i+3}']}],
            'actions': ['推开门。'], 'dialogue': [{'speaker_id': None, 'text': '进来吧。'}],
            'importance': 3, 'role': 'setup', 'causes': [], 'pending_causes': [],
            'internal_causes': [], 'narrative_eligible': i != 3,
        })
    return {
        'bible_id': 'compact-fixture', 'events': events,
        'panels': [{'source_panel_id': str(i), 'order': i-1} for i in range(1, 4)],
        'characters': [{'character_id': 'C1', 'name': '未知人物C1', 'name_status': 'unknown',
                        'gender': None, 'description': '戴围巾的人。', 'aliases': [],
                        'name_claims': [], 'label_claims': [], 'pending_names': [],
                        'source_panel_ids': ['1', '2']}],
        'relationships': [], 'prior_context': {'status': 'no_history', 'chapter': 1},
    }


def test_compaction_preserves_events_facets_and_local_evidence_without_mutation(chapter):
    settings = {'revision': 'settings-version', 'background': '世界规则。', 'characters': '主角设定。'}
    before, settings_before = copy.deepcopy(chapter), copy.deepcopy(settings)
    material = planning_material(chapter, settings)
    assert [e['event_id'] for e in material['events']] == ['E1', 'E2']
    for original, compact in zip(chapter['events'], material['events']):
        assert compact['summary'] == original['summary']
        assert [(f['role'], f['text'], f['status']) for f in compact['facets']] == [
            (f['role'], f['text'], f['status']) for f in original['facets']]
        assert 'actions' not in compact and 'fragment_ids' not in planning_json(compact)
        assert 'source_panel_ids' not in compact and 'reveal_panel_id' not in compact
        assert compact['reveal_order'] == original['reveal_order']
    assert material['settings'] == {'background': '世界规则。', 'characters': '主角设定。'}
    assert chapter == before and settings == settings_before
    assert chapter['events'][0]['facets'][0]['fragment_ids'] == ['F1']
    material['events'][0]['facets'][0]['text'] = '修改视图'
    material['characters']['characters'][0]['name'] = '修改视图'
    assert chapter == before


@pytest.mark.parametrize('actions', [['推开门。', '又推了一次。'], ['推开门。', '推开门。']])
def test_unmatched_and_repeated_actions_are_kept(chapter, actions):
    chapter['events'][0]['actions'] = actions
    assert planning_material(chapter)['events'][0]['actions'] == actions


def test_repeated_dialogue_in_different_events_and_conflicting_speakers_are_not_merged(chapter):
    chapter['events'][1]['dialogue'][0]['speaker_id'] = 'C1'
    material = planning_material(chapter)
    assert [e['facets'][0]['text'] for e in material['events']] == ['推开门。', '推开门。']
    assert material['events'][0]['dialogue'] == [{'speaker_id': None, 'text': '进来吧。'}]
    assert material['events'][1]['dialogue'] == [{'speaker_id': 'C1', 'text': '进来吧。'}]


def test_only_empty_list_defaults_are_removed_and_unknowns_and_history_survive(chapter):
    event = chapter['events'][0]
    event.update(importance=0, status='unknown', pending_causes=[{
        'status': 'needs_history', 'source_event_id': 'previous-event'}])
    material = planning_material(chapter)
    compact = material['events'][0]
    assert compact['importance'] == compact['reveal_order'] == 0
    assert compact['status'] == 'unknown'
    assert compact['time'] == {'original_time': '未知', 'status': 'unknown'}
    assert compact['pending_causes'] == event['pending_causes']
    assert 'causes' not in compact and 'internal_causes' not in compact
    assert material['characters']['characters'][0]['gender'] is None
    assert material['characters']['characters'][0]['name_status'] == 'unknown'
    assert material['history'] == {'status': 'no_history', 'chapter': 1}
    assert material['identity_positions'] == {'status': 'no_confirmed_reveals'}


def test_mixed_story_and_visual_details_are_not_deleted_by_keywords(chapter):
    facet = chapter['events'][0]['facets'][1]
    facet.update(text='他用衣服颜色作为约定信号；墙上的署名是否为暗号尚不清楚。', status='conflicting')
    assert planning_material(chapter)['events'][0]['facets'][1]['text'] == facet['text']
    assert planning_material(chapter)['events'][0]['facets'][1]['status'] == 'conflicting'


def test_legacy_events_without_facets_keep_their_actions_and_causes(chapter):
    event = chapter['events'][0]
    event.pop('facets')
    event['causes'] = [{'event_id': 'earlier', 'evidence': '听到约定口令后才开门。'}]
    compact = planning_material(chapter)['events'][0]
    assert compact['actions'] == event['actions']
    assert compact['causes'] == event['causes']


def test_real_reveals_and_unverified_legacy_identity_are_not_replaced_with_empty_status(chapter):
    character = chapter['characters'][0]
    character.update(name='陈默', name_status='confirmed', name_claims=[{
        'name': '陈默', 'source_panel_id': '2', 'reveal_order': 1,
        'quote': '我叫陈默。', 'attribution': {'character_id': 'C1', 'source_panel_ids': ['2']}}])
    legacy = {'character_id': 'C2', 'name': '旧称呼', 'source_panel_ids': ['1'],
              'identity_evidence': [{'detail': '旧版未经确认的识别记录。'}]}
    chapter['characters'].append(legacy)
    before = identity_context(chapter, [{'event_id': 'E1'}, {'event_id': 'E2'}])
    after = planning_material(chapter)['identity_positions']
    assert after[0]['not_yet_revealed'] == before[0]['not_yet_revealed']
    assert 'available_by_node_end' not in after[0]
    assert after[1]['available_by_node_end'] == before[1]['available_by_node_end']
    assert after[0]['legacy_needs_verification'] == [legacy]
    assert after[1]['legacy_needs_verification'] == [legacy]


def test_director_uses_compact_input_and_resolves_full_sources_without_extra_calls(story):
    _, state, model, pipeline = story
    integrated = pipeline.integrate(state['revision'])
    bible = integrated['bible']
    before = copy.deepcopy(bible)
    model.calls.clear()
    settings = {'revision': 'version', 'background': 'WORLD_SENTINEL', 'characters': 'PEOPLE_SENTINEL'}
    plan = NarrativeDirector(model, settings).plan(bible)
    assert len(model.calls) == 1 and model.calls[0][0] == 'direct' and not model.calls[0][2]
    prompt = model.calls[0][1]
    data = json.JSONDecoder().raw_decode(prompt.split('整话证据事件（按原作顺序）：', 1)[1])[0]
    assert [e['event_id'] for e in data] == [e['event_id'] for e in bible['events']]
    assert 'fragment_ids' not in prompt and 'no_confirmed_reveals' in prompt
    assert prompt.count('WORLD_SENTINEL') == prompt.count('PEOPLE_SENTINEL') == 1
    assert plan['beats'][0]['source_panel_ids'] == ['1', '2']
    assert bible == before


def test_previous_settings_contract_checkpoint_is_not_reused_or_overwritten(story):
    store, state, model, pipeline = story
    grounding_model(story)
    integrated = pipeline.integrate(state['revision'])
    bible = integrated['bible']
    legacy = NarrativeDirector(model).plan(bible)
    legacy['planning_validation']['version'] = 3
    dependencies = {'contract': 3, 'bible': bible, 'style': 'faithful', 'max_events': None, 'settings': None}
    validated_result(store, model, 'direct', dependencies, lambda: copy.deepcopy(legacy))
    old_files = {p: p.read_bytes() for p in (store.root/'cache').glob('*') if p.is_file()}
    before = counts(model)
    generated = pipeline.generate(integrated['revision'])
    assert counts(model)['direct'] - before['direct'] == 1
    assert counts(model)['event_integration'] == before['event_integration']
    assert generated['plan']['planning_validation']['version'] == PLANNING_CONTRACT_VERSION
    assert all(p.read_bytes() == data for p, data in old_files.items())
    assert validated_result(store, model, 'direct', dependencies,
                            lambda: pytest.fail('Old checkpoint must remain readable')) == legacy
    before = counts(model)
    assert pipeline.generate(generated['revision'])['script'] == generated['script']
    assert counts(model)['direct'] == before['direct']
