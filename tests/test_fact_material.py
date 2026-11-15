"""Offline retention, provenance and cache contracts for compact fact inputs."""
import copy
import json
import socket

import pytest

from pipeline.fact_material import fact_material, FACT_CONTRACT_VERSION
from pipeline.narrative_director import NarrativeDirector
from pipeline.prompt_material import context_catalog
from pipeline.script_writer import ScriptWriter
from pipeline.story_model import validated_result
from test_engagement_flow import story
from test_planning_material import chapter


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('Fact compaction must not call live models or TTS')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)


@pytest.fixture
def source(chapter):
    for i, event in enumerate(chapter['events']):
        if i == 1:
            event['dialogue'][0]['speaker_id'] = 'C1'
        chapter['panels'][i]['dialogue'] = copy.deepcopy(event['dialogue'])
        event['evidence'] = [{'kind': 'dialogue', 'detail': '进来吧。', 'source_panel_id': str(i+1)}]
    chapter['events'][0]['evidence'] += [
        {'kind': 'visual', 'detail': '桌下还有一个药盒。', 'source_panel_id': '1'},
        {'kind': 'caption', 'detail': '进来吧。', 'source_panel_id': '1'}]
    chapter['events'][2].update(kind='historical_explanation', historical=True)
    plan = {'bible_id': chapter['bible_id'], 'style': 'faithful', 'version': 5,
            'settings_basis': {'revision': 'settings-version', 'hash': 'settings-hash'},
            'planning_validation': {'version': 4}, 'essential_event_ids': ['E1', 'E2'],
            'ending_rule': '保留既有结果。', 'milestones': {'climax': 'E2'},
            'beats': [{'event_id': 'E1', 'event_ids': ['E1'], 'context_event_ids': [],
                       'section': 'opening', 'title': '先理解约定', 'payoff': '出现信号',
                       'viewer_question': '约定能否奏效？', 'reason': '说明行动条件',
                       'source_panel_ids': ['1']},
                      {'event_id': 'E2', 'event_ids': ['E2'], 'context_event_ids': ['E3'],
                       'section': 'ending', 'title': '照约定行动', 'source_panel_ids': ['2', '3']}]}
    return chapter, plan


def test_scope_order_unknowns_plan_intent_and_unique_text_survive_without_source_mutation(source):
    bible, plan = source
    before, before_plan = copy.deepcopy(bible), copy.deepcopy(plan)
    bible['events'][0].update(status='unknown', pending_causes=[{'status': 'needs_history'}])
    before = copy.deepcopy(bible)
    material = fact_material(bible, plan)
    assert [e['event_id'] for e in material['events']] == ['E1', 'E2', 'E3']
    assert material['events'][2]['kind'] == 'historical_explanation'  # Context is not eligibility-filtered.
    assert material['events'][0]['status'] == 'unknown'
    assert material['events'][0]['time']['status'] == 'unknown'
    assert material['events'][0]['pending_causes'] == [{'status': 'needs_history'}]
    assert material['events'][0]['reveal_order'] == 0
    assert material['characters'][0]['gender'] is None
    assert material['characters'][0]['name_status'] == 'unknown'
    assert material['plan']['essential_event_ids'] == ['E1', 'E2']
    assert material['plan']['beats'][1]['context_event_ids'] == ['E3']
    assert material['plan']['beats'][0]['viewer_question'] == '约定能否奏效？'
    assert 'settings_basis' not in material['plan'] and 'planning_validation' not in material['plan']
    assert material['identity_availability'] == {'status': 'no_confirmed_reveals'}
    material['events'][0]['facets'][0]['text'] = '只修改返回视图'
    material['plan']['essential_event_ids'].append('only-view')
    assert bible == before and plan == before_plan


def test_only_same_source_dialogue_copy_is_omitted_and_unique_modalities_stay(source):
    bible, plan = source
    material = fact_material(bible, plan)
    assert material['events'][0]['evidence'] == [
        {'kind': 'visual', 'detail': '桌下还有一个药盒。'},
        {'kind': 'caption', 'detail': '进来吧。'}]
    assert 'evidence' not in material['events'][1]
    # Repeated text in two events still retains each event's own speaker.
    assert material['events'][0]['dialogue'] == [{'speaker_id': None, 'text': '进来吧。'}]
    assert material['events'][1]['dialogue'] == [{'speaker_id': 'C1', 'text': '进来吧。'}]
    assert material['events'][0]['facets'][0]['text'] == material['events'][1]['facets'][0]['text']


@pytest.mark.parametrize('kind', [None, 'current_action'])
def test_historical_actions_keep_origin_even_with_legacy_or_current_action_kind(source, kind):
    bible, plan = source
    event = bible['events'][2]
    event['history_origin'] = {'chapter': 1, 'revision': 'prior-version', 'event_id': 'original-event'}
    if kind is None:
        event.pop('kind')
    else:
        event['kind'] = kind
    compact = fact_material(bible, plan)['events'][2]
    assert compact['historical'] is True
    assert compact['history_origin'] == event['history_origin']
    assert compact.get('kind') == kind
    assert compact['time'] == event['time']


@pytest.mark.parametrize('case', ['other_source', 'speaker_conflict', 'non_exact_quote',
                                'uncertain_note', 'extra_provenance', 'unknown_source'])
def test_ambiguous_or_nonidentical_evidence_is_kept_with_its_source(source, case):
    bible, plan = source
    item = bible['events'][0]['evidence'][0]
    if case == 'other_source': item['source_panel_id'] = '2'
    elif case == 'speaker_conflict': bible['events'][0]['dialogue'][0]['speaker_id'] = 'C1'
    elif case == 'non_exact_quote': item['detail'] = '进来吧'
    elif case == 'uncertain_note': item['status'] = 'uncertain'
    elif case == 'extra_provenance': item['history_origin'] = {'chapter': 9, 'revision': 'other-version'}
    elif case == 'unknown_source': item['source_panel_id'] = 'missing'
    assert fact_material(bible, plan)['events'][0]['evidence'][0] == item


def test_nonempty_identity_claims_and_legacy_ambiguity_are_preserved(source):
    bible, plan = source
    claim = {'name': '陈默', 'source_panel_id': '2', 'reveal_order': 1,
             'quote': '我叫陈默。', 'attribution': {'character_id': 'C1', 'source_panel_ids': ['2']}}
    bible['characters'][0].update(name='陈默', name_status='confirmed', name_claims=[claim])
    legacy = {'character_id': 'C2', 'name': '旧称呼', 'source_panel_ids': ['1'],
              'identity_evidence': [{'detail': '旧版待核记录'}]}
    bible['characters'].append(legacy)
    rows = fact_material(bible, plan)['identity_availability']
    assert rows[0]['not_yet_revealed'][0] == {'character_id': 'C1', **claim}
    assert rows[1]['available_by_node_end'][0] == {'character_id': 'C1', **claim}
    assert rows[0]['legacy_needs_verification'] == [legacy]


def test_context_provenance_and_conflicting_versions_are_not_compacted(source):
    bible, plan = source
    bible['prior_context'] = {'chapter': 2, 'status': 'available',
        'previous_chapter': {'chapter': 1, 'revision': 'old-version'}, 'background': [
            {'text': '旧规则尚未核实。', 'event_ids': ['E1'], 'status': 'uncertain',
             'source_version': 'other-version', 'sha256': 'other-original', 'source_panel_ids': ['other-panel']}]}
    rows = fact_material(bible, plan)['context_catalog']
    assert rows == context_catalog(bible)
    assert rows[0]['status'] == 'uncertain'
    assert rows[0]['source']['source_version'] == 'other-version'
    assert rows[0]['source']['sha256'] == 'other-original'
    assert rows[0]['source']['source_panel_ids'] == ['other-panel']


@pytest.mark.parametrize('has_context', [False, True])
def test_digest_uses_compact_input_and_only_adds_catalog_instructions_when_needed(story, has_context):
    _, state, model, _ = story
    bible = copy.deepcopy(state['bible'])
    if has_context:
        bible['prior_context'] = {'background': [{'text': '待核旧规则', 'event_ids': [], 'status': 'uncertain'}]}
    plan = NarrativeDirector(model).plan(bible)
    model.calls.clear()
    facts = ScriptWriter(model).prepare_facts(bible, plan)
    assert len(facts) == len(plan['beats'])
    assert len(model.calls) == 1 and model.calls[0][0] == 'story_digest' and not model.calls[0][2]
    prompt = model.calls[0][1]
    material = json.JSONDecoder().raw_decode(prompt.split('资料：', 1)[1])[0]
    assert prompt.startswith('Story Digest v3')
    assert bool(material['context_catalog']) == has_context
    assert ('逐条决定用途' in prompt) == has_context
    assert [b['event_id'] for b in material['plan']['beats']] == [b['event_id'] for b in plan['beats']]


def test_old_digest_cache_is_not_reused_or_overwritten_and_new_result_is_shared(story):
    store, state, model, _ = story
    bible = state['bible']; plan = NarrativeDirector(model).plan(bible)
    facts = ScriptWriter(model).prepare_facts(bible, plan)
    legacy = copy.deepcopy(facts); legacy[0]['facts'][0]['text'] = 'OLD_CACHE_SENTINEL'
    dependencies = {'contract': 3, 'bible': bible, 'plan': plan}
    validated_result(store, model, 'story_digest', dependencies, lambda: legacy)
    old_files = {p: p.read_bytes() for p in (store.root/'cache').glob('*') if p.is_file()}
    writer = ScriptWriter(model, store)
    before = sum(c[0] == 'story_digest' for c in model.calls)
    assert writer.prepare_facts(bible, plan) == facts
    assert sum(c[0] == 'story_digest' for c in model.calls) == before+1
    assert FACT_CONTRACT_VERSION != 3
    assert all(p.read_bytes() == data for p, data in old_files.items())
    assert writer.prepare_facts(bible, plan) == facts
    assert sum(c[0] == 'story_digest' for c in model.calls) == before+1
    assert validated_result(store, model, 'story_digest', dependencies,
                            lambda: pytest.fail('Old cache must remain readable')) == legacy


def test_compacted_away_source_metadata_still_invalidates_digest_cache(story):
    store, state, model, _ = story
    bible = state['bible']; plan = NarrativeDirector(model).plan(bible)
    writer = ScriptWriter(model, store)
    writer.prepare_facts(bible, plan)
    changed = copy.deepcopy(bible)
    changed['events'][0]['importance'] += 1
    assert fact_material(changed, plan) == fact_material(bible, plan)
    before = sum(c[0] == 'story_digest' for c in model.calls)
    writer.prepare_facts(changed, plan)
    assert sum(c[0] == 'story_digest' for c in model.calls) == before+1
    assert store.read() == state
