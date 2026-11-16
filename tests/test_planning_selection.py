"""Offline selection, request, validation and cache contracts, not model-quality tests."""
import copy
import json
import socket
from collections import Counter

import pytest

import pipeline.story_model as roles
import pipeline.narrative_director as director_module
import pipeline.story_pipeline as pipeline_module
from pipeline.chapter_memory import planning_settings
from pipeline.narrative_director import NarrativeDirector, PLANNING_CONTRACT_VERSION
from pipeline.story_bible import StoryBible, StoryError
from pipeline.story_model import StoryModel, validated_result
from test_engagement_flow import story
from test_story_reuse import grounding_model, counts
from tools.audit_planning_inputs import audit


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('Selection tests cannot use real APIs')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)


@pytest.fixture
def chapter():
    summaries = [
        '旧桥每逢涨水便会封闭，强行过桥可能被困；这是多年留下的限制。' * 25,
        '送药人不肯冒险上桥。',
        '桌角摆着一枚新出现的蓝色回形针，与人物辨认、行动和情绪无关。',
        '听到对岸孩子急需药物，他答应趁封桥前过去。',
        '他把药物包好，带着同伴上桥。',
        '二人一边互相挤兑，一边拉紧绳子继续前进。',
        '同伴及时拉住险些跌倒的送药人，药物也保住了。',
        '脱险后，同伴抱着他哭了一阵，他安慰对方，两人静下来。',
        '同伴这才自我介绍：我叫小林。',
        '二人平复心情，带着保住的药继续去对岸。',
        '宣传页：关注连载。',
    ]
    events = []
    for i, summary in enumerate(summaries, 1):
        events.append({'event_id': f'E{i}', 'order': i-1, 'summary': summary,
                       'source_panel_ids': [str(i)], 'character_ids': [],
                       'importance': 5 if i == 3 else 1,
                       'role': 'climax' if i == 3 else 'setup',
                       'kind': 'world_rule' if i == 1 else 'current_action',
                       'time': {'original_time': '多年以前' if i == 1 else '未知',
                                'status': 'explicit' if i == 1 else 'unknown'},
                       'status': 'uncertain' if i == 6 else 'supported',
                       'dialogue': [{'speaker_id': None, 'text': '你可别掉下去。'}] if i == 6 else [],
                       'actions': [summary], 'causes': [], 'narrative_eligible': i != 11})
    return {'bible_id': 'selection-fixture', 'events': events,
            'panels': [{'source_panel_id': str(i), 'order': i-1} for i in range(1, 12)],
            'characters': [], 'relationships': [], 'prior_context': {'status': 'no_history', 'chapter': 1}}


@pytest.fixture
def response():
    return {'beats': [
        {'event_ids': ['E1', 'E2', 'E4'], 'section': 'opening', 'title': '从拒绝到决定送药',
         'viewer_question': '急需的药让冒险过桥有了代价与理由。', 'payoff': '解释旧桥限制及改变决定的消息。',
         'reason': '先建立危险与选择，承接上桥行动。'},
        {'event_ids': ['E5', 'E6', 'E7'], 'section': 'climax', 'title': '互相挤兑的同伴救下送药人',
         'viewer_question': '两人得带着药安全过桥。', 'payoff': '以拉住同伴兑现协作，保住药物。',
         'reason': '沿上桥行动推进到脱险，为事后情绪留出落点。'},
        {'event_ids': ['E8', 'E9', 'E10'], 'section': 'ending', 'title': '平复心情后继续送药',
         'viewer_question': '这场惊险让两人的相处有了变化。', 'payoff': '先安慰与自我介绍，再继续原有任务。',
         'reason': '兑现脱险后的情绪及关系变化，再接下一步目标。'}],
        'essential_event_ids': ['E1', 'E4', 'E5', 'E7', 'E8', 'E10'],
        'milestones': {'conflict': 'E2', 'climax': 'E7', 'rules': 'E1', 'goal_or_stakes': 'E4', 'ending': 'E10'},
        'rationale': '危险限制—改变决定—协作脱险—安慰与继续送药。'}


class Transport:
    kind = 'test'
    host = 'offline.invalid'

    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, **request):
        self.calls.append(request)
        return json.dumps(self.response, ensure_ascii=False)


def make_model(tmp_path, response):
    transport = Transport(response)
    return StoryModel(StoryBible(tmp_path/'chapter'), transport, writer_model='offline-fixture'), transport


def selected(plan):
    return [eid for beat in plan['beats'] for eid in beat['event_ids']]


def test_actual_request_allows_unique_omission_and_continuous_multi_event_beats(tmp_path, chapter, response):
    model, transport = make_model(tmp_path, response)
    before = copy.deepcopy(chapter)
    plan = NarrativeDirector(model).plan(chapter, max_events=2)
    assert plan['omitted_event_ids'] == ['E3']  # Unique, high old score/role, no storytelling function.
    assert selected(plan) == ['E1', 'E2', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10']
    assert plan['essential_event_ids'] == response['essential_event_ids']
    for beat, spec in zip(plan['beats'], response['beats']):
        assert len(beat['event_ids']) == 3
        assert all(beat[field] == spec[field] for field in ('title', 'viewer_question', 'payoff', 'reason'))
    assert len(transport.calls) == 1
    request = transport.calls[0]
    assert request['system'] == roles.PLAN_SYSTEM and request['images'] == []
    events = json.JSONDecoder().raw_decode(request['prompt'].split('整话证据事件（按原作顺序）：', 1)[1])[0]
    assert [e['event_id'] for e in events] == [f'E{i}' for i in range(1, 11)]
    assert events[0]['summary'] == chapter['events'][0]['summary']  # Long useful background intact.
    assert events[0]['time'] == chapter['events'][0]['time']
    assert events[5]['status'] == 'uncertain' and events[5]['dialogue'][0]['speaker_id'] is None
    assert chapter == before  # Contextual interpretation never edits the stored evidence.
    traces = [json.loads(p.read_text(encoding='utf-8')) for p in (model.store.root/'calls').glob('*.json')]
    assert [trace['request']['stage'] for trace in traces] == ['direct']
    assert 'completeness_review' not in plan


@pytest.mark.parametrize('eid', ['E1', 'E4', 'E5', 'E8'])
def test_declared_background_trigger_action_and_emotional_payoff_cannot_be_dropped(tmp_path, chapter, response, eid):
    for beat in response['beats']:
        beat['event_ids'] = [e for e in beat['event_ids'] if e != eid]
    model, transport = make_model(tmp_path, response)
    with pytest.raises(StoryError, match='缺少关键动机'):
        NarrativeDirector(model).plan(chapter)
    assert len(transport.calls) == 3
    assert eid in transport.calls[-1]['prompt'].split('上一版需修正：')[1]
    assert all(r['system'] == roles.PLAN_SYSTEM for r in transport.calls)


@pytest.mark.parametrize('key', ['conflict', 'climax', 'rules', 'goal_or_stakes', 'ending'])
def test_every_nonempty_milestone_requires_a_selected_event(tmp_path, chapter, response, key):
    response['essential_event_ids'] = []
    response['milestones'][key] = 'E3'
    model, _ = make_model(tmp_path, response)
    with pytest.raises(StoryError, match='缺少关键动机'):
        NarrativeDirector(model).plan(chapter)


@pytest.mark.parametrize('problem', ['unknown', 'non_story', 'duplicate_within', 'duplicate_across', 'reverse', 'interleave'])
def test_reference_and_range_constraints_survive_selection_changes(tmp_path, chapter, response, problem):
    if problem == 'unknown': response['beats'][0]['event_ids'].append('E999')
    if problem == 'non_story': response['beats'][-1]['event_ids'].append('E11')
    if problem == 'duplicate_within': response['beats'][0]['event_ids'].append('E1')
    if problem == 'duplicate_across': response['beats'][1]['event_ids'].append('E4')
    if problem == 'reverse': response['beats'].reverse()
    if problem == 'interleave':
        response['beats'][0]['event_ids'].append('E6')
        response['beats'][1]['event_ids'].remove('E6')
    model, _ = make_model(tmp_path, response)
    with pytest.raises(StoryError):
        NarrativeDirector(model).plan(chapter)


def test_no_selection_or_essential_count_target_is_imposed(tmp_path, chapter, response):
    response['beats'][0]['event_ids'].insert(2, 'E3')
    response['essential_event_ids'] = []
    model, _ = make_model(tmp_path, response)
    plan = NarrativeDirector(model).plan(chapter, max_events=2)
    assert plan['omitted_event_ids'] == [] and len(selected(plan)) == 10
    assert plan['essential_event_ids'] == []


def test_planning_system_and_contract_isolate_both_caches_without_invalidating_integration(story, monkeypatch):
    store, state, fixture, pipeline = story
    integrated = pipeline.integrate(state['revision'])
    bible = integrated['bible']
    raw = fixture.ask('direct', 'offline response fixture')
    transport = Transport(raw)
    model = StoryModel(store, transport, writer_model='offline-fixture')
    others = {stage: model.cache_identity(stage) for stage in ('event_integration', 'write_story', 'script_polish', 'check')}
    dependencies = {'contract': 4, 'bible': bible, 'style': 'faithful', 'max_events': None,
                    'settings': planning_settings(store)}
    new_system = roles.PLAN_SYSTEM
    with monkeypatch.context() as old:
        old.setattr(roles, 'PLAN_SYSTEM', '旧规划角色：仅供离线缓存隔离测试。')
        legacy = validated_result(store, model, 'direct', dependencies, lambda: NarrativeDirector(model).plan(bible))
        legacy_identity = model.cache_identity('direct')
    assert model.cache_identity('direct') != legacy_identity
    assert model.cache_identity('direct')['system'] == new_system
    assert all(model.cache_identity(stage) == identity for stage, identity in others.items())
    preserved = {p: p.read_bytes() for folder in ('cache', 'calls', 'revisions', 'evidence')
                 for p in (store.root/folder).rglob('*') if p.is_file()}
    # Even the identical task prompt gets a new raw response cache after PLAN_SYSTEM changes.
    new = validated_result(store, model, 'direct', dependencies, lambda: NarrativeDirector(model).plan(bible))
    assert len(transport.calls) == 2
    assert transport.calls[0]['prompt'] == transport.calls[1]['prompt']
    dependencies = dict(dependencies, contract=PLANNING_CONTRACT_VERSION)
    assert PLANNING_CONTRACT_VERSION != 4
    assert validated_result(store, model, 'direct', dependencies, lambda: NarrativeDirector(model).plan(bible)) == new
    assert len(transport.calls) == 2  # New checkpoint, new prompt/system raw cache already available.
    assert validated_result(store, model, 'direct', dependencies, lambda: pytest.fail('Must reuse new checkpoint')) == new
    assert all(p.read_bytes() == data for p, data in preserved.items())
    assert store.read(integrated['revision']) == integrated
    assert legacy['bible_id'] == new['bible_id'] == bible['bible_id']
    before = Counter(call[0] for call in fixture.calls)
    assert pipeline._integrate(copy.deepcopy(integrated)) == integrated
    assert Counter(call[0] for call in fixture.calls) == before
    assert {json.loads(p.read_text(encoding='utf-8'))['request']['stage']
            for p in (store.root/'calls').glob('*.json')} == {'direct'}


def test_contract_four_script_and_review_state_remain_in_history_after_replanning(story, monkeypatch):
    store, state, model, pipeline = story
    grounding_model(story)
    with monkeypatch.context() as old:
        old.setattr(pipeline_module, 'PLANNING_CONTRACT_VERSION', 4)
        old.setattr(director_module, 'PLANNING_CONTRACT_VERSION', 4)
        first = pipeline.generate(state['revision'])
    first['script']['segments'][0]['review_status'] = 'rejected'
    first = store.save(first, 'fixture_historical_review', first['revision'])
    previous = copy.deepcopy(first)
    kept = {p: p.read_bytes() for folder in ('cache', 'calls', 'revisions', 'evidence')
            for p in (store.root/folder).rglob('*') if p.is_file()}
    before = counts(model)
    new = pipeline.generate(first['revision'])
    assert new['bible'] == previous['bible'] and store.read(first['revision']) == previous
    assert new['plan']['planning_validation']['version'] == PLANNING_CONTRACT_VERSION
    assert counts(model) - before == {'direct': 1, 'write_story': 1, 'script_polish': 1}
    assert all(p.read_bytes() == data for p, data in kept.items())
    before = counts(model)
    assert pipeline.generate(new['revision'])['script'] == new['script']
    assert counts(model) == before


def test_request_audit_is_read_only_and_stops_before_any_model_response(story):
    store, state, model, pipeline = story
    integrated = pipeline.integrate(state['revision'])
    before = {p: p.read_bytes() for p in store.root.parent.rglob('*') if p.is_file()}
    calls = copy.deepcopy(model.calls)
    result = audit(store.root)
    assert result['report']['source_revision'] == integrated['revision']
    assert result['request']['stage'] == 'direct' and result['request']['system'] == roles.PLAN_SYSTEM
    assert result['report']['model_calls'] == result['report']['project_writes'] == 0
    assert model.calls == calls
    assert {p: p.read_bytes() for p in store.root.parent.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('matching_bible', [True, False])
def test_audit_reads_old_external_plan_only_if_it_matches_stored_integration(story, matching_bible):
    store, state, model, pipeline = story
    integrated = pipeline.integrate(state['revision'])
    plan = NarrativeDirector(model).plan(integrated['bible'])
    if not matching_bible:
        plan['bible_id'] = 'different-integrated-chapter'
    path = store.root.parent/'accepted-plan.json'
    path.write_text(json.dumps(plan, ensure_ascii=False), encoding='utf-8')
    before = {p: p.read_bytes() for p in store.root.parent.rglob('*') if p.is_file()}
    if matching_bible:
        result = audit(store.root, existing_plan=path)
        assert result['report']['existing_plan']['selected_events'] == 5
        assert result['report']['existing_plan']['beats'] == 3
        assert result['report']['request_events_match_saved_planning_view']
    else:
        with pytest.raises(ValueError, match='exact stored integration'):
            audit(store.root, existing_plan=path)
    assert {p: p.read_bytes() for p in store.root.parent.rglob('*') if p.is_file()} == before
