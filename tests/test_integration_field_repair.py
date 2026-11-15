"""Offline field-scope, provenance and pending-state contracts, not semantic evals."""
import copy
import json
import socket

import pytest

import pipeline.integration_repair as repair
from pipeline.event_integration import (OUTPUT_CONTRACT, consume, fragments_for, integrate_chapter,
                                        repair_integration_result)
from pipeline.integration_repair import IntegrationRepairUnresolved, apply_field_patch, field_issues
from pipeline.story_bible import StoryError
from pipeline.story_model import StoryModel
from integration_fixture import integration_response
from test_event_integration import analysis, merged_response
from test_engagement_flow import story


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('Field repair tests must not use real APIs')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)


def payload(prompt):
    return json.JSONDecoder().raw_decode(prompt.split('资料：', 1)[1])[0]


def answer(issue, value):
    return {'patches': [{'issue_id': issue['issue_id'], 'value': value}], 'unresolved': []}


def test_facet_role_sends_only_one_field_its_exact_sources_and_small_readonly_context(analysis):
    fragments = fragments_for(analysis)
    correct = merged_response(analysis)
    raw = copy.deepcopy(correct)
    raw['events'][1]['facets'][0]['role'] = 'invalid'
    raw['events'][0]['summary'] = 'UNRELATED_EVENT_SENTINEL'
    original = copy.deepcopy(raw)
    calls, audit = [], []
    class Model:
        def ask(self, stage, prompt, images=None, fresh=False):
            assert stage == 'event_integration' and fresh and not images
            calls.append(prompt)
            data = payload(prompt)
            assert 'UNRELATED_EVENT_SENTINEL' not in prompt
            assert not {'events', 'material', 'problems', 'replacements', 'additions'} & data.keys()
            assert len(data['issues']) == 1
            issue = data['issues'][0]
            assert issue['path'] == '/events/1/facets/0/role' and issue['current_value'] == 'invalid'
            assert issue['readonly_context'] == {k: original['events'][1]['facets'][0][k] for k in ('text', 'status')}
            assert {r[0] for r in data['fragments']} == set(original['events'][1]['facets'][0]['fragment_ids'])
            return answer(issue, correct['events'][1]['facets'][0]['role'])
    actual = repair_integration_result(Model(), raw, analysis, fragments, field_repairs=audit)
    expected = copy.deepcopy(original)
    expected['events'][1]['facets'][0]['role'] = correct['events'][1]['facets'][0]['role']
    assert actual == consume(expected, analysis, fragments) and raw == original
    assert len(calls) == len(audit) == 1 and audit[0]['path'] == '/events/1/facets/0/role'


def test_all_independent_bad_fields_share_a_call_and_valid_siblings_are_untouched(analysis):
    fragments = fragments_for(analysis)
    correct = merged_response(analysis)
    raw = copy.deepcopy(correct)
    raw['events'][0]['summary'] = None
    raw['events'][0]['time']['status'] = []
    raw['events'][1]['facets'][0]['role'] = {}
    expected = {'/events/0/summary': correct['events'][0]['summary'],
                '/events/0/time/status': correct['events'][0]['time']['status'],
                '/events/1/facets/0/role': correct['events'][1]['facets'][0]['role']}
    calls = []
    class Model:
        def ask(self, stage, prompt, **kwargs):
            calls.append(stage)
            issues = payload(prompt)['issues']
            assert {i['path'] for i in issues} == set(expected)
            return {'patches': [{'issue_id': i['issue_id'], 'value': expected[i['path']]} for i in issues], 'unresolved': []}
    assert repair_integration_result(Model(), raw, analysis, fragments) == consume(correct, analysis, fragments)
    assert calls == ['event_integration']


def test_second_call_sends_only_still_invalid_field_and_preserves_first_patch(analysis):
    fragments = fragments_for(analysis)
    raw = merged_response(analysis)
    correct = copy.deepcopy(raw)
    raw['events'][0]['summary'] = ''
    raw['events'][0]['facets'][0]['role'] = 'invalid'
    calls, audit = [], []
    class Model:
        def ask(self, stage, prompt, **kwargs):
            issues = payload(prompt)['issues']
            calls.append(issues)
            if len(calls) == 1:
                assert len(issues) == 2
                return {'patches': [{'issue_id': i['issue_id'], 'value': 'FIRST_PATCH_PRESERVED'
                                     if i['path'].endswith('/summary') else []} for i in issues], 'unresolved': []}
            assert len(issues) == 1 and issues[0]['path'] == '/events/0/facets/0/role'
            assert 'FIRST_PATCH_PRESERVED' not in prompt
            return answer(issues[0], correct['events'][0]['facets'][0]['role'])
    result = repair_integration_result(Model(), raw, analysis, fragments, field_repairs=audit)
    correct['events'][0]['summary'] = 'FIRST_PATCH_PRESERVED'
    assert result == consume(correct, analysis, fragments) and len(calls) == len(audit) == 2


@pytest.mark.parametrize('attack', ['legacy_event', 'extra_event', 'extra_path', 'foreign_issue', 'duplicate',
                                   'missing', 'both', 'whole_event_value', 'invalid_enum', 'wrong_type'])
def test_bad_protocol_or_value_cannot_overwrite_fields_or_expand_scope(analysis, attack):
    fragments = fragments_for(analysis)
    raw = merged_response(analysis)
    raw['events'][0]['facets'][0]['role'] = 'invalid'
    original, calls = copy.deepcopy(raw), []
    class Model:
        def ask(self, stage, prompt, **kwargs):
            calls.append(stage)
            issue = payload(prompt)['issues'][0]
            response = answer(issue, 'setting')
            p = response['patches'][0]
            if attack == 'legacy_event':
                return {'replacements': [{'index': 0, 'event': original['events'][0]}], 'additions': []}
            if attack == 'extra_event': response['events'] = []
            elif attack == 'extra_path': p['path'] = '/events/1/summary'
            elif attack == 'foreign_issue': p['issue_id'] = 'OTHER'
            elif attack == 'duplicate': response['patches'].append(copy.deepcopy(p))
            elif attack == 'missing': response['patches'] = []
            elif attack == 'both': response['unresolved'] = [{'issue_id': issue['issue_id'], 'status': 'insufficient_evidence', 'reason': '待补'}]
            elif attack == 'whole_event_value': p['value'] = original['events'][0]
            elif attack == 'invalid_enum': p['value'] = 'invented_role'
            elif attack == 'wrong_type': p['value'] = 1
            return response
    with pytest.raises(StoryError):
        repair_integration_result(Model(), raw, analysis, fragments)
    assert raw == original and len(calls) == 2


@pytest.mark.parametrize('attack', ['foreign_id', 'foreign_event_id', 'extra_program_field', 'empty_refs'])
def test_new_facet_field_cannot_inject_sources_or_program_fields(analysis, attack):
    fragments = fragments_for(analysis)
    correct = merged_response(analysis)
    raw = copy.deepcopy(correct)
    raw['events'][0]['facets'] = None
    class Model:
        def ask(self, stage, prompt, **kwargs):
            issue = payload(prompt)['issues'][0]
            value = copy.deepcopy(correct['events'][0]['facets'])
            if attack == 'foreign_id': value[0]['fragment_ids'] = ['F99999']
            elif attack == 'foreign_event_id': value[0]['fragment_ids'] = correct['events'][-1]['fragment_ids']
            elif attack == 'empty_refs': value[0]['fragment_ids'] = []
            else: value[0]['event_id'] = 'INVENTED'
            return answer(issue, value)
    with pytest.raises(StoryError):
        repair_integration_result(Model(), raw, analysis, fragments)
    assert raw['events'][0]['facets'] is None


@pytest.mark.parametrize('status', sorted(repair.UNRESOLVED))
def test_unresolved_is_explicit_and_stops_after_one_call_without_partial_commit(analysis, status):
    fragments = fragments_for(analysis)
    raw = merged_response(analysis)
    raw['events'][0]['summary'] = ''
    before, calls = copy.deepcopy(raw), []
    class Model:
        def ask(self, stage, prompt, **kwargs):
            calls.append(stage)
            issue = payload(prompt)['issues'][0]
            return {'patches': [], 'unresolved': [{'issue_id': issue['issue_id'], 'status': status, 'reason': '需要核对相关对白的说话人'}]}
    with pytest.raises(IntegrationRepairUnresolved) as exc:
        repair_integration_result(Model(), raw, analysis, fragments)
    assert calls == ['event_integration'] and raw == before
    assert exc.value.issues[0]['path'] == '/events/0/summary' and exc.value.issues[0]['status'] == status


def test_missing_entire_event_is_scope_change_without_model_guessing(analysis):
    fragments = fragments_for(analysis)
    raw = merged_response(analysis)
    raw['events'].pop()
    class Model:
        def ask(self, *args, **kwargs): raise AssertionError('No field can authorize new event placement')
    with pytest.raises(IntegrationRepairUnresolved, match='遗漏') as exc:
        repair_integration_result(Model(), raw, analysis, fragments)
    assert exc.value.issues[0]['status'] == 'requires_scope_change'


def test_patch_for_changed_value_is_rejected(analysis):
    fragments = fragments_for(analysis)
    raw = merged_response(analysis)
    raw['events'][0]['summary'] = ''
    issues = field_issues(raw, OUTPUT_CONTRACT, set(), fragments)
    raw['events'][0]['summary'] = '人工已修改'
    with pytest.raises(StoryError, match='已变化'):
        apply_field_patch(raw, issues, answer(issues[0], '模型的旧修正'))
    assert raw['events'][0]['summary'] == '人工已修改'


def test_patch_reference_format_and_status_are_fixed_locally_without_a_second_call(analysis):
    fragments = fragments_for(analysis)
    correct = merged_response(analysis)
    raw = copy.deepcopy(correct)
    raw['events'][1]['facets'] = None
    calls, local = [], []
    class Model:
        def ask(self, stage, prompt, **kwargs):
            calls.append(stage)
            issue = payload(prompt)['issues'][0]
            facets = copy.deepcopy(correct['events'][1]['facets'])
            for facet in facets:
                facet['fragment_ids'] = facet['fragment_ids'][0]
                facet['status'] = 'supported'
            return answer(issue, facets)
    result = repair_integration_result(Model(), raw, analysis, fragments, local_repairs=local)
    assert result == consume(correct, analysis, fragments) and calls == ['event_integration']
    assert {c['reason'] for c in local} == {'exact_reference_format', 'source_certainty_ceiling'}


def test_field_prompt_adds_only_exact_dialogue_dependency_and_no_full_chapter(analysis):
    analysis['events'][0]['dialogue'] = copy.deepcopy(analysis['panels'][0]['dialogue'])
    fragments = fragments_for(analysis)
    correct = merged_response(analysis)
    raw = copy.deepcopy(correct)
    raw['events'][0]['facets'][0]['role'] = 'invalid'
    class Model:
        def ask(self, stage, prompt, **kwargs):
            data = payload(prompt)
            issue = data['issues'][0]
            assert {r[1] for r in data['fragments']} == {'event', 'dialogue'}
            assert {r[2] for r in data['fragments']} == {'E1', '1'}
            assert set(issue['fragment_ids']) == {r[0] for r in data['fragments']}
            assert 'dialogue' not in next(r[4] for r in data['fragments'] if r[1] == 'event')
            return answer(issue, correct['events'][0]['facets'][0]['role'])
    assert repair_integration_result(Model(), raw, analysis, fragments) == consume(correct, analysis, fragments)


def test_unresolved_transport_trace_is_preserved_without_checkpoint_or_revision(story):
    store, state, _, _ = story
    before = {p: p.read_bytes() for p in store.root.rglob('*') if p.is_file()}
    class Transport:
        kind, host = 'test', 'offline.invalid'
        calls = []
        def chat(self, **request):
            self.calls.append(request)
            if len(self.calls) == 1:
                response = integration_response(request['prompt'])
                response['events'][0]['summary'] = ''
            else:
                issue = payload(request['prompt'])['issues'][0]
                response = {'patches': [], 'unresolved': [{'issue_id': issue['issue_id'],
                            'status': 'insufficient_evidence', 'reason': '说话人待核对'}]}
            return json.dumps(response, ensure_ascii=False)
    transport = Transport()
    with pytest.raises(IntegrationRepairUnresolved):
        integrate_chapter(store, StoryModel(store, transport), state['bible'], state['revision'])
    assert len(transport.calls) == 2 and store.read() == state
    assert all(p.read_bytes() == value for p, value in before.items())
    assert not list((store.root/'cache').glob('valid-*.json'))
    traces = [json.loads(p.read_text(encoding='utf-8')) for p in (store.root/'calls').glob('*.json')]
    assert len(traces) == 2 and any('unresolved' in json.loads(t['response']) for t in traces)


def test_field_repair_journal_cache_and_prompt_dependencies_keep_old_results(story, monkeypatch):
    store, state, _, _ = story
    class Model:
        calls = []
        def ask(self, stage, prompt, **kwargs):
            self.calls.append(prompt)
            if 'Event Integration Field Repair v1' in prompt:
                issue = payload(prompt)['issues'][0]
                return answer(issue, '修正的合成事件梳理')
            result = integration_response(prompt)
            result['events'][0]['summary'] = ''
            return result
    model = Model()
    result = integrate_chapter(store, model, state['bible'], state['revision'])
    assert len(result['event_integration']['field_repairs']) == 1 and len(model.calls) == 2
    assert integrate_chapter(store, model, result, state['revision']) == result and len(model.calls) == 2
    old = {p: p.read_bytes() for p in (store.root/'cache').glob('*.json')}
    monkeypatch.setattr(repair, 'REPAIR_INSTRUCTIONS', 'NEW_FIELD_PROMPT\n' + repair.REPAIR_INSTRUCTIONS)
    updated = integrate_chapter(store, model, result, state['revision'])
    assert len(model.calls) == 4 and model.calls[-1].startswith('NEW_FIELD_PROMPT')
    assert all(p.read_bytes() == value for p, value in old.items()) and store.read() == state
    assert result['events'] == updated['events']
    assert result['event_integration']['repair_prompt_hash'] != updated['event_integration']['repair_prompt_hash']
