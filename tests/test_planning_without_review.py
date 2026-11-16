"""Offline contracts for planning with program checks and no model reviewer."""
import copy
import socket

import pytest

from pipeline.narrative_director import NarrativeDirector, PLANNING_CONTRACT_VERSION
from pipeline.story_bible import StoryError
from pipeline.story_model import validated_result
from test_engagement_flow import story
from test_story_reuse import grounding_model, counts


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('Planning checks cannot use real APIs')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)


@pytest.mark.parametrize('problem', ['unknown_id', 'duplicate_id', 'non_story', 'reversed_order',
                                    'missing_essential', 'missing_milestone'])
def test_program_constraints_still_reject_invalid_plans_without_a_model_review(story, problem):
    _, state, model, _ = story
    bible = copy.deepcopy(state['bible'])
    if problem == 'non_story':
        bible['events'][-1]['narrative_eligible'] = False
    before = copy.deepcopy(bible)
    original = model.ask
    def ask(stage, prompt, **kwargs):
        assert stage == 'direct'
        raw = original(stage, prompt, **kwargs)
        if problem == 'unknown_id': raw['beats'][0]['event_ids'] = ['E99999']
        elif problem == 'duplicate_id': raw['beats'][0]['event_ids'].append('E1')
        elif problem == 'reversed_order': raw['beats'][0], raw['beats'][1] = raw['beats'][1], raw['beats'][0]
        elif problem == 'missing_essential':
            raw['essential_event_ids'] = ['E2']
            raw['beats'][0]['event_ids'] = ['E1']
        elif problem == 'missing_milestone': raw['beats'][0]['event_ids'] = ['E2']
        return raw
    model.ask = ask
    with pytest.raises(StoryError):
        NarrativeDirector(model).plan(bible)
    assert len(model.calls) == 3 and all(c[0] == 'direct' for c in model.calls)
    assert bible == before


def test_program_check_does_not_claim_semantic_completeness_or_add_a_review_call(story):
    _, state, model, _ = story
    original = model.ask
    def ask(stage, prompt, **kwargs):
        assert stage == 'direct'
        raw = original(stage, prompt, **kwargs)
        raw['beats'][0]['event_ids'] = ['E1']  # E2 is not declared mandatory by this fake model.
        return raw
    model.ask = ask
    plan = NarrativeDirector(model).plan(state['bible'])
    assert [c[0] for c in model.calls] == ['direct']
    assert 'E2' in plan['omitted_event_ids'] and 'completeness_review' not in plan
    assert plan['planning_validation'] == {'version': PLANNING_CONTRACT_VERSION, 'mode': 'program_constraints'}


def test_new_planning_cache_does_not_reuse_old_reviewed_contract_or_overwrite_history(story):
    store, state, model, pipeline = story
    grounding_model(story)
    integrated = pipeline.integrate(state['revision'])
    bible = integrated['bible']
    legacy_plan = NarrativeDirector(model).plan(bible)
    legacy_plan.pop('planning_validation')
    legacy_plan['completeness_review'] = {'passed': True, 'missing_event_ids': [], 'issues': []}
    old_dependencies = {'contract': 1, 'bible': bible, 'style': 'faithful', 'max_events': None}
    validated_result(store, model, 'direct', old_dependencies, lambda: copy.deepcopy(legacy_plan))
    old_files = {p: p.read_bytes() for p in store.root.rglob('*') if p.is_file()}
    before = counts(model)
    generated = pipeline.generate(integrated['revision'])
    assert counts(model)['direct'] - before['direct'] == 1
    assert counts(model)['story_plan_review'] == 0
    assert counts(model)['story_digest'] == 0 and counts(model)['script_polish'] == 1
    assert 'completeness_review' not in generated['plan']
    assert generated['plan']['planning_validation']['version'] == PLANNING_CONTRACT_VERSION
    assert all(p.read_bytes() == data for p, data in old_files.items() if p.name != 'current.json')
    assert store.read(integrated['revision']) == integrated
    assert validated_result(store, model, 'direct', old_dependencies,
                            lambda: pytest.fail('Old history must remain readable')) == legacy_plan
    before = counts(model)
    repeated = pipeline.generate(generated['revision'])
    assert repeated['script'] == generated['script']
    assert all(counts(model)[s] == before[s] for s in ('direct','story_digest','write_story','script_polish','story_plan_review'))
