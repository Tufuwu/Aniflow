"""No implicit digest, source retention and backwards-compatible draft handling."""
import copy
import json
import socket

import pytest

from pipeline.fact_material import writing_material, source_material_basis
from pipeline.narrative_director import NarrativeDirector
from pipeline.script_writer import ScriptWriter
from pipeline.story_bible import StoryConflict, StoryError, digest
from pipeline.prompt_material import current_fact_sheet, text_evidence
from pipeline.story_candidates import compare_candidates
from test_candidate_reuse import candidate_model, counts
from test_engagement_flow import story


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('Program writing tests cannot call a live API')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)


def payload(prompt, marker='叙事依据：'):
    return json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])[0]


def forbid_digest(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail('Default generation must not call prepare_facts')
    monkeypatch.setattr(ScriptWriter, 'prepare_facts', fail)


def test_default_candidate_flow_skips_digest_and_shares_raw_source_dependencies(story, monkeypatch):
    store, state, model, pipeline = story
    candidate_model(story)
    forbid_digest(monkeypatch)
    before = copy.deepcopy(state)
    result = pipeline.generate(state['revision'])
    stages = [c[0] for c in model.calls]
    assert stages == ['event_integration','direct','write_story','script_review',
                      'write_story','script_review','candidate_compare']
    for c in result['candidates']:
        assert 'fact_sheet' not in c['script'] and 'fact_sheet_basis' not in c['script']
        assert c['script']['source_material_basis'] == source_material_basis(result['bible'], c['plan'])
        assert set(c['script']['source_material_basis']) == {'version', 'hash'}
    for stage, prompt, images, _ in model.calls:
        if stage == 'write_story':
            data = payload(prompt)
            assert not images and 'nodes' not in data and 'fact_sheet' not in data
            assert 'basis_mode' not in data
            assert len(data['plan']['beats']) == 3 and len(data['events']) == 5
            assert data['events'][1]['summary'] == result['bible']['events'][1]['summary']
            assert '模型处理不等于内容正确' in prompt
    assert store.read(before['revision']) == before


def test_compatible_verified_flow_keeps_review_and_original_image_checks(story, monkeypatch):
    _, state, model, pipeline = story
    candidate_model(story)
    forbid_digest(monkeypatch)
    result = pipeline.generate_verified(state['revision'])
    stages = [c[0] for c in model.calls]
    assert 'story_digest' not in stages and 'script_review' in stages and 'check' in stages
    assert stages.index('write_story') < stages.index('script_review') < stages.index('check')
    assert any(images for stage, _, images, _ in model.calls if stage == 'check')
    assert 'fact_sheet' not in result['script']
    assert result['script']['source_material_basis'] == source_material_basis(result['bible'], result['plan'])


def test_standalone_writer_calls_write_directly_when_facts_are_none(story, monkeypatch):
    _, state, model, _ = story
    plan = NarrativeDirector(model).plan(state['bible'])
    model.calls.clear()
    forbid_digest(monkeypatch)
    segments = ScriptWriter(model).write_story(state['bible'], plan)
    assert len(segments) == 3 and [c[0] for c in model.calls] == ['write_story']
    assert [s['event_id'] for s in segments] == [b['event_id'] for b in plan['beats']]


@pytest.mark.parametrize('problem', ['unknown_event', 'missing_essential'])
def test_bad_packaging_scope_is_rejected_before_any_model_call(story, problem):
    _, state, model, _ = story
    plan = NarrativeDirector(model).plan(state['bible'])
    if problem == 'unknown_event': plan['beats'][0]['context_event_ids'].append('missing')
    else:
        plan['essential_event_ids'] = ['E2']
        plan['beats'][0]['event_ids'] = ['E1']
    model.calls.clear()
    before = copy.deepcopy(state['bible'])
    with pytest.raises(StoryError): ScriptWriter(model).write_story(state['bible'], plan)
    assert model.calls == [] and state['bible'] == before


@pytest.mark.parametrize('chapter', [1, 2])
def test_history_and_unverified_references_keep_sources_without_inventing_a_recap(story, chapter):
    _, state, model, _ = story
    bible = copy.deepcopy(state['bible'])
    bible['events'][1].update(status='unknown', pending_causes=[{'status':'needs_history'}])
    prior = {'chapter':chapter,'status':'no_history','series_notes':{
        'revision':'ref-v1','characters':'未知姓名不能据此绑定人物'}}
    if chapter == 2:
        old_panel = copy.deepcopy(bible['panels'][0])
        old_panel.update(source_panel_id='H1', order=-1, historical=True)
        bible['panels'].insert(0, old_panel)
        old_event = copy.deepcopy(bible['events'][0])
        old_event.update(event_id='H_E1', source_panel_ids=['H1'], historical=True,
                         history_origin={'chapter':1,'revision':'old-v1','event_id':'E1'})
        bible['events'].insert(0, old_event)
        prior.update(status='available', previous_chapter={'chapter':1,'revision':'old-v1'},
                     background=[{'text':'上一话的危险限制','event_ids':['H_E1']}])
    bible['prior_context'] = prior
    plan = NarrativeDirector(model).plan(bible)
    if chapter == 2: plan['beats'][0]['context_event_ids'].append('H_E1')
    before = copy.deepcopy(bible)
    data = writing_material(bible, plan)
    assert data['reference_context'][0]['status'] == 'unverified_reference'
    assert data['reference_context'][0]['source']['revision'] == 'ref-v1'
    assert data['reference_context'][0]['selection'] == 'uncertain'
    if chapter == 1: assert data['prior_context'] == []
    else:
        assert data['prior_context'][0]['source']['events'][0]['origin']['revision'] == 'old-v1'
        historical = next(e for e in data['events'] if e['event_id'] == 'H_E1')
        assert historical['historical'] is True and historical['history_origin']['revision'] == 'old-v1'
    event = next(e for e in data['events'] if e['event_id'] == 'E2')
    assert event['status'] == 'unknown' and event['pending_causes'] == [{'status':'needs_history'}]
    assert bible == before


def test_legacy_model_sheet_is_a_reference_and_is_paired_with_original_events(story):
    _, state, model, _ = story
    bible = state['bible']; plan = NarrativeDirector(model).plan(bible)
    facts = ScriptWriter(model).prepare_facts(bible, plan)
    facts[0]['facts'][0]['text'] = 'MODEL_MAY_BE_WRONG_SENTINEL'
    model.calls.clear()
    writer = ScriptWriter(model)
    segments = writer.write_story(bible, plan, facts=facts)
    data = payload(model.calls[-1][1])
    assert 'MODEL_MAY_BE_WRONG_SENTINEL' in model.calls[-1][1]
    assert data['events'][0]['summary'] == bible['events'][0]['summary']
    assert '模型处理不等于内容正确' in model.calls[-1][1]
    writer.review_story(bible, plan, segments, facts=facts)
    data = payload(model.calls[-1][1], '资料：')
    assert data['fact_sheet'][0]['facts'][0]['text'] == 'MODEL_MAY_BE_WRONG_SENTINEL'
    assert data['events'][0]['summary'] == bible['events'][0]['summary']
    assert all(c[0] != 'story_digest' for c in model.calls)


def test_old_candidates_and_model_cache_stay_readable_but_are_not_reused_as_new_drafts(story):
    store, state, model, pipeline = story
    candidate_model(story)
    first = pipeline.generate(state['revision'])
    plan = first['candidates'][0]['plan']; bible = first['bible']
    facts = ScriptWriter(model, store).prepare_facts(bible, plan)
    for c in first['candidates']:
        c['script'].pop('source_material_basis')
        c['script'].update(fact_sheet=copy.deepcopy(facts), fact_sheet_basis=digest({'bible':bible,'plan':plan}))
    legacy = store.save(first, 'fixture_old_digest_candidates', first['revision'])
    kept = copy.deepcopy(legacy['candidates'])
    old_cache = {p:p.read_bytes() for p in (store.root/'cache').glob('*') if p.is_file()}
    model.calls.clear()
    result = pipeline.generate(legacy['revision'])
    assert result['candidate_set'] == ['K003','K004'] and result['candidates'][:2] == kept
    assert counts(model)['story_digest'] == counts(model)['direct'] == 0
    assert counts(model)['write_story'] == 2
    assert current_fact_sheet(bible, plan, kept[0]['script']) == facts
    assert store.read(legacy['revision']) == legacy
    assert all(p.read_bytes() == data for p, data in old_cache.items())


def test_program_input_version_changes_candidate_group_without_replanning(story, monkeypatch):
    import pipeline.fact_material as material_module
    _, state, model, pipeline = story
    candidate_model(story)
    first = pipeline.generate(state['revision']); kept = copy.deepcopy(first['candidates'])
    next_version = material_module.SOURCE_MATERIAL_VERSION + 1
    monkeypatch.setattr(material_module, 'SOURCE_MATERIAL_VERSION', next_version)
    model.calls.clear()
    second = pipeline.generate(first['revision'])
    assert second['candidate_set'] == ['K003','K004'] and second['candidates'][:2] == kept
    assert counts(model)['direct'] == counts(model)['story_digest'] == 0
    assert second['candidates'][2]['script']['source_material_basis']['version'] == next_version


def test_full_source_metadata_invalidates_basis_even_if_omitted_from_model_input(story):
    _, state, model, _ = story
    bible = state['bible']; plan = NarrativeDirector(model).plan(bible)
    changed = copy.deepcopy(bible); changed['events'][0]['importance'] += 1
    assert writing_material(bible, plan) == writing_material(changed, plan)
    assert source_material_basis(bible, plan) != source_material_basis(changed, plan)


def test_selected_human_lock_still_blocks_whole_generation(story):
    store, state, model, pipeline = story
    candidate_model(story)
    generated = pipeline.generate(state['revision'])
    selected = pipeline.select(generated['revision'], 'K001')
    selected['script']['segments'][0]['review_status'] = 'approved'
    locked = store.save(selected, 'fixture_human_lock', selected['revision'])
    model.calls.clear()
    with pytest.raises(StoryConflict): pipeline.generate(locked['revision'])
    assert model.calls == [] and store.read() == locked


def test_comparison_separates_legacy_sheet_and_raw_sources_with_same_plan(story):
    _, state, model, pipeline = story
    candidate_model(story)
    result = pipeline.generate(state['revision'])
    mixed = copy.deepcopy(result['candidates'])
    facts = ScriptWriter(model).prepare_facts(result['bible'], mixed[0]['plan'])
    mixed[0]['script'].pop('source_material_basis')
    mixed[0]['script'].update(fact_sheet=facts,
        fact_sheet_basis=digest({'bible':result['bible'],'plan':mixed[0]['plan']}))
    model.calls.clear()
    comparison = compare_candidates(model, result['bible'], mixed)
    assert comparison['status'] == 'compared'
    data = payload(model.calls[-1][1], '资料：')
    assert len(data['shared_material']) == 2
    assert len({c['material_id'] for c in data['candidates']}) == 2
    assert all(c['material_id'] in data['shared_material'] for c in data['candidates'])
