"""Source-aware text views; all stories and models here are offline fixtures."""
import copy
import json
import socket

import pytest

from pipeline.prompt_material import (context_catalog, supplemental_context, bind_context_usage,
                                     editor_material, fact_sheet_view, current_fact_sheet)
from pipeline.script_writer import ScriptWriter
from pipeline.story_bible import StoryError
from pipeline.fact_checker import FactChecker
from test_engagement_flow import story
from test_candidate_reuse import candidate_model, counts


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*a, **k):
        raise AssertionError('No real API or network in offline tests')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)


def payload(prompt, marker):
    return json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])[0]


def with_context(story, rows, decisions):
    store, state, model, pipeline = story
    state['bible']['prior_context'] = {'chapter':1, 'status':'no_history', 'background':rows}
    original = model.ask
    def ask(stage, prompt, images=None, fresh=False):
        raw = original(stage, prompt, images, fresh)
        if stage == 'story_digest':
            catalog = payload(prompt, '资料：')['context_catalog']
            raw['nodes'][0]['context_usage'] = [
                {'context_id':r['context_id'], 'usage':usage, 'reason':reason}
                for r,(usage,reason) in zip(catalog,decisions)]
        return raw
    model.ask = ask
    state = store.save(state, 'fixture_context', state['revision'])
    return state


def test_same_chapter_paraphrase_is_not_sent_again_or_as_character_forensics(story):
    candidate_model(story)
    state = with_context(story, [
        {'text':'DUPLICATE_PARAPHRASED_WORLD_AND_ACTION', 'event_ids':['E1','E2'], 'status':'confirmed'}],
        [('covered','本规划首节点的必要环境、拒绝和触发事实已完整涵盖')])
    state['bible']['characters'] = [{'character_id':'C1','name':'未知人物C1','name_status':'unknown',
        'description':'FORENSICS_DESCRIPTION','identity_evidence':[{'text':'FORENSICS_HISTORY'}],
        'name_claims':[], 'label_claims':[]}]
    story[0].save(state, 'fixture_characters', state['revision'])
    saved = story[3].generate(story[0].read()['revision'])
    assert counts(story[2])['story_digest'] == 1
    for stage,prompt,_,_ in story[2].calls:
        if stage in {'write_story','script_review'}:
            info = payload(prompt, '叙事依据：' if stage == 'write_story' else '资料：')
            assert info['prior_context'] == []
            assert info['reference_context'] == []
            assert info['nodes' if stage == 'write_story' else 'fact_sheet'] == fact_sheet_view(
                saved['candidates'][0]['script']['fact_sheet'],spoken=stage == 'write_story')
            assert not any(s in prompt for s in ('DUPLICATE_PARAPHRASED_WORLD_AND_ACTION',
                                                 'FORENSICS_HISTORY','FORENSICS_DESCRIPTION'))
    assert saved['bible']['prior_context']['background'][0]['text'] == 'DUPLICATE_PARAPHRASED_WORLD_AND_ACTION'
    assert saved['candidates'][0]['script']['fact_sheet'][0]['context_usage'] == []
    comparison = next(p for stage,p,_,_ in story[2].calls if stage == 'candidate_compare')
    assert len(payload(comparison,'资料：')['shared_material']) == 1
    assert 'DUPLICATE_PARAPHRASED_WORLD_AND_ACTION' not in comparison and 'FORENSICS_HISTORY' not in comparison


def test_first_chapter_without_history_keeps_context_empty(story):
    candidate_model(story)
    saved = story[3].generate(story[1]['revision'])
    script = saved['candidates'][0]['script']
    assert context_catalog(saved['bible']) == []
    assert editor_material(saved['bible'],saved['candidates'][0]['plan'],script['fact_sheet'])['prior_context'] == []
    assert 'context_usage' not in script['fact_sheet'][0]


def test_first_chapter_external_reference_does_not_become_a_fabricated_recap(story):
    bible = copy.deepcopy(story[1]['bible'])
    bible['prior_context'] = {'chapter':1,'status':'no_history',
        'series_notes':{'revision':'ref-v1','characters':'称呼尚未归属，不能提前确认身份'}}
    plan = {'beats':[{'event_id':'E1','event_ids':['E1']}]}
    nodes = [{'event_id':'E1','facts':[{'text':'当前事实','event_ids':['E1']}]}]
    material = editor_material(bible,plan,nodes)
    assert material['prior_context'] == []
    assert material['reference_context'][0]['status'] == 'unverified_reference'
    assert material['reference_context'][0]['selection'] == 'uncertain'


@pytest.mark.parametrize('conflict', [
    {'chapter':9}, {'source_revision':'different-version'}, {'source_version':'different-version'},
    {'sha256':'different-original'}, {'history_origin':{'chapter':9}}, {'certainty':'model_guess'},
    {'source_panel_ids':['different-panel']}])
def test_event_id_collision_cannot_discard_different_source_or_uncertainty(story, conflict):
    state = with_context(story, [{'text':'不能静默丢弃的规则','status':'confirmed','event_ids':['E1'], **conflict}],
                         [('covered','模型声称重复但来源有冲突')])
    nodes = [{'event_id':'E1','facts':[{'text':'本话事实','event_ids':['E1']}],
              'context_usage':[{'context_id':'X0001','usage':'covered','reason':'模型声称重复'}]}]
    rows = supplemental_context(state['bible'],nodes)
    assert rows[0]['selection'] == 'uncertain' and rows[0]['text'] == '不能静默丢弃的规则'
    assert all(rows[0]['source'][k] == v for k,v in conflict.items())


def test_matching_text_without_provenance_is_unresolved_not_keyword_deduplicated(story):
    bible = copy.deepcopy(story[1]['bible'])
    bible['prior_context'] = {'background':[{'text':'桥头禁行','event_ids':['missing']}],
                              'series_notes':{'revision':'ref-v7','background':'桥头禁行','characters':''}}
    nodes = [{'event_id':'E1','facts':[{'text':'桥头禁行','event_ids':['E1']}]}]
    rows = supplemental_context(bible,nodes)
    assert len(rows) == 2 and all(r['selection'] == 'uncertain' for r in rows)
    assert rows[1]['status'] == 'unverified_reference' and rows[1]['source']['revision'] == 'ref-v7'
    assert rows[1]['source']['unit_ids'] == ['background:0:4']


def test_needed_old_rule_character_state_and_unknown_limit_keep_sources(story):
    bible = copy.deepcopy(story[1]['bible'])
    old = copy.deepcopy(bible['events'][0]); old.update(event_id='H00001_E1', historical=True,
        history_origin={'chapter':1,'revision':'old-v2','event_id':'E1'})
    bible['events'].append(old)
    bible['prior_context'] = {'chapter':2,'status':'available','previous_chapter':{'chapter':1,'revision':'old-v2'},
        'background':[{'text':'更早建立的桥索承重限制','event_ids':[old['event_id']]}],
        'character_states':[{'character_id':'C1','facts':[{'text':'仍需送药，目标未完成','event_ids':[old['event_id']]}]}],
        'series_notes':{'revision':'ref-v1','unknowns':[{'text':'闻溪尚未对应人物，不可确认关系',
                                                       'unit_ids':['R0007'],'status':'confirmed'}]}}
    nodes = [{'event_id':'E1','facts':[{'text':'本话选择','event_ids':['E1']}], 'context_usage':[
        {'context_id':r['context_id'],'usage':'supplement','reason':'理解本话限制和选择必需，简表未重述'}
        for r in context_catalog(bible)]}]
    rows = supplemental_context(bible,nodes)
    assert len(rows) == 3 and rows[0]['status'] == 'reviewed_history'
    assert rows[0]['source']['events'][0]['origin']['revision'] == 'old-v2'
    assert rows[0]['source']['events'][0]['panels'][0]['sha256'] == bible['panels'][0]['sha256']
    assert rows[1]['character_id'] == 'C1'
    assert rows[2]['status'] == 'unverified_reference' and rows[2]['source']['unit_ids'] == ['R0007']
    assert rows[2]['source']['reported_status'] == 'confirmed'  # Reference cannot self-promote.


def test_same_event_additional_necessary_fact_is_not_automatically_discarded(story):
    bible = copy.deepcopy(story[1]['bible'])
    bible['prior_context'] = {'background':[{'text':'同事件尚缺的必要危险限制','event_ids':['E1'],'status':'confirmed'}]}
    nodes = [{'event_id':'E1','facts':[{'text':'人物只执行了行动','event_ids':['E1']}],
              'context_usage':[{'context_id':'X0001','usage':'supplement','reason':'简表未保留危险限制'}]}]
    assert supplemental_context(bible,nodes)[0]['selection'] == 'supplement'


def test_irrelevant_selection_is_recorded_locally_without_resending_text(story):
    state = with_context(story,[{'text':'无关年代的建港史','event_ids':['E1'],'status':'confirmed'}],
                         [('irrelevant','不影响当前选择、因果、身份或限制')])
    nodes = ScriptWriter(story[2],story[0]).prepare_facts(state['bible'], {'style':'faithful', 'beats':[
        {'event_id':eid,'event_ids':ids} for eid,ids in [('E1',['E1','E2']),('E3',['E3','E4']),('E5',['E5'])]]})
    assert nodes[0]['context_usage'][0]['usage'] == 'irrelevant'
    assert supplemental_context(state['bible'],nodes) == []
    assert state['bible']['prior_context']['background'][0]['text'] == '无关年代的建港史'


@pytest.mark.parametrize('mutation',['unknown_id','duplicate','invalid_usage'])
def test_invalid_reference_selection_is_rejected(story,mutation):
    bible = copy.deepcopy(story[1]['bible'])
    bible['prior_context'] = {'background':[{'text':'规则','event_ids':['E1']}]}
    row = {'context_id':'X0001','usage':'supplement','reason':'必要'}
    decisions = [row]
    if mutation == 'unknown_id': row['context_id'] = 'X9000'
    if mutation == 'duplicate': decisions.append(copy.deepcopy(row))
    if mutation == 'invalid_usage': row['usage'] = 'confirmed_fact'
    with pytest.raises(StoryError):
        bind_context_usage(bible,[{'event_id':'E1','facts':[], 'context_usage':decisions}])


def test_text_projection_keeps_reveal_order_and_never_changes_strict_check(story):
    candidate_model(story)
    saved = story[3].generate(story[1]['revision'])
    script = saved['candidates'][0]['script']; plan = saved['candidates'][0]['plan']
    bible = saved['bible']; segment = script['segments'][1]
    bible['characters'] = [{'character_id':'C1','name':'闻溪','name_status':'confirmed','name_claims':[
        {'name':'闻溪','source_panel_id':'5','reveal_panel_id':'5','reveal_order':4,
         'attribution':{'source_panel_ids':['5'],'basis':'完整原图归属取证'}}], 'label_claims':[]}]
    before = copy.deepcopy(bible)
    checker = FactChecker(story[0],story[2])
    checker.check(bible,segment)
    first = story[2].calls[-1]
    material = editor_material(bible,plan,script['fact_sheet'])
    assert material['identity_availability'][0]['not_yet_revealed'][0]['reveal_order'] == 4
    assert '完整原图归属取证' not in json.dumps(material,ensure_ascii=False)
    checker.check(bible,segment)
    assert story[2].calls[-1][1:3] == first[1:3] and first[2]
    assert bible == before
    assert {e['event_id'] for e in material['plan']['beats']} == {b['event_id'] for b in plan['beats']}


def test_context_and_plan_dependencies_invalidate_without_overwriting_old_candidates(story):
    candidate_model(story)
    state = with_context(story,[{'text':'必要但待核验的旧规则','event_ids':[]}],[('uncertain','来源不足')])
    saved = story[3].generate(state['revision'])
    candidate = saved['candidates'][0]; writer = ScriptWriter(story[2],story[0])
    before = copy.deepcopy(saved); facts = writer.prepare_facts(saved['bible'],candidate['plan'])
    assert counts(story[2])['story_digest'] == 1
    changed = copy.deepcopy(saved['bible']); changed['prior_context']['background'][0]['text'] += '修订'
    writer.prepare_facts(changed,candidate['plan'])
    assert counts(story[2])['story_digest'] == 2
    assert current_fact_sheet(changed,candidate['plan'],candidate['script']) is None
    changed_plan = copy.deepcopy(candidate['plan']); changed_plan['beats'][0]['title'] += '新组织'
    writer.prepare_facts(saved['bible'],changed_plan)
    assert counts(story[2])['story_digest'] == 3
    assert writer.prepare_facts(saved['bible'],candidate['plan']) == facts
    assert story[0].read(saved['revision']) == before


def test_changed_plan_uses_new_candidate_group_and_preserves_previous_versions(story):
    candidate_model(story)
    first = story[3].generate(story[1]['revision'])
    kept = copy.deepcopy(first['candidates'])
    original = story[2].ask
    def ask(stage,prompt,images=None,fresh=False):
        raw = original(stage,prompt,images,fresh)
        if stage == 'direct':
            raw['beats'][0]['title'] = '新规划：先理解拒绝，再听触发消息'
        return raw
    story[2].ask = ask
    changed = story[3].generate(first['revision'],max_events=3)
    assert changed['candidates'][:2] == kept
    assert changed['candidate_set'] == ['K003','K004']
    assert counts(story[2])['story_digest'] == 2
    assert changed['candidates'][2]['plan']['beats'][0]['title'] != kept[0]['plan']['beats'][0]['title']
    assert story[0].read(first['revision']) == first


def test_audit_counts_three_inputs_separately_and_does_not_write_project(story):
    from tools.audit_prompt_inputs import audit
    from pipeline.story_pipeline import StoryPipeline
    store,state,model,pipeline = story
    candidate_model(story)
    state = pipeline.generate(state['revision'])
    state = pipeline.select(state['revision'],'K001')
    before = {p.relative_to(store.root):p.read_bytes() for p in store.root.rglob('*') if p.is_file()}
    calls = len(model.calls)
    result = audit(store.root)
    assert result['writing']['attached_images'] == result['whole_review']['attached_images'] == 0
    check = result['original_image_verification']
    assert check['calls_if_all_checked'] == 3 and check['attached_images_sum_with_repeated_use'] > 0
    for row in [result['writing'],result['whole_review'],*check['per_node']]:
        assert sum(row['components_chars'].values()) + row['instructions_keys_and_layout_chars'] == row['prompt_chars']
    assert len(model.calls) == calls
    assert {p.relative_to(store.root):p.read_bytes() for p in store.root.rglob('*') if p.is_file()} == before
