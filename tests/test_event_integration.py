"""Offline program contracts; hand-authored responses do not prove semantics."""
import copy
import json
import socket
import subprocess
from pathlib import Path

import pytest

import pipeline.event_integration as integration
from pipeline.event_integration import (chapter_input, fragments_for, consume, integrate_chapter, resolve_fragment,
                                        repair_integration_result, normalize_integration_result)
from pipeline.story_bible import StoryBible, StoryError, StoryConflict
from pipeline.story_model import StoryModel, validated_result
from pipeline.story_pipeline import StoryPipeline
from pipeline.chapter_memory import prompt_context
from pipeline.prompt_material import context_catalog
from integration_fixture import integration_response
from test_engagement_flow import story
from test_story_reuse import grounding_model as candidate_model, counts


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*a, **k):
        raise AssertionError('Offline: no model, TTS or network')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)


@pytest.fixture
def analysis():
    panels = [{'source_panel_id': str(i), 'order': i-1, 'sha256': f'image-{i}',
               'dialogue': [], 'visible_text': [], 'uncertainties': [], 'scene': ''}
              for i in range(1, 6)]
    panels[0]['dialogue'] = [{'speaker_id': None, 'text': '过桥前必须固定绳索。'}]
    panels[0]['visible_text'] = ['沙流倒转时，桥只能承受一人。']
    panels[2]['uncertainties'] = ['喊话者身份未知，未显示落水结果。']
    summaries = ['沙流倒转，过桥受限', '守桥人拒绝放行', '旁人拉索并喊话，守桥人停下',
                 '回忆中桥曾断裂', '场景转到屋内，有人寻找药盒']
    events = [{'event_id': f'E{i}', 'order': i-1, 'summary': s, 'character_ids': [f'C{i}'],
               'actions': [s], 'source_panel_ids': [str(i)], 'importance': 1 if i == 3 else 3,
               'role': 'setup', 'evidence': [{'kind': 'visual', 'detail': s, 'source_panel_id': str(i)}],
               'causes': []} for i, s in enumerate(summaries, 1)]
    events[2]['causes'] = [{'event_id': 'E2', 'evidence': '明确要求先固定绳索', 'source_panel_id': '3'}]
    bible = {'panels': panels, 'events': events, 'characters': [], 'relationships': [], 'analysis_version': 6}
    return chapter_input(bible, 'source-v1')


def event(fragments, ids, summary='连贯事件', kind='current_action', role='action', status=None):
    chosen = [f for f in fragments if f['fragment_id'] in ids]
    return {'summary': summary, 'fragment_ids': [f['fragment_id'] for f in chosen], 'kind': kind,
            'time': {'original_time': '当年' if kind == 'recollection' else '未知', 'status': 'unknown'},
            'facets': [{'text': summary, 'role': role, 'fragment_ids': [f['fragment_id']],
                        'status': status or f['status']} for f in chosen]}


def by_panels(fragments, *pids):
    return [f['fragment_id'] for f in fragments if set(f['source_panel_ids']) <= set(pids)]


def merged_response(analysis):
    f = fragments_for(analysis)
    return {'events': [event(f, by_panels(f, '1'), '沙流倒转限制桥上通行', 'world_rule', 'setting'),
                       event(f, by_panels(f, '2', '3'), '守桥人拒绝放行，旁人拉索喊话后守桥人停下'),
                       event(f, by_panels(f, '4'), '回忆里桥曾断裂', 'recollection', 'premise'),
                       event(f, by_panels(f, '5'), '屋内的人寻找药盒')]}


def test_merge_split_preserve_rules_supporting_action_order_sources_and_original_time(analysis):
    before = copy.deepcopy(analysis)
    f = fragments_for(analysis)
    raw = merged_response(analysis)
    result = consume(raw, analysis, f)
    assert len(result) == 4 and result[1]['source_event_ids'] == ['E2', 'E3']
    assert result[1]['source_panel_ids'] == ['2', '3'] and result[1]['character_ids'] == ['C2', 'C3']
    assert result[1]['status'] == 'uncertain'
    assert result[0]['kind'] == 'world_rule' and result[0]['facets'][0]['role'] == 'setting'
    assert result[0]['dialogue'] == analysis['panels'][0]['dialogue']
    assert result[0]['evidence'] == analysis['events'][0]['evidence']
    assert result[2]['time']['original_time'] == '当年' and result[2]['reveal_panel_id'] == '4'
    assert [e['order'] for e in result] == [0, 1, 2, 3]
    assert not any(e['causes'] for e in result)  # Internal explicit cause is retained in original analysis.
    assert result[1]['internal_causes'][0]['event_id'] == 'E2'
    # One extracted fragment can contain multiple distinct processes.
    raw['events'][-1:] = [event(f, by_panels(f, '5'), '屋内的人寻找药盒'),
                          event(f, by_panels(f, '5'), '仍未找到药盒', kind='unresolved', role='unknown', status='unknown')]
    split = consume(raw, analysis, f)
    assert len(split) == 5 and len({e['event_id'] for e in split}) == 5
    assert split[-1]['source_event_ids'] == ['E5'] and split[-1]['source_panel_ids'] == ['5']
    assert analysis == before


@pytest.mark.parametrize('change', ['missing', 'unknown_ref', 'wrong_order', 'crossed_order', 'promoted', 'invented_cause', 'empty_facets'])
def test_invalid_coverage_references_order_and_uncertainty_rejected(analysis, change):
    f = fragments_for(analysis)
    raw = merged_response(analysis)
    if change == 'missing':
        raw['events'].pop(0)
    elif change == 'unknown_ref':
        raw['events'][0]['fragment_ids'] = ['F99999']
    elif change == 'wrong_order':
        raw['events'].reverse()
    elif change == 'crossed_order':
        raw['events'] = [event(f, by_panels(f, '1', '3')), event(f, by_panels(f, '2', '4', '5'))]
    elif change == 'promoted':
        for facet in raw['events'][1]['facets']:
            facet['status'] = 'supported'
    elif change == 'invented_cause':
        raw['events'][0]['causes'] = [{'event_id': 'E5', 'evidence': '猜测'}]
    else:
        raw['events'][0]['facets'] = []
    with pytest.raises(StoryError):
        consume(raw, analysis, f)


def test_complete_panel_information_without_extracted_event_cannot_disappear(analysis):
    analysis['events'] = analysis['events'][:-1]
    analysis['panels'][-1]['dialogue'] = [{'speaker_id': None, 'text': '这层房间也不能点火。'}]
    f = fragments_for(analysis)
    raw = merged_response(analysis)
    result = consume(raw, analysis, f)
    assert result[-1]['source_event_ids'] == [] and result[-1]['dialogue'][0]['text'] == '这层房间也不能点火。'
    raw['events'].pop()
    with pytest.raises(StoryError, match='遗漏'):
        consume(raw, analysis, f)


@pytest.mark.parametrize('status', ['unknown', 'uncertain', 'inferred', 'needs_review', 'unverified_reference'])
def test_unknown_source_status_cannot_be_promoted(analysis, status):
    analysis['events'][0]['certainty'] = status
    f = fragments_for(analysis)
    raw = merged_response(analysis)
    for facet in raw['events'][0]['facets']:
        facet['status'] = 'supported'
    with pytest.raises(StoryError, match='不能升级'):
        consume(raw, analysis, f)


def test_cause_only_inherits_explicit_target_and_ambiguous_split_stays_pending(analysis):
    f = fragments_for(analysis)
    rows = [event(f, by_panels(f, str(i))) for i in range(1, 6)]
    result = consume({'events': rows}, analysis, f)
    assert result[2]['causes'] == analysis['events'][2]['causes']
    assert not result[1]['causes']
    rows.insert(2, event(f, by_panels(f, '2'), '同一原片段的另一过程'))
    split = consume({'events': rows}, analysis, f)
    assert not split[3]['causes'] and split[3]['pending_causes'][0]['status'] == 'uncertain'


def test_input_uses_full_chapter_not_old_summaries_or_history(story):
    _, state, _, _ = story
    bible = copy.deepcopy(state['bible'])
    bible['prior_context'] = {'series_notes': {'background': 'OLD_BACKGROUND_SENTINEL', 'characters': 'OLD_CHAR_SENTINEL'}}
    bible['events'][0]['causes'] = [{'event_id': 'H1', 'evidence': 'HISTORY_PREMISE_SENTINEL'}]
    bible['events'].append({'historical': True, 'event_id': 'H1', 'summary': 'HISTORY_EVENT_SENTINEL'})
    bible['panels'].append({'historical': True, 'source_panel_id': 'H1', 'dialogue': 'HISTORY_TEXT_SENTINEL'})
    data = chapter_input(bible, state['revision'])
    serialized = json.dumps(data)
    assert all(s not in serialized for s in ('OLD_BACKGROUND_SENTINEL', 'OLD_CHAR_SENTINEL', 'HISTORY_PREMISE_SENTINEL', 'HISTORY_EVENT_SENTINEL', 'HISTORY_TEXT_SENTINEL'))
    assert len(data['events']) == 5 and len(data['panels']) == 5
    assert data['panels'][0]['dialogue'] == bible['panels'][0]['dialogue']
    assert data['pending_history'] == [{'event_id': 'E1', 'status': 'needs_history'}]
    f = fragments_for(data)
    raw = {'events': [event(f, by_panels(f, str(i))) for i in range(1, 6)]}
    result = consume(raw, data, f)
    assert result[0]['pending_causes'][0]['status'] == 'needs_history'
    assert result[0]['status'] == 'uncertain' and not result[0]['causes']


def test_identical_titles_do_not_guess_or_conflate_source_links(analysis):
    for source in analysis['events']:
        source['summary'] = '完全相同的标题和关键词'
    f = fragments_for(analysis)
    rows = [event(f, by_panels(f, str(i)), '完全相同的标题和关键词') for i in range(1, 6)]
    result = consume({'events': rows}, analysis, f)
    assert [e['source_event_ids'] for e in result] == [[f'E{i}'] for i in range(1, 6)]
    rows[0]['fragment_ids'] = ['完全相同的标题和关键词']
    with pytest.raises(StoryError, match='引用无效'):
        consume({'events': rows}, analysis, f)


def test_non_story_boundary_is_inherited_cannot_be_invented_or_merged_away(analysis):
    f = fragments_for(analysis)
    rows = [event(f, by_panels(f, str(i))) for i in range(1, 6)]
    rows[0]['kind'] = 'non_story'
    with pytest.raises(StoryError, match='不能自行排除'):
        consume({'events': rows}, analysis, f)
    rows[0]['kind'] = 'world_rule'
    analysis['events'][-1].update(narrative_eligible=False, exclusion_reason='片尾互动题')
    result = consume({'events': rows}, analysis, f)
    assert result[-1]['narrative_eligible'] is False and '片尾互动题' in result[-1]['exclusion_reason']
    rows[-2:] = [event(f, by_panels(f, '4', '5'))]
    with pytest.raises(StoryError, match='边界'):
        consume({'events': rows}, analysis, f)


class Model:
    def __init__(self):
        self.calls = []
        self.version = 'model-v1'
        self.bad = False

    def cache_identity(self, stage):
        return {'model': self.version, 'stage': stage}

    def ask(self, stage, prompt, images=None, fresh=False):
        assert stage == 'event_integration' and not images
        self.calls.append((prompt, fresh))
        if 'Event Integration Field Repair v1' in prompt:
            raise StoryError('事件整合遗漏本话信息片段')
        raw = integration_response(prompt)
        if self.bad:
            raw['events'].pop()
        return raw


@pytest.mark.parametrize('change', ['input', 'image_hash', 'source_version', 'prompt', 'model', 'schema'])
def test_cache_binds_actual_input_source_version_prompt_model_and_output_contract(story, monkeypatch, change):
    store, state, _, _ = story
    model = Model()
    bible = state['bible']
    before = copy.deepcopy(bible)
    integrated = integrate_chapter(store, model, bible, 'analysis-v1')
    assert integrate_chapter(store, model, integrated, 'analysis-v1') == integrated and len(model.calls) == 1
    old = {p: p.read_bytes() for p in (store.root/'cache').glob('*.json')}
    changed = copy.deepcopy(integrated)
    revision = 'analysis-v1'
    if change == 'input':
        changed['analysis_events'][0]['summary'] += '本话真实输入更新'
    elif change == 'image_hash':
        changed['panels'][0]['sha256'] = 'different-image'
    elif change == 'source_version':
        revision = 'analysis-v2'
    elif change == 'prompt':
        monkeypatch.setattr(integration, 'INSTRUCTIONS', '提示词变更\n' + integration.INSTRUCTIONS)
    elif change == 'model':
        model.version = 'model-v2'
    else:
        monkeypatch.setattr(integration, 'OUTPUT_CONTRACT', {**integration.OUTPUT_CONTRACT, 'version': integration.VERSION + 1})
    integrate_chapter(store, model, changed, revision)
    assert len(model.calls) == 2 and all(p.read_bytes() == content for p, content in old.items())
    assert bible == before and integrated['analysis_events'] == bible['events']


def test_old_reference_cache_is_not_event_integration_and_missing_event_stays_pending(story):
    store, state, _, _ = story
    model = Model()
    validated_result(store, model, 'context_digest', {}, lambda: {'background': ['旧摘要'], 'characters': []})
    model.bad = True
    with pytest.raises(StoryError, match='遗漏'):
        integrate_chapter(store, model, state['bible'], 'source')
    assert len(model.calls) == 1 and [fresh for _, fresh in model.calls] == [False]
    model.bad = False
    result = integrate_chapter(store, model, state['bible'], 'source')
    assert len(model.calls) == 2 and result['event_integration']['source_revision'] == 'source'
    assert store.read() == state


def test_separate_first_step_saves_no_script_keeps_old_versions_and_reuses_on_generate(story):
    store, state, model, pipeline = story
    candidate_model(story)
    before = {p: p.read_bytes() for folder in ('evidence', 'revisions') for p in (store.root/folder).glob('*')}
    integrated = pipeline.integrate(state['revision'])
    assert integrated['stage'] == 'integrated' and integrated['script'] is None and integrated['plan'] is None
    assert counts(model) == {'event_integration': 1}
    assert store.read(state['revision']) == state
    assert all(p.read_bytes() == data for p, data in before.items())
    assert pipeline.integrate(integrated['revision']) == integrated
    generated = pipeline.generate(integrated['revision'])
    assert counts(model)['event_integration'] == 1
    assert generated['bible']['events'] == integrated['bible']['events']
    stages = [c[0] for c in model.calls]
    assert stages.index('event_integration') < stages.index('direct') < stages.index('write_story')
    assert 'story_digest' not in stages
    assert 'context_digest' not in stages and 'analyze' not in stages and not any(c[2] for c in model.calls)


def test_same_chapter_notes_never_resent_in_planning_facts_writing_review_or_comparison(story):
    store, state, model, pipeline = story
    candidate_model(story)
    old = {'series_notes': {'background': 'OLD_BACKGROUND_SENTINEL'*300, 'characters': 'OLD_CHAR_SENTINEL'*300},
           'background': [{'text': 'SAME_CHAPTER_JSON_SENTINEL', 'event_ids': ['E1']}]}
    state['bible']['prior_context'] = old
    state = store.save(state, 'fixture_old_reference', state['revision'])
    saved = pipeline.generate(state['revision'])
    for stage, prompt, images, _ in model.calls:
        assert not any(s in prompt for s in ('OLD_BACKGROUND_SENTINEL', 'OLD_CHAR_SENTINEL', 'SAME_CHAPTER_JSON_SENTINEL'))
        assert not images and stage not in {'context_digest', 'analyze'}
    assert saved['bible']['prior_context'] == old and context_catalog(saved['bible']) == []


def test_prior_chapter_remains_available_only_after_integration(story):
    store, state, _, _ = story
    state['bible']['prior_context'] = {'previous_chapter': {'chapter': 1, 'revision': 'old'},
                                     'background': [{'text': 'PREVIOUS_CHAPTER_RULE', 'event_ids': ['H1']}],
                                     'series_notes': {'background': 'OLD_SAME_CHAPTER_SUMMARY'}}
    integrated = integrate_chapter(store, Model(), state['bible'], 'source')
    assert prompt_context(integrated)['background'][0]['text'] == 'PREVIOUS_CHAPTER_RULE'
    assert 'series_notes' not in prompt_context(integrated)


def test_exact_source_lookup_survives_history_offsets_and_derived_event_replacement(story):
    store, state, _, _ = story
    bible = copy.deepcopy(state['bible'])
    bible['panels'].insert(0, {'historical': True, 'source_panel_id': 'H1', 'dialogue': []})
    bible['events'].insert(0, {'historical': True, 'event_id': 'H1', 'source_panel_ids': ['H1']})
    bible['panels'][1]['scene'] = '当前桥头'
    integrated = integrate_chapter(store, Model(), bible, 'source')
    for fragment in integrated['event_integration']['fragment_index']:
        original = resolve_fragment(bible, fragment)
        assert resolve_fragment(integrated, fragment) == original
        assert 'H1' not in json.dumps(original, ensure_ascii=False)
    scene = next(f for f in integrated['event_integration']['fragment_index'] if f['path'].endswith('/scene'))
    assert resolve_fragment(bible, scene) == '当前桥头'


def test_changed_actual_analysis_pins_new_version_and_keeps_previous_source(story):
    store, state, model, pipeline = story
    first = pipeline.integrate(state['revision'])
    changed = copy.deepcopy(first)
    changed['bible']['analysis_events'][0]['summary'] += '；本话来源资料修正'
    source = store.save(changed, 'fixture_corrected_analysis', first['revision'])
    updated = pipeline.integrate(source['revision'])
    assert updated['bible']['event_integration']['source_revision'] == source['revision']
    assert first['bible']['event_integration']['source_revision'] == state['revision']
    assert store.read(first['revision']) == first and store.read(state['revision']) == state
    assert pipeline.integrate(updated['revision']) == updated


def test_manual_lock_and_stale_revision_block_integration_without_model_calls(story):
    store, state, _, _ = story
    model = Model()
    pipeline = StoryPipeline(store, model)
    state['script'] = {'segments': [{'review_status': 'approved', 'text': '人工审核原文'}]}
    locked = store.save(state, 'fixture_lock', state['revision'])
    with pytest.raises(StoryConflict, match='已审核'):
        pipeline.integrate(locked['revision'])
    with pytest.raises(StoryConflict, match='已变化'):
        pipeline.integrate(state['revision'])
    with pytest.raises(StoryConflict, match='已审核'):
        pipeline.generate(locked['revision'])
    assert not model.calls and store.read() == locked


def test_real_transport_role_is_text_only_and_keeps_old_traces(story):
    store, state, _, _ = story
    class Transport:
        kind = 'test'
        host = 'offline.invalid'
        calls = []
        def chat(self, **request):
            self.calls.append(request)
            return json.dumps(integration_response(request['prompt']), ensure_ascii=False)
    transport = Transport()
    model = StoryModel(store, transport)
    result = integrate_chapter(store, model, state['bible'], 'source')
    assert len(transport.calls) == 1 and transport.calls[0]['images'] == []
    assert transport.calls[0]['system'] == model.cache_identity('event_integration')['system']
    previous = {p: p.read_bytes() for p in (store.root/'calls').glob('*.json')}
    assert integrate_chapter(store, model, result, 'source') == result
    assert len(transport.calls) == 1 and all(p.read_bytes() == data for p, data in previous.items())


def test_review_page_displays_integration_time_unknowns_and_pending_history_safely(story):
    store, state, _, pipeline = story
    saved = pipeline.integrate(state['revision'])
    saved['history'] = store.history()
    first = saved['bible']['events'][0]
    first.update(kind='historical_explanation', status='unknown',
                 time={'original_time': '原作时间未知', 'status': 'unknown'},
                 pending_causes=[{'status': 'needs_history'}])
    first['facets'][0].update(text='<img src=x onerror=alert(1)>', role='setting', status='conflicting')
    code = r'''
const fs=require('fs'), vm=require('vm'), assert=require('assert');
const state=JSON.parse(fs.readFileSync(0,'utf8'));
const node={value:'',textContent:''}, host={innerHTML:'',querySelector:()=>node,querySelectorAll:()=>[]};
const window={};
vm.runInNewContext(fs.readFileSync('static/js/story_review.js','utf8'),{
  window,document:{contains:()=>true},clearTimeout,setTimeout,
  fetch:async(url,options)=>{assert(!options);return {ok:true,json:async()=>state};}
});
(async()=>{
  await window.StoryReview.mount(host,'fixture');
  assert.equal(node.textContent,'');
  for(const text of ['本话信息整合与事件还原','历史说明','原作时间未知','待补历史','存在矛盾','必要设定及作用'])
    assert(host.innerHTML.includes(text),text);
  assert(host.innerHTML.includes('&lt;img src=x onerror=alert(1)&gt;'));
  assert(!host.innerHTML.includes('<img src=x onerror=alert(1)>'));
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    result = subprocess.run(['node', '-e', code], input=json.dumps(saved), text=True,
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_targeted_integration_repair_retains_good_rows_and_only_fills_bad_summary(analysis):
    f = fragments_for(analysis)
    original = merged_response(analysis)
    broken = copy.deepcopy(original)
    broken['events'][0]['summary'] = ''
    preserved = copy.deepcopy(broken['events'][1:])
    class Repair:
        def ask(self, stage, prompt, images=None, fresh=False):
            assert stage == 'event_integration' and not images and fresh
            payload = json.JSONDecoder().raw_decode(prompt.split('资料：', 1)[1])[0]
            assert len(payload['issues']) == 1 and payload['issues'][0]['path'] == '/events/0/summary'
            return {'patches': [{'issue_id': payload['issues'][0]['issue_id'],
                                 'value': original['events'][0]['summary']}], 'unresolved': []}
    repaired = repair_integration_result(Repair(), broken, analysis, f)
    assert repaired == consume(original, analysis, f)
    assert broken['events'][1:] == preserved


@pytest.mark.parametrize('attack', ['correct_row', 'change_sources', 'duplicate_missing'])
def test_targeted_repair_cannot_touch_correct_events_or_change_sources(analysis, attack):
    f = fragments_for(analysis)
    original = merged_response(analysis)
    broken = copy.deepcopy(original)
    broken['events'][1]['summary'] = ''
    patch = {'replacements': [{'index': 1, 'event': copy.deepcopy(original['events'][1])}], 'additions': []}
    if attack == 'correct_row':
        patch['replacements'].append({'index': 0, 'event': original['events'][0]})
    elif attack == 'change_sources':
        patch['replacements'][0]['event']['fragment_ids'].pop()
    elif attack == 'duplicate_missing':
        patch['additions'] = [original['events'][0]]
    class Repair:
        def ask(self, *args, **kwargs):
            return copy.deepcopy(patch)
    before = copy.deepcopy(broken)
    with pytest.raises(StoryError):
        repair_integration_result(Repair(), broken, analysis, f)
    assert broken == before


class NoRepairModel:
    def ask(self, *args, **kwargs):
        raise AssertionError('Deterministic validation must not call a model')


def test_facet_subset_keeps_every_event_source_without_repeated_coverage_or_model_call(analysis):
    fragments = fragments_for(analysis)
    raw = merged_response(analysis)
    # The setting text already expresses this rule. Its original visible-text
    # ID remains at event level; no duplicate facet is required for that ID.
    raw['events'][0]['facets'] = raw['events'][0]['facets'][:1]
    before = copy.deepcopy(raw)
    result = repair_integration_result(NoRepairModel(), raw, analysis, fragments)
    assert result[0]['fragment_ids'] == before['events'][0]['fragment_ids']
    assert result[0]['facets'] == before['events'][0]['facets']
    assert {f for e in result for f in e['fragment_ids']} == {f['fragment_id'] for f in fragments}
    assert result[0]['dialogue'] == analysis['panels'][0]['dialogue'] and raw == before


def test_exact_reference_format_is_repaired_locally_with_audit_and_no_input_mutation(analysis):
    fragments = fragments_for(analysis)
    raw = merged_response(analysis)
    raw['events'][0]['fragment_ids'].reverse()
    raw['events'][0]['fragment_ids'].append(raw['events'][0]['fragment_ids'][0])
    raw['events'][0]['facets'][0]['fragment_ids'] = raw['events'][0]['facets'][0]['fragment_ids'][0]
    before = copy.deepcopy(raw)
    audit = []
    result = repair_integration_result(NoRepairModel(), raw, analysis, fragments, local_repairs=audit)
    assert result == consume(merged_response(analysis), analysis, fragments)
    assert len(audit) == 2 and {c['reason'] for c in audit} == {'exact_reference_format'}
    assert raw == before
    normalized, _ = normalize_integration_result(raw, fragments)
    assert normalize_integration_result(normalized, fragments) == (normalized, [])


@pytest.mark.parametrize('note', ['人物所处具体场景未知', '无法确认场景位置', '快看为平台水印，非剧情文字',
                                 '画面左下角为水印', 'SPAN、ORIGINAL为品牌字样，与人物身份无关'])
def test_legacy_uncertainty_annotation_status_is_lowered_locally_without_rewriting_text(analysis, note):
    analysis['panels'][0]['uncertainties'] = [note]
    fragments = fragments_for(analysis)
    raw = merged_response(analysis)
    facet = next(f for f in raw['events'][0]['facets'] if f['status'] == 'uncertain')
    facet.update(text=note, role='setting', status='supported')
    before = copy.deepcopy(raw)
    audit = []
    result = repair_integration_result(NoRepairModel(), raw, analysis, fragments, local_repairs=audit)
    actual = next(f for f in result[0]['facets'] if f['text'] == note)
    assert actual == {**facet, 'status': 'uncertain'} and result[0]['status'] == 'uncertain'
    assert len(audit) == 1 and audit[0]['reason'] == 'source_certainty_ceiling'
    assert raw == before and analysis['panels'][0]['uncertainties'] == [note]


@pytest.mark.parametrize('status', ['unknown', 'uncertain', 'conflicting', 'needs_history'])
def test_local_status_repair_preserves_source_restrictions_and_never_promotes(status, analysis):
    analysis['events'][0]['status'] = status
    fragments = fragments_for(analysis)
    raw = merged_response(analysis)
    raw['events'][0]['facets'][0]['status'] = 'supported'
    normalized, audit = normalize_integration_result(raw, fragments)
    assert normalized['events'][0]['facets'][0]['status'] == status
    assert len(audit) == 1
    for facet in raw['events'][0]['facets']:
        facet['status'] = 'unknown'
    normalized, audit = normalize_integration_result(raw, fragments)
    assert all(f['status'] == 'unknown' for f in normalized['events'][0]['facets']) and audit == []


@pytest.mark.parametrize('field', ['event', 'facet', 'cross_event', 'global_order'])
def test_ambiguous_sources_and_global_order_fail_without_paid_guessing(analysis, field):
    fragments = fragments_for(analysis)
    raw = merged_response(analysis)
    if field == 'event':
        raw['events'][0]['fragment_ids'][0] = 'F99999'
    elif field == 'facet':
        raw['events'][0]['facets'][0]['fragment_ids'] = ['F99999']
    elif field == 'cross_event':
        raw['events'][0]['facets'][0]['fragment_ids'] = raw['events'][-1]['fragment_ids']
    else:
        raw['events'].reverse()
    with pytest.raises(StoryError):
        repair_integration_result(NoRepairModel(), raw, analysis, fragments)


def test_missing_story_content_still_needs_model_and_model_status_error_is_repaired_locally(analysis):
    fragments = fragments_for(analysis)
    correct = merged_response(analysis)
    raw = copy.deepcopy(correct)
    raw['events'][1]['facets'] = None
    calls = []
    class Repair:
        def ask(self, stage, prompt, **kwargs):
            calls.append(stage)
            payload = json.JSONDecoder().raw_decode(prompt.split('资料：', 1)[1])[0]
            assert len(payload['issues']) == 1 and payload['issues'][0]['path'] == '/events/1/facets'
            replacement = copy.deepcopy(correct['events'][1]['facets'])
            for facet in replacement:
                facet['status'] = 'supported'
            return {'patches': [{'issue_id': payload['issues'][0]['issue_id'], 'value': replacement}], 'unresolved': []}
    audit = []
    result = repair_integration_result(Repair(), raw, analysis, fragments, local_repairs=audit)
    assert calls == ['event_integration'] and result == consume(correct, analysis, fragments)
    assert len(audit) == 1 and audit[0]['after'] == 'uncertain'


def test_local_repair_audit_is_cached_and_keeps_source_revisions_and_raw_transport_trace(story):
    store, state, _, _ = story
    before = copy.deepcopy(state)
    class Transport:
        kind = 'test'
        host = 'offline.invalid'
        calls = []
        def chat(self, **request):
            self.calls.append(request)
            raw = integration_response(request['prompt'])
            raw['events'][0]['facets'][0]['fragment_ids'] = raw['events'][0]['facets'][0]['fragment_ids'][0]
            return json.dumps(raw, ensure_ascii=False)
    transport = Transport()
    model = StoryModel(store, transport)
    integrated = integrate_chapter(store, model, state['bible'], state['revision'])
    audit = integrated['event_integration']['local_repairs']
    assert len(audit) == 1 and audit[0]['reason'] == 'exact_reference_format'
    traces = {p: p.read_bytes() for p in (store.root/'calls').glob('*.json')}
    trace = json.loads(next(iter(traces.values())))
    assert isinstance(json.loads(trace['response'])['events'][0]['facets'][0]['fragment_ids'], str)
    assert integrate_chapter(store, model, integrated, state['revision']) == integrated
    assert len(transport.calls) == 1 and state == before and store.read() == state
    assert all(p.read_bytes() == data for p, data in traces.items())
