"""Opening continuity and original-image grounding, using offline fixtures."""
import copy
import json
import socket

import pytest

from pipeline.fact_checker import FactChecker, sentences
from pipeline.fact_material import writing_material, source_material_basis
from pipeline.narrative_director import NarrativeDirector
from pipeline.opening_recap import with_opening_recap
from pipeline.script_writer import ScriptWriter
from pipeline.semantic_grounding import grounded_units
from pipeline.story_bible import StoryBible, StoryError
from pipeline.story_pipeline import StoryPipeline
from integration_fixture import integration_response
from test_story_reuse import grounding_model, counts
from test_chapter_memory import MemoryModel, remembered, next_analysis, data


RECAP = '上一话，沙流倒转后，桥边的人报出姓名。'
CURRENT = '如今，他继续拉紧绳索。'


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('Opening recap tests must stay offline')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)


class RecapModel(MemoryModel):
    def ask(self, stage, prompt, images=None, fresh=False):
        if stage in {'script_polish','chapter_memory','chapter_memory_review'}:
            return super().ask(stage, prompt, images, fresh)
        self.calls.append((stage,prompt,images,fresh))
        if stage == 'event_integration':
            return integration_response(prompt)
        if stage == 'direct':
            return {'beats':[{'event_ids':['E00001'],'section':'opening','title':'继续拉索'}],
                    'essential_event_ids':['E00001'],
                    'milestones':{'conflict':'E00001','climax':'E00001','ending':'E00001',
                                  'rules':None,'goal_or_stakes':None}}
        if stage in {'write_story','rewrite_story'}:
            return {'segments':[{'event_id':'E00001','text':RECAP+CURRENT}]}
        if stage == 'write':
            return {'text':RECAP+CURRENT}
        if stage == 'check':
            parts = json.loads(prompt.split('待核对句子：',1)[1].split('\n返回',1)[0])
            return {'sentences':[{'sentence_index':p['sentence_index'],'verdict':'supported',
                                 'evidence':'离线模拟，不能证明实际内容正确','issues':[]} for p in parts]}
        raise AssertionError(stage)


@pytest.fixture
def continuation(tmp_path):
    first = StoryBible(tmp_path/'book'/'Chapter_00001')
    _, previous, _ = remembered(first)
    store = StoryBible(tmp_path/'book'/'Chapter_00002')
    state = next_analysis(store)
    model = RecapModel()
    return first, previous, store, state, model


def test_recap_uses_previous_summary_sources_without_changing_nodes_or_source_data(continuation):
    _, _, _, state, model = continuation
    bible = state['bible']; plan = NarrativeDirector(model).plan(bible)
    before = copy.deepcopy((bible,plan))
    packed = writing_material(bible,plan)
    enriched = with_opening_recap(bible,plan)
    first = enriched['beats'][0]
    historical = enriched['opening_recap']['event_ids']
    assert historical == ['H00001_E00001','H00001_E00002']
    assert len(enriched['beats']) == len(plan['beats']) == 1
    assert first['event_id'] == plan['beats'][0]['event_id'] == 'E00001'
    assert set(first['event_ids']) == set(historical+['E00001'])
    assert set(first['source_panel_ids']) == {'1','H00001_1','H00001_2'}
    assert all(e['historical'] for e in packed['events'] if e['event_id'] in historical)
    assert with_opening_recap(bible,enriched) == enriched
    assert source_material_basis(bible,plan) == source_material_basis(bible,enriched)
    assert (bible,plan) == before


@pytest.mark.parametrize('missing', ['first_chapter','needs_review','no_summary','unknown_ref',
                                    'mixed_refs','wrong_chapter','wrong_revision','missing_image'])
def test_unusable_history_never_creates_filler_or_partial_recap(continuation, missing):
    _, _, _, state, model = continuation
    bible = copy.deepcopy(state['bible']); plan = NarrativeDirector(model).plan(bible)
    prior = bible['prior_context']
    if missing == 'first_chapter': prior['chapter'] = 1
    elif missing == 'needs_review': prior['status'] = 'needs_review'
    elif missing == 'no_summary': prior['summary'] = []
    elif missing == 'unknown_ref': prior['summary'][0]['event_ids'] = ['absent']
    elif missing == 'mixed_refs': prior['summary'][0]['event_ids'].append('E00001')
    elif missing == 'wrong_chapter': prior['previous_chapter']['chapter'] = 0
    elif missing == 'wrong_revision': prior['previous_chapter']['revision'] = 'different'
    elif missing == 'missing_image': bible['panels'] = [p for p in bible['panels'] if p['source_panel_id'] != 'H00001_1']
    result = with_opening_recap(bible,plan)
    assert result == plan and 'opening_recap' not in writing_material(bible,plan)['plan']


def test_unavailable_recap_restores_original_causal_context(continuation):
    _, _, _, state, model = continuation
    bible = copy.deepcopy(state['bible'])
    bible['events'][-1]['causes'] = [{'event_id':'H00001_E00001','evidence':'已有的原作因果'}]
    plan = NarrativeDirector(model).plan(bible)
    enriched = with_opening_recap(bible,plan)
    bible['prior_context']['status'] = 'needs_review'
    assert with_opening_recap(bible,enriched) == plan


@pytest.mark.parametrize('mode', ['single','verified'])
def test_generation_persists_first_sentence_scope_and_preserves_previous_chapter(continuation, mode):
    first, previous, store, state, model = continuation
    pipeline = StoryPipeline(store,model)
    grounding_model((store,state,model,pipeline))
    saved = pipeline.generate(state['revision']) if mode == 'single' else pipeline.generate_verified(state['revision'])
    drafts = [saved]
    for draft in drafts:
        segment = draft['script']['segments'][0]
        assert segment['text'] == RECAP+CURRENT
        assert segment['event_ids'] == draft['plan']['beats'][0]['event_ids']
        assert 'H00001_E00001' in segment['event_ids']
    assert counts(model)['story_digest'] == 0 and first.read() == previous
    assert store.read(state['revision']) == state
    if mode == 'single':
        writes = counts(model)['write_story']
        again = pipeline.generate(saved['revision'])
        assert again['script'] == saved['script'] and counts(model)['write_story'] == writes
    else:
        assert grounded_units(saved['bible'],saved['script']['segments'][0])


def test_recap_sentence_is_checked_against_historical_original_images(continuation):
    _, _, store, state, model = continuation
    bible = state['bible']; plan = NarrativeDirector(model).plan(bible)
    segment = ScriptWriter(model).write_story(bible,plan)[0]
    class Checker:
        def ask(self, stage, prompt, images=None, **kwargs):
            assert stage == 'check' and len(images) == 3
            assert 'H00001_1' in prompt and 'H00001_2' in prompt
            rows = []
            for i, wording in enumerate(sentences(segment['text'])):
                ids = ['H00001_E00001','H00001_E00002'] if i == 0 else ['E00001']
                panels = ['H00001_1','H00001_2'] if i == 0 else ['1']
                rows.append({'sentence_index':i,'verdict':'supported','evidence':'离线模拟对应原图','issues':[],
                             'units':[{'text':wording,'event_ids':ids,'display_panel_ids':panels,
                                       'background_event_ids':[]}]})
            return {'sentences':rows}
    segment['fact_check'] = FactChecker(store,Checker()).check(bible,segment)
    assert segment['fact_check']['status'] == 'passed'
    units = grounded_units(bible,segment)
    assert units[0]['display_panel_ids'] == ['H00001_1','H00001_2']
    assert units[1]['display_panel_ids'] == ['1']


def test_targeted_opening_repair_adds_sources_and_preserves_locked_other_segments(continuation):
    _, _, _, state, model = continuation
    bible = state['bible']; plan = NarrativeDirector(model).plan(bible)
    writer = ScriptWriter(model)
    segments = writer.write_story(bible,plan)
    # A legacy first paragraph lacks both recap text and its historical scope.
    segments[0].update(text=CURRENT,event_ids=['E00001'],source_panel_ids=['1'],fact_check={'status':'passed'})
    untouched = {**copy.deepcopy(segments[0]),'segment_id':'S_other','event_id':'other',
                 'text':'需要保持的已审核段落。','review_status':'approved'}
    segments.append(untouched)
    before = copy.deepcopy(segments)
    fixed = writer.revise_story(bible,plan,segments,{'issues':[{'event_id':'E00001','detail':'缺少一句前情回顾'}]})
    assert fixed[0]['text'] == RECAP+CURRENT and 'H00001_E00001' in fixed[0]['event_ids']
    assert fixed[0]['fact_check'] is None and fixed[0]['review_status'] == 'pending'
    assert fixed[1] == untouched and segments == before
    info = data(model.calls[-1][1])
    assert any(e['event_id'] == 'H00001_E00001' for e in info['events'])
    segments[0]['review_status'] = 'approved'
    before_calls = len(model.calls)
    with pytest.raises(StoryError,match='已审核'):
        writer.revise_story(bible,plan,segments,{'issues':[{'event_id':'E00001','detail':'修改回顾'}]})
    assert len(model.calls) == before_calls
