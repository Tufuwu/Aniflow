"""Offline program constraints; synthetic model output says nothing about appeal."""
import copy
import json
import socket
from collections import Counter

from PIL import Image
import pytest

from pipeline.story_analyzer import StoryAnalyzer
from pipeline.story_bible import StoryBible, StoryError, StoryConflict, atomic_json
from pipeline.story_model import StoryModel
from pipeline.story_pipeline import StoryPipeline
from pipeline.script_writer import ScriptWriter
from pipeline.chapter_memory import build_memory, prompt_context
from pipeline.prompt_material import character_index, text_evidence
from pipeline.reference_brief import prepare_reference_brief
from pipeline.semantic_grounding import grounded_units
from tools.audit_story_usage import audit
from test_engagement_flow import story
from test_character_introduction import panel
from test_chapter_memory import MemoryModel, ready_state, data


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*a,**k):raise AssertionError('No API or TTS in offline tests')
    monkeypatch.setattr(socket.socket,'connect',deny)
    monkeypatch.setattr(socket,'create_connection',deny)


def candidate_model(story):
    model=story[2];original=model.ask
    def ask(stage,prompt,images=None,fresh=False):
        if stage=='candidate_compare':
            model.calls.append((stage,prompt,images,fresh));info=data(prompt)
            return {'candidates':[{'candidate_id':c['candidate_id'],
                'citations':[{'event_id':c['script'][0]['event_id'],'text':c['script'][0]['text']}],
                'strengths':'模拟：背景支持当前选择','problems':'模拟：措辞仍需人审','fact_risks':'尚未核对原图'}
                for c in info['candidates']], 'recommended_candidate_id':info['candidates'][0]['candidate_id'],
                'reason':'模拟比较引用与背景用途，不证明真实吸引力'}
        result=original(stage,prompt,images,fresh)
        if stage=='check':
            events=json.JSONDecoder().raw_decode(prompt.split('\n事件：',1)[1])[0]
            parts=json.loads(prompt.split('待核对句子：',1)[1].split('\n返回',1)[0])
            for row,part in zip(result['sentences'],parts):
                row['units']=[{'text':part['text'],'event_ids':[e['event_id'] for e in events],
                    'display_panel_ids':list(dict.fromkeys(p for e in events for p in e['source_panel_ids'])),
                    'background_event_ids':[]}]
        return result
    model.ask=ask
    return model


def counts(model):return Counter(c[0] for c in model.calls)


def test_two_candidates_share_plan_facts_without_any_images_and_supplement_only_third(story):
    store,state,model,pipeline=story;candidate_model(story)
    first=pipeline.generate(state['revision'])
    assert counts(model)=={'event_integration':1,'direct':1,'story_plan_review':1,'story_digest':1,'write_story':2,'script_review':2,'candidate_compare':1}
    assert first['script'] is None and first['stage']=='pending_selection'
    assert len(first['candidates'])==2
    before=copy.deepcopy(first['candidates'])
    third=pipeline.generate(first['revision'],candidate_count=3)
    assert third['candidates'][:2]==before and len(third['candidates'])==3
    assert counts(model)['write_story']==3 and counts(model)['story_digest']==1 and counts(model)['direct']==1
    assert all(not images for _,_,images,_ in model.calls)
    assert store.read(first['revision'])==first


def test_only_selected_candidate_checked_checks_reused_and_selection_is_not_approval(story):
    store,state,model,pipeline=story;candidate_model(story)
    generated=pipeline.generate(state['revision'])
    selected=pipeline.select(generated['revision'],'K002')
    assert counts(model)['check']==0 and all(s['review_status']=='pending' for s in selected['script']['segments'])
    verified=pipeline.verify_selected(selected['revision'])
    assert counts(model)['check']==3
    assert all(grounded_units(verified['bible'],s) for s in verified['script']['segments'])
    assert all(s['fact_check'] is None for s in verified['candidates'][0]['script']['segments'])
    again=pipeline.verify_selected(verified['revision'])
    assert counts(model)['check']==3
    switched=pipeline.select(again['revision'],'K001')
    assert switched['script']==switched['candidates'][0]['script']
    pipeline.verify_selected(switched['revision'])
    assert counts(model)['check']==6  # Same events and even identical text do not copy checks between candidates.


def test_selected_edit_invalidates_comparison_only_rechecks_changed_text_and_protects_versions(story):
    store,state,model,pipeline=story;candidate_model(story)
    generated=pipeline.generate(state['revision'])
    selected=pipeline.select(generated['revision'],'K001')
    verified=pipeline.verify_selected(selected['revision'])
    before=copy.deepcopy(verified)
    changed=pipeline.segment_action('S_E1','edit',verified['revision'],'暴雨还在下，陈默听到送药的消息后开了门。')
    assert counts(model)['check']==4
    assert changed['candidates'][1]==before['candidates'][1]
    assert changed['script']['segments'][1:]==before['script']['segments'][1:]
    assert changed['candidate_comparison']['status']=='needs_review'
    assert changed['candidate_comparison']['recommended_candidate_id'] is None
    assert store.read(before['revision'])==before
    locked=pipeline.segment_action('S_E1','approve',changed['revision'])
    with pytest.raises(StoryConflict):pipeline.select(locked['revision'],'K002')
    with pytest.raises(StoryConflict):pipeline.generate(locked['revision'],candidate_count=3)


def test_plan_dependent_digest_never_reused_for_another_plan(story):
    store,state,model,pipeline=story;candidate_model(story)
    generated=pipeline.generate(state['revision'])
    writer=ScriptWriter(model,store);bible=generated['bible'];plan=generated['candidates'][0]['plan']
    writer.prepare_facts(bible,plan)
    assert counts(model)['story_digest']==1
    different=copy.deepcopy(plan);different['beats'][0]['title']='本节点背景怎样影响选择'
    writer.prepare_facts(bible,different)
    assert counts(model)['story_digest']==2
    writer.prepare_facts(bible,plan)
    assert counts(model)['story_digest']==2


def test_missing_review_keeps_candidate_pending_without_rewriting_everything(story):
    store,state,model,pipeline=story;candidate_model(story);original=model.ask
    def ask(stage,*a,**k):return {'passed':True,'issues':[]} if stage=='script_review' else original(stage,*a,**k)
    model.ask=ask
    saved=pipeline.generate(state['revision'])
    assert saved['candidate_comparison']['status']=='needs_review'
    assert all(c['status']=='needs_review' and c['script']['segments'] for c in saved['candidates'])
    pipeline.generate(saved['revision'])
    assert counts(model)['write_story']==2 and counts(model)['check']==0


def test_comparison_must_quote_current_candidate_and_cannot_auto_select(story):
    _,state,model,pipeline=story;candidate_model(story);original=model.ask
    def ask(stage,*a,**k):
        raw=original(stage,*a,**k)
        if stage=='candidate_compare':raw['candidates'][0]['citations'][0]['text']='伪造的原句'
        return raw
    model.ask=ask;saved=pipeline.generate(state['revision'])
    assert saved['candidate_comparison']['status']=='needs_review' and not saved.get('selected_candidate_id')
    with pytest.raises(StoryError):pipeline.verify_selected(saved['revision'])


def test_failed_candidate_can_resume_without_regenerating_successful_sibling(story):
    _,state,model,pipeline=story;candidate_model(story);original=model.ask;fail=True
    def ask(stage,prompt,*a,**k):
        if stage=='write_story' and 'K002' in prompt and fail:raise StoryError('模拟第二候选中断')
        return original(stage,prompt,*a,**k)
    model.ask=ask;first=pipeline.generate(state['revision'])
    assert first['candidates'][0]['script']['segments'] and not first['candidates'][1]['script']['segments']
    kept=copy.deepcopy(first['candidates'][0]);fail=False
    resumed=pipeline.generate(first['revision'])
    assert resumed['candidates'][0]==kept and len(resumed['candidates'])==2
    assert resumed['candidates'][1]['candidate_id']=='K002' and resumed['candidates'][1]['script']['segments']
    assert counts(model)['write_story']==2


def test_unrelated_event_change_keeps_scoped_check_but_continuity_image_change_invalidates(story):
    _,state,model,pipeline=story;candidate_model(story)
    state=pipeline.generate(state['revision']);state=pipeline.select(state['revision'],'K001')
    verified=pipeline.verify_selected(state['revision']);segment=verified['script']['segments'][0]
    bible=copy.deepcopy(verified['bible']);bible['events'][-1]['summary']='改变后续未讲述的事件依据'
    assert grounded_units(bible,segment)
    bible['panels'][0]['sha256']='changed'
    assert not grounded_units(bible,segment)


class AnalysisModel:
    def __init__(self):self.calls=[];self.fail=None;self.version='vision-a'
    def cache_identity(self,stage):return {'vision':self.version}
    def ask(self,stage,prompt,images=None,fresh=False):
        self.calls.append((stage,prompt,fresh))
        if stage=='analyze':
            pid=prompt.split('按阅读顺序分析当前来源 ',1)[1].split('，',1)[0]
            if self.fail==pid:raise StoryError('模拟此图中断')
            raw=panel();raw['analyzed_source_panel_id']=pid
            raw['characters'][0]['identity_match']={'method':'continuous_action','basis':'仍握同一根桥索，连续拉索动作'}
            raw['events'][0]['summary']='桥索被拉紧，沙流仍在倒转'
            return raw
        if stage=='reconcile':
            if self.fail=='boundary':raise StoryError('模拟边界复核中断')
            ids=json.loads(prompt.split('按顺序附上以下来源图：',1)[1].split('。',1)[0])
            return {'panels':[{'source_panel_id':p,'classification':'story','reason':'桥头行动'} for p in ids]}
        raise AssertionError(stage)


def panels(tmp_path):
    result=[]
    for i in range(3):
        path=tmp_path/f'{i}.png';Image.new('RGB',(20,30),(20+i,40,60)).save(path)
        result.append({'panel_id':str(i+1),'path':str(path),'derivation':{'version':1}})
    return result


@pytest.mark.parametrize('failure',['2','boundary'])
def test_analysis_resume_uses_valid_panel_checkpoints_after_interruption(tmp_path,failure):
    store=StoryBible(tmp_path/'chapter');model=AnalysisModel();model.fail=failure
    analyzer=StoryAnalyzer(store,model);inputs=panels(tmp_path)
    with pytest.raises(StoryError):analyzer.analyze(inputs)
    before=Counter(c[0] for c in model.calls)
    model.fail=None;result=analyzer.analyze(inputs)
    assert len(result['events'])==3
    delta=Counter(c[0] for c in model.calls)-before
    assert delta['analyze']==(2 if failure=='2' else 0)
    before=len(model.calls);assert analyzer.analyze(inputs)==result
    assert len(model.calls)==before


@pytest.mark.parametrize('change',['image','preparation','model','history'])
def test_analysis_dependencies_invalidate_only_affected_prefix_successors(tmp_path,change):
    store=StoryBible(tmp_path/'chapter');model=AnalysisModel();inputs=panels(tmp_path)
    analyzer=StoryAnalyzer(store,model);analyzer.analyze(inputs);before=len(model.calls)
    history={}
    if change=='image':Image.new('RGB',(20,30),'red').save(inputs[1]['path'])
    if change=='preparation':inputs[1]['derivation']['version']=2
    if change=='model':model.version='vision-b'
    if change=='history':history={'background':[{'text':'桥索只能由一人拉动','event_ids':['history']} ]}
    analyzer.analyze(inputs,history=history)
    calls=model.calls[before:];assert sum(c[0]=='analyze' for c in calls)==(2 if change in ('image','preparation') else 3)


def test_identity_failure_retries_identity_without_reanalyzing_panels_or_boundary(tmp_path,monkeypatch):
    store=StoryBible(tmp_path/'chapter');model=AnalysisModel();analyzer=StoryAnalyzer(store,model)
    calls=[]
    def fail(bible):calls.append(1);raise StoryError('模拟身份格式错误')
    monkeypatch.setattr(analyzer,'reconcile_identity',fail)
    inputs=panels(tmp_path);first=analyzer.analyze(inputs)
    assert first['identity_review']['status']=='needs_review'
    before=len(model.calls)
    monkeypatch.setattr(analyzer,'reconcile_identity',lambda bible:{**bible,'identity_review':{'status':'reviewed'}})
    fixed=analyzer.analyze(inputs)
    assert fixed['identity_review']['status']=='reviewed' and len(model.calls)==before


def test_compact_prompts_keep_unknowns_reveals_sources_but_no_nested_checks(story,tmp_path):
    store,state,model,pipeline=story;candidate_model(story)
    state['bible']['characters']=[{'character_id':'C9','name':None,'name_status':'unknown','gender':None,
        'pending_names':[{'name':'闻溪','source_panel_id':'5'}],'identity_evidence':[{'detail':'UNRELATED_PROOF_HISTORY'}]}]
    state=store.save(state,'fixture_unknown',state['revision'])
    generated=pipeline.generate(state['revision']);script=generated['candidates'][0]['script']
    for s in script['segments']:s['fact_check']={'sentences':['NESTED_CHECK_SENTINEL'],'grounding':{'units':['UNITS_SENTINEL']}}
    ScriptWriter(model).review_story(generated['bible'],generated['candidates'][0]['plan'],script['segments'])
    prompt=model.calls[-1][1]
    assert all(x not in prompt for x in ('NESTED_CHECK_SENTINEL','UNITS_SENTINEL','UNRELATED_PROOF_HISTORY'))
    assert 'pending_names' in prompt and 'unknown' in prompt and 'source_panel_id' in prompt
    prior={'summary':[{'text':'上一话过桥','event_ids':['H1']}], 'background':[{'text':'更早的沙流规则','event_ids':['H0']}],
        'character_states':[{'character_id':'C9','facts':[{'text':'姓名仍未确认','source_panel_ids':['H图']}]}],
        'open_threads':[{'text':'寻找断绳','event_ids':['H1']}],'events':['HUGE_EVENT_TABLE'],'panels':['HUGE_PANEL_TABLE']}
    brief=prompt_context({'prior_context':prior})
    assert 'events' not in brief and 'panels' not in brief
    assert all(brief[k]==prior[k] for k in ('summary','background','character_states','open_threads'))


def test_summary_patches_only_problem_entries_and_rechecks_only_changed_claim(tmp_path):
    store=StoryBible(tmp_path/'Chapter_00001');state=ready_state(store);model=MemoryModel();original=model.ask
    first_review=True
    def ask(stage,prompt,images=None,fresh=False):
        nonlocal first_review
        if stage=='chapter_memory' and 'Chapter Memory Patch' in prompt:
            model.calls.append((stage,prompt,images,fresh));info=data(prompt)
            assert len(info['problems'])==1
            c=info['problems'][0]
            return {'patches':[{'claim_id':c['claim_id'],'replacement':{'text':'沙流倒转带来拉索限制。','event_ids':c['event_ids']}}]}
        raw=original(stage,prompt,images,fresh)
        if stage=='chapter_memory_review' and first_review:
            first_review=False;raw['claims'][1]['verdict']='issues'
        return raw
    model.ask=ask;memory=build_memory(store,model,state)
    assert memory['status']=='reviewed'
    reviews=[data(c[1]) for c in model.calls if c[0]=='chapter_memory_review']
    assert len(reviews[0]['claims'])==3 and len(reviews[1]['claims'])==1
    assert memory['summary'][0]['text']=='沙流倒转后，桥边的人报出姓名。'
    assert memory['background'][0]['text']=='沙流倒转带来拉索限制。'


def test_failed_summary_patch_is_pending_and_next_refresh_resumes_problem_entries(tmp_path):
    store=StoryBible(tmp_path/'Chapter_00001');state=ready_state(store);model=MemoryModel();original=model.ask
    bad=True;initial=True;patch_calls=[]
    def ask(stage,prompt,images=None,fresh=False):
        nonlocal initial
        if stage=='chapter_memory' and 'Chapter Memory Patch' in prompt:
            info=data(prompt);patch_calls.append(info)
            claim=info['problems'][0]
            return {'patches':[{'claim_id':'M0001' if bad else claim['claim_id'],
                'replacement':{'text':'沙流倒转带来拉索限制。','event_ids':claim['event_ids']}}]}
        raw=original(stage,prompt,images,fresh)
        if stage=='chapter_memory_review' and initial:
            initial=False;raw['claims'][1]['verdict']='issues'
        return raw
    model.ask=ask;memory=build_memory(store,model,state)
    assert memory['status']=='needs_review' and memory['repair_context']
    assert memory['candidate']['summary'][0]['text']=='沙流倒转后，桥边的人报出姓名。'
    state['chapter_memory']=memory;bad=False
    fixed=build_memory(store,model,state,True)
    assert fixed['status']=='reviewed' and len(patch_calls)==2
    assert all(len(p['problems'])==1 for p in patch_calls)
    assert sum(c[0]=='chapter_memory' for c in model.calls)==1  # No whole-summary regeneration on refresh.


def test_usage_audit_marks_unknown_and_counts_cache_without_inventing_tokens(tmp_path):
    store=StoryBible(tmp_path/'chapter')
    class Transport:
        kind='test';host='test'
        def chat(self,**k):return '{}'
    model=StoryModel(store,Transport());model.ask('write','本地模拟');model.ask('write','本地模拟')
    result=audit(store.root)['stages']['write']
    assert result['requests']==1 and result['usage_unknown']==1
    assert result['input_tokens']==0 and result['reuse']['cache_hit']==1
    assert result['reasoning_unknown']==1 and result['finish_reasons']['unknown']==1


def test_validated_analysis_avoids_old_invalid_response_but_keeps_raw_trace(tmp_path):
    store=StoryBible(tmp_path/'chapter');inputs=panels(tmp_path)[:1]
    class Transport:
        kind='test';host='test';count=0
        def chat(self,**k):
            self.count+=1
            if 'Story Analyzer' in k['prompt']:
                raw=panel();raw['analyzed_source_panel_id']='1'
                if self.count==1:raw['events'][0]['evidence']=[{'kind':'dialogue','detail':'伪造的引文'}]
                return json.dumps(raw,ensure_ascii=False)
            return json.dumps({'panels':[{'source_panel_id':'1','classification':'story','reason':'桥索仍在拉动'}]})
    transport=Transport();model=StoryModel(store,transport);analyzer=StoryAnalyzer(store,model)
    first=analyzer.analyze(inputs);calls=transport.count
    raw_records={p.name:p.read_bytes() for p in (store.root/'calls').glob('*.json')}
    assert calls==3 and analyzer.analyze(inputs)==first and transport.count==calls
    assert all((store.root/'calls'/name).read_bytes()==value for name,value in raw_records.items())
    assert audit(store.root)['stages']['analyze']['reuse']['checkpoint_reuse']==1


def test_long_reference_is_distilled_once_keeps_unknowns_and_originals_local(tmp_path):
    store=StoryBible(tmp_path/'chapter')
    notes={'revision':'a'*32,'background':'河水在夜间倒流，不能将先后当因果。'*150,
           'characters':'闻溪这个称呼尚未对应具体人物，不能提前使用。'*100}
    before=copy.deepcopy(notes);calls=[]
    class Model:
        def ask(self,stage,prompt,images=None,fresh=False):
            assert stage=='context_digest' and not images
            calls.append(prompt);units=data(prompt)['units']
            return {'background':[{'text':'夜间河水倒流。','unit_ids':[units[0]['unit_id']]}],
                'characters':[], 'unknowns':[{'text':'称呼尚未归属人物，不能提前使用；先后不等于因果。',
                                              'unit_ids':[u['unit_id'] for u in units]}],
                'coverage':[{'unit_id':u['unit_id'],'usage':'brief','reason':'保留环境及未知限制'} for u in units]}
    model=Model();brief=prepare_reference_brief(store,model,notes)
    assert prepare_reference_brief(store,model,notes)==brief and len(calls)==1 and notes==before
    assert brief['status']=='unverified_reference' and brief['unknowns'] and brief['source_ref']
    transmitted=prompt_context({'prior_context':{'series_notes':brief,'reference_notes':notes}})
    assert 'reference_notes' not in transmitted and 'source_index' not in transmitted['series_notes']
    assert len(json.dumps(transmitted,ensure_ascii=False)) < len(json.dumps(notes,ensure_ascii=False))
    changed=copy.deepcopy(notes);changed['background']+='桥的承重有限。'
    prepare_reference_brief(store,model,changed);assert len(calls)==2


def test_text_review_uses_fact_table_and_dialogue_once_without_visual_forensics(story):
    store,state,model,pipeline=story;candidate_model(story)
    bible=state['bible']
    for e in bible['events']:
        e['evidence']=[{'kind':'visual','detail':'FULL_VISUAL_FORENSICS_SENTINEL'}]
        e['dialogue']=[{'text':'DUPLICATE_EVENT_DIALOGUE_SENTINEL'}]
    state=store.save(state,'fixture_full_local_evidence',state['revision'])
    saved=pipeline.generate(state['revision'])
    review_prompt=next(p for stage,p,_,_ in model.calls if stage=='script_review')
    info=data(review_prompt)
    assert info['fact_sheet']==saved['candidates'][0]['script']['fact_sheet'] and 'sources' not in info
    fallback=text_evidence(bible,saved['candidates'][0]['plan']['beats'])
    assert fallback['sources'][0]['dialogue']==bible['panels'][0]['dialogue']
    assert 'FULL_VISUAL_FORENSICS_SENTINEL' not in review_prompt
    assert 'DUPLICATE_EVENT_DIALOGUE_SENTINEL' not in review_prompt
    assert all('evidence' not in e and 'dialogue' not in e for e in info['events'])
    assert saved['bible']['analysis_events']==bible['events']
