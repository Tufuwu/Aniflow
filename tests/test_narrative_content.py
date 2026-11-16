"""One-call interest polishing and independent fact-verification contracts."""
import copy
import json
import socket
import pytest
from pipeline.narration_review import polish_basis, polish_ready
from pipeline.narrative_director import NarrativeDirector
from pipeline.script_writer import ScriptWriter
from pipeline.semantic_grounding import grounded_units
from pipeline.shot_pipeline import ShotEditingPipeline
from pipeline.shot_selector import ShotSelectorAgent, GlobalShotOptimizer
from pipeline.story_bible import StoryBible, StoryConflict, StoryError
from pipeline.story_pipeline import StoryPipeline
from pipeline.timeline_director import semantic_alignment
from test_engagement_flow import story, generate, polish_payload
from test_semantic_alignment import mapped

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*a,**k):raise AssertionError('Polish contracts cannot use live API')
    monkeypatch.setattr(socket.socket,'connect',deny);monkeypatch.setattr(socket,'create_connection',deny)

def data_after(prompt,marker='资料：'):
    return json.JSONDecoder().raw_decode(prompt.split(marker,1)[1])[0]


def test_one_call_merges_partial_final_text_and_preserves_other_paragraphs(story):
    state=generate(story);segments=state['script']['segments'];before=copy.deepcopy(segments);calls=[]
    class Model:
        def ask(self,stage,prompt,images=None,**kwargs):
            calls.append(stage);assert stage=='script_polish' and not images
            info=data_after(prompt);assert set(info)=={'style','polish_options','script'}
            assert all(set(row)=={'event_id','text','locked'} for row in info['script'])
            assert 'SOURCE_SENTINEL' not in prompt and 'PLANNING_SENTINEL' not in prompt
            return {'segments':[{'event_id':'E1','text':'他刚才还拒绝开门，听见送药的消息，立刻改变了主意。'}]}
    bible=copy.deepcopy(state['bible']);bible['events'][0]['summary']='SOURCE_SENTINEL'
    plan={**state['plan'],'secret':'PLANNING_SENTINEL'}
    result=ScriptWriter(Model()).polish_story(bible,plan,segments,facts=[{'text':'SOURCE_SENTINEL'}])
    assert calls==['script_polish'] and segments==before
    assert result['segments'][1:]==before[1:]
    assert result['segments'][0]['fact_check'] is None and result['segments'][0]['review_status']=='pending'
    assert result['segments'][0]['event_ids']==before[0]['event_ids']
    assert result['completion']['engagement_polish']['changed_event_ids']==['E1']
    assert polish_ready(result['completion'],bible,plan,result['segments'])


def test_empty_response_keeps_text_checks_and_locks_exactly(story):
    state=generate(story);segments=copy.deepcopy(state['script']['segments'])
    segments[0]['review_status']='approved';before=copy.deepcopy(segments)
    class Model:
        def ask(self,stage,prompt,**kwargs):
            assert data_after(prompt)['script'][0]['locked']
            return {'segments':[]}
    result=ScriptWriter(Model()).polish_story(state['bible'],state['plan'],segments)
    assert result['segments']==before and segments==before
    assert result['completion']['engagement_polish']['changed_event_ids']==[]


@pytest.mark.parametrize('bad',['critique','missing','type','unknown','duplicate','extra_source','empty_text','locked','extra_root','unchanged'])
def test_invalid_patches_never_partially_change_the_current_draft(story,bad):
    state=generate(story);segments=copy.deepcopy(state['script']['segments']);before=copy.deepcopy(segments)
    row={'event_id':'E1','text':'利用现有反差润色后的正文。'}
    raw={'segments':[row]}
    if bad=='critique':raw={'engagement':{'verdict':'issues','detail':'要更有趣'}}
    if bad=='missing':raw={}
    if bad=='type':raw={'segments':'修改意见'}
    if bad=='unknown':row['event_id']='unknown'
    if bad=='duplicate':raw['segments'].append(copy.deepcopy(row))
    if bad=='extra_source':row['source_panel_ids']=['new']
    if bad=='empty_text':row['text']=' '
    if bad=='locked':segments[0]['review_status']='approved';before=copy.deepcopy(segments)
    if bad=='extra_root':raw['issues']=['先评审再返修']
    if bad=='unchanged':row['text']=segments[0]['text']
    class Model:
        def ask(self,*a,**k):return raw
    with pytest.raises(StoryError):ScriptWriter(Model()).polish_story(state['bible'],state['plan'],segments)
    assert segments==before


def test_default_generation_calls_writer_and_polisher_once_and_saves_both_versions(story):
    from test_story_reuse import grounding_model,counts
    store,state,model,pipeline=story;grounding_model(story);original=model.ask
    def ask(stage,prompt,images=None,fresh=False):
        if stage=='script_polish':
            model.calls.append((stage,prompt,images,fresh))
            return {'segments':[{'event_id':'E1','text':'他刚说不肯开门，听到送药消息又改了主意。'}]}
        if stage in {'script_review','rewrite_story'}:pytest.fail('No critique/rewrite pipeline')
        return original(stage,prompt,images,fresh)
    model.ask=ask;result=pipeline.generate(state['revision'])
    assert counts(model)=={'event_integration':1,'direct':1,'write_story':1,'script_polish':1}
    draft=next(store.read(h['revision']) for h in store.history() if h['action']=='script_written')
    assert draft['script']['segments'][0]['text']==model.narrations['E1']
    assert result['script']['segments'][0]['text']!=draft['script']['segments'][0]['text']
    before=counts(model);assert pipeline.generate(result['revision'])==result and counts(model)==before


def test_explicit_polish_is_one_call_and_defers_image_verification(story):
    from test_story_reuse import grounding_model,counts
    store,state,model,pipeline=story;grounding_model(story);state=pipeline.generate(state['revision'])
    before=copy.deepcopy(state);original=model.ask;calls=counts(model)
    def ask(stage,prompt,images=None,fresh=False):
        assert stage=='script_polish' and fresh
        model.calls.append((stage,prompt,images,fresh))
        return {'segments':[{'event_id':'E1','text':'送药的消息一来，他刚才的拒绝就变了。'}]}
    model.ask=ask;result=pipeline.revise(state['revision'])
    assert counts(model)-calls=={'script_polish':1}
    assert result['script']['segments'][1:]==before['script']['segments'][1:]
    assert result['script']['segments'][0]['fact_check'] is None
    assert store.read(before['revision'])==before


def test_polish_failure_keeps_original_draft_and_has_no_automatic_second_call(story):
    store,state,model,pipeline=story;original=model.ask;calls=[]
    def ask(stage,prompt,images=None,fresh=False):
        if stage=='script_polish':calls.append(fresh);raise StoryError('模拟润色不可用')
        if stage=='rewrite_story':pytest.fail('No separate rewrite after failure')
        return original(stage,prompt,images,fresh)
    model.ask=ask;result=pipeline.generate(state['revision'])
    assert calls==[False]
    assert [s['text'] for s in result['script']['segments']]==list(model.narrations.values())
    assert result['script']['narrative_review']['engagement_polish']['status']=='unavailable'
    assert '模拟润色不可用' in result['script']['narrative_review']['error']


def test_legacy_review_only_resumes_polishing_without_rewriting_the_draft(story):
    from test_story_reuse import grounding_model,counts
    store,state,model,pipeline=story;grounding_model(story);state=pipeline.generate(state['revision'])
    state['script']['narrative_review']={'passed':True,'issues':[],'engagement_review':{'version':1,'status':'reviewed'}}
    saved=store.save(state,'legacy_review',state['revision']);before=counts(model)
    result=pipeline.generate(saved['revision'])
    assert counts(model)-before=={'script_polish':1} and result['script']['segments']==saved['script']['segments']
    assert store.read(saved['revision'])==saved
    assert polish_ready(result['script']['narrative_review'],result['bible'],result['plan'],result['script']['segments'])


def test_image_and_human_gates_are_independent_of_a_model_quality_judgment(tmp_path,mapped):
    source,segment,_,_=mapped;source['plan']={'version':5,'beats':[{'event_id':'E2'}]}
    source['script'].pop('narrative_review',None)
    store=StoryBible(tmp_path/'chapter');saved=store.save(source,'fixture',None)
    approved=StoryPipeline(store,object()).segment_action(segment['segment_id'],'approve',saved['revision'])
    assert approved['script']['segments'][0]['review_status']=='approved'
    ShotEditingPipeline.approved(approved)
    approved['script']['segments'][0]['fact_check'].pop('grounding')
    with pytest.raises(StoryError,match='来源关联'):ShotEditingPipeline.approved(approved)


@pytest.mark.parametrize('field,value', [('event_ids',['E1','E2']),('context_event_ids',['E1']),('source_panel_ids',['2'])])
def test_same_text_mapping_does_not_survive_changed_event_or_source_scope(mapped,field,value):
    source,segment,_,_=mapped
    assert grounded_units(source['bible'],segment)
    segment[field]=value
    assert grounded_units(source['bible'],segment)==[]


@pytest.mark.parametrize('action',['edit','regenerate'])
def test_same_text_result_keeps_existing_verification_and_review_state(story,action):
    state=generate(story)
    original=story[2].ask
    segment=state['script']['segments'][0]
    count=story[2].check_calls
    def ask(stage,prompt,images=None,fresh=False):
        if stage=='write':return {'text':segment['text']}
        return original(stage,prompt,images,fresh)
    story[2].ask=ask
    result=story[3].segment_action(segment['segment_id'],action,state['revision'],segment['text'])
    assert result['script']['segments'][0]==segment
    assert story[2].check_calls==count
    assert state==story[0].read(state['revision'])


def test_writer_receives_confirmed_identity_and_filters_verification_only_background(story):
    state=generate(story)
    state['bible']['characters']=[{'character_id':'C1','canonical_name':'陈默','name_status':'confirmed','evidence':'原文称呼'}]
    original=story[2].ask
    def ask(stage,prompt,images=None,fresh=False):
        result=original(stage,prompt,images,fresh)
        if stage=='story_digest':
            result['nodes'][0]['facts'].append({'text':'城镇百年前的建港历史，仅供核验','event_ids':['E1'],'usage':'verification_only'})
        if stage=='write_story':
            assert '城镇百年前的建港历史' not in prompt
            assert 'canonical_name' in prompt and '身份对应证据' in prompt
        return result
    story[2].ask=ask
    writer = ScriptWriter(story[2])
    facts = writer.prepare_facts(state['bible'],state['plan'])
    writer.write_story(state['bible'],state['plan'],facts=facts)


def test_optional_digest_essential_trigger_cannot_be_hidden_in_verification_only_facts(story):
    state=generate(story)
    state['plan']['essential_event_ids']=['E2']
    original=story[2].ask
    def ask(stage,prompt,images=None,fresh=False):
        result=original(stage,prompt,images,fresh)
        if stage=='story_digest':result['nodes'][0]['facts'][0]['usage']='verification_only'
        return result
    story[2].ask=ask
    with pytest.raises(StoryError,match='遗漏关键转折'):
        ScriptWriter(story[2]).prepare_facts(state['bible'],state['plan'])


def test_planner_cannot_move_reveal_before_earlier_action(story):
    original=story[2].ask
    def ask(stage,prompt,images=None,fresh=False):
        result=original(stage,prompt,images,fresh)
        if stage=='direct':result['beats'][0],result['beats'][1]=result['beats'][1],result['beats'][0]
        return result
    story[2].ask=ask
    with pytest.raises(StoryError,match='信息揭示顺序'):
        NarrativeDirector(story[2]).plan(story[1]['bible'])


@pytest.mark.parametrize('time,nid,text,event,correct,wrong',[
    (116.5,'S_E00029','总理成为政府首脑。','E00034','27','28'),
    (333.5,'S_E00085','男子坐在桌前应了一声“好！”。','E00084','63','64'),
])
def test_previous_preview_mismatch_cannot_hide_in_same_paragraph(time,nid,text,event,correct,wrong):
    # IDs/text taken from the preserved manifest and source ledger. A known
    # current claim is required; the old manifest alone has no sentence mapping.
    frame=round(time*30)
    timeline={'narration':[{'narration_id':nid,'text':text,'start_frame':frame-1,'end_frame':frame+1}],
              'semantic_units':[{'narration_id':nid,'unit_id':'u','text':text,'event_ids':[event],
                                 'display_panel_ids':[correct],'start_frame':frame-1,'end_frame':frame+1}],
              'shots':[{'narration_id':nid,'panel_id':wrong,'unit_ids':['u'],'display_event_ids':[event],
                        'start_frame':frame-1,'end_frame':frame+1}]}
    assert semantic_alignment(timeline)['status']=='needs_review'
    timeline['shots'][0]['panel_id']=correct
    assert semantic_alignment(timeline)['status']=='aligned'


def test_reveal_in_actual_crop_stays_unresolved_even_after_full_image_fallback(tmp_path,mapped):
    from PIL import Image
    source,_,_,library=mapped
    for panel in library:
        path=tmp_path/(panel['panel_id']+'.png')
        Image.new('RGB',(360,640),'gray').save(path)
        panel['evidence_path']=str(path)
    groups=GlobalShotOptimizer().optimize(source,library,ShotSelectorAgent())
    groups[0]['selected'][0]['crop']=[0,0,1,.5]
    calls=[]
    class Model:
        def ask(self,stage,prompt,images=None,**kwargs):
            assert stage=='visual_review' and '不能提前展示' in prompt
            data=data_after(prompt)
            assert data['semantic_units'] and data['shot_units']
            calls.append(images)
            return {'checks':[{'panel_id':pid,'usable':False,'reason':'当前讲电池到达，画面已展示灯塔亮起的后续结果'}
                              for pid in data['panels']]}
    result=ShotSelectorAgent(Model()).review_selected(source,library,groups)
    assert len(calls)==2 and result['status']=='needs_review'
    assert groups[0]['review_status']=='pending'


@pytest.mark.parametrize('style',['faithful','payoff','suspense'])
def test_polish_direction_is_independent_of_planning_and_writing_style(story,style):
    from pipeline.narration_review import ENGAGEMENT_POLISH_STYLE
    from pipeline.narrative_director import STYLES
    _,state,model,pipeline=story
    result=pipeline.generate(state['revision'],style=style)
    writing=next(prompt for stage,prompt,_,_ in model.calls if stage=='write_story')
    polishing=next(prompt for stage,prompt,_,_ in model.calls if stage=='script_polish')
    assert data_after(writing,'叙事依据：')['style']==STYLES[style]
    assert data_after(polishing)['style']==ENGAGEMENT_POLISH_STYLE=='生动有趣的故事解说'
    assert result['plan']['style']==result['script']['style']==style
    assert result['script']['narrative_review']['engagement_polish']['style']==ENGAGEMENT_POLISH_STYLE


def test_old_inherited_style_receipt_cannot_be_reused_for_new_polish_direction(story):
    from pipeline.story_bible import digest
    from pipeline.narration_review import ENGAGEMENT_POLISH_VERSION
    _,state,model,_=story
    segments=[{'event_id':'A','text':'他刚说不去，一听有肉罐头就问起了地址。'}]
    old_basis=digest({'polish_contract':ENGAGEMENT_POLISH_VERSION,'style':'faithful','script':segments})
    old={'basis_hash':old_basis,'engagement_polish':{'status':'completed','version':ENGAGEMENT_POLISH_VERSION}}
    assert not polish_ready(old,state['bible'],{'style':'faithful'},segments)
    current=ScriptWriter(model).polish_story(state['bible'],{'style':'faithful'},segments)
    assert polish_ready(current['completion'],state['bible'],{'style':'suspense'},segments)
    assert current['segments']==segments and old['basis_hash']==old_basis


def test_prompt_v1_completion_only_resumes_new_polish_and_keeps_old_versions(story):
    from test_story_reuse import grounding_model,counts
    from pipeline.story_bible import digest
    from pipeline.narration_review import ENGAGEMENT_POLISH_STYLE,ENGAGEMENT_POLISH_VERSION
    store,state,model,pipeline=story;grounding_model(story)
    first=pipeline.generate(state['revision'])
    record=first['script']['narrative_review']
    record['engagement_polish'].pop('prompt_version')
    record['basis_hash']=digest({'polish_contract':ENGAGEMENT_POLISH_VERSION,'style':ENGAGEMENT_POLISH_STYLE,
        'script':[{k:s[k] for k in ('event_id','text')} for s in first['script']['segments']]})
    old=store.save(first,'fixture_prompt_v1_polish',first['revision'])
    before=counts(model);result=pipeline.generate(old['revision'])
    assert counts(model)-before=={'script_polish':1}
    assert result['script']['segments']==old['script']['segments']
    assert result['script']['narrative_review']['engagement_polish']['prompt_version']==5
    assert store.read(old['revision'])==old


@pytest.mark.parametrize('bad',[None,'unknown','duplicate','extra_field','locked'])
def test_mixed_unchanged_rows_are_filtered_only_after_full_validation(story,bad):
    state=generate(story);original=copy.deepcopy(state['script']['segments'])
    if bad=='locked':original[0]['review_status']='approved'
    before=copy.deepcopy(original)
    changed={'event_id':original[0]['event_id'],'text':'他刚才还在讲大道理，一听到送药就改了主意。'}
    echoed={'event_id':original[1]['event_id'],'text':original[1]['text']}
    rows=[changed,echoed]
    if bad=='unknown':rows.append({'event_id':'missing','text':'不应接受的正文。'})
    if bad=='duplicate':rows.append(copy.deepcopy(echoed))
    if bad=='extra_field':echoed['source_panel_ids']=['forged']
    class Model:
        def ask(self,*args,**kwargs):return {'segments':rows}
    if bad:
        with pytest.raises(StoryError):ScriptWriter(Model()).polish_story(state['bible'],state['plan'],original)
    else:
        result=ScriptWriter(Model()).polish_story(state['bible'],state['plan'],original)
        assert result['segments'][0]['text']==changed['text']
        assert result['segments'][1:]==original[1:]
        assert result['completion']['engagement_polish']['ignored_unchanged_event_ids']==[echoed['event_id']]
        assert result['completion']['engagement_polish']['changed_event_ids']==[changed['event_id']]
    assert original==before
