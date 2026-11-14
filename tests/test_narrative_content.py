"""Offline review contracts. Fake judgments do not measure real narration quality."""
import copy
import json
import socket

import pytest

from pipeline.narration_review import content_issues, information_issues, review_basis, review_ready
from pipeline.narrative_director import NarrativeDirector
from pipeline.script_writer import ScriptWriter
from pipeline.semantic_grounding import grounded_units
from pipeline.shot_pipeline import ShotEditingPipeline
from pipeline.shot_selector import ShotSelectorAgent, GlobalShotOptimizer
from pipeline.story_bible import StoryBible, StoryConflict, StoryError
from pipeline.story_pipeline import StoryPipeline
from pipeline.timeline_director import semantic_alignment
from test_engagement_flow import story, generate, review_payload
from test_semantic_alignment import mapped


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('Local content tests must not access APIs')
    monkeypatch.setattr(socket.socket,'connect',deny)
    monkeypatch.setattr(socket,'create_connection',deny)


def data_after(prompt, marker='资料：'):
    return json.JSONDecoder().raw_decode(prompt.split(marker,1)[1])[0]


@pytest.mark.parametrize('aspect,detail,suggestion',[
    ('opening','反复查看消息，尚未讲清落脚点将不可用','保留环境压力，合并无变化的查看过程'),
    ('turns','拒绝后直接行动，丢掉了送药消息这一转折触发','依据E2补回收到消息与改变决定，不解释心理'),
    ('exposition','开门任务被无关的城镇历史打断','区分当前选择必需、稍后解释和仅供核验的背景'),
    ('references','两个角色都叫青年，听者无法持续分辨','依据已有身份使用稳定称呼，不补姓名性别或关系'),
    ('action','手脚姿态与拟声淹没了开门拿药的动作','保留有危险或变化作用的细节，不补造动作主体'),
    ('emotion','接药后的反应被办公用品清单淹没','给已有反应与安慰留空间，保留有作用的停顿'),
    ('ending','重复交代接药与搬运，掩盖寻找家人的新目标','删重复安排，保留下一步寻找家人的目标'),
])
def test_each_content_finding_overrides_a_model_pass_and_targets_only_affected_node(story,aspect,detail,suggestion):
    state=generate(story)
    model=story[2]
    original=model.ask
    segments=state['script']['segments']
    def ask(stage,prompt,images=None,fresh=False):
        if stage=='script_review':
            raw=review_payload(segments,['E1','E2','E3','E4','E5'])
            check=next(c for c in raw['content_checks'] if c['aspect']==aspect)
            check.update(verdict='issues',detail=detail,issues=[{'event_id':'E1','detail':detail,'suggestion':suggestion}])
            return raw  # Deliberately contradictory top-level passed=True.
        if stage=='rewrite_story':
            data=data_after(prompt)
            assert data['revision_event_ids']==['E1']
            assert data['current_script'][0]['text']==segments[0]['text']
            assert {e['event_id'] for e in data['events']}=={'E1','E2'}
            assert data['characters']==state['bible']['characters']
            assert suggestion in prompt
            return {'segments':[{'event_id':'E1','text':'门外暴雨不停。他原本拒绝开门，听到来人送药，才打开门。'}]}
        return original(stage,prompt,images,fresh)
    model.ask=ask
    writer=ScriptWriter(model)
    review=writer.review_story(state['bible'],state['plan'],segments)
    assert not review['passed']
    assert review['content_review']['status']=='reviewed'
    assert [i['aspect'] for i in review['issues']]==[aspect]
    repaired=writer.revise_story(state['bible'],state['plan'],segments,review)
    assert repaired[1:]==segments[1:]
    assert repaired[0]['fact_check'] is None and repaired[0]['review_status']=='pending'


def test_incomplete_review_stays_pending_without_rewriting_correct_draft(story):
    state=generate(story)
    original=story[2].ask
    def ask(stage,prompt,images=None,fresh=False):
        if stage=='script_review':return {'passed':True,'issues':[]}
        if stage in {'rewrite_story','write_story','story_digest','check'}:
            pytest.fail('Missing review metadata must not cause a whole-story rewrite or recheck')
        return original(stage,prompt,images,fresh)
    story[2].ask=ask
    result=story[3].revise(state['revision'])
    assert result['script']['segments']==state['script']['segments']
    assert result['script']['narrative_review']['content_review']['status']=='needs_review'
    assert not result['script']['narrative_review']['passed']
    assert result['stage']=='review'


@pytest.mark.parametrize('corrupt', ['quote','evidence','unanchored','unrelated_citation','contradiction','missing','skip_opening'])
def test_content_review_cannot_use_fabricated_or_unactionable_proof(corrupt):
    segments=[{'event_id':'A','text':'潮水上涨，守灯人必须守住信号。'}]
    raw=review_payload(segments,['A'])
    check=raw['content_checks'][0]
    if corrupt=='quote':check['citations'][0]['text']='正文没有的句子'
    if corrupt=='evidence':check['evidence_event_ids']=['unknown']
    if corrupt=='unanchored':
        check.update(verdict='issues',issues=[{'event_id':'unknown','detail':'缺口','suggestion':'修改'}])
    if corrupt=='unrelated_citation':
        segments.append({'event_id':'B','text':'这是未引用的正确段落。'})
        check.update(verdict='issues',issues=[{'event_id':'B','detail':'缺口','suggestion':'修改'}])
    if corrupt=='contradiction':check['verdict']='issues'
    if corrupt=='missing':raw['content_checks'].pop()
    if corrupt=='skip_opening':check['verdict']='not_applicable'
    with pytest.raises(StoryError):content_issues(segments,{'events':[{'event_id':'A'}]},raw)


def test_original_quiet_scene_long_background_and_neutral_labels_are_allowed(story):
    state=generate(story)
    segments=copy.deepcopy(state['script']['segments'])
    segments[0]['text']=('潮水将淹没堤岸，守灯人要把最后一条船引回港口。旧航道每逢涨潮便会封闭，只有旧灯塔能给出返港方向。'
        '港口有两道入口，东侧通往暗礁，西侧才是避风的内湾。白天能看见分隔水道的标记，入夜后标记就会被浪头盖住。'
        '过去船只靠岸边的钟声辨别方向，如今钟楼已在风暴中倒塌。守灯人因此把灯移到西窗，让返航的船沿着灯光找到内湾。'
        '这盏灯不能挪向东边，否则船头就会对准暗礁。岸上的人守着西窗，等待最后一条船的回应。')
    segments[1]['text']='岸上的人看着灯光，久久没有开口。守灯人也静静等着，直到那人终于点了点头。'
    segments[2]['text']='灯光照到了船头，岸上的人哭出了声。守灯人放下提灯，等哭声渐渐平息。'
    for event in state['bible']['events']:
        index=0 if event['event_id'] in {'E1','E2'} else 1 if event['event_id'] in {'E3','E4'} else 2
        event.update(summary=segments[index]['text'],actions=[segments[index]['text']])
    class Model:
        def ask(self,stage,prompt,**kwargs):
            assert '不按长度全删' in prompt and '每句话都新增客观事件' in prompt
            assert '持续分清人物' in prompt and '身份对应证据' in prompt
            raw=review_payload(segments,['E1','E3','E5'])
            # Same paragraph contains purposeful emphasis, not cross-paragraph redundancy.
            for c in raw['content_checks']:
                c['detail']='当前航道信息解释为什么必须守灯；中性称呼可分辨两人；沉默与哭声兑现等待。'
                if c['aspect']=='turns':
                    c.update(verdict='not_applicable',citations=[],evidence_event_ids=[],detail='原作没有拒绝或决定转折，不制造转折')
            return raw
    review=ScriptWriter(Model()).review_story(state['bible'],state['plan'],segments)
    assert review_ready(review,state['bible'],state['plan'],segments)


def test_unavailable_review_preserves_generated_draft_as_pending(story):
    original=story[2].ask
    def ask(stage,prompt,images=None,fresh=False):
        if stage=='script_review':raise StoryError('复核响应格式损坏')
        if stage=='rewrite_story':pytest.fail('Unavailable reviewer cannot authorize rewriting')
        return original(stage,prompt,images,fresh)
    story[2].ask=ask
    result=generate(story)
    assert [s['text'] for s in result['script']['segments']]==list(story[2].narrations.values())
    assert result['stage']=='review' and not result['script']['narrative_review']['passed']
    assert '格式损坏' in result['script']['narrative_review']['error']
    assert all(s['review_status']=='pending' for s in result['script']['segments'])


def test_semantic_callback_needs_explicit_citation_of_new_understanding():
    segments=[{'event_id':'A','text':'她等着远处的灯光。'}, {'event_id':'B','text':'她再次望向灯塔，终于松开了紧握的双手。'}]
    raw={'information_flow':[
        {'event_id':'A','new_information':['她等着远处的灯光。'],'repeats':[]},
        {'event_id':'B','new_information':['终于松开了紧握的双手。'],'repeats':[
            {'earlier_event_id':'A','purpose':'payoff','detail':'等待后的释然'}]}]}
    issues,_=information_issues(segments,raw,[])
    assert issues  # Previously any nonempty additions could excuse this repeat.
    raw['information_flow'][1]['repeats'][0]['new_information']='终于松开了紧握的双手。'
    assert information_issues(segments,raw,[])[0]==[]


@pytest.mark.parametrize('change',['text','scope','character','evidence','plan'])
def test_review_is_bound_to_current_script_evidence_and_identity(story,change):
    state=generate(story)
    review=state['script']['narrative_review']
    segments=state['script']['segments']
    assert review_ready(review,state['bible'],state['plan'],segments)
    if change=='text':segments[-1]['text']='另一个目标。'
    if change=='scope':segments[-1]['event_ids']=['E4','E5']
    if change=='character':state['bible']['characters'].append({'character_id':'C2','name':'未证实'})
    if change=='evidence':state['bible']['panels'][0]['sha256']='changed'
    if change=='plan':state['plan']['essential_event_ids'].append('E2')
    assert not review_ready(review,state['bible'],state['plan'],segments)


def test_full_content_review_is_required_at_story_and_editing_gates(tmp_path,mapped):
    source,segment,_,_=mapped
    source['plan']={'version':5,'beats':[{'event_id':'E2'}]}
    source['script']['narrative_review']={'passed':True,'issues':[], 'information_review':{'status':'reviewed'}}
    store=StoryBible(tmp_path/'chapter')
    saved=store.save(source,'fixture',None)
    with pytest.raises(StoryConflict,match='叙事内容复核'):
        StoryPipeline(store,object()).segment_action(segment['segment_id'],'approve',saved['revision'])
    segment['review_status']='approved'
    with pytest.raises(StoryError,match='叙事内容复核'):ShotEditingPipeline.approved(source)
    review=source['script']['narrative_review']
    review.update(content_review={'status':'reviewed','version':2},basis_hash=review_basis(source['bible'],source['plan'],[segment]))
    ShotEditingPipeline.approved(source)
    assert store.read()==saved


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
    ScriptWriter(story[2]).write_story(state['bible'],state['plan'])


def test_essential_trigger_cannot_be_hidden_in_verification_only_facts(story):
    state=generate(story)
    state['plan']['essential_event_ids']=['E2']
    original=story[2].ask
    def ask(stage,prompt,images=None,fresh=False):
        result=original(stage,prompt,images,fresh)
        if stage=='story_digest':result['nodes'][0]['facts'][0]['usage']='verification_only'
        return result
    story[2].ask=ask
    with pytest.raises(StoryError,match='遗漏关键转折'):
        ScriptWriter(story[2]).write_story(state['bible'],state['plan'])


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
