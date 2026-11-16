"""Neighbor images fill editorial gaps without becoming current event evidence."""
import copy
import json

from PIL import Image
import pytest

from pipeline.shot_selector import ShotSelectorAgent,GlobalShotOptimizer,coverage_report
from pipeline.shot_text_ranges import validate_ranges,mapped_text_units
from pipeline.story_bible import StoryError,digest
from pipeline.timeline_director import TimelineDirector,validate_timeline
from test_semantic_alignment import mapped
from test_shot_text_ranges import span,audio
from test_shot_editing import chapter
from test_manual_review_flow import ShotModel
from pipeline.shot_text_ranges import text_units


@pytest.fixture
def context_case(tmp_path,mapped):
    source,_,_,library=mapped
    texts=['他看着手机。','消息又来了。肉罐头！看来值得。','他们继续上路。']
    segments=[{'segment_id':f'S_E{i}','event_id':f'E{i}','event_ids':[f'E{i}'],
               'context_event_ids':[],'source_panel_ids':[str(i)],'text':text,'fact_check':None,
               'review_status':'pending'} for i,text in enumerate(texts)]
    source['script']['segments']=segments
    source['plan']['beats']=[{'event_id':s['event_id'],'event_ids':s['event_ids']} for s in segments]
    current=segments[1]
    def association(value,kind):
        return {**span(current['text'],value),'match_type':kind,'reason':'延续相关画面，不提前展示消息'}
    ranges={'0':[association('消息又来了。','transition'),association('看来值得。','commentary')],
            '1':[association('肉罐头！','direct')],'2':[]}
    for panel in library:
        path=tmp_path/(panel['panel_id']+'.png')
        Image.new('RGB',(360,640),'gray').save(path)
        panel['evidence_path']=str(path)

    class Model:
        def __init__(self):self.calls=[];self.ranges=ranges;self.reject_future=False
        def ask(self,stage,prompt,images=None):
            data=json.loads(prompt.split('资料：',1)[1])
            self.calls.append((stage,data,images))
            if stage=='shot_select':
                assert {n['direction'] for n in data['adjacent_narrations']}=={'previous','next'}
                assert [p.source_panel_id for p in images]==[p['panel_id'] for p in data['panels']]
                return {'candidates':[{'panel_id':p['panel_id'],'score':.99 if p.get('context_only') else .8,
                    'reason':'当前内容或相邻承接','visual_role':'reaction','crop':[0,0,1,1],
                    'text_ranges':copy.deepcopy(self.ranges[p['panel_id']])} for p in data['panels']]}
            if stage=='shot_text_fill':
                return {'assignments':[]}
            assert stage=='visual_review'
            assert data['shot_context'] and data['adjacent_narrations']
            return {'checks':[{'panel_id':pid,'usable':not (self.reject_future and pid=='2'),
                'reason':'画面提前揭晓结果' if self.reject_future and pid=='2' else '承接方式合理，原文与画面一致'}
                for pid in data['panels']]}
    return source,current,library,Model()


def chosen(case):
    source,current,library,model=case
    selector=ShotSelectorAgent(model)
    rows,required=selector.candidates(source,library,current)
    class Saved:
        def candidates(self,*args):return copy.deepcopy(rows),list(required)
    subset={**source,'script':{'segments':[current]}}
    return GlobalShotOptimizer().optimize(subset,library,Saved())[0],selector,rows


def test_previous_image_fills_transition_and_commentary_gaps(context_case):
    source,current,library,model=context_case
    group,selector,rows=chosen(context_case)
    assert [s['panel_id'] for s in group['selected']]==['0','1']
    borrowed=group['selected'][0]
    assert borrowed['context_only'] and borrowed['context_origins'][0]['event_id']=='E0'
    assert [r['match_type'] for r in borrowed['text_ranges']]==['transition','commentary']
    assert group['text_mapping']['status']=='mapped'
    report=coverage_report([group],library)
    assert report['required_event_ids']==['E1'] and report['missing_text_ranges']==[]
    assert next(r for r in rows if r['panel_id']=='0')['covers']==[]
    assert selector.review_selected(source,library,[group])['status']=='passed'
    assert model.calls[-1][1]['shot_context'][0]['context_only']
    assert current['fact_check'] is None


def test_transition_and_commentary_follow_measured_audio_without_extra_cut(context_case):
    _,_,library,_=context_case
    group,_,_=chosen(context_case)
    performance=next(iter(audio(['消息又来了。','肉罐头！','看来值得。']).values()))
    timeline=TimelineDirector().build([group],{'S_E1':performance},library)
    assert [(s['panel_id'],s['start_frame'],s['end_frame']) for s in timeline['shots']]==[
        ('0',0,30),('1',30,60),('0',60,90)]
    assert [u['associations'][0]['match_type'] for u in timeline['shot_text_units']]==['transition','direct','commentary']
    assert validate_timeline(timeline)['shot_text_alignment']['status']=='aligned'


def test_context_image_is_not_added_when_current_images_cover_all_words(context_case):
    _,current,_,model=context_case
    model.ranges['1']=[{**span(current['text'],current['text']),'match_type':'direct','reason':'当前画面可持续承接'}]
    group,_,_=chosen(context_case)
    assert [s['panel_id'] for s in group['selected']]==['1']


def test_gaps_are_not_filled_by_unproposed_neighbor_images(context_case):
    context_case[3].ranges['0']=[]
    group,_,_=chosen(context_case)
    assert group['text_mapping']['status']=='needs_review'
    assert [s['panel_id'] for s in group['selected']]==['1']


def test_borrowed_image_cannot_claim_new_current_facts(context_case):
    context_case[3].ranges['0'][0]['match_type']='direct'
    group,_,rows=chosen(context_case)
    row=next(r for r in rows if r['panel_id']=='0')
    assert not row['eligible'] and '借用相邻镜头' in row['context_validation_error']
    assert group['text_mapping']['status']=='needs_review'


@pytest.mark.parametrize('kind,phrase',[('callback','看来值得。'),('commentary','看来值得。'),('transition','消息又来了。')])
def test_future_neighbor_is_restricted_to_tail_transition(context_case,kind,phrase):
    _,current,_,model=context_case
    model.ranges['0']=model.ranges['0'][:1]
    model.ranges['2']=[{**span(current['text'],phrase),'match_type':kind,'reason':'下一段候选'}]
    group,_,rows=chosen(context_case)
    row=next(r for r in rows if r['panel_id']=='2')
    assert not row['eligible'] and '后段镜头' in row['context_validation_error']
    assert '2' not in [s['panel_id'] for s in group['selected']]


def test_future_transition_still_needs_visual_reveal_review(context_case):
    source,current,library,model=context_case
    model.ranges['0']=model.ranges['0'][:1]
    model.ranges['2']=[{**span(current['text'],'看来值得。'),'match_type':'transition','reason':'段尾转场'}]
    model.reject_future=True
    group,selector,_=chosen(context_case)
    assert [s['panel_id'] for s in group['selected']]==['0','1','2']
    assert selector.review_selected(source,library,[group])['status']=='needs_review'
    assert group['review_status']=='pending'


def test_callback_reuse_is_explicit_and_not_penalized_as_accidental_duplicate(context_case):
    _,_,library,model=context_case
    model.ranges['0'][-1]['match_type']='callback'
    group,_,_=chosen(context_case)
    previous={'narration_id':'S_E0','text':'他看着手机。','required_event_ids':['E0'],
              'selected':[{'panel_id':'0'}]}
    report=coverage_report([previous,group],library)
    assert report['intentional_reuse_count']==1 and report['duplicate_count']==0


def test_borrowed_origin_and_reason_are_bound_to_saved_mapping(context_case):
    group,_,_=chosen(context_case)
    group['selected'][0]['context_origins'][0]['event_id']='changed'
    with pytest.raises(StoryError,match='过期'):
        mapped_text_units(group)


def test_old_version_one_mapping_remains_readable_without_rewriting():
    text='甲。';ranges=[{'start_char':0,'end_char':2,'text':text}]
    group={'narration_id':'S','text':text,'selected':[{'panel_id':'1','text_ranges':ranges}],
           'locked':True,'review_status':'approved'}
    units=[{'start_char':0,'end_char':2,'text':text,'display_panel_ids':['1'],'unit_id':'S:R1'}]
    group['text_mapping']={'version':1,'offset_unit':'unicode_codepoint','end_exclusive':True,
        'text_hash':digest(text),'basis_hash':digest({'text':text,'shots':[{'panel_id':'1','text_ranges':ranges}]}),
        'status':'mapped','issues':[],'units':units}
    before=copy.deepcopy(group)
    assert mapped_text_units(group)==units and group==before


def test_manual_borrow_can_be_saved_pending_and_completed_before_approval(chapter):
    _,source,pipeline=chapter
    pipeline.model=ShotModel()
    state=pipeline.select(None,source['revision'],True)
    group=state['groups'][0]
    row=next(r for r in group['decisions'] if r.get('context_only'))
    proposed=copy.deepcopy(group['selected'])+[{'panel_id':row['panel_id'],'crop':[0,0,1,1],
        'motion':'auto','transition':'cut','duration_frames':None,'text_ranges':[]}]
    state=pipeline.review(group['narration_id'],'update',state['revision'],{'selected':proposed})
    assert state['groups'][0]['text_mapping']['status']=='needs_review'
    with pytest.raises(StoryError,match='完整关联'):
        pipeline.review(group['narration_id'],'approve',state['revision'],{})
    proposed=copy.deepcopy(state['groups'][0]['selected'])
    proposed[-1]['text_ranges']=[{**text_units(group['text'])[-1],'match_type':'transition',
                                  'reason':'段尾接入相邻场景，人工确认无提前揭晓'}]
    state=pipeline.review(group['narration_id'],'update',state['revision'],{'selected':proposed})
    assert state['groups'][0]['text_mapping']['status']=='mapped'
    state=pipeline.review(group['narration_id'],'approve',state['revision'],{})
    assert state['groups'][0]['review_status']=='approved'


def test_explicit_gap_fill_uses_previous_reaction_and_preserves_existing_ranges(context_case):
    source,current,library,model=context_case
    model.ranges['0']=[]
    group,selector,_=chosen(context_case)
    original=copy.deepcopy(group['selected'][0]['text_ranges'])
    ask=model.ask
    def fill(stage,prompt,images=None):
        if stage!='shot_text_fill':return ask(stage,prompt,images)
        data=json.loads(prompt.split('资料：',1)[1])
        assert '2' not in [p['panel_id'] for p in data['panels']]
        return {'assignments':[{'panel_id':'0','text_ranges':[
            {**r,'match_type':'transition' if i==0 else 'commentary','reason':'延用之前的反应承接文字'}
            for i,r in enumerate(data['missing_text_ranges'])]}]}
    model.ask=fill
    result=selector.fill_text_gaps(source,library,group)
    assert result['text_mapping']['status']=='mapped' and result['text_fill']['attempts']==1
    assert next(s for s in result['selected'] if s['panel_id']=='1')['text_ranges']==original
    borrowed=next(s for s in result['selected'] if s['panel_id']=='0')
    assert borrowed['context_only'] and len(borrowed['text_ranges'])==2


def test_invalid_gap_fill_never_partially_applies_or_overwrites_existing_text(context_case):
    source,current,library,model=context_case
    model.ranges['0']=[]
    group,selector,_=chosen(context_case)
    before=copy.deepcopy(group['selected'])
    def fill(stage,prompt,images=None):
        data=json.loads(prompt.split('资料：',1)[1])
        return {'assignments':[
            {'panel_id':'0','text_ranges':[{**data['missing_text_ranges'][0],
                'match_type':'transition','reason':'延用上一镜'}]},
            {'panel_id':'1','text_ranges':[{**span(current['text'],current['text']),
                'match_type':'commentary','reason':'非法覆盖已配正文'}]}]}
    model.ask=fill
    result=selector.fill_text_gaps(source,library,group)
    assert result['selected']==before and result['text_mapping']['status']=='needs_review'
    assert '超出指定原文范围' in result['text_fill']['errors'][0]


def test_gap_fill_cannot_import_a_future_picture(context_case):
    source,current,library,model=context_case
    model.ranges['0']=[]
    group,selector,_=chosen(context_case)
    def fill(stage,prompt,images=None):
        data=json.loads(prompt.split('资料：',1)[1])
        return {'assignments':[{'panel_id':'2','text_ranges':[
            {**data['missing_text_ranges'][-1],'match_type':'transition','reason':'非法引用下一段'}]}]}
    model.ask=fill
    result=selector.fill_text_gaps(source,library,group)
    assert '未知或重复图片' in result['text_fill']['errors'][0]
    assert result['text_mapping']['status']=='needs_review'


@pytest.mark.parametrize('annotation',[{'match_type':'wrong','reason':'有理由'}, {'match_type':'transition'}, {'match_type':'commentary','reason':''}])
def test_new_model_associations_require_valid_mode_and_reason(annotation):
    with pytest.raises(StoryError,match='承接'):
        validate_ranges('甲。',[{'start_char':0,'end_char':2,'text':'甲。',**annotation}],require_annotations=True)
