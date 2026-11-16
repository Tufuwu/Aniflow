"""Exact editorial text ranges survive selection, review, and measured scheduling."""
import copy
import json

from PIL import Image
import pytest

from pipeline.shot_selector import ShotSelectorAgent, GlobalShotOptimizer, coverage_report
from pipeline.shot_text_ranges import text_units, validate_ranges, build_mapping, mapped_text_units
from pipeline.story_bible import StoryError
from pipeline.timeline_director import TimelineDirector, validate_timeline
from test_semantic_alignment import mapped
from test_shot_editing import chapter
from test_manual_review_flow import ShotModel


def span(text, value, start=0):
    offset = text.index(value, start)
    return {'start_char': offset, 'end_char': offset+len(value), 'text': value}


@pytest.fixture
def ranged(tmp_path, mapped):
    source, segment, _, library = mapped
    segment.update(event_ids=['E1','E2'], fact_check=None)
    text = segment['text']
    ranges = {'1': [span(text,'备用电池送到了。')], '2': [span(text,'他重新点亮灯塔。')]}
    for panel in library:
        path = tmp_path / (panel['panel_id']+'.png')
        Image.new('RGB',(360,640),'gray').save(path)
        panel['evidence_path'] = str(path)

    class Model:
        def __init__(self):
            self.calls = []
            self.ranges = ranges

        def ask(self, stage, prompt, images=None):
            data = json.loads(prompt.split('资料：',1)[1])
            self.calls.append((stage,data,images))
            ids = [p['panel_id'] for p in data['panels']] if stage=='shot_select' else data['panels']
            assert [image.source_panel_id for image in images] == ids
            if stage=='shot_select':
                assert ''.join(u['text'] for u in data['text_units']) == segment['text']
                return {'candidates':[{'panel_id':pid,'score':.9,'reason':'原图对应当前原文','crop':[0,0,1,1],
                                      'visual_role':'action','text_ranges':[
                                          {'match_type':'direct','reason':'当前原文的动作',**copy.deepcopy(r)}
                                          for r in self.ranges.get(pid,[])]} for pid in ids]}
            if stage=='shot_text_fill':
                return {'assignments':[]}
            assert stage=='visual_review'
            assert all(row['text_ranges'] for row in data['shot_text_ranges'])
            return {'checks':[{'panel_id':pid,'usable':True,'reason':'当前原句对应的动作完整'} for pid in ids]}

    return source, segment, library, Model()


def select(ranged):
    source, _, library, model = ranged
    selector = ShotSelectorAgent(model)
    return GlobalShotOptimizer().optimize(source,library,selector), selector


def audio(texts):
    timings = [{'text':text,'start_sample':i*1000,'speech_end_sample':i*1000+900,'end_sample':(i+1)*1000}
               for i,text in enumerate(texts)]
    return {'S_E2': {'duration':float(len(texts)), 'path':'local.wav','sha256':'audio',
                     'sample_rate':1000,'unit_timings':timings}}


def test_model_ranges_reach_selected_shots_and_actual_crop_review(ranged):
    source, segment, library, model = ranged
    groups, selector = select(ranged)
    group = groups[0]
    assert group['text_mapping']['status']=='mapped'
    assert group['text_mapping']['offset_unit']=='unicode_codepoint'
    assert [s['text_ranges'][0]['text'] for s in group['selected']]==['备用电池送到了。','他重新点亮灯塔。']
    assert coverage_report(groups,library)['missing_text_ranges']==[]
    result = selector.review_selected(source,library,groups)
    assert result['status']=='passed'
    assert model.calls[-1][1]['shot_text_ranges'][0]['text_ranges']==group['selected'][0]['text_ranges']
    assert segment['fact_check'] is None  # Editorial association never invents factual approval.


def test_ranges_drive_real_audio_cut_points_and_subtitles(ranged):
    groups, _ = select(ranged)
    timeline = TimelineDirector().build(groups,audio(['备用电池送到了。','他重新点亮灯塔。']),ranged[2])
    assert [(s['panel_id'],s['start_frame'],s['end_frame']) for s in timeline['shots']]==[('1',0,30),('2',30,60)]
    assert timeline['subtitle_alignment']=='measured_shot_text_ranges'
    assert validate_timeline(timeline)['shot_text_alignment']['status']=='aligned'
    timeline['shots'][0]['panel_id']='2'
    with pytest.raises(StoryError,match='原文范围排轴'):
        validate_timeline(timeline,manual_narration_ids=['S_E2'])


def test_same_image_can_cover_disjoint_ranges_in_narration_order(ranged):
    _, segment, library, model = ranged
    text = '灯亮了。电池送达。灯又灭了。'
    segment['text']=text
    model.ranges={'1':[span(text,'电池送达。')], '2':[span(text,'灯亮了。'),span(text,'灯又灭了。')]}
    groups, _ = select(ranged)
    timeline = TimelineDirector().build(groups,audio(['灯亮了。','电池送达。','灯又灭了。']),library)
    assert [s['panel_id'] for s in timeline['shots']]==['2','1','2']
    assert len({s['shot_id'] for s in timeline['shots']})==3
    assert validate_timeline(timeline)['shot_text_alignment']['status']=='aligned'


def test_overlapping_ranges_can_use_multiple_images_in_the_shared_interval(ranged):
    _, segment, library, model = ranged
    model.ranges['1']=[span(segment['text'],segment['text'])]
    groups, _ = select(ranged)
    timeline = TimelineDirector().build(groups,audio(['备用电池送到了。','他重新点亮灯塔。']),library)
    assert [s['panel_id'] for s in timeline['shots']]==['1','1','2']
    assert validate_timeline(timeline)['shot_text_alignment']['status']=='aligned'


def test_missing_audio_edges_do_not_guess_times_from_character_lengths(ranged):
    groups, _ = select(ranged)
    info=audio(['备用电池送到了。他重新点亮灯塔。'])
    with pytest.raises(StoryError,match='实测配音对齐'):
        TimelineDirector().build(groups,info,ranged[2])


@pytest.mark.parametrize('change', ['text','range'])
def test_saved_associations_expire_when_text_or_ranges_change(ranged,change):
    groups, _ = select(ranged)
    group=groups[0]
    if change=='text':
        group['text']+='新句。'
    else:
        group['selected'][0]['text_ranges']=[span(group['text'],group['text'])]
    with pytest.raises(StoryError,match='过期'):
        mapped_text_units(group)


def test_unmapped_selected_picture_and_uncovered_words_stay_reviewable(ranged):
    ranged[3].ranges['2']=[]
    groups, _ = select(ranged)
    assert groups[0]['text_mapping']['status']=='needs_review'
    assert coverage_report(groups,ranged[2])['missing_text_ranges']
    with pytest.raises(StoryError,match='完整关联'):
        mapped_text_units(groups[0])


def test_unicode_positions_use_code_points_and_preserve_whitespace():
    text='🎬 道夏来了。\n\n“肉罐头？”  他停下。'
    units=text_units(text)
    assert ''.join(u['text'] for u in units)==text
    value=span(text,'道夏来了。')
    assert value['start_char']==2
    assert validate_ranges(text,[value])==[value]
    bad={**value,'start_char':3,'end_char':value['end_char']+1}
    with pytest.raises(StoryError,match='不一致'):
        validate_ranges(text,[bad])
    group={'narration_id':'S','text':text,'selected':[{'panel_id':'1','text_ranges':[span(text,text.strip())]}]}
    assert build_mapping(group)['status']=='mapped'


@pytest.mark.parametrize('bad', [None, [{'start_char':True,'end_char':2,'text':'甲乙'}],
    [{'start_char':0,'end_char':9,'text':'甲乙。'}], [{'start_char':0,'end_char':2,'text':'改写'}],
    [{'start_char':0,'end_char':2,'text':'甲乙'},{'start_char':1,'end_char':3,'text':'乙。'}]])
def test_invalid_model_ranges_are_rejected(bad):
    with pytest.raises(StoryError):
        validate_ranges('甲乙。',bad)


def test_locked_shot_associations_survive_reruns(ranged):
    groups, _ = select(ranged)
    groups[0].update(locked=True,review_status='approved')
    before=copy.deepcopy(groups[0])
    ranged[3].ask=lambda *a,**k:pytest.fail('Locked choices must not call a model')
    result=GlobalShotOptimizer().optimize(ranged[0],ranged[2],ShotSelectorAgent(ranged[3]),{'S_E2':groups[0]})
    assert result==[before]


def test_manual_range_edit_is_saved_checked_and_preserved_when_locked(chapter):
    _, source, pipeline = chapter
    pipeline.model=ShotModel()
    state=pipeline.select(None,source['revision'],True)
    group=state['groups'][0]
    previous=copy.deepcopy(group)
    shot=copy.deepcopy(group['selected'][0])
    text=group['text']
    cut=text.index('，')+1
    shot['text_ranges']=[span(text,text[:cut]),span(text,text[cut:])]
    state=pipeline.review(group['narration_id'],'update',state['revision'],{'selected':[shot]})
    assert len(state['groups'][0]['selected'][0]['text_ranges'])==2
    assert mapped_text_units(state['groups'][0])
    state=pipeline.review(group['narration_id'],'approve',state['revision'],{})
    locked=copy.deepcopy(state['groups'][0])
    state=pipeline.select(state['revision'],source['revision'],True)
    assert state['groups'][0]==locked and locked['locked']
    assert previous['selected'][0]['text_ranges']!=locked['selected'][0]['text_ranges']


def test_invalid_manual_range_edit_never_commits_a_partial_update(chapter):
    _, source, pipeline = chapter
    pipeline.model=ShotModel()
    state=pipeline.select(None,source['revision'],True)
    group=state['groups'][0]
    shot=copy.deepcopy(group['selected'][0])
    shot['text_ranges']=[{'start_char':0,'end_char':999,'text':'越界'}]
    with pytest.raises(StoryError,match='越界'):
        pipeline.review(group['narration_id'],'update',state['revision'],{'selected':[shot]})
    assert pipeline.store.read()==state
