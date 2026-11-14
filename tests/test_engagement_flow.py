"""Offline story-retention contracts, with original synthetic story fixtures."""
import copy
import hashlib
import json
import math
from pathlib import Path
import socket
import struct
import wave

from PIL import Image,ImageDraw
import numpy as np
import pytest

from pipeline.story_bible import StoryBible,StoryError,atomic_json,digest
from pipeline.story_pipeline import StoryPipeline
from pipeline.narrative_director import NarrativeDirector
from pipeline.fact_checker import FactChecker
from pipeline.script_writer import ScriptWriter,narration_lint
from pipeline.shot_selector import ShotSelectorAgent,GlobalShotOptimizer,build_library,required_events
from pipeline.story_inputs import presentation_regions,compact_white_gutters
from pipeline.timeline_compositor import ass_text
from pipeline.timeline_director import subtitle_chunks
from pipeline.preview_pipeline import preview_chapter
from pipeline.story_model import StoryModel,SYSTEM,WRITER_SYSTEM
from pipeline.narration_review import CONTENT_ASPECTS, CONTENT_DETAILS


def review_payload(segments, evidence_ids):
    """A structurally complete fake review; never evidence of model quality."""
    cite = {'event_id':segments[0]['event_id'],'text':segments[0]['text']}
    return {'passed':True,'issues':[], 'information_flow':[
        {'event_id':s['event_id'],'new_information':[s['text']],'repeats':[]} for s in segments],
        'content_checks':[{'aspect':aspect,'verdict':'passed','detail':'离线模拟复核的具体判断',
                           'assessments':{k:'模拟判断：仅测试字段约束，不证明内容质量' for k in CONTENT_DETAILS.get(aspect,())},
                           'citations':[cite],'evidence_event_ids':evidence_ids[:1],'issues':[]}
                          for aspect in CONTENT_ASPECTS]}


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def deny(*args,**kwargs):
        raise AssertionError('Offline acceptance must never access a network')
    monkeypatch.setattr(socket.socket,'connect',deny)
    monkeypatch.setattr(socket,'create_connection',deny)


class StoryFixtureModel:
    def __init__(self):
        self.calls = []
        self.uncertain_once = False
        self.missing_turn_once = False
        self.direct_calls = 0
        self.check_calls = 0
        self.narrations = {
            'E1':'门外大雨不停，陈默原本不肯开门。听到来人是来送药的，他改变了主意。',
            'E3':'他推开门，接过药盒。',
            'E5':'拿到药后，他准备去找家人。'}

    def ask(self,stage,prompt,images=None,fresh=False):
        self.calls.append((stage,prompt,images,fresh))
        if stage == 'event_integration':
            from integration_fixture import integration_response
            return integration_response(prompt)
        if stage == 'direct':
            self.direct_calls += 1
            ids = ['E1'] if self.missing_turn_once and self.direct_calls == 1 else ['E1','E2']
            return {'beats':[{'event_ids':ids,'section':'opening','title':'为何改变决定'},
                             {'event_ids':['E3','E4'],'section':'climax','title':'开门拿药'},
                             {'event_ids':['E5'],'section':'ending','title':'去找家人'}],
                    'essential_event_ids':[],
                    'milestones':{'conflict':'E1','climax':'E3','rules':None,'goal_or_stakes':None,'ending':'E5'}}
        if stage == 'story_plan_review':
            if self.missing_turn_once and self.direct_calls == 1:
                return {'passed':False,'missing_event_ids':['E2'],'issues':['缺少改变主意的原因：来人送药']}
            return {'passed':True,'missing_event_ids':[],'issues':[]}
        if stage in {'write_story','rewrite_story'}:
            return {'segments':[{'event_id':eid,'text':text} for eid,text in self.narrations.items()]}
        if stage == 'story_digest':
            return {'nodes':[{'event_id':eid,'facts':[{'text':text,'event_ids':ids}]} for (eid,text),ids in
                              zip(self.narrations.items(),[['E1','E2'],['E3','E4'],['E5']])]}
        if stage == 'write':
            return {'text':self.narrations['E1']}
        if stage == 'script_review':
            data = json.JSONDecoder().raw_decode(prompt.split('资料：',1)[1])[0]
            return review_payload(data['script'],[e['event_id'] for e in data['events']])
        if stage == 'check':
            self.check_calls += 1
            parts = json.loads(prompt.split('待核对句子：',1)[1].split('\n返回',1)[0])
            uncertain = self.uncertain_once and self.check_calls == 1
            return {'sentences':[{'sentence_index':p['sentence_index'],'verdict':'uncertain' if uncertain else 'supported',
                                  'evidence':'当前原图的动作和对白','issues':[]} for p in parts]}
        if stage == 'shot_select':
            data = json.loads(prompt.split('资料：',1)[1])
            return {'candidates':[{'panel_id':p['panel_id'],'score':.9,'reason':'有动作和人物主体',
                                   'visual_role':'action','crop':[0,0,1,1]} for p in data['panels']]}
        if stage == 'visual_review':
            data = json.loads(prompt.split('资料：',1)[1])
            return {'checks':[{'panel_id':pid,'usable':True,'reason':'主体和动作完整'} for pid in data['panels']]}
        raise AssertionError(stage)


@pytest.fixture
def story(tmp_path):
    store = StoryBible(tmp_path/'Chapter_00001')
    panels = []
    for i in range(1,6):
        image = Image.new('RGB',(360,640),(30+i*20,50+i*15,90+i*10))
        draw = ImageDraw.Draw(image)
        draw.rectangle((35,45,320,590),outline='white',width=7)
        draw.ellipse((100,100+i*20,260,300+i*20),fill=(220,150,i*30))
        path = tmp_path/f'panel_{i}.png';image.save(path)
        panels.append({'source_panel_id':str(i),'order':i-1,**store.snapshot(path),
                       'dialogue':[{'speaker_id':'C1','text':['不开门。','我来送药。','那就开门。','药给你。','去找家人。'][i-1]}],
                       'uncertainties':[],'panel_type':'story'})
    summaries = ['陈默拒绝在暴雨中开门','来人说明来送药','陈默听到送药后开门','陈默接过药盒','陈默决定去找家人']
    events = [{'event_id':f'E{i}','order':i-1,'summary':summary,'character_ids':['C1'],
               'actions':[summary],'importance':1 if i == 2 else 4,'role':'climax' if i == 3 else 'setup',
               'source_panel_ids':[str(i)],'dialogue':panels[i-1]['dialogue'],
               'causes':[{'event_id':'E2','evidence':'听到送药才开门'}] if i == 3 else []}
              for i,summary in enumerate(summaries,1)]
    bible = {'bible_id':'offline-fixture','panels':panels,'events':events,'characters':[], 'relationships':[]}
    state = store.save({'bible':bible,'plan':None,'script':None,'stage':'analyzed'},'fixture',None)
    model = StoryFixtureModel()
    return store,state,model,StoryPipeline(store,model)


def generate(story):
    store,state,model,pipeline = story
    return pipeline.generate_verified(state['revision'],max_events=2)


def test_plan_keeps_low_importance_turn_and_ignores_legacy_count_limit(story):
    state = generate(story)
    assert len(state['plan']['beats']) == 3
    assert 'E2' in state['plan']['beats'][0]['event_ids']
    assert state['plan']['objective']=='continuous_story_engagement'
    assert all('target_chars' not in b for b in state['plan']['beats'])
    assert state['plan']['completeness_review']['passed']


def test_missing_motivation_is_replanned_instead_of_shortened(story):
    store,state,model,pipeline = story
    model.missing_turn_once = True
    plan = NarrativeDirector(model).plan(state['bible'],max_events=2)
    assert model.direct_calls == 2 and 'E2' in plan['beats'][0]['event_ids']
    retries = [c for c in model.calls if c[0]=='direct']
    assert retries[-1][3] and '来人送药' in retries[-1][1]


def test_production_model_cannot_silently_skip_scene_planning(story):
    class LegacyResponse:
        planning_version = 5
        def ask(self,*a,**kw):
            return {'event_ids':['E1','E3','E5']}
    with pytest.raises(StoryError,match='不能退回旧'):
        NarrativeDirector(LegacyResponse()).plan(story[1]['bible'])


def test_plan_reviewer_cannot_report_already_selected_turn_as_missing(story):
    model = story[2]
    original, reviews = model.ask, []
    def ask(stage,prompt,images=None,fresh=False):
        if stage == 'story_plan_review':
            reviews.append((prompt,fresh))
            if len(reviews)==1:
                return {'passed':False,'missing_event_ids':['E2'],'issues':['没有列在essential标签']}
        return original(stage,prompt,images,fresh)
    model.ask = ask
    plan = NarrativeDirector(model).plan(story[1]['bible'])
    assert plan['completeness_review']['passed'] and model.direct_calls==1
    assert len(reviews)==2 and reviews[-1][1] and '确定的误报' in reviews[-1][0]


def test_whole_story_is_composed_before_factual_checks(story):
    result = generate(story)
    stages = [c[0] for c in story[2].calls]
    assert stages.count('write_story')==1 and 'write' not in stages
    assert stages.index('write_story') < stages.index('check')
    assert result['script']['narrative_review']['passed']
    assert all(s['review_status']=='pending' for s in result['script']['segments'])


def test_story_revision_receives_current_draft_and_preserves_grounding(story):
    model = story[2]
    original,reviews = model.ask,[]
    def ask(stage,prompt,images=None,fresh=False):
        if stage=='script_review':
            reviews.append(prompt)
            if len(reviews)==1:
                return {'passed':False,'blocking_issues':[{'event_id':'E1','detail':'需要改善衔接'}],'suggestions':[]}
        return original(stage,prompt,images,fresh)
    model.ask = ask
    state = generate(story)
    stages = [c[0] for c in model.calls]
    assert stages.count('write_story')==1 and stages.count('rewrite_story')==1
    rewrite = next(c for c in model.calls if c[0]=='rewrite_story')
    assert model.narrations['E1'] in rewrite[1] and '需要改善衔接' in rewrite[1] and rewrite[3]
    assert state['script']['segments'][0]['event_ids']==['E1','E2']
    assert state['script']['narrative_review']['passed']


def test_rejected_story_is_saved_for_review_without_approval(story):
    model = story[2]
    original = model.ask
    def ask(stage,prompt,images=None,fresh=False):
        if stage=='script_review':
            return {'passed':False,'issues':[{'event_id':'E1','detail':'真实的主线问题'}]}
        return original(stage,prompt,images,fresh)
    model.ask = ask
    with pytest.raises(StoryError,match='连贯性'):
        generate(story)
    saved = story[0].read()
    assert saved['stage']=='review' and len(saved['script']['segments'])==3
    assert not saved['script']['narrative_review']['passed']
    assert all(s['review_status']=='pending' and s['fact_check'] is None for s in saved['script']['segments'])


def test_writer_role_differs_from_fact_checker_and_thinking_stays_scoped(tmp_path):
    class Transport:
        kind='openai'
        host='https://api.deepseek.com'
        def __init__(self):self.calls=[]
        def chat(self,**kwargs):self.calls.append(kwargs);return '{}'
    transport = Transport()
    model = StoryModel(StoryBible(tmp_path/'roles'),transport=transport)
    model.ask('write_story','draft')
    model.ask('check','verify')
    assert transport.calls[0]['system']==WRITER_SYSTEM and transport.calls[1]['system']==SYSTEM
    assert transport.calls[0]['temperature']==.5
    assert transport.calls[0]['extra_body']['thinking']=={'type':'disabled'}
    assert transport.calls[1]['extra_body']['thinking']=={'type':'enabled'}
    assert transport.calls[1]['extra_body']['reasoning_effort']=='low'


def test_uncertain_sentence_is_repaired_with_new_request(story):
    story[2].uncertain_once = True
    result = generate(story)
    assert any(c[0]=='write' and c[3] for c in story[2].calls)
    assert any(c[0]=='write' and '当前段原稿' in c[1] and story[2].narrations['E1'] in c[1] for c in story[2].calls)
    assert all(s['fact_check']['status']=='passed' for s in result['script']['segments'])


def test_transient_check_failure_retries_without_rewriting_correct_text(story):
    model,original,checks = story[2],story[2].ask,[]
    def ask(stage,prompt,images=None,fresh=False):
        if stage=='check':
            checks.append(fresh)
            if len(checks)==1:
                raise RuntimeError('temporary malformed response')
        return original(stage,prompt,images,fresh)
    model.ask = ask
    state = generate(story)
    assert checks[:2]==[False,True] and not any(c[0]=='write' for c in model.calls)
    assert all(s['fact_check']['status']=='passed' for s in state['script']['segments'])


def test_group_grounding_rejects_unknown_member_even_if_model_would_approve(story):
    result = generate(story)
    segment = copy.deepcopy(result['script']['segments'][0])
    segment['event_ids'].append('invented')
    assert FactChecker(story[0],story[2]).check(result['bible'],segment)['status']=='issues'


def test_context_is_kept_for_fact_checking_without_forcing_repeated_shots(story):
    result = generate(story)
    segment = result['script']['segments'][1]
    assert segment['context_event_ids']==['E2']
    assert set(segment['source_panel_ids'])=={'2','3','4'}
    assert required_events(result,segment)==['E3','E4']


def test_visual_selector_receives_actual_candidate_images(story):
    result = generate(story)
    library = build_library(story[0],result)
    groups = GlobalShotOptimizer().optimize(result,library,ShotSelectorAgent(story[2]))
    calls = [c for c in story[2].calls if c[0]=='shot_select']
    assert calls and all(c[2] and all(Path(p).is_file() for p in c[2]) for c in calls)
    assert all(any(d.get('visual_reviewed') for d in g['decisions']) for g in groups)


def test_reading_explanations_are_rejected_even_if_model_review_says_pass(story):
    result = generate(story)
    result['script']['segments'][0]['text']='一段红色对白写着：救命。是谁还没交代。'
    review = ScriptWriter(story[2]).review_story(result['bible'],result['plan'],result['script']['segments'])
    assert not review['passed'] and review['issues']
    assert not narration_lint([{'event_id':'E1','text':'他听说来人带着药，终于打开了门。'}])


def test_rejected_actual_crop_is_restored_and_rechecked(story):
    result = generate(story)
    library = build_library(story[0],result)
    selector = ShotSelectorAgent(story[2])
    groups = GlobalShotOptimizer().optimize(result,library,selector)
    first = groups[0]['selected'][0]
    first['crop'] = [.2,.2,.4,.4]
    sizes, original = [], story[2].ask
    def review(stage,prompt,images=None,fresh=False):
        if stage != 'visual_review':
            return original(stage,prompt,images,fresh)
        data = json.loads(prompt.split('资料：',1)[1])
        sizes.append(Image.open(images[0]).size)
        return {'checks':[{'panel_id':pid,'usable':len(sizes)>1,'reason':'主体完整' if len(sizes)>1 else '主体被截断'}
                           for pid in data['panels']]}
    story[2].ask = review
    report = selector.review_selected(result,library,groups[:1])
    assert sizes[0][0] < sizes[1][0] and first['crop']==[0,0,1,1]
    assert report['status']=='passed'


def test_visual_review_cannot_invent_source_ids(story):
    result = generate(story)
    library = build_library(story[0],result)
    groups = GlobalShotOptimizer().optimize(result,library,ShotSelectorAgent())
    class BadReviewer:
        def ask(self,*a,**kw):
            return {'checks':[{'panel_id':'invented','usable':True,'reason':'虚构来源'} for _ in groups[0]['selected']]}
    with pytest.raises(StoryError,match='未知或重复'):
        ShotSelectorAgent(BadReviewer()).review_selected(result,library,groups[:1])


def test_manual_edit_invalidates_previous_whole_story_review(story):
    result = generate(story)
    result = story[3].segment_action(result['script']['segments'][0]['segment_id'],'edit',result['revision'],
                                    '一段红色对白写着：救命。')
    assert not result['script']['narrative_review']['passed']
    assert result['stage']=='review'


def test_complete_detected_panel_is_never_cut_by_fixed_height():
    image = Image.new('RGB',(200,4200),'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((15,400,185,3600),fill='#777777')
    draw.rectangle((0,1780,199,1830),fill='white')
    box = {'x':15,'y':400,'w':170,'h':3200}
    regions = presentation_regions(image,[box])
    assert any(r['y']<=400 and r['y']+r['h']>=3600 for r in regions)
    assert sum(r['h'] for r in regions)==image.height


def test_gutter_compaction_keeps_every_nonwhite_row_and_provenance():
    image = Image.new('RGB',(100,2000),'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((20,100,80,300),fill='red')
    draw.point((99,1500),fill=(254,254,254))
    original = np.asarray(image).copy()
    packed,spans = compact_white_gutters(image)
    retained = {y for a,b in spans for y in range(a,b)}
    assert all(y in retained for y in range(image.height) if (original[y]!=255).any())
    assert np.count_nonzero(np.asarray(packed)!=255)==np.count_nonzero(original!=255)
    assert packed.height < 500 and np.array_equal(original,np.asarray(image))


def test_subtitles_use_pixel_canvas_escape_tags_and_keep_latin_words():
    chunks = subtitle_chunks('青年问：楼上写着SOS，那里有人吗？')
    assert ''.join(chunks)=='青年问：楼上写着SOS，那里有人吗？'
    assert any('SOS' in c for c in chunks)
    long_word = subtitle_chunks('W'*60)
    assert ''.join(long_word)=='W'*60 and max(map(len,long_word)) <= 18
    ass = ass_text({'subtitles':[{'start_frame':0,'end_frame':30,'text':r'{\fs999}正文'}]})
    assert 'PlayResX: 1080' in ass and 'PlayResY: 1920' in ass and 'YaHei,52,' in ass
    assert r'{\fs999}' not in ass


def test_offline_preview_cache_miss_never_calls_speech_or_model(story,tmp_path):
    generate(story)
    def fail(*a,**kw):
        pytest.fail('Offline preview attempted online synthesis')
    with pytest.raises(StoryError,match='未调用任何配音API'):
        preview_chapter(story[0],tmp_path/'preview',offline=True,speech=fail,model=object())


def test_offline_preview_renders_cached_audio_and_never_changes_review_state(story,tmp_path):
    source = generate(story)
    root = tmp_path/'preview';(root/'audio').mkdir(parents=True)
    opts = {'engine':'edge','voice':'zh-CN-YunxiNeural','rate':'+25%','pitch':'-2Hz'}
    for segment in source['script']['segments']:
        key = digest({'text':segment['text'],'options':opts})
        path = root/'audio'/f'{key}.mp3'
        with wave.open(str(path),'wb') as stream:
            stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(16000)
            stream.writeframes(b''.join(struct.pack('<h',int(4000*math.sin(i*math.tau*330/16000))) for i in range(12000)))
        atomic_json(path.with_suffix('.json'),{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'duration':999})
    before = story[0].read()
    result = preview_chapter(story[0],root,offline=True)
    assert result['offline'] and not result['human_approved'] and not result['models_used']
    assert result['report']['complete'] and abs(result['report']['duration']-2.3)<.1
    assert result['report']['presentation']['subtitle_canvas']==[1080,1920]
    assert result['report']['presentation']['subtitle_font_size']==52
    assert Path(result['video']).stat().st_size>1024
    assert (root/'latest.json').is_file() and story[0].read()==before
