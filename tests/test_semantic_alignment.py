"""Offline acceptance of claim/display/timing contracts, not model quality."""
import copy
import hashlib
import json
from pathlib import Path
import socket

import pytest

from pipeline.fact_checker import FactChecker
from pipeline.narration_review import repeat_candidates
from test_engagement_flow import polish_payload
from pipeline.script_writer import ScriptWriter, narration_lint
from pipeline.semantic_grounding import make_grounding, grounded_units, measured_spans, attach_tts_boundaries
from pipeline.shot_selector import GlobalShotOptimizer, ShotSelectorAgent, required_events, coverage_report
from pipeline.story_bible import StoryBible, StoryError, StoryConflict, digest
from pipeline.story_pipeline import StoryPipeline
from pipeline.timeline_director import TimelineDirector, validate_timeline
from test_engagement_flow import polish_payload


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('No API calls allowed in local acceptance')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)


def run_mock_stream(coroutine):
    # The fake async iterator completes synchronously. No Windows event-loop
    # self-pipe socket is needed while all socket connections are forbidden.
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    raise AssertionError('Mock speech stream unexpectedly requested external I/O')


@pytest.fixture
def mapped():
    # Original lighthouse story: old storm background is NOT spoken; the
    # context fact "battery arrived" IS spoken, followed by a new action.
    bible = {'events':[
        {'event_id':'E0','order':0,'source_panel_ids':['0'],'causes':[], 'character_ids':[], 'importance':2},
        {'event_id':'E1','order':1,'source_panel_ids':['1'],'causes':[{'event_id':'E0'}], 'character_ids':[], 'importance':2},
        {'event_id':'E2','order':2,'source_panel_ids':['2'],'causes':[{'event_id':'E1'}], 'character_ids':[], 'importance':5}],
        'panels':[{'source_panel_id':str(i),'sha256':str(i)} for i in range(3)], 'characters':[]}
    segment = {'segment_id':'S_E2','event_id':'E2','event_ids':['E2'], 'context_event_ids':['E0','E1'],
               'source_panel_ids':['0','1','2'],'text':'备用电池送到了。他重新点亮灯塔。','review_status':'pending'}
    rows = [{'text':'备用电池送到了。','units':[{'text':'备用电池送到了。','event_ids':['E1'],
             'display_panel_ids':['1'],'background_event_ids':['E0']}]},
            {'text':'他重新点亮灯塔。','units':[{'text':'他重新点亮灯塔。','event_ids':['E2'],
             'display_panel_ids':['2'],'background_event_ids':['E1','E0']}]}]
    segment['fact_check'] = {'status':'passed','sentences':rows,'grounding':make_grounding(bible,segment,rows)}
    source = {'bible':bible,'plan':{'beats':[]},'script':{'segments':[segment]}}
    library = [{'panel_id':str(i),'order':i,'width':360,'height':640,'event_ids':[f'E{i}'],
                'sha256':str(i),'fingerprint':str(i)*256,'color_signature':[i*100]*192,
                'quality':.8,'visual':.6,'character_ids':[],'actions':[],'dialogue':[], 'scene':''} for i in range(3)]
    return source, segment, rows, library


def directed(mapped):
    source, segment, rows, library = mapped
    groups = GlobalShotOptimizer().optimize(source,library,ShotSelectorAgent())
    audio = {'S_E2':{'path':'local.wav','sha256':'audio','duration':4.,'sample_rate':1000,'unit_timings':[
        {'text':rows[0]['text'],'start_sample':0,'speech_end_sample':900,'end_sample':1000},
        {'text':rows[1]['text'],'start_sample':1000,'speech_end_sample':3800,'end_sample':4000}]}}
    return TimelineDirector().build(groups,audio,library), groups, audio


def test_spoken_context_has_display_but_recursive_background_does_not(mapped):
    source, segment, _, _ = mapped
    assert required_events(source,segment)==['E1','E2']
    timeline, groups, _ = directed(mapped)
    assert {s['panel_id'] for s in timeline['shots']}=={'1','2'}
    assert timeline['shots'][0]['end_frame']==30  # Not the high-importance event's whole-paragraph weight.
    assert timeline['shots'][1]['start_frame']==30
    assert validate_timeline(timeline,require_semantic=True)['semantic_alignment']['status']=='aligned'
    assert coverage_report(groups,mapped[3])['missing_semantic_units']==[]


def test_duration_perfect_but_swapped_shots_cannot_pass(mapped):
    timeline, _, _ = directed(mapped)
    timeline['shots'][0]['panel_id'],timeline['shots'][1]['panel_id']='2','1'
    report = validate_timeline(timeline)
    assert report['timeline_complete'] and report['max_audio_visual_error']==0
    assert report['semantic_alignment']['status']=='needs_review'
    assert any('错配' in r['detail'] for r in report['semantic_alignment']['issues'])
    with pytest.raises(StoryError,match='语义对应'):
        validate_timeline(timeline,require_semantic=True)


def test_missing_audio_boundaries_stays_unresolved(mapped):
    _,groups,audio = directed(mapped)
    audio['S_E2'].pop('unit_timings')
    timeline = TimelineDirector().build(groups,audio,mapped[3])
    result = validate_timeline(timeline)
    assert result['timeline_complete'] and result['semantic_alignment']['status']=='needs_review'


def test_missing_context_display_is_rejected(mapped):
    source,segment,rows,_ = mapped
    rows[0]['units'][0]['display_panel_ids']=['2']
    with pytest.raises(StoryError,match='展示来源'):
        make_grounding(source['bible'],segment,rows)


def test_partial_event_display_cannot_hide_behind_node_coverage(mapped):
    source,segment,rows,_ = mapped
    rows[0]['units'][0]['event_ids']=['E1','E2']
    with pytest.raises(StoryError,match='缺少展示依据'):
        make_grounding(source['bible'],segment,rows)


@pytest.mark.parametrize('change',['text','event','image'])
def test_stale_mapping_is_never_reused(mapped,change):
    source,segment,_,_ = mapped
    if change=='text':segment['text']='新说法。'
    elif change=='event':source['bible']['events'][0]['summary']='修改后的证据'
    else:source['bible']['panels'][1]['sha256']='changed'
    assert grounded_units(source['bible'],segment)==[]


def test_missing_sentence_mapping_does_not_fake_whole_paragraph_sources(mapped):
    source,segment,rows,_ = mapped
    rows[0].pop('units')
    mapping = make_grounding(source['bible'],segment,rows)
    assert mapping['status']=='needs_review' and not mapping['units']


def test_measured_word_edges_align_without_resynthesizing(mapped):
    source,segment,_,_ = mapped
    units = grounded_units(source['bible'],segment)
    audio = {'duration':4.,'sha256':'audio','boundary_audio_sha256':'audio','boundary_text_hash':digest(segment['text']),
             'boundaries':[{'text':'备用电池送到了','start':.1,'end':.9}, {'text':'他重新点亮灯塔','start':1.,'end':3.8}]}
    spans,problem = measured_spans(units,audio)
    assert problem is None and spans[0]['start']==0 and spans[0]['end']==1 and spans[-1]['end']==4
    audio['boundaries']=[{'text':segment['text'],'start':.1,'end':3.8}]
    spans,problem = measured_spans(units,audio)
    assert not spans and '内部' in problem  # Never manufacture timing by character proportions.


def test_boundary_sidecar_is_bound_to_audio_bytes_and_text(tmp_path):
    path = tmp_path/'cached.mp3';path.write_bytes(b'local audio')
    info = {'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    metadata = {'audio_sha256':info['sha256'],'text_hash':digest('原稿。'),
                'boundaries':[{'text':'原稿','start':0.,'end':1.}]}
    path.with_suffix('.boundaries.json').write_text(json.dumps(metadata),encoding='utf-8')
    assert attach_tts_boundaries(info,path,'原稿。')['boundaries']
    assert 'boundaries' not in attach_tts_boundaries(info,path,'改稿。')
    assert 'boundaries' not in attach_tts_boundaries({'sha256':'different'},path,'原稿。')


def test_same_stream_records_word_boundaries_once(tmp_path,monkeypatch):
    import pipeline.tts_engine as tts
    calls=[]
    class Communicate:
        def __init__(self,**kwargs):calls.append(kwargs)
        async def stream(self):
            yield {'type':'audio','data':b'audio-bytes'}
            yield {'type':'WordBoundary','offset':0,'duration':10000000,'text':'灯亮了'}
    monkeypatch.setattr(tts.edge_tts,'Communicate',Communicate)
    monkeypatch.setattr(tts,'_get_audio_duration',lambda path:1.)
    monkeypatch.setattr(tts,'apply_audio_profile',lambda *args:None)
    async def inline(function,*args):
        return function(*args)
    monkeypatch.setattr(tts.asyncio,'to_thread',inline)
    path=tmp_path/'speech.mp3'
    result=run_mock_stream(tts._generate_speech_edge('灯亮了。',str(path),rate='+25%'))
    assert len(calls)==1 and calls[0]['boundary']=='WordBoundary' and calls[0]['rate']=='+25%'
    assert result['boundaries']==[{'text':'灯亮了','start':0.,'end':1.}]
    assert json.loads(path.with_suffix('.boundaries.json').read_text('utf-8'))['audio_sha256']==result['audio_sha256']


def test_partial_speech_stream_failure_is_not_cached_as_success(tmp_path,monkeypatch):
    import pipeline.tts_engine as tts
    class Broken:
        def __init__(self,**kwargs):pass
        async def stream(self):
            yield {'type':'audio','data':b'partial'}
            raise RuntimeError('interrupted stream')
    monkeypatch.setattr(tts.edge_tts,'Communicate',Broken)
    path=tmp_path/'partial.mp3'
    with pytest.raises(RuntimeError,match='interrupted'):
        run_mock_stream(tts._generate_speech_edge('原稿',str(path)))
    assert not path.with_suffix('.boundaries.json').exists()











@pytest.mark.parametrize('value',['他背对画面站着。','她背对镜头走远。','画面出现字样。'])
def test_editorial_voice_is_blocked(value):
    assert narration_lint([{'event_id':'A','text':value}])


def test_targeted_repair_preserves_unchanged_checks_and_rechecks_changed_only(tmp_path,mapped):
    source,segment,rows,_=mapped
    other=copy.deepcopy(segment)
    other.update(event_id='E1',event_ids=['E1'],segment_id='S_E1',text='电池已经送达。',source_panel_ids=['0','1'],context_event_ids=['E0'])
    other_rows=[{'text':other['text'],'units':[{'text':other['text'],'event_ids':['E1'],
        'display_panel_ids':['1'],'background_event_ids':['E0']}]}]
    other['fact_check']={'status':'passed','unchanged_marker':True,
                         'grounding':make_grounding(source['bible'],other,other_rows),'sentences':other_rows}
    source.update(plan={'version':5,'beats':[{'event_id':'E1'},{'event_id':'E2'}]},revision=None,stage='review')
    source['script']['segments']=[other,segment]
    class Model:
        def __init__(self):self.checks=0;self.reviews=0
        def ask(self,stage,prompt,images=None,fresh=False):
            if stage=='script_polish':
                self.reviews+=1
                return {'segments':[{'event_id':'E2','text':'他重新点亮灯塔。'}]}
            if stage=='check':
                self.checks+=1
                return {'sentences':[{'sentence_index':0,'verdict':'supported','evidence':'原图灯亮', 'issues':[],
                                     'units':rows[1]['units']}]}
            raise AssertionError(stage)
    store=StoryBible(tmp_path/'chapter')
    # FactChecker receives only local evidence paths; mocked model never reads/network-calls them.
    store.evidence_path=lambda p:Path(tmp_path)/f"{p['source_panel_id']}.png"
    saved=store.save(source,'fixture',None)
    model=Model()
    result=StoryPipeline(store,model).revise(saved['revision'])
    assert model.checks==0 and model.reviews==1
    assert result['script']['segments'][0]==other
    assert result['script']['segments'][1]['fact_check'] is None
    result=StoryPipeline(store,model).verify(result['revision'])
    assert model.checks==1 and model.reviews==1 and result['script']['segments'][0]==other
    changed=result['script']['segments'][1]
    assert changed['fact_check']['status']=='passed'
    assert grounded_units(result['bible'],changed)[0]['display_panel_ids']==['2']
    assert changed['review_status']=='pending'
    assert store.read(saved['revision'])['script']['segments'][1]['text']==segment['text']


def test_targeted_repair_rejects_changes_outside_feedback(mapped):
    source,segment,_,_=mapped
    other={**copy.deepcopy(segment),'event_id':'E1','segment_id':'S_E1','text':'正确原句。'}
    class Model:
        def ask(self,*args,**kwargs):
            return {'segments':[{'event_id':'E1','text':'被擅自改写。'},{'event_id':'E2','text':'返修句子。'}]}
    with pytest.raises(StoryError,match='未被指出'):
        ScriptWriter(Model()).revise_story(source['bible'], {'beats':[]}, [other,segment], {'issues':[{'event_id':'E2'}]})


def test_manual_approval_does_not_require_model_fact_mapping(tmp_path,mapped):
    source,segment,_,_=mapped
    source['plan']={'version':5,'beats':[{'event_id':'E2'}]}
    source['script'].pop('narrative_review',None)
    source['script']['segments'][0]['fact_check'].pop('grounding')
    store=StoryBible(tmp_path/'chapter');state=store.save(source,'legacy_check',None)
    saved = StoryPipeline(store,object()).segment_action(segment['segment_id'],'approve',state['revision'])
    assert saved['script']['segments'][0]['review_status']=='approved'
    assert 'grounding' not in saved['script']['segments'][0]['fact_check']
    assert store.read(state['revision'])['script']['segments'][0]['review_status']=='pending'


def test_existing_approved_text_cannot_be_changed_by_whole_draft_revision(tmp_path,mapped):
    source,segment,_,_=mapped
    segment['review_status']='approved'
    store=StoryBible(tmp_path/'chapter')
    saved=store.save(source,'fixture',None)
    with pytest.raises(StoryConflict,match='锁定'):
        StoryPipeline(store,object()).revise(saved['revision'])
    assert store.read()==saved


def test_unit_timing_does_not_force_source_order_when_story_returns_to_old_event(mapped):
    source,segment,_,library=mapped
    rows=[{'text':'他重新点亮灯塔。','units':[{'text':'他重新点亮灯塔。','event_ids':['E2'],'display_panel_ids':['2']}]},
          {'text':'备用电池送到了。','units':[{'text':'备用电池送到了。','event_ids':['E1'],'display_panel_ids':['1']}]}]
    segment['text']=''.join(r['text'] for r in rows)
    segment['fact_check']={'status':'passed','grounding':make_grounding(source['bible'],segment,rows)}
    groups=GlobalShotOptimizer().optimize(source,library,ShotSelectorAgent())
    audio={'S_E2':{'path':'local','sha256':'local','duration':3.,'sample_rate':1000,'unit_timings':[
        {'text':rows[0]['text'],'start_sample':0,'speech_end_sample':900,'end_sample':1000},
        {'text':rows[1]['text'],'start_sample':1000,'speech_end_sample':2900,'end_sample':3000}]}}
    timeline=TimelineDirector().build(groups,audio,library)
    assert [s['panel_id'] for s in timeline['shots']]==['2','1']
    assert validate_timeline(timeline,require_semantic=True)['timeline_complete']
