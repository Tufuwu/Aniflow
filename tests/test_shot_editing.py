"""Isolated single-chapter acceptance: no cloud calls or existing review-data writes."""
import copy
import hashlib
import json
from pathlib import Path
import threading
import time
import wave

from flask import Flask
from PIL import Image, ImageDraw
import pytest

import config
from pipeline.shot_api import create_shot_blueprint
from pipeline.shot_pipeline import ShotEditingPipeline
from pipeline.shot_selector import (ShotSelectorAgent, GlobalShotOptimizer, build_library,
                                    coverage_report, same_information, visually_similar)
from pipeline.story_bible import StoryBible, StoryConflict, StoryError
from pipeline.timeline_director import TimelineDirector, audio_duration, validate_timeline


def speech_fixture(text, output_path, **kwargs):
    # Deliberately return a wrong duration: the director must measure the real waveform.
    duration = .57 if text.startswith('雨') else .81
    with wave.open(output_path, 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        import math
        import struct
        stream.writeframes(b''.join(struct.pack('<h', int(5000 * math.sin(i * math.tau * 440 / 16000)))
                                    for i in range(round(duration * 16000))))
    return {"duration": 999}


@pytest.fixture
def chapter(tmp_path):
    store = StoryBible(tmp_path / 'Chapter_00001')
    panels = []
    for i in range(1, 6):
        image = Image.new('RGB', (360, 640) if i != 3 else (40, 60), '#d4e7db')
        draw = ImageDraw.Draw(image)
        draw.rectangle((5, 10, image.width - 10, image.height - 15), outline='#0c5b48', width=4)
        draw.ellipse((10, 20, image.width - 15, image.height // 2), fill='#7681ac')
        if i in (4, 5):
            draw.rectangle((40, 300, 300, 550), fill='#f8c878')
        path = tmp_path / f'original_{i}.png'
        image.save(path)
        panels.append({'source_panel_id': str(i), 'order': i-1, **store.snapshot(path),
                       'panel_type': 'story', 'boundary_uncertain': i == 3,
                       'dialogue': [{'text': '额外线索' if i == 5 else '雨来了' if i <= 2 else '别动'}]})
    events = [{'event_id': f'E{i}', 'order': i-1, 'summary': '下雨' if i == 1 else '规则' if i == 2 else '开门',
               'character_ids': ['C1'], 'actions': ['等雨' if i == 1 else '警告' if i == 2 else '开门'],
               'source_panel_ids': ['1','2'] if i == 1 else ['3'] if i == 2 else ['4','5'],
               'importance': 5 if i == 3 else 3, 'role': 'climax' if i == 3 else 'setup',
               'causes': [{'event_id':'E2'}] if i == 3 else []} for i in range(1,4)]
    segments = [{'segment_id': f'S_E{i}', 'event_id': f'E{i}',
                 'text': '雨落下，他走到门口。' if i == 1 else '她警告他别动，但门还是开了。',
                 'context_event_ids': ['E2'] if i == 3 else [],
                 'review_status': 'approved', 'fact_check': {'status': 'passed'}} for i in (1,3)]
    state = {'bible': {'bible_id':'fixture', 'panels':panels, 'characters':[], 'events':events},
             'plan': {'beats':[{'event_id':s['event_id']} for s in segments],
                      'milestones':{'rules':'E2','climax':'E3'}, 'opening_event_id':'E1','ending_event_id':'E3'},
             'script':{'segments':segments}, 'stage':'approved'}
    source = store.save(state, 'fixture_approved', None)
    pipeline = ShotEditingPipeline(store, speech=speech_fixture)
    return store, source, pipeline


def selected(chapter):
    store, source, pipeline = chapter
    return pipeline.select(None, source['revision'], use_model=False)


def reviewed(chapter):
    state = selected(chapter)
    pipeline = chapter[2]
    for group in state['groups']:
        state = pipeline.review(group['narration_id'], 'approve', state['revision'], {})
    return state


def test_library_keeps_all_and_low_quality_causal_event_is_mandatory(chapter):
    state = selected(chapter)
    assert len(state['library']) == 5
    assert state['metrics']['coverage_rate'] == 1
    assert state['metrics']['missing_by_narration'] == {}
    group = state['groups'][1]
    assert 'E2' in group['required_event_ids']
    assert '3' in {s['panel_id'] for s in group['selected']}
    assert state['library'][2]['quality'] < state['library'][0]['quality']
    assert all(len(g['decisions']) == 5 and all(r['reason'] for r in g['decisions']) for g in state['groups'])


def test_duplicate_compression_preserves_new_dialogue(chapter):
    state = selected(chapter)
    library = state['library']
    assert visually_similar(library[0], library[1]) and same_information(library[0], library[1])
    assert visually_similar(library[3], library[4]) and not same_information(library[3], library[4])
    assert len(state['groups'][0]['selected']) == 1
    assert any('压缩重复' in r['reason'] for r in state['groups'][0]['decisions'])
    assert state['metrics']['duplicate_rate'] == 0


def test_cannot_remove_necessary_story_and_failed_update_does_not_commit(chapter):
    pipeline = chapter[2]
    state = selected(chapter)
    group = state['groups'][1]
    with pytest.raises(StoryError, match='必要剧情'):
        pipeline.review(group['narration_id'], 'update', state['revision'], {'selected':[{'panel_id':'4'}]})
    assert pipeline.store.read()['revision'] == state['revision']


def test_manual_crop_selection_order_reasons_and_locks_survive_ai_rerun(chapter):
    store, source, pipeline = chapter
    originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in store.root.parent.glob('original*')}
    source_bytes = (store.root / 'revisions' / f"{source['revision']}.json").read_bytes()
    state = selected(chapter)
    group = state['groups'][1]
    state = pipeline.review(group['narration_id'], 'update', state['revision'], {'selected':[
        {'panel_id':'3','crop':[.1,.1,.8,.8],'motion':'pan_down','transition':'fade','duration_frames':9},
        {'panel_id':'5','motion':'zoom_out'}]})
    state = pipeline.review(group['narration_id'], 'approve', state['revision'], {})
    locked = copy.deepcopy(state['groups'][1])
    state = pipeline.select(state['revision'], source['revision'], use_model=False)
    assert state['groups'][1] == locked
    assert originals == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in originals}
    assert store.read()['revision'] == source['revision']
    assert (store.root / 'revisions' / f"{source['revision']}.json").read_bytes() == source_bytes
    with pytest.raises(StoryConflict):
        pipeline.select(None, source['revision'], use_model=False)


def test_unapproved_text_or_incomplete_milestones_are_blocked(chapter):
    store, source, pipeline = chapter
    source['script']['segments'][0]['review_status'] = 'pending'
    pending = store.save(source, 'pending', source['revision'])
    with pytest.raises(StoryError, match='审核'):
        pipeline.select(None, pending['revision'], use_model=False)
    pending['script']['segments'][0]['review_status'] = 'approved'
    pending['plan']['milestones']['rules'] = 'E404'
    invalid = store.save(pending, 'missing_milestone', pending['revision'])
    with pytest.raises(StoryError, match='里程碑'):
        pipeline.select(None, invalid['revision'], use_model=False)


def test_missing_event_evidence_fails_closed(chapter):
    store, source, pipeline = chapter
    source['bible']['events'][1]['source_panel_ids'] = ['404']
    changed = store.save(source, 'missing_evidence', source['revision'])
    with pytest.raises(StoryError, match='必要剧情'):
        pipeline.select(None, changed['revision'], use_model=False)
    assert pipeline.store.read()['revision'] is None


def test_actual_tts_measurement_frame_sync_and_cache(chapter):
    pipeline = chapter[2]
    state = reviewed(chapter)
    state = pipeline.direct(state['revision'])
    assert [round(a['duration'],2) for a in state['audio'].values()] == [.57,.81]
    assert validate_timeline(state['timeline'])['max_audio_visual_error'] < 1/30
    assert state['timeline']['total_frames'] == 18+25
    assert state['timeline']['subtitles'][-1]['end_frame'] == 43
    first = {n:a['sha256'] for n,a in state['audio'].items()}
    pipeline.speech = lambda *a,**kw: pytest.fail('valid cached audio must not regenerate')
    repeated = pipeline.direct(state['revision'])
    assert first == {n:a['sha256'] for n,a in repeated['audio'].items()}
    assert state['timeline'] == repeated['timeline']


def test_manual_duration_conflict_and_audio_tamper_block_export(chapter):
    pipeline = chapter[2]
    state = selected(chapter)
    group = state['groups'][0]
    shot = copy.deepcopy(group['selected'][0]);shot['duration_frames']=300
    state = pipeline.review(group['narration_id'], 'update', state['revision'], {'selected':[shot]})
    with pytest.raises(StoryError, match='人工固定'):
        pipeline.direct(state['revision'])
    shot['duration_frames']=None
    state = pipeline.review(group['narration_id'], 'update', state['revision'], {'selected':[shot]})
    for group in state['groups']:
        state = pipeline.review(group['narration_id'], 'approve', state['revision'], {})
    state = pipeline.direct(state['revision'])
    Path(next(iter(state['audio'].values()))['path']).write_bytes(b'tampered')
    with pytest.raises(StoryError, match='音频内容'):
        pipeline.export(state['revision'])


def test_global_optimization_considers_future_continuity(chapter):
    store, source, _ = chapter
    library = build_library(store, source)
    library[0]['character_ids']=['C_other'];library[0]['visual']=.9
    library[1]['character_ids']=['C1'];library[1]['visual']=.2
    for panel in library[2:]:panel['character_ids']=['C1']
    class Selector(ShotSelectorAgent):
        def candidates(self, source, library, segment):
            rows, required = super().candidates(source, library, segment)
            for row in rows:
                row['score']=.9 if row['panel_id']=='1' else .89
            return rows, required
    groups=GlobalShotOptimizer().optimize(source,library,Selector())
    assert groups[0]['selected'][0]['panel_id']=='2'


def test_model_cannot_invent_panel_ids(chapter):
    store, source, pipeline = chapter
    class Model:
        def ask(self,*a,**kw):return {'candidates':[{'panel_id':'invented','score':1,'reason':'伪造'}]}
    pipeline.model=Model()
    with pytest.raises(StoryError,match='无证据'):
        pipeline.select(None,source['revision'],True)


def test_long_narration_can_add_distinct_evidence_but_never_redundant_images(chapter):
    store,source,_=chapter
    source['script']['segments'][0]['text'] *= 8
    source['script']['segments'][1]['text'] *= 8
    library=build_library(store,source)
    groups=GlobalShotOptimizer().optimize(source,library,ShotSelectorAgent())
    assert len(groups[0]['selected'])==1  # Same image and same information cannot fill a pacing budget.
    assert {s['panel_id'] for s in groups[1]['selected']}=={'3','4','5'}


def test_color_and_story_information_prevent_false_duplicate_merges(chapter):
    state=selected(chapter)
    a,b=copy.deepcopy(state['library'][:2])
    b['sha256']='different'
    b['color_signature']=[255-v for v in b['color_signature']]
    assert not visually_similar(a,b)


def test_sync_validator_rejects_shifted_audio_and_missing_subtitles(chapter):
    pipeline=chapter[2]
    state=pipeline.direct(reviewed(chapter)['revision'])
    changed=copy.deepcopy(state['timeline'])
    changed['narration'][0]['start_frame']=1
    with pytest.raises(StoryError,match='旁白时间轴'):
        validate_timeline(changed)
    changed=copy.deepcopy(state['timeline'])
    changed['subtitles'].pop()
    with pytest.raises(StoryError,match='字幕未完整'):
        validate_timeline(changed)


def wait_job(client, base, response):
    assert response.status_code == 202, response.get_json()
    jid = response.get_json()['job_id']
    for _ in range(1000):
        result=client.get(f'{base}/jobs/{jid}').get_json()
        if result['status'] != 'running':return result
        time.sleep(.01)
    pytest.fail('editing job timed out')


def test_api_single_chapter_history_conflict_and_media_access(chapter, monkeypatch):
    store, source, _ = chapter
    monkeypatch.setattr(config,'STORY_DIR',str(store.root.parent.parent))
    monkeypatch.setattr('pipeline.shot_api.story_store',lambda project:store if project=='test' else (_ for _ in ()).throw(StoryError('非法项目')))
    app=Flask(__name__,template_folder=str(Path(__file__).parents[1]/'templates'))
    app.register_blueprint(create_shot_blueprint(speech=speech_fixture))
    client=app.test_client();base='/api/editing/test'
    assert client.get('/shot-review').status_code==200
    assert client.post(base+'/select',json={}).status_code==422
    result=wait_job(client,base,client.post(base+'/select',json={'revision':None,'source_revision':source['revision'],'use_model':False}))
    assert result['status']=='completed',result
    state=client.get(base).get_json(); revision=state['revision']
    assert client.get(base+'/evidence/3').status_code==200
    assert client.get(base+'/preview/S_E3/3').status_code==200
    assert client.get(base+'/preview/S_E3/404').status_code==404
    assert client.get(base+'/evidence/404').status_code==404
    assert client.post(base+'/select',json={'revision':None}).status_code==409
    group=state['groups'][0]
    result=wait_job(client,base,client.post(base+f"/groups/{group['narration_id']}/approve",json={'revision':revision}))
    assert result['status']=='completed'
    assert client.get(base+'?revision='+revision).get_json()['groups'][0]['review_status']=='pending'
    assert client.get(base+'?revision=../../config.py').status_code==422
    assert client.get(base+'/exports/'+'a'*32+'/video.mp4').status_code==404
    assert client.get(base+'/exports/'+'a'*32+'/config.py').status_code==404


def test_parallel_edit_is_rejected(chapter, monkeypatch):
    store, source, pipeline=chapter
    state=selected(chapter)
    monkeypatch.setattr('pipeline.shot_api.story_store',lambda project:store)
    entered,release=threading.Event(),threading.Event()
    def blocking_speech(*a,**kw):entered.set();release.wait(5);return speech_fixture(*a,**kw)
    app=Flask(__name__);app.register_blueprint(create_shot_blueprint(speech=blocking_speech))
    client=app.test_client();base='/api/editing/test'
    response=client.post(base+'/direct',json={'revision':state['revision']})
    assert entered.wait(3)
    assert client.post(base+'/select',json={'revision':state['revision']}).status_code==409
    release.set()
    assert wait_job(client,base,response)['status']=='completed'


def test_export_requires_shot_review(chapter):
    pipeline=chapter[2]
    state=selected(chapter)
    with pytest.raises(StoryError,match='逐段审核'):
        pipeline.export(state['revision'])


def test_real_portrait_export_decodes_completely_and_preserves_previous_export(chapter):
    pytest.importorskip('moviepy')
    store,source,pipeline=chapter
    state=reviewed(chapter)
    # Exercise crop, push/pull, pan and transition in the real encoder.
    group=state['groups'][1]
    shots=copy.deepcopy(group['selected'])
    shots[0].update(crop=[.1,.1,.8,.8],motion='pan_down',transition='fade')
    state=pipeline.review(group['narration_id'],'update',state['revision'],{'selected':shots})
    state=pipeline.review(group['narration_id'],'approve',state['revision'],{})
    state=pipeline.direct(state['revision'])
    state=pipeline.export(state['revision'])
    report=state['export']['report']
    assert report['complete'] and (report['width'],report['height'])==(1080,1920)
    assert report['codec']=='h264' and report['decoded_frames']==43
    root=pipeline.store.root/'exports'/state['export']['export_id']
    assert (root/'video.mp4').stat().st_size>1024
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['source_revision']==source['revision']
    assert '雨' in (root/'subtitles.srt').read_text(encoding='utf-8')
    saved=(root/'video.mp4').read_bytes()
    truncated=root/'truncated.mp4'
    truncated.write_bytes(saved[:len(saved)//2])
    from pipeline.timeline_compositor import inspect_export
    with pytest.raises(StoryError):
        inspect_export(truncated,state['timeline'])
    # Force a failed rerender; the last valid export pointer and file must remain intact.
    import pipeline.timeline_compositor as compositor
    original=compositor.render_timeline
    compositor.render_timeline=lambda *a,**kw: (_ for _ in ()).throw(StoryError('模拟编码失败'))
    try:
        with pytest.raises(StoryError,match='编码失败'):pipeline.export(state['revision'])
    finally:compositor.render_timeline=original
    assert pipeline.store.read()['export']==state['export']
    assert (root/'video.mp4').read_bytes()==saved
