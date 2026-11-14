"""Voice direction acceptance: isolated source, fake provider, real audio samples/encoder."""
import copy
import hashlib
import json
from pathlib import Path
import re
import wave

from flask import Flask
import pytest

from test_shot_editing import chapter, speech_fixture, selected, wait_job
from pipeline.elevenlabs_tts import ElevenLabsTTS
from pipeline.shot_api import create_shot_blueprint
from pipeline.story_bible import StoryError, StoryConflict
from pipeline.timeline_director import validate_timeline, TimelineDirector
from pipeline.voice_audio import audio_review, synthesize_performance
from pipeline.voice_director import VoiceDirector, expressive_text, plan_hash, validate_plan

OPTIONS = {'engine':'elevenlabs','model':'eleven_v4','voice':'fixture_voice'}


def planned(chapter):
    state=selected(chapter)
    return chapter[2].plan_voice(state['revision'],'suspense',False)


class FixtureDirector:
    def ask(self, stage, prompt):
        assert stage=='voice_direct'
        data=json.loads(prompt.split('\n资料：')[1])
        segment=data['narration']
        return {'units':[{'text':segment['text'],'emotion':'surprised','intensity':.6,'rate':.92,
                          'emphasis':[segment['text'][:2]],'pause_after_ms':0,
                          'reason':'依据本段事件，为揭示信息安排克制的惊讶语气',
                          'evidence_event_ids':[segment['event_id']]}]}


def test_model_plans_only_performance_and_keeps_approved_script(chapter):
    store,source,pipeline=chapter
    state=selected(chapter)
    pipeline.model=FixtureDirector()
    before=(store.root/'revisions'/f"{source['revision']}.json").read_bytes()
    state=pipeline.plan_voice(state['revision'],'hotblood',True)
    for segment in source['script']['segments']:
        plan=state['voice_direction']['plans'][segment['segment_id']]
        assert plan['origin']=='model'
        assert ''.join(u['text'] for u in plan['units'])==segment['text']
        assert plan['review_status']=='pending'
    assert store.read()['revision']==source['revision']
    assert (store.root/'revisions'/f"{source['revision']}.json").read_bytes()==before


@pytest.mark.parametrize('failure',['text','event','pause','emphasis','intensity'])
def test_invalid_performance_metadata_cannot_commit(chapter,failure):
    source=chapter[1];segment=source['script']['segments'][0]
    plan=VoiceDirector().plan(source,segment,'suspense',False)
    unit=plan['units'][0]
    if failure=='text':unit['text']+='凭空出现的未来剧情'
    if failure=='event':unit['evidence_event_ids']=['E404']
    if failure=='pause':unit['pause_after_ms']=9999
    if failure=='emphasis':unit['emphasis']=['原文没有的词']
    if failure=='intensity':unit['intensity']=float('nan')
    with pytest.raises(StoryError):validate_plan(plan,segment,source)


def test_locked_manual_direction_survives_rerun_and_stale_edits_fail(chapter):
    pipeline=chapter[2];state=planned(chapter);nid='S_E1'
    units=copy.deepcopy(state['voice_direction']['plans'][nid]['units'])
    units[0].update(emotion='serious',intensity=.7,emphasis=['门口'],pause_after_ms=410)
    state=pipeline.review_voice(state['revision'],nid,'update',{'units':units})
    state=pipeline.review_voice(state['revision'],nid,'approve',{})
    locked=copy.deepcopy(state['voice_direction']['plans'][nid])
    stale=state['revision']
    state=pipeline.plan_voice(state['revision'],'deadpan',False)
    assert state['voice_direction']['plans'][nid]==locked
    with pytest.raises(StoryConflict):pipeline.plan_voice(stale,'hotblood',False)


def test_unconfigured_model_does_not_silently_fall_back_to_rules(chapter):
    pipeline=chapter[2];state=selected(chapter)
    with pytest.raises(StoryError,match='配置剧情模型'):
        pipeline.plan_voice(state['revision'],'suspense',True)
    assert pipeline.store.read()['revision']==state['revision']


def test_audio_tags_preserve_original_text_and_never_speak_reason(chapter):
    plan=planned(chapter)['voice_direction']['plans']['S_E1']
    unit=plan['units'][0];unit.update(emphasis=['门口'],intensity=.8)
    compiled=expressive_text(unit)
    assert '[emphasizing]门口' in compiled
    assert re.sub(r'\[[^\]]*\]','',compiled)==unit['text']
    assert unit['reason'] not in compiled and 'E1' not in compiled


def test_exact_added_pause_is_silence_and_unit_timings_cover_waveform(chapter,tmp_path):
    state=planned(chapter)
    plan=state['voice_direction']['plans']['S_E1']
    text=plan['units'][0]['text']
    unit=copy.deepcopy(plan['units'][0]);unit.update(text='雨落下，',pause_after_ms=325)
    end=copy.deepcopy(unit);end.update(text='他走到门口。',pause_after_ms=0)
    plan['units']=[unit,end]
    path=tmp_path/'sample.wav'
    info=synthesize_performance(plan,path,{'engine':'edge','voice':'zh-CN-YunxiNeural'},speech_fixture)
    first=info['unit_timings'][0]
    assert first['pause_samples']==15600
    with wave.open(str(path),'rb') as stream:
        assert stream.getnframes()==info['unit_timings'][-1]['end_sample']
        stream.setpos(first['speech_end_sample'])
        assert set(stream.readframes(first['pause_samples']))=={0}
    report=audio_review(path,text,lambda path:text)
    assert report['transcript_status']=='matched' and report['manual_listen_required']
    timeline=TimelineDirector().build([state['groups'][0]],{'S_E1':{**info,'path':str(path),
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}},state['library'])
    assert validate_timeline(timeline)['timeline_complete']
    second=next(s for s in timeline['subtitles'] if s['text']=='他走到门口。')
    assert second['start_frame']==round(first['end_sample'] * 30 / 48000)


def test_asr_differences_and_unrun_status_are_truthful(chapter,tmp_path):
    path=tmp_path/'audio.wav';speech_fixture('雨落下',str(path))
    result=audio_review(path,'雨落下。',lambda path:'whispering 雨没落下')
    assert result['transcript_status']=='needs_review' and result['differences']
    result=audio_review(path,'雨落下。',lambda path:None)
    assert result['transcript_status']=='not_run'


def test_three_style_auditions_do_not_replace_locked_plan_or_selected_audio(chapter):
    pipeline=chapter[2];state=planned(chapter);nid='S_E1'
    state=pipeline.review_voice(state['revision'],nid,'approve',{})
    before=copy.deepcopy(state['voice_direction']['plans'][nid])
    state=pipeline.audition_voice(state['revision'],nid,{'engine':'edge','voice':'zh-CN-YunxiNeural'},True)
    assert {t['plan']['style'] for t in state['voice_samples'][nid]}=={'suspense','hotblood','deadpan'}
    assert state['voice_direction']['plans'][nid]==before
    assert not state.get('voice_selected')
    for take in state['voice_samples'][nid]:assert Path(take['path']).is_file()
    pipeline.speech=lambda *a,**kw:pytest.fail('must reuse existing audition cache')
    state=pipeline.audition_voice(state['revision'],nid,{'engine':'edge','voice':'zh-CN-YunxiNeural'},True)
    assert len(state['voice_samples'][nid])==3


def test_audio_approval_requires_listening_and_mismatch_note(chapter):
    pipeline=chapter[2];state=planned(chapter);nid='S_E1'
    pipeline.transcriber=lambda path:'错误的读音'
    state=pipeline.review_voice(state['revision'],nid,'approve',{})
    state=pipeline.audition_voice(state['revision'],nid,OPTIONS)
    take=state['voice_samples'][nid][-1]
    state=pipeline.review_voice(state['revision'],nid,'select_take',{'take_id':take['take_id']})
    with pytest.raises(StoryError,match='先试听'):pipeline.review_voice(state['revision'],nid,'approve_audio',{})
    with pytest.raises(StoryError,match='试听结论'):
        pipeline.review_voice(state['revision'],nid,'approve_audio',{'listened':True})
    state=pipeline.review_voice(state['revision'],nid,'approve_audio',{'listened':True,'review_note':'人工核对原文，确认转写误识别'})
    assert state['voice_samples'][nid][-1]['review_status']=='approved'


def accepted(chapter):
    pipeline=chapter[2];state=planned(chapter)
    for nid,plan in list(state['voice_direction']['plans'].items()):
        state=pipeline.review_voice(state['revision'],nid,'approve',{})
        pipeline.transcriber=lambda path,text=''.join(u['text'] for u in plan['units']):text
        state=pipeline.audition_voice(state['revision'],nid,OPTIONS)
        take=state['voice_samples'][nid][-1]
        state=pipeline.review_voice(state['revision'],nid,'select_take',{'take_id':take['take_id']})
        state=pipeline.review_voice(state['revision'],nid,'approve_audio',{'listened':True})
    return state


def test_accepted_voice_drives_timeline_and_ai_rerun_preserves_audio(chapter):
    pipeline=chapter[2];state=accepted(chapter)
    selection=copy.deepcopy(state['voice_selected']);samples=copy.deepcopy(state['voice_samples'])
    state=pipeline.direct(state['revision'])
    assert state['timeline']['subtitle_alignment']=='measured_semantic_units'
    assert validate_timeline(state['timeline'])['timeline_complete']
    assert all(a['take_id'] in selection.values() for a in state['audio'].values())
    state=pipeline.plan_voice(state['revision'],'hotblood',False)
    assert state['voice_selected']==selection and state['voice_samples']==samples
    state=pipeline.review_voice(state['revision'],'S_E1','update',{'units':state['voice_direction']['plans']['S_E1']['units']})
    assert state['timeline'] is None and state.get('voice_selected',{}).get('S_E1') is None
    with pytest.raises(StoryError):pipeline.direct(state['revision'])


def test_fresh_take_keeps_old_audio_bytes_and_plan(chapter):
    pipeline=chapter[2];state=accepted(chapter);nid='S_E1'
    before=copy.deepcopy(state['voice_samples'][nid]);chosen=state['voice_selected'][nid]
    hashes={t['path']:hashlib.sha256(Path(t['path']).read_bytes()).hexdigest() for t in before}
    state=pipeline.audition_voice(state['revision'],nid,OPTIONS,fresh=True)
    assert len(state['voice_samples'][nid])==len(before)+1
    assert state['voice_selected'][nid]==chosen
    assert hashes=={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in hashes}


def test_v4_provider_payload_filters_unsupported_settings_and_measures_real_audio(tmp_path,monkeypatch):
    sample=tmp_path/'source.wav';speech_fixture('雨',str(sample));seen={}
    class Response:
        content=sample.read_bytes()
        def raise_for_status(self):pass
    def post(url,**kwargs):seen.update(url=url,**kwargs);return Response()
    monkeypatch.setattr('pipeline.elevenlabs_tts.requests.post',post)
    engine=ElevenLabsTTS();engine.set_api_key('test-secret')
    result=engine.generate_speech('[tense]雨',str(tmp_path/'out.mp3'),voice='fixture_voice',model='eleven_v4',
                                  previous_text='此前',next_text='后来',seed=123)
    assert seen['json']['model_id']=='eleven_v4'
    assert set(seen['json']['voice_settings'])=={'stability','similarity_boost'}
    assert seen['json']['previous_text']=='此前' and seen['json']['seed']==123
    assert result['duration']==pytest.approx(.57)
    assert seen['params']['output_format']=='mp3_44100_128'


def test_voice_api_jobs_audio_access_and_key_never_enters_history(chapter,monkeypatch):
    store,source,pipeline=chapter;state=selected(chapter)
    monkeypatch.setattr('pipeline.shot_api.story_store',lambda project:store)
    monkeypatch.setattr('pipeline.elevenlabs_tts._elevenlabs_tts',ElevenLabsTTS())
    app=Flask(__name__);app.register_blueprint(create_shot_blueprint(lambda store:FixtureDirector(),speech_fixture))
    client=app.test_client();base='/api/editing/demo'
    assert 'suspense' in client.get('/api/editing/voice-capabilities').get_json()['presets']
    assert client.post('/api/editing/voice-key',json={'key':'test-do-not-save-secret'}).status_code==200
    result=wait_job(client,base,client.post(base+'/voice/plan',json={'revision':state['revision'],'use_model':True}))
    assert result['status']=='completed',result
    state=client.get(base).get_json()
    result=wait_job(client,base,client.post(base+'/voice/S_E1/audition',json={'revision':state['revision'],'tts':OPTIONS}))
    assert result['status']=='completed',result
    state=client.get(base).get_json();take=state['voice_samples']['S_E1'][0]
    assert client.get(base+f"/voice-audio/S_E1/{take['take_id']}").status_code==200
    assert client.get(base+'/voice-audio/S_E1/missing').status_code==422
    assert all('test-do-not-save-secret' not in p.read_text(encoding='utf-8') for p in pipeline.store.root.rglob('*.json'))


def test_reviewed_voice_export_contains_auditable_performance_plan(chapter):
    pipeline=chapter[2];state=accepted(chapter)
    for group in state['groups']:state=pipeline.review(group['narration_id'],'approve',state['revision'],{})
    state=pipeline.direct(state['revision']);state=pipeline.export(state['revision'])
    root=pipeline.store.root/'exports'/state['export']['export_id']
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['voice_selected']==state['voice_selected']
    assert manifest['voice_direction']['plans']['S_E1']['review_status']=='approved'
    assert state['export']['report']['complete']
