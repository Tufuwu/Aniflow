"""Offline continuity contracts with an original sand-bridge story."""
import copy
import json
import socket

import pytest
from flask import Flask
from PIL import Image

import config
from pipeline.chapter_memory import (generation_context, memory_usable,
                                     save_series_notes, series_notes, context_current)
from pipeline.character_identity import identity_context
from pipeline.fact_checker import FactChecker
from pipeline.script_writer import ScriptWriter
from pipeline.semantic_grounding import make_grounding
from pipeline.story_api import create_story_blueprint, story_store
from pipeline.story_bible import StoryBible, StoryConflict, digest
from pipeline.story_pipeline import StoryPipeline
from test_character_introduction import panel, character, revealed_bible
from test_engagement_flow import story, generate, polish_payload


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args,**kwargs):raise AssertionError('Chapter memory tests must stay offline')
    monkeypatch.setattr(socket.socket,'connect',deny)
    monkeypatch.setattr(socket,'create_connection',deny)


def data(prompt):
    return json.JSONDecoder().raw_decode(prompt.split('资料：',1)[1])[0]


class MemoryModel:
    def __init__(self):
        self.calls=[];self.bad=None

    def ask(self,stage,prompt,images=None,fresh=False):
        self.calls.append((stage,prompt,images,fresh))
        info=data(prompt)
        if stage=='chapter_memory':
            ids=[e['event_id'] for e in info['events']]
            cid=info['characters'][0]['character_id'] if info['characters'] else None
            result={'summary':[{'text':'沙流倒转后，桥边的人报出姓名。','event_ids':ids}],
                    'background_updates':[{'text':'沙流会倒转。','event_ids':ids[:1]}],
                    'character_updates':[{'character_id':cid,'facts':[{'text':'桥边的人报出了姓名。','event_ids':ids[-1:]}]}] if cid else [],
                    'main_character_ids':[cid] if cid else [],'open_threads':[]}
            if self.bad=='event':result['summary'][0]['event_ids']=['invented']
            if self.bad=='character':result['character_updates'][0]['character_id']='invented'
            return result
        if stage=='chapter_memory_review':
            assert images and all(info['image_panel_ids'])
            rows=[{'claim_id':c['claim_id'],'verdict':'supported','evidence':'离线模拟原图核对，不代表实际质量'} for c in info['claims']]
            if self.bad=='uncertain':rows[-1]['verdict']='uncertain'
            if self.bad=='missing':rows.pop()
            if self.bad=='outage':raise RuntimeError('simulated memory review outage')
            return {'claims':rows}
        if stage=='script_polish':
            return polish_payload(info['script'])
        raise AssertionError(stage)


def ready_state(store,bible=None):
    bible=copy.deepcopy(bible or revealed_bible())
    for i,p in enumerate(bible['panels']):
        if 'sha256' not in p:
            path=store.root/f'fixture-{i}.png';Image.new('RGB',(45,60),(50+i,70,80)).save(path)
            p.update(store.snapshot(path))
    current=[e for e in bible['events'] if not e.get('historical')]
    if len(current)>1:current[-1]['summary']='桥边的人说自己叫闻溪'
    bible['bible_id']=digest(bible)
    beats=[{'event_id':e['event_id'],'event_ids':[e['event_id']], 'source_panel_ids':e['source_panel_ids'],
            'context_event_ids':[], 'section':'opening' if i==0 else 'ending'} for i,e in enumerate(current)]
    plan={'version':5,'style':'faithful','beats':beats,'bible_id':bible['bible_id']}
    segments=[]
    for e in current:
        segment={'segment_id':'S_'+e['event_id'],'event_id':e['event_id'],'event_ids':[e['event_id']],
                 'context_event_ids':[], 'source_panel_ids':e['source_panel_ids'], 'text':e['summary']+'。',
                 'section':'development','review_status':'pending'}
        rows=[{'text':segment['text'],'units':[{'text':segment['text'],'event_ids':[e['event_id']],
                'display_panel_ids':e['source_panel_ids'],'background_event_ids':[]}]}]
        segment['fact_check']={'status':'passed','grounding':make_grounding(bible,segment,rows),'sentences':rows}
        segments.append(segment)
    review=ScriptWriter(MemoryModel()).polish_story(bible,plan,segments)['completion']
    state=store.read()
    state.update(bible=bible,plan=plan,script={'segments':segments,'narrative_review':review},stage='review')
    return store.save(state,'fixture_ready',state['revision'])


def remembered(store):
    state=ready_state(store)
    model=MemoryModel()
    saved=StoryPipeline(store,model).remember(state['revision'])
    assert memory_usable(saved)
    return state,saved,model


def test_series_notes_are_versioned_and_shared_only_within_one_project(tmp_path):
    first=StoryBible(tmp_path/'book'/'Chapter_00001')
    saved=save_series_notes(first,'沙海世界，桥索有断裂风险。','闻溪：姓名归属仍须原图核对。',None)
    second=StoryBible(tmp_path/'book'/'Chapter_00002')
    assert series_notes(second)['background']==saved['background']
    with pytest.raises(StoryConflict):save_series_notes(second,'覆盖旧资料','',None)
    assert not series_notes(StoryBible(tmp_path/'other'/'Chapter_00001'))['background']
    assert StoryBible(first.root.parent/'series_context').read(saved['revision'])['characters']==saved['characters']


def test_memory_generated_and_reviewed_without_touching_script_or_human_status(tmp_path):
    store=StoryBible(tmp_path/'book'/'Chapter_00001')
    before,saved,model=remembered(store)
    assert [c[0] for c in model.calls]==['chapter_memory','chapter_memory_review']
    assert saved['script']==before['script'] and saved['stage']==before['stage']
    assert saved['chapter_memory']['character_states'][0]['character_id']=='C001'
    assert store.read(before['revision'])==before
    assert all(c[3] for c in model.calls)


def test_generation_automatically_saves_memory_after_final_script_polish(story):
    model=story[2];original=model.ask;memory=MemoryModel()
    def ask(stage,prompt,images=None,fresh=False):
        if stage.startswith('chapter_memory'):return memory.ask(stage,prompt,images,fresh)
        result=original(stage,prompt,images,fresh)
        if stage=='check':
            events=json.JSONDecoder().raw_decode(prompt.split('\n事件：',1)[1])[0]
            for row,part in zip(result['sentences'],json.loads(prompt.split('待核对句子：',1)[1].split('\n返回',1)[0])):
                row['units']=[{'text':part['text'],'event_ids':[e['event_id'] for e in events],
                               'display_panel_ids':list(dict.fromkeys(p for e in events for p in e['source_panel_ids'])),
                               'background_event_ids':[]}]
        return result
    model.ask=ask
    saved=generate(story)
    assert memory_usable(saved) and len(memory.calls)==2
    assert saved['action']=='chapter_memory' and all(s['review_status']=='pending' for s in saved['script']['segments'])


@pytest.mark.parametrize('bad',['event','character','uncertain','missing','outage'])
def test_invalid_or_uncertain_memory_is_saved_pending_and_not_inherited(tmp_path,bad):
    first=StoryBible(tmp_path/'book'/'Chapter_00001');before=ready_state(first)
    model=MemoryModel();model.bad=bad
    saved=StoryPipeline(first,model).remember(before['revision'])
    assert saved['chapter_memory']['status']=='needs_review' and not memory_usable(saved)
    assert saved['script']==before['script']
    context=generation_context(StoryBible(tmp_path/'book'/'Chapter_00002'))
    assert context['status']=='needs_review' and context['characters']==[] and context['events']==[]


def test_summary_review_cannot_promote_an_unverified_source_draft(tmp_path):
    first=StoryBible(tmp_path/'book'/'Chapter_00001');state=ready_state(first)
    state['script']['segments'][0]['fact_check']['status']='uncertain'
    state=first.save(state,'fixture_uncertain',state['revision'])
    saved=StoryPipeline(first,MemoryModel()).remember(state['revision'])
    assert saved['chapter_memory']['status']=='reviewed' and not memory_usable(saved)


def next_analysis(store):
    class Model:
        def ask(self,stage,prompt,images=None,**kwargs):
            if stage=='analyze':
                assert 'H00001_E00001' in prompt and '"history_available": true' in prompt
                assert '沙流会倒转' in prompt
                c=character();c['identity_match']={'method':'original_image_comparison','basis':'与附图中自我介绍者对应，继续握同一根绳索'}
                raw=panel(c,'绳索还拉得动。')
                raw['events'][0]['summary']='桥边的人继续拉紧绳索'
                return raw
            if stage=='reconcile':
                assert 'H00001' not in prompt.split('按顺序附上以下来源图：',1)[1].split('。',1)[0]
                return {'panels':[{'source_panel_id':'1','classification':'story','reason':'当前话桥边行动'}]}
            pytest.fail(stage)
    return StoryPipeline(store,Model()).analyze([{'panel_id':'1','path':str(store.root.parent/'Chapter_00001'/'fixture-0.png')}],None)


def test_prior_identities_reveals_and_original_sources_survive_into_next_chapter(tmp_path):
    first=StoryBible(tmp_path/'book'/'Chapter_00001');_,old,_=remembered(first)
    second=StoryBible(tmp_path/'book'/'Chapter_00002');state=next_analysis(second)
    bible=state['bible'];current=[e for e in bible['events'] if not e.get('historical')]
    assert [e['event_id'] for e in current]==['E00001']
    assert bible['characters'][0]['character_id']=='C001' and bible['characters'][0]['name']=='闻溪'
    assert bible['characters'][0]['name_claims'][0]['reveal_order']<0
    assert identity_context(bible,[{'event_id':'E00001'}])[0]['not_yet_revealed']==[]
    for p in bible['panels']:
        assert second.evidence_path(p).exists()
        if p.get('historical'):assert p['source_panel_id'].startswith('H00001_')
    assert first.read()==old and bible['prior_context']['previous_chapter']['revision']==old['revision']


def test_third_chapter_receives_early_background_and_cumulative_character_facts(tmp_path):
    first=StoryBible(tmp_path/'book'/'Chapter_00001');remembered(first)
    second=StoryBible(tmp_path/'book'/'Chapter_00002');state=next_analysis(second)
    ready=ready_state(second,state['bible'])
    saved=StoryPipeline(second,MemoryModel()).remember(ready['revision']);assert memory_usable(saved)
    context=generation_context(StoryBible(tmp_path/'book'/'Chapter_00003'))
    assert context['status']=='available' and len(context['background'])==2
    assert any('H00001_E00001' in f['event_ids'] for f in context['background'])
    assert any('H00002_E00001' in f['event_ids'] for f in context['background'])
    assert len(context['character_states'][0]['facts'])==2
    assert len(context['events'])==3 and len({e['event_id'] for e in context['events']})==3


def test_memory_review_receives_original_historical_cause_not_only_a_summary(tmp_path):
    first=StoryBible(tmp_path/'book'/'Chapter_00001');remembered(first)
    second=StoryBible(tmp_path/'book'/'Chapter_00002');state=next_analysis(second)
    state['bible']['events'][-1]['causes']=[{'event_id':'H00001_E00001','evidence':'原图中仍在倒转的沙流构成拉索限制'}]
    ready=ready_state(second,state['bible']);model=MemoryModel()
    saved=StoryPipeline(second,model).remember(ready['revision'])
    assert memory_usable(saved)
    _,prompt,images,_=model.calls[-1];info=data(prompt)
    assert info['causal_context'][0]['event_id']=='H00001_E00001'
    assert set(info['image_panel_ids'])=={'1','H00001_1'} and len(images)==2


def test_current_fact_check_receives_prior_original_name_proof(tmp_path):
    first=StoryBible(tmp_path/'book'/'Chapter_00001');remembered(first)
    second=StoryBible(tmp_path/'book'/'Chapter_00002');state=next_analysis(second)
    calls=[]
    class Model:
        def ask(self,stage,prompt,images=None,**kwargs):
            calls.append((prompt,images))
            return {'sentences':[{'sentence_index':0,'verdict':'supported','evidence':'模拟人物对应与姓名原图',
                'issues':[],'units':[{'text':'闻溪拉紧绳索。','event_ids':['E00001'],
                                     'display_panel_ids':['1'],'background_event_ids':[]}]}]}
    segment={'segment_id':'S_E00001','event_id':'E00001','event_ids':['E00001'],
             'context_event_ids':[],'source_panel_ids':['1'],'text':'闻溪拉紧绳索。'}
    check=FactChecker(second,Model()).check(state['bible'],segment)
    assert check['status']=='passed'
    assert check['sentences'][0]['verification_panel_ids']==['1','H00001_2']
    assert check['grounding']['units'][0]['display_panel_ids']==['1']
    assert len(calls[0][1])==2


def test_current_saved_settings_enter_planning_once_without_repeating_in_other_stages(story):
    store,state,model,pipeline=story
    save_series_notes(store,'海雾会切断渡口。','渡口的人暂未确认姓名。',None)
    saved=generate(story)
    planning = next(prompt for s,prompt,_,_ in model.calls if s=='direct')
    assert planning.count('海雾会切断渡口') == 1
    assert all(stage != 'story_digest' for stage,_,_,_ in model.calls)
    for stage in ('direct','write_story','script_polish'):
        if stage != 'direct':
            assert '海雾会切断渡口' not in next(prompt for s,prompt,_,_ in model.calls if s==stage)
    writer=ScriptWriter(model)
    writer.revise_story(saved['bible'],saved['plan'],saved['script']['segments'],{'issues':[{'event_id':'E1','detail':'模拟称呼核对'}]})
    assert '海雾会切断渡口' not in model.calls[-1][1]
    assert series_notes(store)['background'] == '海雾会切断渡口。'


def test_previous_text_change_invalidates_downstream_context_without_changing_old_versions(tmp_path):
    first=StoryBible(tmp_path/'book'/'Chapter_00001');_,old,_=remembered(first)
    second=StoryBible(tmp_path/'book'/'Chapter_00002');state=next_analysis(second)
    state=ready_state(second,state['bible'])
    changed=copy.deepcopy(old);changed['script']['segments'][0]['text']='新的证据表述。'
    first.save(changed,'fixture_corrected',old['revision'])
    assert not context_current(second,state['bible'])
    with pytest.raises(StoryConflict,match='前话资料已变化'):
        StoryPipeline(second,object()).generate_verified(state['revision'])
    with pytest.raises(StoryConflict,match='前话资料已变化'):
        StoryPipeline(second,object()).segment_action(state['script']['segments'][0]['segment_id'],'approve',state['revision'])
    assert first.read(old['revision'])==old and second.read()==state


def test_no_future_chapter_or_other_book_is_injected(tmp_path):
    future=StoryBible(tmp_path/'book'/'Chapter_00003');remembered(future)
    assert generation_context(StoryBible(tmp_path/'book'/'Chapter_00001'))['status']=='no_history'
    assert generation_context(StoryBible(tmp_path/'other'/'Chapter_00004'))['characters']==[]


def test_previously_missing_history_requires_refresh_once_predecessor_finishes(tmp_path):
    second=StoryBible(tmp_path/'book'/'Chapter_00002')
    context=generation_context(second)
    bible={'prior_context':{k:v for k,v in context.items() if k not in {'panels','events','characters','relationships'}}}
    assert context_current(second,bible)
    remembered(StoryBible(tmp_path/'book'/'Chapter_00001'))
    assert not context_current(second,bible)


def test_corrected_memory_alone_invalidates_downstream_context(tmp_path):
    first=StoryBible(tmp_path/'book'/'Chapter_00001');_,before,_=remembered(first)
    second=StoryBible(tmp_path/'book'/'Chapter_00002');state=next_analysis(second)
    model=MemoryModel();original=model.ask
    def ask(stage,prompt,images=None,fresh=False):
        raw=original(stage,prompt,images,fresh)
        if stage=='chapter_memory':raw['background_updates'][0]['text']='沙流倒转会拉扯桥头的绳索。'
        return raw
    model.ask=ask
    saved=StoryPipeline(first,model).remember(before['revision'])
    assert saved['script']==before['script'] and memory_usable(saved)
    assert not context_current(second,state['bible'])


def test_refresh_failure_does_not_erase_previous_memory_or_approved_text_history(tmp_path):
    store=StoryBible(tmp_path/'book'/'Chapter_00001');_,old,_=remembered(store)
    approved=copy.deepcopy(old)
    for s in approved['script']['segments']:s['review_status']='approved'
    approved=store.save(approved,'fixture_human_approved',old['revision'])
    model=MemoryModel();model.bad='outage'
    saved=StoryPipeline(store,model).remember(approved['revision'])
    assert saved['script']==approved['script'] and saved['chapter_memory']['status']=='needs_review'
    assert store.read(old['revision'])['chapter_memory']==old['chapter_memory']


def test_context_notes_and_chapter_endpoints_keep_legacy_first_chapter_and_isolate_versions(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'STORY_DIR',str(tmp_path/'stories'))
    app=Flask(__name__);app.register_blueprint(create_story_blueprint(model_factory=lambda store:MemoryModel()))
    client=app.test_client();base='/api/story/book'
    response=client.put(base+'/context',json={'revision':None,'background':'沙海','characters':'桥边的人'})
    assert response.status_code==200
    assert client.get(base+'/chapters/2').get_json()['series_notes']['background']=='沙海'
    assert client.put(base+'/context',json={'revision':None,'background':'旧窗口覆盖'}).status_code==409
    assert client.get(base).get_json()['revision'] is None
    second=story_store('book',2);ready=ready_state(second)
    assert client.get(base+'/chapters/2').get_json()['revision']==ready['revision']
    assert client.get(base).get_json()['revision'] is None
    assert client.get('/api/story/other/context').get_json()['background']==''
    assert client.get(base+'/chapters/0').status_code==422
