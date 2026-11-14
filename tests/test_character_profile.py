"""Typed current records, source gates and direct updates; no real model calls."""
import copy
import json
import socket
import subprocess
from pathlib import Path

import pytest
from flask import Flask

import config
from pipeline.character_profile import character_records, world_background
from pipeline.chapter_memory import build_memory, generation_context, merge_memory_patches
from pipeline.prompt_material import context_catalog, editor_material
from pipeline.story_api import create_story_blueprint
from pipeline.story_bible import StoryBible
from pipeline.story_pipeline import StoryPipeline
from test_chapter_memory import MemoryModel, ready_state, data
from test_character_introduction import revealed_bible


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*a,**k):raise AssertionError('Offline profiles must not call a real API')
    monkeypatch.setattr(socket.socket,'connect',deny)
    monkeypatch.setattr(socket.socket,'connect_ex',deny)
    monkeypatch.setattr(socket,'create_connection',deny)


def attribute(field,key,certainty='explicit'):
    return {'field':field,'key':key,'certainty':certainty}


def model_for(definitions, background=None):
    model = MemoryModel(); original = model.ask
    def ask(stage,prompt,images=None,fresh=False):
        raw = original(stage,prompt,images,fresh)
        if stage == 'chapter_memory' and 'Chapter Memory Patch' not in prompt:
            info = data(prompt); ids = [e['event_id'] for e in info['events']]
            raw['character_updates'] = [{'character_id':'C001','facts':[
                {'text':value,'event_ids':ids[-1:],'attribute':copy.deepcopy(meta)}
                for value,meta in definitions]}]
            raw['summary'][0]['role'] = 'outcome'
            raw['summary'][0]['certainty'] = 'observed'
            if background is not None:
                raw['background_updates'] = [{**background,'event_ids':ids[:1]}]
        return raw
    model.ask = ask
    return model


def memory_with(tmp_path, definitions, prior=None, background=None):
    store = StoryBible(tmp_path/'book'/'Chapter_00001')
    bible = revealed_bible()
    if prior is not None:
        bible['prior_context'] = copy.deepcopy(prior)
    state = ready_state(store,bible)
    model = model_for(definitions,background)
    memory = build_memory(store,model,state)
    return store,state,model,memory


def records(state,memory,usable=True):
    return character_records(state['bible'],memory['character_states'],usable)


def test_name_identity_personality_ability_are_structured_without_rewriting_script(tmp_path):
    store,state,model,memory = memory_with(tmp_path,[
        ('渡口守灯人',attribute('identity','occupation')),
        ('原作明确说明性格沉稳',attribute('personality','temperament')),
        ('只能在夜间感知沙流方向',attribute('ability','sand_sense','observed'))])
    profile = records(state,memory)[0]
    assert profile['name']['value'] == '闻溪' and profile['name']['status'] == 'confirmed'
    assert profile['identity'][0]['key'] == 'occupation'
    assert profile['personality'][0]['value'] == '原作明确说明性格沉稳'
    assert profile['ability'][0]['value'] == '只能在夜间感知沙流方向'
    assert profile['state'] == []
    for field in ('identity','personality','ability'):
        row = profile[field][0]
        assert row['status'] == 'confirmed' and row['current']
        assert row['source']['event_ids'] and row['source']['panels'][0]['sha256']
        assert row['source']['origin']['revision'] == state['revision']
        assert row['reveal_panel_id'] and row['reveal_order'] >= 0
    assert [c[0] for c in model.calls] == ['chapter_memory','chapter_memory_review']
    assert store.read() == state and all(s['review_status']=='pending' for s in state['script']['segments'])
    review_prompt = model.calls[-1][1]
    assert 'attribute' in review_prompt and '性格不凭一次行为确认' in review_prompt


@pytest.mark.parametrize('field,certainty,status',[
    ('personality','observed','uncertain'),('personality','inferred','uncertain'),
    ('identity','inferred','uncertain'),('ability','unknown','unknown')])
def test_supported_image_review_does_not_promote_inference_or_single_behavior(tmp_path,field,certainty,status):
    _,state,_,memory = memory_with(tmp_path,[('不可升级成确认设定',attribute(field,'test',certainty))])
    char = memory['character_states'][0]
    assert char['facts'] == [] and len(char['pending_facts']) == 1
    entry = records(state,memory)[0][field][0]
    assert entry['status'] == status and not entry['current']


def test_empty_fields_and_old_free_text_stay_unknown_and_unclassified():
    bible = {'panels':[], 'characters':[{'character_id':'C1','name':'猜测姓名','name_status':'confirmed',
             'description':'勇敢的法师，似乎会飞','name_claims':[]}]}
    old = [{'character_id':'C1','facts':[{'text':'看起来很勇敢，也许会飞','event_ids':['E1']}]}]
    before = copy.deepcopy((bible,old))
    record = character_records(bible,old,True)[0]
    assert record['name']['status'] == 'needs_verification'
    assert record['identity'] == record['personality'] == record['ability'] == []
    assert record['other'][0]['status'] == 'needs_verification'
    assert (bible,old) == before


def test_only_main_characters_keep_profiles_and_new_roster_replaces_old_one(tmp_path):
    store=StoryBible(tmp_path/'Chapter_00001');bible=revealed_bible()
    side=copy.deepcopy(bible['characters'][0]);side.update(character_id='C002',name_status='unknown',name='未知人物C002',name_claims=[])
    bible['characters'].append(side)
    bible['prior_context']={'main_character_ids':['C001','C002'],'character_states':[
        {'character_id':'C002','facts':[{'text':'旧名单内的配角记录','event_ids':['E00001']}]}]}
    state=ready_state(store,bible);before=copy.deepcopy(state)
    model=model_for([('守灯人',attribute('identity','occupation'))]);original=model.ask
    def ask(stage,prompt,images=None,fresh=False):
        raw=original(stage,prompt,images,fresh)
        if stage=='chapter_memory':
            raw['main_character_ids']=['C001']
            raw['character_updates'].append({'character_id':'C002','facts':[
                {'text':'不应建立长期档案','event_ids':['E00001'],'attribute':attribute('identity','occupation')}]})
        return raw
    model.ask=ask;memory=build_memory(store,model,state)
    assert memory['status']=='reviewed' and memory['main_character_ids']==['C001']
    assert [c['character_id'] for c in memory['character_states']]==['C001']
    profiles=character_records(state['bible'],memory['character_states'],True,memory['main_character_ids'])
    assert [c['character_id'] for c in profiles]==['C001']
    assert len(state['bible']['characters'])==2 and state==before and store.read()==before


def test_no_main_roster_does_not_promote_all_recognized_characters_to_profiles():
    bible=revealed_bible()
    legacy=[{'character_id':'C001','facts':[{'text':'未分类的旧识别记录'}]}]
    assert character_records(bible,legacy,False,[])==[]


@pytest.mark.parametrize('meta',[
    {'field':'name','key':'name','certainty':'explicit'},
    {'field':'ability','key':'flight','certainty':'confirmed'},
    {'field':'identity','key':'','certainty':'explicit'},
    {'field':'ability','key':'flight','certainty':'explicit','status':'confirmed'}])
def test_invalid_fields_or_self_confirmed_metadata_keep_memory_pending(tmp_path,meta):
    store,state,_,memory = memory_with(tmp_path,[('不能绕过属性校验',meta)])
    assert memory['status'] == 'needs_review'
    assert state == store.read()


def test_current_attribute_is_replaced_by_exact_key_and_other_properties_survive(tmp_path):
    _,first,_,old = memory_with(tmp_path,[
        ('左臂受伤',attribute('state','health')),
        ('能感知沙流',attribute('ability','sand_sense','observed'))])
    preserved = copy.deepcopy(old)
    prior = {'status':'available','character_states':old['character_states']}
    _,second,_,updated = memory_with(tmp_path,[('左臂已经康复',attribute('state','health'))],prior)
    values = records(second,updated)[0]
    assert [r['value'] for r in values['state']] == ['左臂已经康复']
    assert [r['value'] for r in values['ability']] == ['能感知沙流']
    assert old == preserved and records(first,old)[0]['state'][0]['value'] == '左臂受伤'
    assert len(updated['character_states'][0]['facts']) == 2


def test_pending_update_does_not_overwrite_confirmed_current_value(tmp_path):
    _,_,_,old = memory_with(tmp_path,[('守灯人',attribute('identity','occupation'))])
    _,state,_,updated = memory_with(tmp_path,[('也许是使者',attribute('identity','occupation','inferred'))],
        {'status':'available','character_states':old['character_states']})
    record = records(state,updated)[0]
    assert [(r['value'],r['current']) for r in record['identity']] == [('守灯人',True),('也许是使者',False)]
    bible = copy.deepcopy(state['bible'])
    bible['prior_context']['character_states'] = updated['character_states']
    catalog = context_catalog(bible)
    assert catalog[-1]['status'] == 'uncertain' and catalog[-1]['source']['certainty'] == 'inferred'


def test_forced_refresh_keeps_omitted_current_fields_and_saved_old_revision(tmp_path):
    store,state,model,_=memory_with(tmp_path,[('守灯人',attribute('identity','occupation'))])
    saved=StoryPipeline(store,model).remember(state['revision'])
    refreshed=StoryPipeline(store,model_for([])).remember(saved['revision'])
    assert refreshed['chapter_memory']['character_states'][0]['facts']==saved['chapter_memory']['character_states'][0]['facts']
    assert store.read(saved['revision'])==saved and refreshed['script']==saved['script']


def test_duplicate_current_keys_are_rejected_instead_of_using_return_order(tmp_path):
    _,_,_,memory = memory_with(tmp_path,[
        ('受伤',attribute('state','health')),('康复',attribute('state','health'))])
    assert memory['status'] == 'needs_review' and '一个当前值' in memory['error']


def test_world_overview_and_main_events_reuse_existing_fields_and_update_same_rule(tmp_path):
    _,_,_,old = memory_with(tmp_path,[],background={
        'text':'夜间沙流倒转','rule_key':'night_sand','certainty':'explicit'})
    _,state,_,updated = memory_with(tmp_path,[],prior={'background':old['background']},background={
        'text':'夜间沙流仅在桥下倒转','rule_key':'night_sand','certainty':'explicit'})
    world = world_background(state['bible'],updated,True)
    assert [r['text'] for r in world['overview']] == ['夜间沙流仅在桥下倒转']
    assert world['main_events'][0]['role'] == 'outcome'
    assert world['main_events'][0]['source']['origin']['revision'] == state['revision']
    assert world['overview'][0]['source']['panels'][0]['sha256']
    assert old['background'][0]['text'] == '夜间沙流倒转'


@pytest.mark.parametrize('certainty,status',[('inferred','uncertain'),('unknown','unknown')])
def test_world_proposal_cannot_overwrite_or_promote_a_confirmed_rule(tmp_path,certainty,status):
    _,_,_,old = memory_with(tmp_path,[],background={
        'text':'已确认夜间沙流倒转','rule_key':'night_sand','certainty':'explicit'})
    _,state,_,updated = memory_with(tmp_path,[],prior={'background':old['background']},background={
        'text':'可能不再倒转','rule_key':'night_sand','certainty':certainty})
    assert updated['background'][0]['text']=='已确认夜间沙流倒转'
    bible=copy.deepcopy(state['bible']);bible['prior_context']={'status':'available','background':updated['background']}
    assert context_catalog(bible)[-1]['status']==status
    assert world_background(bible,updated,True)['overview'][-1]['status']==status


def test_duplicate_world_rule_keys_are_rejected(tmp_path):
    store=StoryBible(tmp_path/'Chapter_00001');state=ready_state(store)
    model=model_for([],background={'text':'倒转规则','rule_key':'night_sand','certainty':'explicit'})
    original=model.ask
    def ask(stage,prompt,images=None,fresh=False):
        raw=original(stage,prompt,images,fresh)
        if stage=='chapter_memory':raw['background_updates']*=2
        return raw
    model.ask=ask
    assert '一个当前值' in build_memory(store,model,state)['error']


def test_without_strict_source_readiness_new_value_is_only_pending(tmp_path):
    store = StoryBible(tmp_path/'Chapter_00001');state=ready_state(store)
    state['script']['segments'][0]['fact_check']['status'] = 'uncertain'
    memory = build_memory(store,model_for([('守灯人',attribute('identity','occupation'))]),state)
    assert memory['status'] == 'reviewed'
    char = memory['character_states'][0]
    assert char['facts'] == [] and char['pending_facts'][0]['verification']['verdict'] == 'supported'
    assert records(state,memory,False)[0]['identity'][0]['status'] == 'needs_verification'


def test_typed_patch_preserves_metadata_and_correct_entries():
    original = {'summary':[{'text':'主线结果','event_ids':['E1']}], 'background_updates':[], 'open_threads':[],
        'character_updates':[{'character_id':'C1','facts':[{'text':'过度推断能力','event_ids':['E1'],
            'attribute':attribute('ability','sand_sense','inferred')}]}]}
    before = copy.deepcopy(original)
    fixed = merge_memory_patches(original,[{'claim_id':'M1'},{'claim_id':'M2'}],
        {'claims':[{'claim_id':'M1','verdict':'supported'},{'claim_id':'M2','verdict':'issues'}]},
        [{'claim_id':'M2','replacement':{'text':'只能感知沙流方向','event_ids':['E1']}}])
    assert fixed['summary'] == original['summary'] and original == before
    assert fixed['character_updates'][0]['facts'][0]['attribute'] == original['character_updates'][0]['facts'][0]['attribute']


def test_attribute_change_invalidates_checked_claim_reuse(tmp_path):
    store = StoryBible(tmp_path/'Chapter_00001');state=ready_state(store)
    model = model_for([('感知沙流',attribute('ability','sand_sense','inferred'))]);original=model.ask
    reviews=[]
    def ask(stage,prompt,images=None,fresh=False):
        if stage == 'chapter_memory' and 'Chapter Memory Patch' in prompt:
            issue=data(prompt)['problems'][0]
            return {'patches':[{'claim_id':issue['claim_id'],'replacement':{
                'text':issue['text'],'event_ids':issue['event_ids'],
                'attribute':attribute('ability','sand_sense','observed')}}]}
        raw=original(stage,prompt,images,fresh)
        if stage=='chapter_memory_review':
            claims=data(prompt)['claims'];reviews.append(claims)
            if len(reviews)==1:raw['claims'][-1]['verdict']='issues'
        return raw
    model.ask=ask;memory=build_memory(store,model,state)
    assert memory['status']=='reviewed' and len(reviews)==2 and len(reviews[1])==1
    assert reviews[1][0]['attribute']['certainty']=='observed'
    assert memory['character_states'][0]['facts'][0]['attribute']['certainty']=='observed'


def test_cross_chapter_typed_sources_are_remapped_without_changing_old_versions(tmp_path):
    store,state,model,_=memory_with(tmp_path,[('守灯人',attribute('identity','occupation'))])
    saved=StoryPipeline(store,model).remember(state['revision'])
    next_store=StoryBible(store.root.parent/'Chapter_00002');context=generation_context(next_store)
    fact=context['character_states'][0]['facts'][0]
    assert fact['event_ids'][0].startswith('H00001_') and fact['reveal_order']<0
    assert fact['attribute']['field']=='identity' and fact['origin']['revision']==state['revision']
    assert fact['fact_id']==saved['chapter_memory']['character_states'][0]['facts'][0]['fact_id']
    assert store.read(saved['revision'])==saved and store.read(state['revision'])==state


def test_api_exposes_structured_profiles_and_world_without_mutating_files_or_calling_models(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'STORY_DIR',str(tmp_path))
    store=StoryBible(tmp_path/'book'/'Chapter_00001');state=ready_state(store)
    saved=StoryPipeline(store,model_for([('守灯人',attribute('identity','occupation'))])).remember(state['revision'])
    before={p.name:p.read_bytes() for p in (store.root/'revisions').glob('*.json')}
    def forbidden(*a,**k):raise AssertionError('Read-only API must not create a model')
    app=Flask(__name__);app.register_blueprint(create_story_blueprint(model_factory=forbidden))
    response=app.test_client().get('/api/story/book').get_json()
    assert response['character_records'][0]['identity'][0]['value']=='守灯人'
    assert response['world_background']['overview'] and response['world_background']['main_events']
    assert {p.name:p.read_bytes() for p in (store.root/'revisions').glob('*.json')}==before
    assert store.read()==saved
    # New profile/world views are never appended wholesale to writing inputs.
    prompt=editor_material(saved['bible'],saved['plan'])
    assert 'character_records' not in prompt and 'world_background' not in prompt


def test_review_page_renders_structured_fields_world_and_escaped_values_offline(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'STORY_DIR',str(tmp_path))
    store=StoryBible(tmp_path/'book'/'Chapter_00001');state=ready_state(store)
    StoryPipeline(store,model_for([('<img src=x onerror=alert(1)>',attribute('identity','occupation'))])).remember(state['revision'])
    app=Flask(__name__);app.register_blueprint(create_story_blueprint())
    response=app.test_client().get('/api/story/book').get_json()
    # The grounding fixture omits display-only check fields; provide the
    # production sentence schema before exercising the full existing page.
    for segment in response['script']['segments']:
        for index,sentence in enumerate(segment['fact_check']['sentences']):
            sentence.update(sentence_index=index,verdict='supported',issues=[],evidence='离线来源')
    script=r'''
const fs=require('fs'), vm=require('vm'), assert=require('assert');
const data=JSON.parse(fs.readFileSync(0,'utf8'));
const node={value:'',textContent:''};
const host={innerHTML:'',querySelector:()=>node,querySelectorAll:()=>[]};
const window={};let calls=0;
vm.runInNewContext(fs.readFileSync('static/js/story_review.js','utf8'),{
  window, document:{contains:()=>true}, clearTimeout, setTimeout,
  fetch:async(url,options)=>{calls++;assert(!options);return {ok:true,json:async()=>data};}
});
(async()=>{
  await window.StoryReview.mount(host,'book');
  assert.equal(node.textContent,'',node.textContent);
  for(const label of ['人物姓名','人物身份','人物性格','人物能力','世界观概述','主线事件'])assert(host.innerHTML.includes(label));
  assert(host.innerHTML.includes('&lt;img src=x onerror=alert(1)&gt;'));
  assert(!host.innerHTML.includes('<img src=x onerror=alert(1)>'));
  assert.equal(calls,1);
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    result=subprocess.run(['node','-e',script],input=json.dumps(response),text=True,
        cwd=Path(__file__).resolve().parents[1],capture_output=True,timeout=15)
    assert result.returncode==0, result.stderr
