"""Offline constraints discovered by actual API validation; no live calls here."""
import copy
import socket

import pytest
from PIL import Image

from pipeline.story_analyzer import StoryAnalyzer
from pipeline.story_bible import StoryError, StoryBible
from pipeline.story_model import StoryModel, EvidenceImage
from pipeline.chapter_memory import merge_memory_patches
from pipeline.script_writer import ScriptWriter
from pipeline.story_pipeline import StoryPipeline
from pipeline.narration_review import review_ready
from test_engagement_flow import review_payload
from pipeline.llm_transport import OpenAICompatTransport
from test_character_introduction import panel


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args,**kwargs):raise AssertionError('Offline regression')
    monkeypatch.setattr(socket.socket,'connect',deny)
    monkeypatch.setattr(socket,'create_connection',deny)


@pytest.mark.parametrize('where',['analysis','evidence'])
def test_reference_source_is_not_silently_reassigned_to_current_image(where):
    raw=panel()
    if where=='analysis':raw['analyzed_source_panel_id']='previous'
    else:raw['events'][0]['evidence'][0]['source_panel_id']='previous'
    with pytest.raises(StoryError,match='参考图'):
        StoryAnalyzer._consume(raw,{}, {}, [],'current')


@pytest.mark.parametrize('kind',['caption','dialogue'])
def test_literal_caption_is_kept_as_caption_without_inventing_a_speaker(kind):
    raw=panel();raw['dialogue']=[];raw['visible_text']=['河上的桥只能承受一辆车。']
    raw['events'][0]['evidence']=[{'kind':kind,'detail':'河上的桥只能承受一辆车。'}]
    entry,_,events=StoryAnalyzer._consume(raw,{}, {}, [],'1')
    assert events[0]['evidence'][0]['kind']=='caption' and entry['dialogue']==[]
    bad=copy.deepcopy(raw);bad['events'][0]['evidence'][0]['detail']='桥上可以通过十辆车。'
    with pytest.raises(StoryError,match='逐字'):StoryAnalyzer._consume(bad,{}, {}, [],'1')


def test_each_image_receives_its_own_adjacent_role_label(monkeypatch):
    sent={}
    def post(url,payload,headers,timeout):
        sent.update(payload)
        return {'choices':[{'message':{'content':'{}'}}]}
    monkeypatch.setattr(OpenAICompatTransport,'_post',staticmethod(post))
    OpenAICompatTransport('http://local.invalid').chat('model','检查当前图',
        [('', 'image/png','空图'),('QUJD','image/png','当前来源'),('REVG','image/jpeg','历史参考')])
    content=sent['messages'][0]['content']
    assert [c['type'] for c in content]==['text','text','image_url','text','image_url']
    assert content[1]['text']=='当前来源' and content[3]['text']=='历史参考'
    assert content[2]['image_url']['url']=='data:image/png;base64,QUJD'


def test_boundary_review_keeps_real_source_ids_and_retries_ordinal_ids(tmp_path):
    store=StoryBible(tmp_path/'chapter');image=tmp_path/'panel.png'
    Image.new('RGB',(30,40),'gray').save(image)
    ids=['1','2','3','64','65','66','67','68']
    bible={'panels':[{'source_panel_id':pid,'dialogue':[],**store.snapshot(image)} for pid in ids], 'events':[]}
    calls=[]
    class Model:
        def ask(self,stage,prompt,images=None,fresh=False):
            calls.append(fresh)
            assert [p.source_panel_id for p in images]==ids
            returned=[str(i+1) for i in range(8)] if len(calls)==1 else ids
            return {'panels':[{'source_panel_id':pid,'classification':'story','reason':'原创测试剧情来源'} for pid in returned]}
    result=StoryAnalyzer(store,Model()).review_boundaries(bible)
    assert calls==[False,True]
    assert [p['source_panel_id'] for p in result['boundary_review']['panels']]==ids


def test_source_label_changes_invalidate_cached_visual_request(tmp_path):
    store=StoryBible(tmp_path/'chapter');image=tmp_path/'panel.png';Image.new('RGB',(30,40),'gray').save(image)
    class Transport:
        kind='test';host='test';calls=0
        def chat(self,**kwargs):
            self.calls+=1
            assert 'source_panel_id=' in kwargs['images'][0][2]
            return '{}'
    transport=Transport();model=StoryModel(store,transport)
    model.ask('reconcile','核对来源',[EvidenceImage(image,'1')])
    model.ask('reconcile','核对来源',[EvidenceImage(image,'64')])
    model.ask('reconcile','核对来源',[EvidenceImage(image,'64')])
    assert transport.calls==2


def test_length_truncation_keeps_metadata_and_never_accepts_partial_json(tmp_path):
    store=StoryBible(tmp_path/'chapter')
    class Transport:
        kind='openai';host='https://api.deepseek.com'
        last_response_metadata={'finish_reason':'length','usage':{'completion_tokens':16000}}
        def chat(self,**kwargs):
            assert kwargs['max_tokens']==16000
            assert kwargs['extra_body']['thinking']=={'type':'enabled'}
            assert kwargs['extra_body']['reasoning_effort']=='low'
            return '{}'
    with pytest.raises(StoryError,match='截断'):StoryModel(store,Transport()).ask('check','核验')
    import json
    record=json.loads(next((store.root/'calls').glob('*.json')).read_text(encoding='utf-8'))
    assert record['response_metadata']['finish_reason']=='length' and 'error' in record


def test_memory_revision_preserves_supported_facts_and_allows_removing_failed_claim():
    old={'summary':[{'text':'过桥','event_ids':['E1']}],
         'background_updates':[{'text':'桥很窄','event_ids':['E1']},{'text':'未核实宽度','event_ids':['E2']}],
         'open_threads':[],'character_updates':[],'main_character_ids':[]}
    claims=[{'claim_id':f'M{i}'} for i in range(3)]
    review={'claims':[{'claim_id':f'M{i}','verdict':'supported' if i<2 else 'issues'} for i in range(3)]}
    revised=copy.deepcopy(old);revised['background_updates'].pop()
    assert merge_memory_patches(old,claims,review,[{'claim_id':'M2','replacement':None}])==revised
    revised['summary'][0]['text']='顺带改写正确摘要'
    with pytest.raises(StoryError,match='正确条目'):merge_memory_patches(old,claims,review,[{'claim_id':'M0','replacement':revised['summary'][0]}])


def test_invalid_review_citation_is_retried_without_rewriting_the_draft():
    segments=[{'segment_id':'S_A','event_id':'A','text':'河水上涨，桥边的人拉住绳索。'}]
    bible={'characters':[],'panels':[{'source_panel_id':'1','sha256':'source','order':0}],
           'events':[{'event_id':'A','source_panel_ids':['1']}]}
    plan={'beats':[{'event_id':'A'}]};calls=[]
    class Model:
        def ask(self,stage,prompt,images=None,fresh=False):
            assert stage=='script_review'
            calls.append(fresh)
            raw=review_payload(segments,['A'])
            if not fresh:raw['content_checks'][0]['citations'][0]['text']='不存在的引文'
            else:assert '上一份复核响应未通过校验' in prompt
            return raw
    before=copy.deepcopy(segments)
    review=StoryPipeline._review(ScriptWriter(Model()),{'bible':bible,'plan':plan},segments)
    assert calls==[False,True] and segments==before and review_ready(review,bible,plan,segments)
