"""Original sand-bridge story: offline program contracts, never writing quality."""
import copy
import json
import socket

import pytest
from PIL import Image

from pipeline.character_identity import confirm_claim, identity_context, premature_name_issues
from pipeline.fact_checker import FactChecker
from pipeline.narration_review import polish_ready
from pipeline.script_writer import ScriptWriter, narration_lint
from pipeline.semantic_grounding import grounded_units
from pipeline.story_analyzer import StoryAnalyzer
from pipeline.story_bible import StoryBible, StoryError
from pipeline.story_pipeline import StoryPipeline
from test_engagement_flow import story, generate, polish_payload
from test_semantic_alignment import mapped


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('Identity and introduction tests must stay offline')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)


def character(name=None):
    return {'character_id':'C001','name':name,'description':'桥边握着绳索的人',
            'identity_evidence':'人物双手握住桥头绳索','aliases':[],
            'narrative_label':'桥边的人','label_evidence':'人物一直站在桥头，尚无职业与性别依据'}


def panel(char=None, quote='沙子开始往天上流，桥上的绳子就会断。'):
    return {'characters':[char or character()], 'dialogue':[{'speaker_id':None,'text':quote}],
            'visible_text':[], 'panel_type':'story', 'scene':'沙桥', 'identity_clues':[],
            'events':[{'summary':'沙流倒转，桥边的人握住绳索','character_ids':['C001'],
                       'actions':['握住绳索'],'importance':4,'role':'setup',
                       'evidence':[{'kind':'visual','detail':'桥头的人握着绳索'}],'causes':[]}],
            'uncertainties':[]}


def claim(quote='我叫闻溪。', source='2', refs=None):
    return {'character_id':'C001','name':'闻溪','source_panel_id':source,'kind':'dialogue','quote':quote,
            'attribution':{'character_id':'C001','method':'self_introduction',
                           'basis':'握绳的人指着自己作自我介绍，口型和气泡连接清晰',
                           'source_panel_ids':refs or [source]}}


def revealed_bible():
    first, chars, events=StoryAnalyzer._consume(panel(),{}, {}, [], '1')
    char=character('闻溪')
    char['identity_match']={'method':'continuous_action','basis':'接上一格双手拉紧绳索的连续动作'}
    char['name_evidence']=claim()
    second,chars,events=StoryAnalyzer._consume(panel(char,'我叫闻溪。'),{},chars,events,'2',[first])
    return {'panels':[first,second],'events':events,'characters':list(chars.values()),'relationships':[]}


def test_empty_history_and_later_new_arrival_use_the_same_temporary_roster(tmp_path):
    store=StoryBible(tmp_path/'chapter')
    source=tmp_path/'bridge.png';Image.new('RGB',(40,60),'tan').save(source)
    calls=[]
    class Model:
        def ask(self,stage,prompt,images=None,fresh=False):
            calls.append(stage)
            if stage=='analyze':
                assert '"history_available": false' in prompt
                return panel()
            if stage=='reconcile':
                return {'panels':[{'source_panel_id':'1','classification':'story','reason':'连续桥头场景'}]}
            pytest.fail('No unresolved identity claim should trigger an extra identity call')
    bible=StoryAnalyzer(store,Model()).analyze([{'panel_id':'1','path':str(source)}])
    c=bible['characters'][0]
    assert c['name_status']=='unknown' and c['gender'] is None
    assert c['character_id']=='C001' and c['source_panel_ids']==['1']
    assert c['identity_evidence'][0]['detail'] and c['label_claims'][0]['label']=='桥边的人'
    assert calls==['analyze','reconcile']
    arrival=panel(character())
    arrival['characters'][0]['character_id']='C002'
    arrival['events'][0]['character_ids']=['C002']
    _,chars,_=StoryAnalyzer._consume(arrival,{}, {'C001':c},bible['events'],'2',bible['panels'])
    assert set(chars)=={'C001','C002'} and chars['C002']['name_status']=='unknown'


def test_first_name_mention_without_attribution_stays_unknown_and_requests_review(tmp_path):
    raw=panel(character('闻溪'),'闻溪，你在哪里？')
    entry,chars,_=StoryAnalyzer._consume(raw,{}, {}, [],'1')
    assert chars['C001']['name_status']=='unknown'
    assert chars['C001']['name_claims']==[] and chars['C001']['pending_names'][0]['name']=='闻溪'
    raw['characters'][0]['name_evidence']=claim('闻溪，你在哪里？','1')
    raw['characters'][0]['name_evidence'].pop('attribution')
    with pytest.raises(StoryError,match='归属'):
        StoryAnalyzer._consume(raw,{}, {}, [],'1')
    # Verbatim matching alone cannot associate the called name with this person.
    assert entry['dialogue'][0]['speaker_id'] is None


@pytest.mark.parametrize('failure',['missing_attribution','unavailable'])
def test_failed_identity_review_preserves_temporary_analysis_without_partial_confirmation(tmp_path,failure):
    store=StoryBible(tmp_path/'chapter')
    image=tmp_path/'bridge.png';Image.new('RGB',(40,60),'tan').save(image)
    class Model:
        def ask(self,stage,prompt,images=None,**kwargs):
            if stage=='analyze':return panel(character('闻溪'),'闻溪，你在哪里？')
            if stage=='reconcile':
                return {'panels':[{'source_panel_id':'1','classification':'story','reason':'桥头求援'}]}
            if stage=='identity':
                if failure=='unavailable':raise RuntimeError('offline simulated outage')
                valid=claim('闻溪，你在哪里？','1')
                return {'names':[valid,{k:v for k,v in valid.items() if k!='attribution'}], 'relationships':[]}
            pytest.fail(stage)
    result=StoryPipeline(store,Model()).analyze([{'panel_id':'1','path':str(image)}],None)
    bible=result['bible']
    assert bible['identity_review']['status']=='needs_review'
    assert bible['characters'][0]['name_status']=='unknown' and bible['characters'][0]['name_claims']==[]
    assert len(bible['events'])==1 and result['stage']=='analyzed'


@pytest.mark.parametrize('corruption',['quote','target','source'])
def test_first_and_later_name_confirmation_share_strict_proof(corruption):
    bible=revealed_bible();sources={p['source_panel_id']:p for p in bible['panels']}
    proof=claim()
    if corruption=='quote':proof['quote']='我是闻溪。'
    if corruption=='target':proof['attribution']['character_id']='C099'
    if corruption=='source':proof['attribution']['source_panel_ids']=['missing']
    with pytest.raises(StoryError):confirm_claim(proof,sources,'C001','闻溪')
    c=character('闻溪');c['name_evidence']=proof
    c['identity_match']={'method':'continuous_action','basis':'仍在拉同一根绳索'}
    with pytest.raises(StoryError):
        StoryAnalyzer._consume(panel(c,'我叫闻溪。'),{}, {}, [],'2',bible['panels'][:1])


def test_plain_name_and_address_change_reach_image_review_without_family_keywords(tmp_path):
    store=StoryBible(tmp_path/'chapter');bible=revealed_bible()
    for p in bible['panels']:
        path=tmp_path/f"{p['source_panel_id']}.png";Image.new('RGB',(30,40),'tan').save(path)
        p.update(store.snapshot(path))
    bible['panels'][0]['identity_clues']=[{'kind':'address','quote':'','character_ids':['C001'],
                                          'needs_review':True,'detail':'后面的自我介绍是否对应桥边的人'}]
    calls=[]
    class Model:
        def ask(self,stage,prompt,images=None,**kwargs):
            calls.append((stage,images))
            assert '不能只凭发色或衣服' in prompt
            return {'names':[claim()], 'relationships':[], 'unresolved':[]}
    result=StoryAnalyzer(store,Model()).reconcile_identity(bible)
    assert calls[0][0]=='identity' and len(calls[0][1])==2
    assert result['characters'][0]['name_claims'][-1]['reveal_order']==1


def test_early_literal_name_does_not_backdate_later_attribution():
    sources={'1':{'order':0,'dialogue':[{'text':'闻溪在哪里？'}]},
             '2':{'order':1,'dialogue':[{'text':'闻溪就是桥边握绳索的人。'}]}}
    proof=claim('闻溪在哪里？','1',['1','2'])
    verified=confirm_claim(proof,sources,'C001','闻溪')
    assert verified['reveal_panel_id']=='2' and verified['reveal_order']==1


def test_confirmed_nickname_has_its_own_proof_and_reveal_without_overwriting_name():
    bible=revealed_bible();c=character('小溪')
    c.update(identity_match={'method':'continuous_action','basis':'桥边的人拉着同一根绳子继续说话'},
             name_evidence={**claim('大家叫我小溪。','3'),'name_kind':'alias'})
    chars={v['character_id']:v for v in bible['characters']}
    _,chars,_=StoryAnalyzer._consume(panel(c,'大家叫我小溪。'),{},chars,bible['events'],'3',bible['panels'])
    assert chars['C001']['name']=='闻溪' and chars['C001']['aliases']==['小溪']
    alias=chars['C001']['name_claims'][-1]
    assert alias['name']=='小溪' and alias['reveal_order']==2
    unsupported=panel(character('闻溪'),'闻溪在哪里？')
    unsupported['characters'][0]['narrative_label']='闻溪'
    with pytest.raises(StoryError,match='绕过姓名归属'):
        StoryAnalyzer._consume(unsupported,{}, {}, [],'1')


def test_same_clothes_do_not_suffice_to_reuse_an_internal_identity():
    first,chars,events=StoryAnalyzer._consume(panel(),{}, {}, [],'1')
    c=character();c['identity_match']={'method':'clothing_similarity','basis':'穿着一样'}
    with pytest.raises(StoryError,match='复用人物编号'):
        StoryAnalyzer._consume(panel(c),{},chars,events,'2',[first])
    # A continuous action provides a distinct kind of evidence, still judged by a model.
    c['identity_match']={'method':'continuous_action','basis':'从握绳到拉紧同一绳索的连续动作'}
    _,result,_=StoryAnalyzer._consume(panel(c),{},chars,events,'2',[first])
    assert result['C001']['source_panel_ids']==['1','2']


def test_future_name_is_blocked_but_original_ambiguous_mention_is_not_rewritten():
    bible=revealed_bible()
    segments=[{'event_id':'E00001','text':'闻溪在桥头拉住绳索。'},
              {'event_id':'E00002','text':'桥边的人说：“我叫闻溪。”'}]
    context=identity_context(bible,segments)
    assert not any(c.get('name') for c in context[0]['available_by_node_end'])
    assert context[0]['not_yet_revealed'][0]['name']=='闻溪'
    assert premature_name_issues(bible,segments)[0]['event_id']=='E00001'
    segments[0]['text']='桥边的人握住绳索。'
    assert premature_name_issues(bible,segments)==[]
    bible['panels'][0]['dialogue']=[{'text':'闻溪在哪里？','speaker_id':None}]
    segments[0]['text']='有人喊：“闻溪在哪里？”'
    assert premature_name_issues(bible,segments)==[]  # Attribution remains a semantic review obligation.


def test_early_name_is_rejected_before_factual_model_call():
    bible=revealed_bible()
    segment={'segment_id':'S_E00001','event_id':'E00001','event_ids':['E00001'],
             'context_event_ids':[],'source_panel_ids':['1'],'text':'闻溪拉住绳子。'}
    class Model:
        def ask(self,*args,**kwargs):pytest.fail('A definite early leak needs no API request')
    check=FactChecker(object(),Model()).check(bible,segment)
    assert check['status']=='issues' and '提前' in check['sentences'][0]['issues'][0]['detail']


def test_unknown_roster_can_write_with_stable_supported_labels():
    bible=revealed_bible()
    bible['characters'][0].update(name='未知人物C001',name_status='unknown',name_claims=[])
    beat={'event_id':'E00001','event_ids':['E00001'],'source_panel_ids':['1'],'section':'opening'}
    class Model:
        def ask(self,stage,prompt,**kwargs):
            assert 'identity_availability' in prompt and '桥边的人' in prompt
            assert '职业、关系或行为身份' in prompt
            return {'text':'沙流开始倒转，桥边的人拉紧绳索，等另一头传来回应。'}
    result=ScriptWriter(Model()).write_segment(bible,{'style':'faithful','beats':[beat]},beat)
    assert narration_lint([result])==[] and result['review_status']=='pending'
    result['text']='未知人物C001拉紧绳索。'
    assert narration_lint([result])








def test_identity_evidence_change_invalidates_sentence_mapping_and_same_text_revision(mapped):
    source,segment,_,_=mapped
    source['bible']['characters'].append({'character_id':'C1','name_status':'unknown','label_claims':[]})
    assert grounded_units(source['bible'],segment)==[]
    class Model:
        def ask(self,*args,**kwargs):
            return {'segments':[{'event_id':segment['event_id'],'text':segment['text']}]}
    result=ScriptWriter(Model()).revise_story(source['bible'],source['plan'],[segment],
                {'issues':[{'event_id':segment['event_id'],'detail':'核对称呼依据'}]})
    assert result[0]['text']==segment['text'] and result[0]['fact_check'] is None
    assert result[0]['review_status']==segment['review_status'] and segment['fact_check']
