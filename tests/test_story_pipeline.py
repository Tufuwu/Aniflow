"""Offline contracts and adversarial fixtures; never spend live API credits."""
import copy
import json
from pathlib import Path
import threading
import time

import pytest
from PIL import Image
from flask import Flask

import config
from pipeline.fact_checker import FactChecker, sentences
from pipeline.narrative_director import NarrativeDirector, STYLES
from pipeline.story_analyzer import StoryAnalyzer
from pipeline.story_api import create_story_blueprint
from pipeline.story_bible import StoryBible, StoryConflict, StoryError
from pipeline.story_model import StoryModel
from pipeline.story_pipeline import StoryPipeline
from pipeline.text_extractor import _parse_vision_response
from story_app import import_chapter


class FixtureModel:
    """Deterministic reviewer simulates each failure category, not model quality."""
    def __init__(self):
        self.seen = []
        self.fail_check = False
        self.bad_cause = False

    def ask(self, stage, prompt, images=None, fresh=False):
        self.seen.append((stage, images, fresh))
        if stage == 'event_integration':
            from integration_fixture import integration_response
            return integration_response(prompt)
        if stage == "reconcile":
            ids=json.loads(prompt.split('按顺序附上以下来源图：',1)[1].split('。',1)[0])
            return {"panels":[{"source_panel_id":pid,"classification":"story","reason":"剧情内容"} for pid in ids]}
        if stage == "direct":
            return {"event_ids":["E00001","E00002","E00003"],"milestones":{"conflict":"E00002","climax":"E00003","rules":None,"goal_or_stakes":None,"ending":"E00003"},"rationale":"开场、冲突和结尾"}
        if stage == "identity":
            return {"names":[],"relationships":[]}
        if stage == "analyze":
            index = sum(s == "analyze" for s, _, _ in self.seen)
            fixture = json.loads((Path(__file__).parent / "fixtures/story_acceptance.json").read_text(encoding="utf-8"))
            item = fixture["panels"][(index - 1) % 3]
            return {"characters": [{"character_id":"C001", "name":"陈默", "description":"佩戴胸牌的青年",
                                     "aliases":[], "identity_evidence":"胸牌上写着陈默",
                                     'identity_match':{'method':'continuous_action','basis':'开门前后持同一把伞的连续行动'},
                                     'name_evidence':{'source_panel_id':'1','kind':'visual_text','quote':'陈默',
                                         'attribution':{'character_id':'C001','method':'name_tag',
                                             'basis':'姓名胸牌佩戴在该人物胸前','source_panel_ids':['1']}} if index % 3 == 1 else None},
                                    {"character_id":"C002", "name":None, "description":"持红伞的女孩",
                                     "aliases":[], "identity_evidence":"对话标注姓名",
                                     'identity_match':{'method':'continuous_action','basis':'交伞后仍站在门边'}}],
                    'visible_text':['陈默'] if index % 3 == 1 else [],
                    'identity_clues':[],
                    "dialogue":[{"speaker_id":None,"text":item["dialogue"]}] if item["dialogue"] else [],
                    "events":[{"summary":item["event"],"character_ids":["C001"],"actions":[item["event"]],
                               "importance":item["importance"],"role":item["role"],
                               "evidence":[{"kind":"visual","detail":item["description"]}],
                               "causes":[{"event_id":"E99999","evidence":"未来事件"}] if self.bad_cause else []}],
                    "uncertainties":[]}
        if stage == "write":
            return {"text":"陈默推开门，看见屋檐下的陌生人。"}
        if self.fail_check:
            raise RuntimeError("backend offline")
        rows = json.loads(prompt.split("待核对句子：",1)[1].split("\n返回",1)[0])
        issues = {"陈默把红伞交给林夏。":"character_confusion", "门外阳光明媚。":"event_contradiction",
                  "陌生人就是陈默的父亲。":"unsupported", "因为陈默开门，徽记才会发光。":"wrong_causality",
                  "陈默明天将加入秘密组织。":"unsupported"}
        return {"sentences":[{"sentence_index":r["sentence_index"], "verdict":"issues" if r["text"] in issues else "supported",
                              "evidence":"对照原图、事件与胸牌文字", "issues":[{"type":issues[r["text"]], "detail":"与来源不符"}]
                              if r["text"] in issues else []} for r in rows]}


@pytest.fixture
def setup_story(tmp_path):
    paths = []
    for i in range(3):
        path = tmp_path / f"{i+1}.png"
        Image.new("RGB", (40, 50), (100 + i, 10, 10)).save(path)
        paths.append({"panel_id":str(i+1),"path":str(path),"text":"繁體中文"})
    store = StoryBible(tmp_path / "story")
    model = FixtureModel()
    return store, model, paths, StoryPipeline(store, model)


def generated(setup_story):
    store, model, panels, pipeline = setup_story
    state = pipeline.analyze(panels, None)
    return pipeline.generate_verified(state["revision"])


def test_analysis_preserves_identity_dialogue_order_and_evidence(setup_story):
    store, model, panels, pipeline = setup_story
    original = Path(panels[0]["path"]).read_bytes()
    state = pipeline.analyze(panels, None)
    bible = state["bible"]
    assert [e["event_id"] for e in bible["events"]] == ["E00001", "E00002", "E00003"]
    assert bible["panels"][0]["dialogue"][0]["text"] == "拿着伞，外面下雨了。"
    assert bible["events"][2]["dialogue"] == []  # silent action must survive
    assert all(not e["causes"] for e in bible["events"])
    assert store.evidence_path(bible["panels"][0]).read_bytes() == original
    assert Path(panels[0]["path"]).read_bytes() == original
    assert bible["characters"][0]["source_panel_ids"] == ["1","2","3"]


def test_analysis_rejects_future_causes_without_committing(setup_story):
    store, model, panels, pipeline = setup_story
    model.bad_cause = True
    with pytest.raises(StoryError, match="因果"):
        pipeline.analyze(panels, None)
    assert store.read()["revision"] is None


def test_director_keeps_causal_prerequisites(setup_story):
    store, model, panels, pipeline = setup_story
    bible = pipeline.analyze(panels, None)["bible"]
    bible["events"][2]["causes"] = [{"event_id":"E00002","evidence":"明确对白"}]
    plan = NarrativeDirector().plan(bible, "suspense", 2)
    assert [b["event_id"] for b in plan["beats"]] == ["E00001","E00002","E00003"]
    assert plan["opening_event_id"] == "E00001"
    assert plan["ending_event_id"] == "E00003"


@pytest.mark.parametrize("style", STYLES)
def test_all_styles_have_program_owned_provenance(setup_story, style):
    store, model, panels, pipeline = setup_story
    state = pipeline.analyze(panels, None)
    state = pipeline.generate_verified(state["revision"], style)
    assert state["script"]["style"] == style
    for s in state["script"]["segments"]:
        e = next(e for e in state["bible"]["events"] if e["event_id"] == s["event_id"])
        assert s["source_panel_ids"] == e["source_panel_ids"]
        assert s["review_status"] == "pending"
        assert s["fact_check"]["status"] == "passed"


def test_checker_covers_four_issue_types_and_every_sentence(setup_story):
    store, model, panels, pipeline = setup_story
    state = generated(setup_story)
    s = copy.deepcopy(state["script"]["segments"][0])
    fixture = json.loads((Path(__file__).parent / "fixtures/story_acceptance.json").read_text(encoding="utf-8"))
    s["text"] = "".join(c["text"] for c in fixture["checks"])
    result = FactChecker(store,model).check(state["bible"],s)
    assert result["status"] == "issues"
    assert len(result["sentences"]) == 6
    assert {i["type"] for r in result["sentences"] for i in r["issues"]} == {
        "unsupported","character_confusion","event_contradiction","wrong_causality"}
    assert model.seen[-1][1]  # reviewer actually receives source images


def test_checker_fails_closed_on_outage_or_missing_sentences(setup_story):
    store, model, panels, pipeline = setup_story
    state = generated(setup_story)
    model.fail_check = True
    s = state["script"]["segments"][0]
    result = FactChecker(store,model).check(state["bible"],s)
    assert result["status"] == "unavailable"
    s["fact_check"] = result
    state = store.save(state,"outage",state["revision"])
    approved = pipeline.segment_action(s["segment_id"],"approve",state["revision"])
    assert approved['script']['segments'][0]['review_status'] == 'approved'
    assert approved['script']['segments'][0]['fact_check']['status'] == 'unavailable'
    model.ask = lambda *args, **kwargs: {"sentences":[]}
    assert FactChecker(store,model).check(state["bible"],s)["status"] == "unavailable"


def test_approved_text_is_locked_and_history_is_immutable(setup_story):
    store, model, panels, pipeline = setup_story
    state = generated(setup_story)
    old_revision = state["revision"]
    segment = state["script"]["segments"][0]
    state = pipeline.segment_action(segment["segment_id"], "approve", state["revision"])
    for action in ("edit", "regenerate", "reject"):
        with pytest.raises(StoryConflict, match="锁定"):
            pipeline.segment_action(segment["segment_id"], action, state["revision"], "任意新内容。")
    with pytest.raises(StoryConflict):
        pipeline.generate_verified(state["revision"])
    with pytest.raises(StoryConflict):
        pipeline.analyze(panels,state["revision"])
    assert store.read(old_revision)["script"]["segments"][0]["review_status"] == "pending"
    assert store.history()[0]["action"].startswith("approve:")


def test_manual_edit_resets_review_and_regeneration_changes_only_one_segment(setup_story):
    store, model, panels, pipeline = setup_story
    state = generated(setup_story)
    segment = state["script"]["segments"][1]
    other = copy.deepcopy(state["script"]["segments"][0])
    state = pipeline.segment_action(segment["segment_id"],"reject",state["revision"])
    state = pipeline.segment_action(segment["segment_id"],"edit",state["revision"],"林夏把红伞交给陈默。")
    assert state["script"]["segments"][1]["review_status"] == "pending"
    state = pipeline.segment_action(segment["segment_id"],"regenerate",state["revision"])
    assert state["script"]["segments"][0] == other
    assert any(stage == "write" and fresh for stage, _, fresh in model.seen)


def test_stale_updates_and_tampered_evidence_are_rejected(setup_story):
    store, model, panels, pipeline = setup_story
    state = generated(setup_story)
    with pytest.raises(StoryConflict):
        store.save(state,"stale",None)
    panel = state["bible"]["panels"][0]
    store.evidence_path(panel).write_bytes(b"changed")
    with pytest.raises(StoryError, match="校验"):
        store.evidence_path(panel)


def test_source_language_ocr_and_sentence_splitting():
    assert _parse_vision_response("Speaker: 我爸媽呢？\n林夏！\n한글\n#@$!!")[0] == "我爸媽呢？\n林夏！\n한글"
    assert sentences("谁来了？他没有回答。\n门开了！") == ["谁来了？","他没有回答。","门开了！"]


def test_model_cache_keys_include_image_model_and_prompt_and_trace_failures(tmp_path):
    class Transport:
        kind, host, api_key = "fake", "http://test", "secret-never-record"
        def __init__(self): self.calls = 0
        def chat(self, **kwargs):
            self.calls += 1
            return '{"text":"测试"}'
    store, transport = StoryBible(tmp_path / "store"), Transport()
    image = tmp_path / "image.png"
    Image.new("RGB", (8,8), "white").save(image)
    model = StoryModel(store,transport,vision_model="v1")
    model.ask("analyze","p1",[str(image)])
    model.ask("analyze","p1",[str(image)])
    assert transport.calls == 1
    model.ask("analyze","p2",[str(image)])
    model.vision_model = "v2"
    model.ask("analyze","p1",[str(image)])
    Image.new("RGB", (8,8), "black").save(image)
    model.ask("analyze","p1",[str(image)])
    assert transport.calls == 4
    assert "secret-never-record" not in "".join(p.read_text(encoding="utf-8") for p in store.root.rglob("*.json"))
    transport.chat = lambda **kwargs: "not JSON"
    with pytest.raises(StoryError):
        model.ask("write","invalid")
    assert any('"error"' in p.read_text(encoding="utf-8") for p in (store.root / "calls").glob("*.json"))


def wait_job(client, project, response):
    assert response.status_code == 202, response.get_json()
    job_id = response.get_json()["job_id"]
    for _ in range(500):
        job = client.get(f"/api/story/{project}/jobs/{job_id}").get_json()
        if job["status"] != "running":
            assert job["status"] == "completed", job
            return job
        time.sleep(.005)
    pytest.fail("background job did not finish")


def test_flask_review_end_to_end_and_evidence_access(setup_story, monkeypatch, tmp_path):
    _, model, panels, _ = setup_story
    monkeypatch.setattr(config,"STORY_DIR", str(tmp_path / "api"))
    app = Flask(__name__, template_folder=str(Path(__file__).parents[1] / "templates"))
    app.register_blueprint(create_story_blueprint(lambda project:panels,lambda store:model))
    client = app.test_client()
    base = "/api/story/test"
    assert client.get("/story-review").status_code == 200
    assert client.post(base+"/analyze",json={}).status_code == 422
    wait_job(client,"test",client.post(base+"/analyze",json={"revision":None}))
    state = client.get(base).get_json()
    wait_job(client,"test",client.post(base+"/integrate",json={"revision":state["revision"]}))
    state = client.get(base).get_json()
    assert state['stage'] == 'integrated' and state['script'] is None
    assert state['bible']['analysis_events'] and state['bible']['event_integration']
    wait_job(client,"test",client.post(base+"/generate",json={"revision":state["revision"]}))
    state = client.get(base).get_json()
    assert state['script']['segments'] and 'candidates' not in state
    assert not any(stage=='check' for stage,_,_ in model.seen)
    assert all(s['fact_check'] is None for s in state['script']['segments'])
    sid = state["script"]["segments"][0]["segment_id"]
    assert client.get(base+"/evidence/1").data == Path(panels[0]["path"]).read_bytes()
    assert client.get(base+"/evidence/does-not-exist").status_code == 404
    wait_job(client,"test",client.post(base+f"/segments/{sid}/approve",json={"revision":state["revision"]}))
    assert not any(stage == 'check' for stage, _, _ in model.seen)
    assert client.post(base+"/generate",json={"revision":state["revision"]}).status_code == 409
    assert client.get(base+"/export").get_json()["segments"][0]["review_status"] == "approved"
    assert client.get(base+"?revision=../../config.py").status_code == 422
    assert client.get("/api/story/invalid.name").status_code == 422


def test_import_uses_natural_order_and_does_not_claim_raw_slices_are_panels(tmp_path, monkeypatch):
    monkeypatch.setattr(config,"STORY_DIR",str(tmp_path / "out"))
    source = tmp_path / "source"
    source.mkdir()
    for filename in ("10.png","2.png","1.png"):
        Image.new("RGB",(10,10),filename[0] == '1' and 'white' or 'black').save(source / filename)
    imported = import_chapter("first",source)
    assert imported["input_kind"] == "raw_slices"
    assert [Path(p["source_page"]).name for p in imported["panels"]] == ["1.png","2.png","10.png"]
    assert len(list(source.iterdir())) == 3


def test_static_establishing_event_can_have_no_actions(setup_story):
    store, model, panels, pipeline = setup_story
    raw = model.ask("analyze", "fixture")
    raw["events"][0]["actions"] = []
    entry, characters, events = StoryAnalyzer._consume(raw, {}, {}, [], "1")
    assert events[0]["actions"] == []


def test_bad_analysis_is_retried_without_partially_merging_characters(setup_story):
    store, model, panels, pipeline = setup_story
    valid = model.ask("analyze", "fixture")
    bad = copy.deepcopy(valid)
    bad["events"][0]["character_ids"] = ["missing"]
    replies = iter([bad, valid, {"panels":[{"source_panel_id":"1","classification":"story","reason":"剧情内容"}]}])
    model.ask = lambda *args, **kwargs: next(replies)
    state = pipeline.analyze(panels[:1], None)
    assert state["bible"]["characters"][0]["source_panel_ids"] == ["1"]


def test_analysis_retries_bypass_invalid_cached_response(setup_story):
    store, model, panels, pipeline = setup_story
    valid = model.ask("analyze", "fixture")
    valid["dialogue"] = [{"speaker_id": None, "text": "拉"}]
    valid["events"][0]["evidence"] = [{"kind": "dialogue", "detail": "拉"}]
    bad = copy.deepcopy(valid)
    bad["events"][0]["evidence"][0]["detail"] = "手旁写有“拉”字"
    replies, cache, attempts = iter([bad, bad, valid]), {}, []
    original_ask = model.ask

    def cached_ask(stage, prompt, images=None, fresh=False):
        if stage != "analyze":
            return original_ask(stage, prompt, images, fresh)
        attempts.append((prompt, fresh))
        if fresh or prompt not in cache:
            cache[prompt] = next(replies)
        return copy.deepcopy(cache[prompt])

    model.ask = cached_ask
    state = pipeline.analyze(panels[:1], None)
    assert [fresh for _, fresh in attempts] == [False, True, True]
    assert "手旁写有“拉”字" in attempts[1][0]
    assert '本区域原文为 ["拉"]' in attempts[1][0]
    assert state["bible"]["events"][0]["evidence"][0]["detail"] == "拉"
    assert store.read()["revision"] == state["revision"]


def test_cover_does_not_pollute_story_identity_roster(setup_story):
    store, model, panels, pipeline = setup_story
    raw = model.ask("analyze", "fixture")
    raw.update(panel_type="cover", events=[], dialogue=[])
    _, characters, events = StoryAnalyzer._consume(raw, {}, {}, [], "1")
    assert characters == {} and events == []


def test_reading_regions_keep_all_rows_including_detector_misses():
    pytest.importorskip("cv2")
    from pipeline.story_inputs import reading_regions
    for height in (100, 2000, 9000):
        image = Image.new("RGB", (80,height), "black")
        regions = reading_regions(image)
        covered = set()
        for region in regions:
            covered.update(range(region["y"],region["y"]+region["h"]))
        assert covered == set(range(height))
        assert max(r["h"] for r in regions) <= 1960


def test_parallel_job_is_rejected_and_failed_job_is_visible(tmp_path, monkeypatch):
    monkeypatch.setattr(config,"STORY_DIR",str(tmp_path / "out"))
    entered, release = threading.Event(), threading.Event()
    def provider(project):
        entered.set()
        release.wait(5)
        raise StoryError("测试来源不可用")
    app = Flask(__name__)
    app.register_blueprint(create_story_blueprint(provider,lambda store:FixtureModel()))
    client=app.test_client()
    response = client.post("/api/story/test/analyze",json={"revision":None})
    assert entered.wait(3)
    assert client.post("/api/story/test/analyze",json={"revision":None}).status_code == 409
    release.set()
    for _ in range(100):
        job=client.get('/api/story/test/jobs/'+response.get_json()['job_id']).get_json()
        if job['status']=='failed':
            break
        time.sleep(.01)
    assert job['status']=='failed' and '来源不可用' in job['error']
    assert client.get('/api/story/test').get_json()['revision'] is None


def test_provider_options_do_not_override_logical_request():
    from pipeline.llm_transport import OpenAICompatTransport
    transport=OpenAICompatTransport('http://fake')
    seen={}
    def post(url,payload,headers,timeout):
        seen.update(payload)
        return {'choices':[{'message':{'content':'{}'}}]}
    transport._post=post
    assert transport.chat('m','p',extra_body={'thinking':{'type':'disabled'},'response_format':{'type':'json_object'}})=='{}'
    assert seen['thinking']=={'type':'disabled'}
    assert seen['messages']==[{'role':'user','content':'p'}]
    with pytest.raises(ValueError):
        transport.chat('m','p',extra_body={'messages':[]})


def test_checker_keeps_uncertainty_instead_of_approving(setup_story):
    store, model, panels, pipeline = setup_story
    state = generated(setup_story)
    model.ask=lambda *args,**kwargs:{'sentences':[{'sentence_index':0,'verdict':'uncertain',
        'evidence':'脸部被遮挡，无法确认身份','issues':[{'type':'character_confusion','detail':'身份待确认'}]}]}
    result=FactChecker(store,model).check(state['bible'],state['script']['segments'][0])
    assert result['status']=='uncertain'


def test_boundary_classification_keeps_story_reactions_and_excludes_end_cards(setup_story):
    store, model, panels, pipeline = setup_story
    state=pipeline.analyze(panels,None)
    model.ask=lambda *args,**kwargs:{'panels':[
        {'source_panel_id':'1','classification':'story','reason':'剧情'},
        {'source_panel_id':'2','classification':'non_story','reason':'点赞关注的片尾互动'},
        {'source_panel_id':'3','classification':'story','reason':'角色夸张反应，不排除'}]}
    bible=StoryAnalyzer(store,model).review_boundaries(state['bible'])
    assert bible['events'][1]['narrative_eligible'] is False
    assert bible['events'][2].get('narrative_eligible',True) is True


def test_confirmed_name_requires_an_exact_original_quote(setup_story):
    store, model, panels, pipeline = setup_story
    bible=pipeline.analyze(panels,None)['bible']
    bible['characters'][0]['name']='未知人物1'
    bible['characters'][0]['name_status']='unknown'
    bible['panels'][0]['dialogue']=[{'speaker_id':None,'text':'用户郁欢触发任务'}]
    bible['panels'][0]['identity_clues']=[{'kind':'name','character_ids':['C001'],'needs_review':True,
                                          'quote':'用户郁欢触发任务','detail':'系统显示姓名，需核对人物对应'}]
    claim={'character_id':'C001','name':'郁欢','source_panel_id':'1','quote':'用户郁欢触发任务',
           'attribution':{'character_id':'C001','method':'explicit_caption','basis':'系统框明确连接当前唯一人物',
                          'source_panel_ids':['1']}}
    model.ask=lambda *args,**kwargs:{'names':[claim],'relationships':[]}
    bible=StoryAnalyzer(store,model).reconcile_identity(bible)
    assert bible['characters'][0]['name']=='郁欢'
    assert bible['characters'][0]['identity_evidence'][-1]['kind']=='confirmed_name'
    model.ask=lambda *args,**kwargs:{'names':[{**claim,'name':'编造姓名'}],'relationships':[]}
    with pytest.raises(StoryError,match='姓名'):
        StoryAnalyzer(store,model).reconcile_identity(bible)


def test_director_causal_context_has_all_its_image_sources(setup_story):
    store, model, panels, pipeline = setup_story
    bible=pipeline.analyze(panels,None)['bible']
    bible['events'][2]['causes']=[{'event_id':'E00002','evidence':'明确前提'}]
    model.ask=lambda *args,**kwargs:{'event_ids':['E00001','E00003'],'milestones':{'conflict':'E00003','climax':'E00003','rules':None,'goal_or_stakes':None,'ending':'E00003'},'rationale':'压缩重复细节'}
    plan=NarrativeDirector(model).plan(bible)
    beat=plan['beats'][-1]
    assert beat['context_event_ids']==['E00002']
    assert beat['source_panel_ids']==['2','3']


def test_chinese_quotes_do_not_create_punctuation_only_sentences():
    assert sentences('她喊：“哥！”。门开了。') == ['她喊：“哥！”。','门开了。']


@pytest.mark.parametrize('chapter_path',['','/chapters/2'])
def test_candidate_api_removed_and_current_verify_available(setup_story, monkeypatch, tmp_path, chapter_path):
    _,model,panels,_=setup_story
    monkeypatch.setattr(config,'STORY_DIR',str(tmp_path/'api'))
    app=Flask(__name__)
    app.register_blueprint(create_story_blueprint(lambda project:panels,lambda store:model))
    client=app.test_client();base='/api/story/test'+chapter_path
    for action in ('compare','select','review_candidate','verify_selected'):
        assert client.post(base+'/candidates/'+action,json={'revision':None}).status_code==404
    for count in (1,2,3):
        response=client.post(base+'/generate',json={'revision':None,'candidate_count':count})
        assert response.status_code==422 and not model.seen
    response=client.post(base+'/verify',json={'revision':None})
    assert response.status_code==202
    job_id=response.get_json()['job_id']
    for _ in range(500):
        job=client.get(base+'/jobs/'+job_id).get_json()
        if job['status']!='running':break
        time.sleep(.005)
    assert job['status']=='failed' and '请先生成稿件' in job['error']
    assert not model.seen


def test_cli_removes_multi_candidate_options_and_exposes_current_verify():
    import subprocess,sys
    root=Path(__file__).parents[1]
    def cli(*args):
        return subprocess.run([sys.executable,str(root/'story_app.py'),*args],cwd=root,
                              capture_output=True,text=True,encoding='utf-8',errors='replace')
    run=cli('run','--help');assert run.returncode==0 and '--candidate-count' not in run.stdout
    verify=cli('verify','--help');assert verify.returncode==0 and '--chapter' in verify.stdout
    assert cli('candidates','--project','test','--action','compare').returncode==2
    assert cli('run','--project','test','--candidate-count','2').returncode==2
