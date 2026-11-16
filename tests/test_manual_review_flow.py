"""The manual script handoff and model shot selection, with no live API calls."""
import copy
import json
import socket
import subprocess
from pathlib import Path

from flask import Flask
import pytest

from pipeline.chapter_memory import source_ready
from pipeline.human_review import manually_approved
from pipeline.shot_api import create_shot_blueprint
from pipeline.shot_pipeline import ShotEditingPipeline
from pipeline.story_bible import StoryConflict, StoryError
from pipeline.story_pipeline import StoryPipeline
from test_engagement_flow import story
from test_shot_editing import chapter, wait_job
from test_voice_direction import accepted


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        pytest.fail('Manual handoff tests must not call live services')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)


class ForbiddenModel:
    def ask(self, *args, **kwargs):
        pytest.fail('Human approval and manual edits must not call a model')


def manual_source(chapter, status=None):
    store, source, _ = chapter
    source = copy.deepcopy(source)
    source['plan']['version'] = 5
    for segment in source['script']['segments']:
        segment.update(review_status='pending', fact_check=None if status is None else {'status': status})
    source = store.save(source, 'manual_review_pending', source['revision'])
    pipeline = StoryPipeline(store, ForbiddenModel())
    for segment in source['script']['segments']:
        source = pipeline.segment_action(segment['segment_id'], 'approve', source['revision'])
    return source


@pytest.mark.parametrize('status', [None, 'issues', 'uncertain', 'unavailable'])
def test_human_can_accept_without_successful_model_check(chapter, status):
    store, before, _ = chapter
    saved = manual_source(chapter, status)
    assert saved['stage'] == 'approved' and source_ready(saved)
    for segment in saved['script']['segments']:
        assert manually_approved(saved['bible'], segment)
        assert segment['fact_check'] == (None if status is None else {'status': status})
    ShotEditingPipeline.approved(saved)
    assert store.read(before['revision']) == before
    with pytest.raises(StoryConflict, match='锁定'):
        StoryPipeline(store, ForbiddenModel()).segment_action('S_E1', 'edit', saved['revision'], '新文案。')


@pytest.mark.parametrize('change', ['text', 'scope', 'evidence'])
def test_human_acceptance_expires_with_text_scope_or_evidence(chapter, change):
    source = manual_source(chapter)
    segment = source['script']['segments'][0]
    if change == 'text':
        segment['text'] += '新文案。'
    elif change == 'scope':
        segment['event_ids'] = ['E1', 'E2']
    else:
        source['bible']['panels'][0]['sha256'] = 'changed'
    with pytest.raises(StoryError, match='重新审核'):
        ShotEditingPipeline.approved(source)


def test_fourth_stage_hands_to_human_and_edit_has_no_hidden_model_calls(story):
    store, state, model, pipeline = story
    state = pipeline.generate(state['revision'])
    assert not any(call[0] == 'check' for call in model.calls)
    pipeline.model = ForbiddenModel()
    original = copy.deepcopy(state)
    sid = state['script']['segments'][0]['segment_id']
    changed = pipeline.segment_action(sid, 'edit', state['revision'], '他推开门，迎面撞上了这场大雨。')
    assert changed['script']['segments'][0]['fact_check'] is None
    assert changed['script']['segments'][1:] == original['script']['segments'][1:]
    for segment in changed['script']['segments']:
        changed = pipeline.segment_action(segment['segment_id'], 'approve', changed['revision'])
    assert changed['stage'] == 'approved'
    ShotEditingPipeline.approved(changed)
    assert store.read(original['revision']) == original


class ShotModel:
    def __init__(self):
        self.calls = []

    def ask(self, stage, prompt, images=None):
        data = json.loads(prompt.split('资料：', 1)[1])
        assert images and all(Path(path).is_file() for path in images)
        assert len(images) == len(data['panels'])
        self.calls.append((stage, data))
        if stage == 'shot_select':
            assert data['narration']['review_status'] == 'approved'
            return {'candidates': [{'panel_id': p['panel_id'], 'score': .8, 'crop': [0, 0, 1, 1],
                                    'reason': '原图保留当前动作与完整对白', 'visual_role': 'action',
                                    'text_ranges':[] if p.get('context_only') else
                                        [{'start_char':0,'end_char':len(data['narration']['text']),
                                          'text':data['narration']['text'],'match_type':'direct','reason':'当前段落的原图'}]}
                                   for p in data['panels']]}
        assert stage == 'visual_review'
        return {'checks': [{'panel_id': pid, 'usable': True, 'reason': '裁切保留当前动作与对白'}
                           for pid in data['panels']]}


def test_rule_shots_are_default_and_model_opt_in_reads_locked_text_and_images(chapter, monkeypatch):
    store, _, pipeline = chapter
    source = manual_source(chapter)
    model = ShotModel()
    monkeypatch.setattr('pipeline.shot_api.story_store', lambda project: store)
    app = Flask(__name__)
    app.register_blueprint(create_shot_blueprint(lambda store: model))
    client = app.test_client()
    base = '/api/editing/manual'
    result = wait_job(client, base, client.post(base + '/select', json={
        'revision': None, 'source_revision': source['revision']}))
    assert result['status'] == 'completed', result
    saved = client.get(base).get_json()
    assert saved['selector'] == 'evidence_metadata'
    assert saved['visual_review']['status'] == 'not_run'
    assert model.calls == []
    assert all(g['review_status'] == 'pending' for g in saved['groups'])
    assert saved['voice_review_required']
    with pytest.raises(StoryError, match='配音试听与审核'):
        pipeline.direct(saved['revision'])
    assert store.read() == source
    # Omitting the pipeline option also uses rules without a configured model.
    default = pipeline.select(saved['revision'], source['revision'])
    assert default['selector'] == 'evidence_metadata'
    result = wait_job(client, base, client.post(base + '/select', json={
        'revision': default['revision'], 'source_revision': source['revision'], 'use_model': True}))
    assert result['status'] == 'completed', result
    selected = client.get(base).get_json()
    assert selected['selector'] == 'metadata+model'
    assert selected['visual_review']['status'] == 'passed'
    assert {stage for stage, _ in model.calls} == {'shot_select', 'visual_review'}


def test_manual_script_voice_review_timeline_and_real_export(chapter):
    store, _, pipeline = chapter
    source = manual_source(chapter)
    # Explicit offline selection is used only for this encoder/voice fixture.
    state = accepted((store, source, pipeline))
    with pytest.raises(StoryError, match='逐段审核'):
        pipeline.export(state['revision'])
    for group in state['groups']:
        state = pipeline.review(group['narration_id'], 'approve', state['revision'], {})
    state = pipeline.direct(state['revision'])
    assert state['sync']['timeline_complete']
    assert state['sync']['semantic_alignment']['status'] == 'manual_review'
    state = pipeline.export(state['revision'])
    assert state['export']['report']['complete']
    assert state['export']['report']['decoded_frames'] == state['timeline']['total_frames']
    assert state['export']['report']['codec'] == 'h264'
    assert store.read() == source


def test_review_page_enables_manual_approval_without_model_checks(story):
    _, state, _, pipeline = story
    source = pipeline.generate(state['revision'])
    source['history'] = []
    code = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const data=JSON.parse(fs.readFileSync(0,'utf8'));
const node={value:'',textContent:''};
const host={innerHTML:'',querySelector:()=>node,querySelectorAll:()=>[]};
const window={};
vm.runInNewContext(fs.readFileSync('static/js/story_review.js','utf8'),{
  window,document:{contains:()=>true},clearTimeout,setTimeout,
  fetch:async()=>({ok:true,json:async()=>data})
});
(async()=>{
  await window.StoryReview.mount(host,'manual');
  assert.equal(node.textContent,'');
  assert(!host.innerHTML.includes('data-action="verify"'));
  assert(!host.innerHTML.includes('data-segment-action="check"'));
  assert(!host.innerHTML.includes('保存并核验'));
  assert(/data-segment-action="approve"\s*>/.test(host.innerHTML));
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    result = subprocess.run(['node', '-e', code], input=json.dumps(source), text=True,
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
