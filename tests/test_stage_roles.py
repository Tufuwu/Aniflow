"""Verify actual transport roles and checkpoint dependencies without a network."""
import socket

import pytest

import pipeline.story_model as prompts
from pipeline.story_bible import StoryBible
from pipeline.story_model import StoryModel, validated_result


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*a,**k):raise AssertionError('Role tests cannot call real APIs')
    monkeypatch.setattr(socket.socket,'connect',deny)
    monkeypatch.setattr(socket,'create_connection',deny)


class Transport:
    kind='test';host='offline.invalid'
    def __init__(self):self.calls=[]
    def chat(self,**request):
        self.calls.append(request)
        return '{"ok":true}'


@pytest.mark.parametrize('stage,role,model',[
    ('event_integration','INTEGRATION_SYSTEM','writer'),
    ('context_digest','CONTEXT_SYSTEM','writer'),('direct','PLAN_SYSTEM','writer'),
    ('story_digest','DIGEST_SYSTEM','writer'),('write','WRITER_SYSTEM','writer'),
    ('write_story','WRITER_SYSTEM','writer'),('rewrite_story','WRITER_SYSTEM','writer'),
    ('script_review','REVIEW_SYSTEM','writer'),('story_plan_review','REVIEW_SYSTEM','writer'),
    ('candidate_compare','REVIEW_SYSTEM','writer'),('chapter_memory','MEMORY_SYSTEM','writer'),
    ('analyze','SYSTEM','vision'),('reconcile','SYSTEM','vision'),('identity','SYSTEM','vision'),
    ('check','SYSTEM','checker'),('chapter_memory_review','SYSTEM','vision'),
    ('shot_select','SYSTEM','vision'),('visual_review','SYSTEM','vision')])
def test_actual_role_and_model_match_checkpoint_identity(tmp_path,stage,role,model):
    transport=Transport();client=StoryModel(StoryBible(tmp_path/'chapter'),transport,
        vision_model='vision',writer_model='writer',checker_model='checker')
    client.ask(stage,'离线结构检查')
    sent=transport.calls[0];identity=client.cache_identity(stage)
    assert sent['system']==getattr(prompts,role)==identity['system']
    assert sent['model']==model==identity['model']
    assert identity['encoder'] is not None if model=='vision' else identity['encoder'] is None


def test_reference_role_change_invalidates_only_its_checkpoint_and_keeps_traces(tmp_path,monkeypatch):
    store=StoryBible(tmp_path/'chapter');transport=Transport();client=StoryModel(store,transport)
    dependencies={'reference_revision':'offline-source-v1'}
    compute=lambda:client.ask('context_digest','整理同一份参考，不写解说')
    first=validated_result(store,client,'context_digest',dependencies,compute)
    assert validated_result(store,client,'context_digest',dependencies,compute)==first
    assert len(transport.calls)==1
    previous={p:p.read_bytes() for folder in ('cache','calls') for p in (store.root/folder).glob('*.json')}
    writer_identity=client.cache_identity('write_story')
    monkeypatch.setattr(prompts,'CONTEXT_SYSTEM',prompts.CONTEXT_SYSTEM+'\n新增离线参考整理约束。')
    validated_result(store,client,'context_digest',dependencies,compute)
    assert len(transport.calls)==2 and transport.calls[0]['system']!=transport.calls[1]['system']
    assert client.cache_identity('write_story')==writer_identity
    assert all(path.read_bytes()==content for path,content in previous.items())


def test_reference_role_preserves_reference_status_and_first_chapter_limit():
    assert '资料整理员' in prompts.CONTEXT_SYSTEM and '不写正式解说' in prompts.CONTEXT_SYSTEM
    assert '同话资料改称跨话前情' in prompts.CONTEXT_SYSTEM
    assert '推测、未知关系保持原状态' in prompts.CONTEXT_SYSTEM
    assert prompts.system_for_stage('write_story')==prompts.WRITER_SYSTEM
    assert '中文故事解说作者' in prompts.WRITER_SYSTEM
