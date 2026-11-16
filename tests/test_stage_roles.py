"""Verify actual transport roles and checkpoint dependencies without a network."""
import socket
import json

import pytest

import pipeline.story_model as prompts
from pipeline.story_bible import StoryBible
from pipeline.story_model import StoryModel, validated_result
from pipeline.llm_transport import OpenAICompatTransport, OllamaTransport


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*a,**k):raise AssertionError('Role tests cannot call real APIs')
    monkeypatch.setattr(socket.socket,'connect',deny)
    monkeypatch.setattr(socket,'create_connection',deny)
    monkeypatch.setattr(prompts.config, 'STORY_INTEGRATION_MAX_OUTPUT_TOKENS', '')


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
    ('script_polish','POLISH_SYSTEM','writer'),('script_polish_analysis','POLISH_ANALYSIS_SYSTEM','writer'),('story_plan_review','REVIEW_SYSTEM','writer'),
    ('chapter_memory','MEMORY_SYSTEM','writer'),
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


@pytest.mark.parametrize('stage,expected', [
    ('write', 'script-pro'), ('write_story', 'script-pro'), ('rewrite_story', 'script-pro'),
    ('direct', 'text-flash'), ('event_integration', 'text-flash'),
    ('story_digest', 'text-flash'), ('script_polish', 'text-flash'),
    ('context_digest', 'text-flash'),
    ('chapter_memory', 'text-flash'), ('check', 'checker'), ('analyze', 'vision')])
def test_script_model_config_routes_actual_requests_only_for_writing(tmp_path, monkeypatch, stage, expected):
    monkeypatch.setattr(prompts.config, 'STORY_WRITER_MODEL', 'text-flash')
    monkeypatch.setattr(prompts.config, 'STORY_SCRIPT_MODEL', 'script-pro')
    transport = Transport()
    model = StoryModel(StoryBible(tmp_path/'chapter'), transport,
                       vision_model='vision', checker_model='checker')
    model.ask(stage, '离线路由验证')
    assert transport.calls[0]['model'] == model.cache_identity(stage)['model'] == expected


def test_script_default_and_explicit_model_overrides_remain_compatible(tmp_path, monkeypatch):
    monkeypatch.setattr(prompts.config, 'STORY_WRITER_MODEL', 'text-flash')
    monkeypatch.setattr(prompts.config, 'STORY_SCRIPT_MODEL', '')
    store = StoryBible(tmp_path/'chapter')
    assert StoryModel(store, Transport())._stage_model('write_story')[0] == 'text-flash'
    monkeypatch.setattr(prompts.config, 'STORY_SCRIPT_MODEL', 'script-pro')
    legacy = StoryModel(store, Transport(), writer_model='explicit-text')
    assert legacy._stage_model('write_story')[0] == legacy._stage_model('direct')[0] == 'explicit-text'
    explicit = StoryModel(store, Transport(), writer_model='explicit-text', script_model='explicit-script')
    assert explicit._stage_model('write_story')[0] == 'explicit-script'
    assert explicit._stage_model('direct')[0] == 'explicit-text'


def test_script_model_change_isolates_writing_cache_and_retains_old_traces(tmp_path):
    store = StoryBible(tmp_path/'chapter')
    transport = Transport()
    old = StoryModel(store, transport, writer_model='text-flash', script_model='script-flash')
    new = StoryModel(store, transport, writer_model='text-flash', script_model='script-pro')
    assert old.cache_identity('direct') == new.cache_identity('direct')
    assert old.cache_identity('script_polish') == new.cache_identity('script_polish')
    for stage in ('write', 'write_story', 'rewrite_story'):
        assert old.cache_identity(stage) != new.cache_identity(stage)
        old.ask(stage, '同一份离线写稿提示')
    previous = {p:p.read_bytes() for name in ('cache', 'calls') for p in (store.root/name).glob('*.json')}
    for stage in ('write', 'write_story', 'rewrite_story'):
        new.ask(stage, '同一份离线写稿提示')
        new.ask(stage, '同一份离线写稿提示')
    assert [r['model'] for r in transport.calls] == ['script-flash']*3 + ['script-pro']*3
    assert all(p.read_bytes() == data for p, data in previous.items())


@pytest.mark.parametrize('stage,expected', [('event_integration', None), ('direct', 16000), ('context_digest', 6000)])
def test_unknown_provider_preserves_default_without_changing_other_stages(tmp_path, stage, expected):
    store = StoryBible(tmp_path/'chapter')
    transport = Transport()
    StoryModel(store, transport).ask(stage, '离线容量检查')
    record = json.loads(next((store.root/'calls').glob('*.json')).read_text(encoding='utf-8'))
    assert transport.calls[0]['max_tokens'] == record['request']['max_tokens'] == expected


@pytest.mark.parametrize('backend', ['openai', 'ollama'])
def test_unknown_provider_does_not_guess_an_output_capacity(tmp_path, monkeypatch, backend):
    payloads = []
    def post(url, payload, headers, timeout):
        payloads.append(payload)
        return {'choices': [{'message': {'content': '{}'}, 'finish_reason': 'stop'}]} if backend == 'openai' else {'message': {'content': '{}'}}
    transport_type = OpenAICompatTransport if backend == 'openai' else OllamaTransport
    monkeypatch.setattr(transport_type, '_post', staticmethod(post))
    model = StoryModel(StoryBible(tmp_path/'chapter'), transport_type('https://offline.invalid'))
    model.ask('event_integration', '离线检查：不设置程序输出上限')
    assert len(payloads) == 1
    assert not any(k in payloads[0] for k in ('max_tokens', 'max_output_tokens', 'max_completion_tokens'))
    assert 'num_predict' not in payloads[0].get('options', {})
    assert model.cache_identity('event_integration')['output_policy']['mode'] == 'provider_default_unknown_maximum'
    assert 'output_policy' not in model.cache_identity('direct')


@pytest.mark.parametrize('model_name', ['deepseek-flash', 'deepseek-v4-pro'])
def test_official_deepseek_requests_advertised_capacity_on_the_actual_wire(tmp_path, monkeypatch, model_name):
    payloads = []
    def post(url, payload, headers, timeout):
        payloads.append(payload)
        return {'choices': [{'message': {'content': '{}'}, 'finish_reason': 'stop'}]}
    monkeypatch.setattr(OpenAICompatTransport, '_post', staticmethod(post))
    store = StoryBible(tmp_path/'chapter')
    model = StoryModel(store, OpenAICompatTransport('https://api.deepseek.com'), writer_model=model_name)
    model.ask('event_integration', '整合和定点纠错共用此阶段策略')
    record = json.loads(next((store.root/'calls').glob('*.json')).read_text(encoding='utf-8'))
    policy = model.cache_identity('event_integration')['output_policy']
    assert payloads[0]['max_tokens'] == record['request']['max_tokens'] == policy['max_tokens'] == 393216
    assert policy['mode'] == 'documented_provider_maximum'
    assert record['request']['output_policy'] == policy
    assert payloads[0]['thinking']['type'] == 'disabled'
    model.ask('direct', '其他阶段沿用原配置')
    assert payloads[1]['max_tokens'] == 16000


def test_configured_capacity_changes_only_integration_checkpoint_and_keeps_old_cache(tmp_path, monkeypatch):
    store = StoryBible(tmp_path/'chapter')
    transport = Transport()
    client = StoryModel(store, transport)
    dependencies = {'input_hash': 'same-source'}
    compute = lambda: client.ask('event_integration', '同一输入')
    validated_result(store, client, 'event_integration', dependencies, compute)
    prior = {p: p.read_bytes() for p in (store.root/'cache').glob('*.json')}
    other = client.cache_identity('direct')
    monkeypatch.setattr(prompts.config, 'STORY_INTEGRATION_MAX_OUTPUT_TOKENS', '65536')
    validated_result(store, client, 'event_integration', dependencies, compute)
    assert len(transport.calls) == 2 and transport.calls[-1]['max_tokens'] == 65536
    assert client.cache_identity('direct') == other
    assert all(p.read_bytes() == value for p, value in prior.items())


@pytest.mark.parametrize('value', ['0', '-1', 'invalid', True, 1.5])
def test_invalid_capacity_never_sends_a_request(tmp_path, monkeypatch, value):
    monkeypatch.setattr(prompts.config, 'STORY_INTEGRATION_MAX_OUTPUT_TOKENS', value)
    transport = Transport()
    with pytest.raises(prompts.StoryError, match='正整数'):
        StoryModel(StoryBible(tmp_path/'chapter'), transport).ask('event_integration', '不得调用')
    assert not transport.calls
