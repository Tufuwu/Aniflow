"""Offline production-route tests; no credentials or real inference are used."""
import copy
import json
import socket

import pytest

import config
from pipeline.event_integration import integrate_chapter
from pipeline.llm_transport import OpenAICompatTransport
from pipeline.story_bible import StoryBible
from pipeline.story_model import StoryModel, ResponseTruncated, validated_result, configured_integration_output_policy
from test_engagement_flow import story
from integration_fixture import integration_response
from test_integration_field_repair import payload, answer


@pytest.fixture(autouse=True)
def offline_routes(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('Stage routing tests cannot use a real network')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)
    values = {
        'STORY_API_BASE': 'https://api.deepseek.com', 'STORY_API_KEY': 'offline-default-secret',
        'STORY_WRITER_MODEL': 'deepseek-flash', 'STORY_SCRIPT_MODEL': 'openai/gpt-6-luna',
        'STORY_SCRIPT_API_BASE': 'https://openrouter.ai/api',
        'STORY_SCRIPT_API_KEY': 'offline-script-secret',
        'STORY_POLISH_MODEL': 'openai/gpt-6-luna',
        'STORY_POLISH_API_BASE': 'https://openrouter.ai/api',
        'STORY_POLISH_API_KEY': 'offline-polish-secret',
        'STORY_VISION_MODEL': 'deepseek-flash', 'STORY_CHECKER_MODEL': 'deepseek-flash',
        'STORY_INTEGRATION_MODEL': 'openai/gpt-6-luna',
        'STORY_INTEGRATION_API_BASE': 'https://openrouter.ai/api',
        'STORY_INTEGRATION_API_KEY': 'offline-integration-secret',
        'STORY_PLANNING_MODEL': 'openai/gpt-6-luna',
        'STORY_PLANNING_API_BASE': 'https://openrouter.ai/api',
        'STORY_PLANNING_API_KEY': 'offline-planning-secret',
        'STORY_INTEGRATION_MAX_OUTPUT_TOKENS': '',
    }
    for key, value in values.items():
        monkeypatch.setattr(config, key, value)


def provider_reply(body=None, finish='stop', usage=None):
    return {'model': 'offline-returned-model', 'usage': usage or {'prompt_tokens': 7, 'completion_tokens': 3},
            'choices': [{'finish_reason': finish, 'message': {'content': json.dumps(body or {'ok': True}, ensure_ascii=False)}}]}


@pytest.mark.parametrize('stage,model,host,key', [
    ('event_integration', 'openai/gpt-6-luna', 'https://openrouter.ai/api', 'offline-integration-secret'),
    ('direct', 'openai/gpt-6-luna', 'https://openrouter.ai/api', 'offline-planning-secret'),
    ('write_story', 'openai/gpt-6-luna', 'https://openrouter.ai/api', 'offline-script-secret'),
    ('write', 'openai/gpt-6-luna', 'https://openrouter.ai/api', 'offline-script-secret'),
    ('rewrite_story', 'openai/gpt-6-luna', 'https://openrouter.ai/api', 'offline-script-secret'),
    ('script_polish', 'openai/gpt-6-luna', 'https://openrouter.ai/api', 'offline-polish-secret'),
    ('script_polish_analysis', 'openai/gpt-6-luna', 'https://openrouter.ai/api', 'offline-polish-secret'),
    ('story_digest', 'deepseek-flash', 'https://api.deepseek.com', 'offline-default-secret'),
    ('context_digest', 'deepseek-flash', 'https://api.deepseek.com', 'offline-default-secret'),
    ('chapter_memory', 'deepseek-flash', 'https://api.deepseek.com', 'offline-default-secret'),
    ('analyze', 'deepseek-flash', 'https://api.deepseek.com', 'offline-default-secret'),
    ('identity', 'deepseek-flash', 'https://api.deepseek.com', 'offline-default-secret'),
    ('check', 'deepseek-flash', 'https://api.deepseek.com', 'offline-default-secret'),
])
def test_default_constructor_routes_actual_http_and_checkpoint_consistently(tmp_path, monkeypatch, stage, model, host, key):
    calls = []
    def post(url, body, headers, timeout):
        calls.append((url, body, headers))
        return provider_reply()
    monkeypatch.setattr(OpenAICompatTransport, '_post', staticmethod(post))
    store = StoryBible(tmp_path/'chapter')
    client = StoryModel(store)
    client.ask(stage, '相同的本话资料')
    url, body, headers = calls[0]
    assert url == host+'/v1/chat/completions'
    assert body['model'] == client.cache_identity(stage)['model'] == model
    assert client.cache_identity(stage)['host'] == host
    assert headers['Authorization'] == 'Bearer '+key
    assert body['response_format'] == {'type': 'json_object'}
    if host == 'https://openrouter.ai/api':
        assert 'temperature' not in body and 'top_p' not in body and 'thinking' not in body
        assert body['max_tokens'] == (128000 if stage == 'event_integration' else 6000 if stage == 'write' else 16000)
        assert client.cache_identity(stage)['request_options']['temperature'] is None
    else:
        assert body['temperature'] == (0.5 if stage in {'write', 'write_story', 'rewrite_story'} else 0.1)
        assert body['top_p'] == 0.9
        assert body['thinking']['type'] == ('enabled' if stage == 'check' else 'disabled')
    trace = json.loads(next((store.root/'calls').glob('*.json')).read_text(encoding='utf-8'))
    assert trace['response_metadata']['usage'] == {'prompt_tokens': 7, 'completion_tokens': 3}
    assert all(secret not in json.dumps(trace) for secret in (
        'offline-default-secret', 'offline-integration-secret', 'offline-planning-secret', 'offline-script-secret',
        'offline-polish-secret'))


def test_stage_output_capacity_uses_actual_model_and_provider(monkeypatch, tmp_path):
    client = StoryModel(StoryBible(tmp_path/'chapter'))
    assert client.output_policy('event_integration') == configured_integration_output_policy()
    assert client.output_policy('event_integration')['max_tokens'] == 128000
    monkeypatch.setattr(config, 'STORY_INTEGRATION_MAX_OUTPUT_TOKENS', '64000')
    assert client.output_policy('event_integration')['max_tokens'] == 64000
    assert client.output_policy('direct') is None


def test_legacy_overrides_and_empty_stage_configuration_still_work(tmp_path, monkeypatch):
    store = StoryBible(tmp_path/'chapter')
    legacy = StoryModel(store, writer_model='explicit-text')
    assert legacy._stage_model('direct')[0] == legacy._stage_model('event_integration')[0] == 'explicit-text'
    assert legacy._stage_transport('direct') is legacy.transport
    injected = OpenAICompatTransport('https://injected.invalid')
    legacy = StoryModel(store, injected)
    assert legacy._stage_transport('event_integration') is injected
    assert legacy._stage_model('direct')[0] == 'deepseek-flash'
    explicit = StoryModel(store, injected, writer_model='explicit-text', planning_model='explicit-plan',
                          planning_transport=OpenAICompatTransport('https://plan.invalid'))
    assert explicit._stage_model('direct')[0] == 'explicit-plan'
    assert explicit._stage_transport('direct').host == 'https://plan.invalid'
    assert explicit._stage_transport('event_integration') is injected
    for prefix in ('STORY_INTEGRATION', 'STORY_PLANNING'):
        monkeypatch.setattr(config, prefix+'_MODEL', '')
        monkeypatch.setattr(config, prefix+'_API_BASE', '')
    fallback = StoryModel(store)
    assert fallback._stage_model('direct')[0] == fallback._stage_model('event_integration')[0] == 'deepseek-flash'
    assert fallback._stage_transport('direct') is fallback._stage_transport('event_integration') is fallback.transport


def test_model_route_change_isolates_first_two_caches_only_and_preserves_old_records(tmp_path, monkeypatch):
    calls = []
    def post(url, body, headers, timeout):
        calls.append((url, body['model']))
        return provider_reply()
    monkeypatch.setattr(OpenAICompatTransport, '_post', staticmethod(post))
    store = StoryBible(tmp_path/'chapter')
    with monkeypatch.context() as old:
        for prefix in ('STORY_INTEGRATION', 'STORY_PLANNING'):
            old.setattr(config, prefix+'_MODEL', '')
            old.setattr(config, prefix+'_API_BASE', '')
        previous = StoryModel(store)
    unchanged = ('analyze', 'check', 'write', 'write_story', 'rewrite_story', 'script_polish',
                 'context_digest', 'story_digest', 'chapter_memory')
    old_identities = {stage: previous.cache_identity(stage) for stage in unchanged}
    for stage in ('event_integration', 'direct'):
        validated_result(store, previous, stage, {'source': 'same'}, lambda: previous.ask(stage, '同一份资料'))
    kept = {p: p.read_bytes() for folder in ('cache', 'calls') for p in (store.root/folder).glob('*.json')}
    current = StoryModel(store)
    assert all(current.cache_identity(stage) == old_identities[stage] for stage in unchanged)
    for stage in ('event_integration', 'direct'):
        assert current.cache_identity(stage) != previous.cache_identity(stage)
        for _ in range(2):
            validated_result(store, current, stage, {'source': 'same'}, lambda: current.ask(stage, '同一份资料'))
        current.ask(stage, '同一份资料')  # Raw cache also reuses only this model/route.
    assert [model for _, model in calls] == ['deepseek-flash']*2+['openai/gpt-6-luna']*2
    assert all(p.read_bytes() == raw for p, raw in kept.items())
    assert current.cache_identity('event_integration') != previous.cache_identity('event_integration')
    monkeypatch.setattr(config, 'STORY_PLANNING_API_KEY', 'offline-rotated-secret')
    rotated = StoryModel(store)
    assert rotated.cache_identity('direct') == current.cache_identity('direct')
    assert 'offline-rotated-secret' not in json.dumps(rotated.cache_identity('direct'))


@pytest.mark.parametrize('stage', ['event_integration', 'direct', 'write_story', 'write', 'rewrite_story',
                                 'script_polish', 'script_polish_analysis'])
def test_truncation_and_usage_follow_active_stage_transport(tmp_path, monkeypatch, stage):
    monkeypatch.setattr(OpenAICompatTransport, '_post', staticmethod(
        lambda *args: provider_reply(finish='length', usage={'completion_tokens': 19})))
    store = StoryBible(tmp_path/'chapter'); client = StoryModel(store)
    client.transport.last_response_metadata = {'finish_reason': 'stop', 'usage': {'wrong_transport': True}}
    with pytest.raises(ResponseTruncated):
        client.ask(stage, '截断响应')
    trace = json.loads(next((store.root/'calls').glob('*.json')).read_text(encoding='utf-8'))
    assert trace['response_metadata']['usage'] == {'completion_tokens': 19}
    assert not list((store.root/'cache').glob('*.json'))


def test_real_integration_and_existing_field_repair_both_use_integration_route(story, monkeypatch):
    store, state, _, _ = story
    calls = []
    def post(url, body, headers, timeout):
        calls.append((url, copy.deepcopy(body), headers['Authorization']))
        if len(calls) == 1:
            raw = integration_response(body['messages'][1]['content'])
            raw['events'][0]['summary'] = ''
        else:
            issue = payload(body['messages'][1]['content'])['issues'][0]
            raw = answer(issue, '修正的合成事件梳理')
        return provider_reply(raw)
    monkeypatch.setattr(OpenAICompatTransport, '_post', staticmethod(post))
    client = StoryModel(store)
    integrated = integrate_chapter(store, client, state['bible'], state['revision'])
    assert len(calls) == 2 and len(integrated['event_integration']['field_repairs']) == 1
    for url, body, auth in calls:
        assert url == 'https://openrouter.ai/api/v1/chat/completions'
        assert body['model'] == 'openai/gpt-6-luna' and body['max_tokens'] == 128000
        assert auth == 'Bearer offline-integration-secret'
    assert integrate_chapter(store, client, integrated, state['revision']) == integrated
    assert len(calls) == 2 and store.read() == state


def test_script_transport_fallback_and_explicit_injection_are_compatible(tmp_path, monkeypatch):
    store=StoryBible(tmp_path/'chapter')
    injected=OpenAICompatTransport('https://injected.invalid')
    legacy=StoryModel(store,injected)
    assert legacy._stage_transport('write_story') is injected
    legacy=StoryModel(store,writer_model='explicit-text')
    assert legacy._stage_model('write_story')[0]=='explicit-text'
    assert legacy._stage_transport('write_story') is legacy.transport
    script=OpenAICompatTransport('https://script.invalid')
    explicit=StoryModel(store,injected,script_model='explicit-script',script_transport=script)
    assert explicit._stage_transport('write_story') is script
    assert explicit._stage_model('write_story')[0]=='explicit-script'
    assert explicit._stage_transport('script_polish') is injected
    monkeypatch.setattr(config,'STORY_SCRIPT_API_BASE','')
    monkeypatch.setattr(config,'STORY_SCRIPT_MODEL','')
    fallback=StoryModel(store)
    assert fallback._stage_model('write_story')[0]=='deepseek-flash'
    assert fallback._stage_transport('write_story') is fallback.transport


def test_script_route_switch_invalidates_only_writing_caches_and_keeps_history(tmp_path, monkeypatch):
    calls=[]
    def post(url,body,headers,timeout):
        calls.append((url,body['model']))
        return provider_reply()
    monkeypatch.setattr(OpenAICompatTransport,'_post',staticmethod(post))
    store=StoryBible(tmp_path/'chapter')
    with monkeypatch.context() as old:
        old.setattr(config,'STORY_SCRIPT_API_BASE','')
        old.setattr(config,'STORY_SCRIPT_MODEL','deepseek-v4-pro')
        previous=StoryModel(store)
    for stage in ('write','write_story','rewrite_story'):
        previous.ask(stage,'相同的本话写稿资料')
    kept={p:p.read_bytes() for folder in ('cache','calls') for p in (store.root/folder).glob('*.json')}
    current=StoryModel(store)
    for stage in ('direct','event_integration','script_polish','check','analyze'):
        assert previous.cache_identity(stage)==current.cache_identity(stage)
    for stage in ('write','write_story','rewrite_story'):
        current.ask(stage,'相同的本话写稿资料')
        current.ask(stage,'相同的本话写稿资料')
    assert calls==[('https://api.deepseek.com/v1/chat/completions','deepseek-v4-pro')]*3 + [
        ('https://openrouter.ai/api/v1/chat/completions','openai/gpt-6-luna')]*3
    assert all(p.read_bytes()==value for p,value in kept.items())


def test_polish_route_switch_isolated_from_other_stages_and_preserves_cache(tmp_path,monkeypatch):
    calls=[]
    def post(url,body,headers,timeout):
        calls.append((url,body['model']))
        return provider_reply()
    monkeypatch.setattr(OpenAICompatTransport,'_post',staticmethod(post))
    store=StoryBible(tmp_path/'chapter')
    with monkeypatch.context() as old:
        old.setattr(config,'STORY_POLISH_MODEL','')
        old.setattr(config,'STORY_POLISH_API_BASE','')
        previous=StoryModel(store)
    current=StoryModel(store)
    for stage in ('direct','event_integration','write_story','check','analyze','chapter_memory'):
        assert current.cache_identity(stage)==previous.cache_identity(stage)
    for stage in ('script_polish','script_polish_analysis'):
        assert current.cache_identity(stage)!=previous.cache_identity(stage)
        previous.ask(stage,'相同第四阶段稿件')
    kept={p:p.read_bytes() for folder in ('cache','calls') for p in (store.root/folder).glob('*.json')}
    for stage in ('script_polish','script_polish_analysis'):
        current.ask(stage,'相同第四阶段稿件');current.ask(stage,'相同第四阶段稿件')
    assert calls==[('https://api.deepseek.com/v1/chat/completions','deepseek-flash')]*2+[
        ('https://openrouter.ai/api/v1/chat/completions','openai/gpt-6-luna')]*2
    assert all(p.read_bytes()==value for p,value in kept.items())


def test_polish_explicit_overrides_and_empty_fallback(tmp_path,monkeypatch):
    store=StoryBible(tmp_path/'chapter');injected=OpenAICompatTransport('https://injected.invalid')
    default=StoryModel(store,injected,writer_model='explicit-text')
    for stage in ('script_polish','script_polish_analysis'):
        assert default._stage_transport(stage) is injected and default._stage_model(stage)[0]=='explicit-text'
    dedicated=OpenAICompatTransport('https://polish.invalid')
    explicit=StoryModel(store,injected,polish_model='explicit-polish',polish_transport=dedicated)
    assert explicit._stage_model('script_polish')[0]=='explicit-polish'
    assert explicit._stage_transport('script_polish_analysis') is dedicated
    monkeypatch.setattr(config,'STORY_POLISH_MODEL','');monkeypatch.setattr(config,'STORY_POLISH_API_BASE','')
    fallback=StoryModel(store)
    assert fallback._stage_model('script_polish')[0]=='deepseek-flash'
    assert fallback._stage_transport('script_polish_analysis') is fallback.transport
