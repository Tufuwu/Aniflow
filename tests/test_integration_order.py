"""Source order is meaningful; fragment table offsets within one source aren't."""
import copy
import json
import socket

import pytest

import pipeline.event_integration as integration
from pipeline.event_integration import consume, fragments_for, integrate_chapter
from pipeline.story_bible import StoryBible, StoryError
from pipeline.story_model import StoryModel
from test_event_integration import analysis, by_panels, event, merged_response


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('Order regression tests must not use a real API')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)


def interleaved_response(analysis):
    analysis['events'][0]['summary'] = '铃声响起'
    message = {**copy.deepcopy(analysis['events'][0]), 'event_id': 'E_MESSAGE', 'summary': '传来求助文字'}
    analysis['events'].insert(1, message)
    analysis['panels'][0]['dialogue'] = [{'speaker_id': None, 'text': '帮帮我'}]
    analysis['panels'][0]['visible_text'] = ['叮铃铃']
    fragments = fragments_for(analysis)
    first = next(f['fragment_id'] for f in fragments if f['source_event_id'] == 'E1')
    bell = next(f['fragment_id'] for f in fragments if f['path'] == 'panels/0/visible_text/0')
    response = merged_response(analysis)
    other = [f for f in by_panels(fragments, '1') if f not in {first, bell}]
    response['events'][:1] = [event(fragments, [first, bell], '铃声响起'),
                              event(fragments, other, '传来求助文字')]
    return fragments, response


class Transport:
    kind, host = 'test', 'offline.invalid'

    def __init__(self, raw):
        self.raw, self.calls = raw, []

    def chat(self, **request):
        self.calls.append(request)
        return json.dumps(self.raw, ensure_ascii=False)


def test_same_source_interleaving_passes_actual_request_without_changing_rows_or_repairing(tmp_path, analysis):
    fragments, raw = interleaved_response(analysis)
    before = copy.deepcopy(raw)
    bible = {k: copy.deepcopy(analysis[k]) for k in ('panels', 'events', 'characters', 'relationships')}
    store, transport = StoryBible(tmp_path), Transport(raw)
    model = StoryModel(store, transport)
    result = integrate_chapter(store, model, bible, 'source-v1')
    assert len(transport.calls) == 1 and transport.calls[0]['images'] == []
    assert result['event_integration']['local_repairs'] == result['event_integration']['field_repairs'] == []
    assert [e['summary'] for e in result['events'][:2]] == ['铃声响起', '传来求助文字']
    for actual, expected in zip(result['events'], before['events']):
        for key in ('summary', 'fragment_ids', 'kind', 'time', 'facets'):
            assert actual[key] == expected[key]
    assert {f for e in result['events'] for f in e['fragment_ids']} == {f['fragment_id'] for f in fragments}
    assert result['analysis_events'] == bible['events'] and raw == before
    assert integrate_chapter(store, model, result, 'source-v1') == result and len(transport.calls) == 1


def test_same_source_event_order_is_not_decided_by_fragment_offsets(analysis):
    fragments, raw = interleaved_response(analysis)
    raw['events'][:2] = reversed(raw['events'][:2])
    result = consume(raw, analysis, fragments)
    assert [e['summary'] for e in result[:2]] == ['传来求助文字', '铃声响起']


@pytest.mark.parametrize('mode', ['reversed', 'crossed', 'shared_source_different_fragments', 'multi_source_fragment'])
def test_cross_source_order_remains_enforced(analysis, mode):
    if mode == 'multi_source_fragment':
        analysis['events'][0]['source_panel_ids'] = ['1', '3']
    fragments = fragments_for(analysis)
    if mode == 'reversed':
        raw = merged_response(analysis)
        raw['events'].reverse()
    elif mode == 'crossed':
        raw = {'events': [event(fragments, by_panels(fragments, '1', '3')),
                          event(fragments, by_panels(fragments, '2', '4', '5'))]}
    elif mode == 'shared_source_different_fragments':
        third = by_panels(fragments, '3')
        raw = {'events': [event(fragments, by_panels(fragments, '1') + third[:1]),
                          event(fragments, by_panels(fragments, '2', '4', '5') + third[1:])]}
    else:
        first = next(f['fragment_id'] for f in fragments if f['source_event_id'] == 'E1')
        second = by_panels(fragments, '2')
        rest = [f['fragment_id'] for f in fragments if f['fragment_id'] not in {first, *second}]
        raw = {'events': [event(fragments, [first]), event(fragments, second), event(fragments, rest)]}
    with pytest.raises(StoryError, match='揭示顺序'):
        consume(raw, analysis, fragments)


def test_exact_shared_multi_source_fragment_can_still_be_split(analysis):
    analysis['events'][0]['source_panel_ids'] = ['1', '2']
    fragments = fragments_for(analysis)
    first = by_panels(fragments, '1', '2')
    raw = {'events': [event(fragments, first, '同一来源过程的前半段'),
                      event(fragments, first, '同一来源过程的后半段'),
                      *[event(fragments, by_panels(fragments, str(i))) for i in (3, 4, 5)]]}
    assert len(consume(raw, analysis, fragments)) == 5


def test_policy_change_revalidates_existing_raw_cache_and_keeps_prior_checkpoint(tmp_path, analysis, monkeypatch):
    bible = {k: copy.deepcopy(analysis[k]) for k in ('panels', 'events', 'characters', 'relationships')}
    raw = merged_response(analysis)
    store, transport = StoryBible(tmp_path), Transport(raw)
    model = StoryModel(store, transport)
    old_contract = {k: copy.deepcopy(v) for k, v in integration.OUTPUT_CONTRACT.items() if k != 'reveal_order'}
    old_contract['version'] = 5
    old_contract['coverage'] = 'every_story_fragment'
    with monkeypatch.context() as old:
        old.setattr(integration, 'VERSION', 5)
        old.setattr(integration, 'OUTPUT_CONTRACT', old_contract)
        previous = integrate_chapter(store, model, bible, 'source-v1')
    old_files = {p: p.read_bytes() for p in (store.root / 'cache').glob('*.json')}
    current = integrate_chapter(store, model, previous, 'source-v1')
    assert len(transport.calls) == 1  # Rechecked raw response; no second API call.
    assert current['event_integration']['version'] == integration.VERSION
    assert previous['event_integration']['output_contract_hash'] != current['event_integration']['output_contract_hash']
    assert current['events'] == previous['events']
    assert len(list((store.root / 'cache').glob('valid-*.json'))) == 2
    assert all(p.read_bytes() == content for p, content in old_files.items())
    activity = [json.loads(p.read_text(encoding='utf-8'))['event'] for p in (store.root / 'usage').glob('*.json')]
    assert activity.count('checkpoint_saved') == 2 and activity.count('cache_hit') == 1
