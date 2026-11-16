"""Lossless writing formats, exceptional metadata and stage isolation."""
import copy
import json
import socket

import pytest

from pipeline.fact_material import fact_material, writing_material
from pipeline.input_compaction import compact_json
from pipeline.planning_material import planning_material
from pipeline.prompt_material import text_evidence
from pipeline.script_writer import ScriptWriter
from test_fact_material import source
from test_planning_material import chapter
from tools.audit_prompt_inputs import Capture, Captured, inventory, json_after


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('Writing compaction tests cannot call real APIs')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)


def expand_events(material):
    """Interpret the declared wire format, independently of the encoder."""
    rows = copy.deepcopy(material['events'])
    schema = material['event_format']
    for event in rows:
        for field in ('facets', 'evidence'):
            if field in event:
                event[field] = [dict(zip(schema[field], item)) if isinstance(item, list) else item
                                for item in event[field]]
        if 'time' not in event:
            event['time'] = copy.deepcopy(schema['omitted_time'])
    return rows


def test_serialized_writing_view_restores_every_event_value_and_order(source):
    bible, plan = source
    bible['events'][0]['facets'].append(copy.deepcopy(bible['events'][0]['facets'][0]))
    bible['events'][0]['causes'] = [{'event_id': 'E2', 'evidence': '仅测试明确因果。'}]
    bible['events'][2].update(history_origin={'chapter': 1, 'revision': 'old-version'},
                             time={'original_time': '两年前', 'status': 'supported'})
    before = copy.deepcopy((bible, plan))
    expected = fact_material(bible, plan)
    packed = json.loads(compact_json(writing_material(bible, plan)))
    assert expand_events(packed) == expected['events']
    assert packed['plan'] == expected['plan']
    assert packed['characters'] == expected['characters']
    assert packed['identity_availability'] == expected['identity_availability']
    assert len(compact_json(packed['events'])) < len(compact_json(expected['events']))
    assert (bible, plan) == before


def test_extended_incomplete_and_nonobject_rows_keep_all_metadata(source):
    bible, plan = source
    event = bible['events'][0]
    event['facets'] += [
        {'role': 'unknown', 'text': '来源有冲突。', 'status': 'conflicting',
         'attribution': {'speaker_id': None}},
        {'text': '没有分类的旧记录。'}, None]
    event['evidence'] += [
        {'kind': 'dialogue', 'detail': '进来吧。', 'source_panel_id': 'other',
         'status': 'uncertain', 'history_origin': {'revision': 'other-version'}},
        {'detail': '缺少类型的旧记录。'}, None]
    expected = fact_material(bible, plan)['events']
    packed = writing_material(bible, plan)
    assert expand_events(packed) == expected
    assert isinstance(packed['events'][0]['facets'][-3], dict)
    assert isinstance(packed['events'][0]['evidence'][-3], dict)


@pytest.mark.parametrize('value', [
    None, {},
    {'original_time': '未知', 'status': 'uncertain'},
    {'original_time': '未知', 'status': 'unknown', 'note': '待补历史。'},
    {'original_time': '两年前', 'status': 'supported'},
    {'original_time': '未知（承接历史说明）', 'status': 'unknown'},
    {'original_time': 0, 'status': 'unknown'},
])
def test_time_other_than_plain_unknown_keeps_exact_value(source, value):
    bible, plan = source
    bible['events'][0]['time'] = copy.deepcopy(value)
    packed = writing_material(bible, plan)
    assert packed['events'][0]['time'] == value
    assert expand_events(packed) == fact_material(bible, plan)['events']


def test_missing_legacy_time_stays_unknown_without_inventing_a_current_time(source):
    bible, plan = source
    bible['events'][0].pop('time')
    packed = writing_material(bible, plan)
    assert 'time' not in packed['events'][0] and 'time' not in bible['events'][0]
    assert packed['event_format']['omitted_time'] == {'original_time': '未知', 'status': 'unknown'}
    assert packed['events'][0]['reveal_order'] == 0


def test_writing_format_is_isolated_from_planning_digest_review_and_source(source):
    bible, plan = source
    before = copy.deepcopy((bible, plan))
    planning_before = planning_material(bible)
    digest_before = fact_material(bible, plan)
    review_before = text_evidence(bible, plan['beats'])
    packed = writing_material(bible, plan)
    packed['events'][0]['facets'][0][2] = '只修改写作视图'
    packed['event_format']['omitted_time']['status'] = 'changed'
    assert (bible, plan) == before
    assert planning_material(bible) == planning_before
    assert fact_material(bible, plan) == digest_before
    assert text_evidence(bible, plan['beats']) == review_before
    assert writing_material(bible, plan)['event_format']['omitted_time']['status'] == 'unknown'
    assert isinstance(digest_before['events'][0]['facets'][0], dict)


def test_actual_writer_declares_format_and_audit_accounts_for_it(source):
    bible, plan = source
    model = Capture()
    with pytest.raises(Captured):
        ScriptWriter(model).write_story(bible, plan)
    stage, prompt, images = model.request
    packed = json_after(prompt, '叙事依据：')
    assert stage == 'write_story' and not images
    assert expand_events(packed) == fact_material(bible, plan)['events']
    assert 'event_format同名字段顺序' in prompt and '对象条目保留额外信息' in prompt
    assert '不表示本话当下或时间已确认' in prompt
    assert 'essential_event_ids' in prompt and '先揭晓后使用' in prompt
    assert '首段第一句' in prompt and '没有目标时长' in prompt
    audit = inventory(model.request)
    assert audit['components_chars']['event_format'] > 0
    assert sum(audit['components_chars'].values()) + audit['instructions_keys_and_layout_chars'] == len(prompt)


def test_audit_still_reads_historical_candidate_heading(source):
    bible, plan = source
    model = Capture()
    with pytest.raises(Captured):
        ScriptWriter(model).write_story(bible, plan)
    stage, prompt, images = model.request
    current = inventory(model.request)
    assert 'candidate_direction' not in current['components_chars']
    historical = prompt + '\n本候选的表达任务（不得改事实/时间/因果/身份揭晓顺序）：{"direction":"历史输入：保留原样"}'
    current = inventory(model.request)
    old = inventory((stage, historical, images))
    assert old['components_chars']['candidate_direction'] > 0
    assert sum(old['components_chars'].values()) + old['instructions_keys_and_layout_chars'] == len(historical)
