"""Current versioned settings enter planning without changing chapter evidence."""
import copy
import json
import socket

import pytest

from pipeline.chapter_memory import planning_settings, save_series_notes, series_notes
from pipeline.story_bible import StoryBible, digest
from test_engagement_flow import story
from test_story_reuse import grounding_model, counts


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('Settings tests must never use real model/TTS APIs')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)


def test_current_settings_enter_second_step_once_and_leave_first_step_and_other_stages_unchanged(story):
    store, state, model, pipeline = story
    grounding_model(story)
    state['bible']['prior_context'] = {'series_notes': {'background': 'OLD_COPIED_JSON_SENTINEL'},
                                     'background': [{'text': 'OLD_SAME_CHAPTER_SENTINEL'}]}
    state = store.save(state, 'fixture_old_snapshot', state['revision'])
    original = copy.deepcopy(state)
    notes = save_series_notes(store, 'WORLD_SETTINGS_SENTINEL', 'MAIN_CHARACTERS_SENTINEL', None)
    result = pipeline.generate(state['revision'])
    assert counts(model)['direct'] == 1 and counts(model)['story_plan_review'] == 0
    for stage, prompt, images, _ in model.calls:
        assert not any(v in prompt for v in ('OLD_COPIED_JSON_SENTINEL', 'OLD_SAME_CHAPTER_SENTINEL'))
        for value in ('WORLD_SETTINGS_SENTINEL', 'MAIN_CHARACTERS_SENTINEL'):
            assert prompt.count(value) == (1 if stage == 'direct' else 0)
        assert not images
    plan = result['plan']
    assert plan['settings_basis'] == {'revision': notes['revision'],
        'hash': digest({k: notes[k] for k in ('background', 'characters')})}
    assert result['bible']['analysis_events'] == original['bible']['events']
    assert result['bible']['characters'] == original['bible']['characters']
    assert result['bible']['prior_context'] == original['bible']['prior_context']
    assert store.read(state['revision']) == original


@pytest.mark.parametrize('role', [None, 'analysis_reference'])
def test_unmarked_old_notes_and_explicit_analysis_copy_are_not_guessed_as_settings(story, role):
    store, state, model, pipeline = story
    notes = {'background': 'OLD_ROOT_REFERENCE_SENTINEL', 'characters': 'OLD_ROOT_CHARACTER_SENTINEL'}
    if role:
        notes['input_role'] = role
    StoryBible(store.root.parent/'series_context').save(notes, 'fixture_legacy_reference', None)
    assert planning_settings(store) is None
    pipeline.generate(state['revision'])
    assert not any('OLD_ROOT_REFERENCE_SENTINEL' in prompt or 'OLD_ROOT_CHARACTER_SENTINEL' in prompt
                   for _, prompt, _, _ in model.calls)


@pytest.mark.parametrize('same_text', [False, True])
def test_settings_revision_invalidates_planning_without_reintegrating_or_overwriting_old_candidates(story, same_text):
    store, state, model, pipeline = story
    grounding_model(story)
    first_notes = save_series_notes(store, 'WORLD_VERSION_ONE', 'PEOPLE_VERSION_ONE', None)
    first = pipeline.generate(state['revision'])
    kept = copy.deepcopy(first)
    again = pipeline.generate(first['revision'])
    assert counts(model)['direct'] == counts(model)['event_integration'] == 1
    assert counts(model)['story_digest'] == 0 and counts(model)['write_story'] == 1
    old = {p: p.read_bytes() for folder in ('cache', 'revisions', 'evidence')
           for p in (store.root/folder).rglob('*') if p.is_file()}
    text = 'WORLD_VERSION_ONE' if same_text else 'WORLD_VERSION_TWO'
    second_notes = save_series_notes(store, text, 'PEOPLE_VERSION_ONE', first_notes['revision'])
    changed = pipeline.generate(again['revision'])
    assert counts(model)['event_integration'] == 1 and counts(model)['direct'] == 2
    assert counts(model)['story_digest'] == 0 and counts(model)['write_story'] == 2
    assert changed['plan']['settings_basis']['revision'] == second_notes['revision']
    assert changed['bible'] == kept['bible'] and store.read(first['revision']) == kept
    assert all(p.read_bytes() == data for p, data in old.items())


def test_settings_reader_has_no_filesystem_side_effects_when_no_settings_exist(story):
    store, _, _, _ = story
    before = {str(p): p.read_bytes() for p in store.root.parent.rglob('*') if p.is_file()}
    assert planning_settings(store) is None
    assert {str(p): p.read_bytes() for p in store.root.parent.rglob('*') if p.is_file()} == before
    assert not (store.root.parent/'series_context').exists()


def test_settings_are_project_scoped_and_empty_settings_can_disable_planning_input(story, tmp_path):
    store, _, _, _ = story
    notes = save_series_notes(store, 'BOOK_A_SETTINGS', 'BOOK_A_PEOPLE', None)
    assert planning_settings(store)['revision'] == notes['revision']
    other = StoryBible(tmp_path/'other_book'/'Chapter_00001')
    assert planning_settings(other) is None
    save_series_notes(store, '', '', notes['revision'])
    assert planning_settings(store) is None
    assert series_notes(store)['background'] == ''


def test_saved_setting_prose_has_no_added_reference_label_and_metadata_stays_out_of_prompt(story):
    store, state, model, pipeline = story
    notes = save_series_notes(store, '世界背景：迷雾与城市。', '主角设定：身份揭晓前不能直接绑定人物。', None)
    assert notes['input_role'] == 'series_settings'
    pipeline.generate(state['revision'])
    prompt = next(p for s, p, _, _ in model.calls if s == 'direct')
    assert notes['background'] in prompt and notes['characters'] in prompt
    assert 'input_role' not in prompt and '外部参考' not in prompt
    assert '不等于本话已经揭晓' in prompt and '不覆盖原分析' in prompt
    assert '上一话剧情与已固定版本的历史资料：' in prompt
