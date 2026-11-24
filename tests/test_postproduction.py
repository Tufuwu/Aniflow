from copy import deepcopy
import hashlib

import cv2
import numpy as np
import pytest
from PIL import Image

from pipeline.postproduction.director import camera_preset, plan
from pipeline.postproduction.media import VideoFrames, ffmpeg, probe
from pipeline.postproduction.render import render, transition_frame
from pipeline.postproduction.timeline import EditError, load_project, sample_curve, save_project, validate


@pytest.fixture
def media(tmp_path):
    image = tmp_path / 'wide.png'
    pixels = np.full((96, 192, 3), (170, 80, 40), dtype=np.uint8)
    pixels[:10] = 0
    pixels[-10:] = 0
    Image.fromarray(pixels).save(image)
    video = tmp_path / 'clip.mp4'
    ffmpeg(['-v', 'error', '-f', 'lavfi', '-i', 'color=blue:size=180x100:rate=12:duration=2',
            '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=2', '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(video)])
    return image, video


@pytest.mark.parametrize(('ease', 'expected'), [('linear', .5), ('ease_in', .25), ('ease_out', .75), ('ease_in_out', .5)])
def test_interpolation(ease, expected):
    values = list(sample_curve([{'frame': 0, 'rate': 1}, {'frame': 4, 'rate': 2, 'ease': ease},
                                {'frame': 6, 'rate': 1}], 8, {'rate': 1}))
    assert [values[i]['rate'] for i in (0, 2, 4, 5, 7)] == [1, 1 + expected, 2, 1.5, 1]


@pytest.mark.parametrize('kind', ['dissolve', 'black', 'slide', 'mask', 'zoom'])
def test_transition_endpoints_and_middle(kind):
    old = np.full((6, 8, 3), 220, np.uint8)
    new = np.full_like(old, 60)
    assert np.array_equal(transition_frame(old, new, kind, 0), old)
    assert np.array_equal(transition_frame(old, new, kind, 1), new)
    middle = transition_frame(old, new, kind, .5)
    assert not np.array_equal(middle, old) and not np.array_equal(middle, new)
    if kind == 'black':
        assert not middle.any()


def test_draft_safety_and_failed_advisor(media):
    image, video = media
    def offline(*_):
        raise OSError('offline')
    draft = plan([{'path': str(image), 'text': '回忆', 'subject': [0, .1, .9, .8]},
                  {'path': str(video), 'text': '追逐', 'scene_change': True, 'source_in': .25},
                  {'path': str(image), 'text': '追逐'}], width=100, height=180, fps=12, advisor=offline)
    first, second, third = draft['tracks'][0]['clips']
    assert first['fit'] == 'contain'
    assert second['transition']['type'] == 'dissolve' and second['source_in'] == 3
    assert third['transition']['type'] == 'cut'
    assert third['fit'] == 'contain'
    assert draft['decisions'][1]['motion'] != draft['decisions'][2]['motion']
    assert draft['decisions'][0]['advisor_error'] == 'OSError'
    assert draft['decisions'][0]['black_edges'][1] > 0
    invalid = deepcopy(draft)
    invalid['tracks'][0]['clips'][1]['start'] += 1
    with pytest.raises(EditError, match='重叠'):
        validate(invalid)


def test_model_advice_is_subject_checked_and_scene_gated(media):
    image, video = media
    samples = []

    def advise(context, frames):
        samples.append((context['scene_change'], len(frames)))
        return {'pace': 'slow', 'motion': 'close_up', 'transition': 'black', 'subject': [0, .2, .9, .6]}

    timeline = plan([{'path': str(image), 'duration': 1}, {'path': str(video), 'duration': 1}],
                    width=100, height=180, fps=12, advisor=advise)
    assert samples == [(False, 1), (False, 3)]
    assert all(d['transition'] == 'cut' and d['subject_source'] == 'vision_model' for d in timeline['decisions'])
    assert all(c['fit'] == 'contain' for c in timeline['tracks'][0]['clips'])


def test_reopen_edit_and_render_media(media, tmp_path):
    image, video = media
    original = [hashlib.sha256(path.read_bytes()).hexdigest() for path in media]
    timeline = plan([{'path': str(image), 'duration': 1}, {'path': str(video), 'duration': 1,
                                                        'scene_change': True}], width=720, height=480, fps=12)
    first, second = timeline['tracks'][0]['clips']
    second['source_in'] = 3
    second['speed'] = [{'frame': 0, 'rate': .5}, {'frame': 11, 'rate': 1.5, 'ease': 'ease_in'}]
    first['camera'] = camera_preset('rotate', first['duration'])[0]
    first['camera'].insert(1, {'frame': 4, 'x': .01, 'y': .5, 'zoom': 1, 'rotation': -2, 'ease': 'ease_out'})
    first['subject'] = [.85, .25, .14, .5]
    timeline['tracks'].extend([
        {'kind': 'video', 'clips': [{'id': 'pip', 'asset': first['asset'], 'start': 0, 'duration': 3,
                                     'rect': [.5, 0, .5, .5]}]},
        {'kind': 'audio', 'clips': [{'id': 'music', 'asset': second['asset'], 'start': 0, 'duration': 8,
                                     'volume': .2}]},
    ])
    project = save_project(timeline, tmp_path / 'edit.json')
    timeline = load_project(project)
    assert save_project(timeline, project, replace=True) == project
    with pytest.raises(EditError):
        render(timeline, video)
    preview = render(timeline, tmp_path / 'preview.mp4', preview=True)
    final = render(timeline, tmp_path / 'final.mp4')
    assert (preview['width'], final['width'], final['height'], final['fps'], final['frames']) == (640, 720, 480, 12, 21)
    assert final['checks'][first['id']]['camera_clamped_frames'] > 0
    assert final['checks'][first['id']]['subject_cropped_frames'] > 0
    assert probe(final['path'])['audio']
    reader = VideoFrames(final['path'], 12, 0)
    try:
        first_image = reader.at(0)
        assert first_image.shape == (480, 720, 3)
        assert np.mean(cv2.absdiff(first_image[:240, :360], first_image[:240, 360:])) > 3
        assert reader.at(20)[240, 360, 0] > 180
        with pytest.raises(EditError, match='末尾'):
            reader.at(21)
    finally:
        reader.close()
    assert original == [hashlib.sha256(path.read_bytes()).hexdigest() for path in media]


def test_failed_render_leaves_no_output(media, tmp_path):
    _, video = media
    timeline = plan([{'path': str(video), 'duration': .5}], width=120, height=80, fps=12)
    timeline['tracks'][0]['clips'][0]['duration'] = 25
    output = tmp_path / 'incomplete.mp4'
    with pytest.raises(EditError, match='末尾'):
        render(timeline, output)
    assert not output.exists()
    assert not list(tmp_path.glob('.postproduction-*'))
