"""Draft edit decisions from storyboard beats, subject candidates and source motion."""

import base64
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .media import VideoFrames, black_edges, camera_matrix, probe, read_image, subject_cropped
from .timeline import CAMERA, FORMAT, number, rect, require, sample_curve, validate


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff', '.gif', '.avif'}
MOTIONS = ('hold', 'push_fast', 'push_slow', 'pull_back', 'pan_horizontal', 'pan_vertical',
           'rotate', 'close_up', 'center', 'shake')
PACE_SECONDS = {'fast': 1.6, 'normal': 2.8, 'slow': 4.5}


def thumbnail(image, limit=560):
    h, w = image.shape[:2]
    scale = min(1., limit / max(h, w))
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def analyze(frames, cascade):
    boxes, grayscale = [], []
    for image in frames:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = () if cascade is None else cascade.detectMultiScale(
            cv2.equalizeHist(gray), scaleFactor=1.08, minNeighbors=5, minSize=(24, 24))
        for x, y, width, height in faces:
            boxes.append((max(0, (x - width * .15) / w), max(0, (y - height * .15) / h),
                          min(1, (x + width * 1.15) / w), min(1, (y + height * 1.15) / h)))
        grayscale.append(cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA))
    subject = None
    if boxes:
        left, top, right, bottom = np.array(boxes)
        subject = [float(left.min()), float(top.min()), float(right.max() - left.min()), float(bottom.max() - top.min())]
    deltas = [cv2.absdiff(left, right).mean() / 255 for left, right in zip(grayscale, grayscale[1:])]
    edges = [max(side) for side in zip(*(black_edges(frame) for frame in frames))]
    detail = cv2.Canny(grayscale[len(grayscale) // 2], 60, 150)
    # Ignore frame edges and letterbox boundaries when estimating a visual focus.
    left, top, right, bottom = [max(5, math.ceil(band * 96) + 2) for band in edges]
    interior = detail[top:96 - bottom, left:96 - right]
    rows, cols = np.nonzero(interior)
    focus = [float((cols.mean() + left + .5) / 96), float((rows.mean() + top + .5) / 96)] if len(rows) >= 10 else [.5, .5]
    return {'subject': subject, 'focus': focus, 'motion_score': float(max(deltas, default=0)), 'black_edges': edges}


def camera_preset(motion, frames, focus=(.5, .5)):
    """Build editable poses in normalized source coordinates."""
    require(motion in MOTIONS and type(frames) is int and frames > 0, '无效的运镜预设或时长')
    x, y = focus
    focused = {'x': x, 'y': y}
    gentle = {'x': .5 + (x - .5) * .45, 'y': .5 + (y - .5) * .45}
    presets = {
        'hold': ({}, {}),
        'push_fast': (gentle, {**focused, 'zoom': 1.15}),
        'push_slow': (gentle, {**gentle, 'zoom': 1.08}),
        'pull_back': ({**gentle, 'zoom': 1.10}, {}),
        'pan_horizontal': ({'x': .46, 'zoom': 1.10}, {'x': .54, 'zoom': 1.10}),
        'pan_vertical': ({'y': .46, 'zoom': 1.10}, {'y': .54, 'zoom': 1.10}),
        'rotate': ({'rotation': -2}, {'rotation': 2}),
        'close_up': ({**gentle, 'zoom': 1.05}, {**focused, 'zoom': 1.20}),
        'center': ({}, {**focused, 'zoom': 1.08}),
        'shake': ({'zoom': 1.08}, {'zoom': 1.08}),
    }
    start, end = presets[motion]
    points = [{'frame': 0, **CAMERA, **start}]
    if frames > 1:
        last = frames - 1
        if motion == 'push_fast':
            last = max(1, last // 2)
        points.append({'frame': last, **CAMERA, **end, 'ease': 'ease_out' if motion == 'push_fast' else 'ease_in_out'})
    return points, .008 if motion == 'shake' else 0.


def _safe_camera(motion, frames, subject, focus, source, target):
    if subject is None:
        aspect_ratio = (source[0] / source[1]) / (target[0] / target[1])
        fit = 'contain' if max(aspect_ratio, 1 / aspect_ratio) > 1.6 else 'cover'
        return motion, fit, *camera_preset(motion, frames, focus)
    x, y, w, h = subject
    focus = x + w / 2, y + h / 2
    for candidate, fit in ((motion, 'cover'), ('hold', 'cover'), (motion, 'contain'), ('hold', 'contain')):
        camera, shake = camera_preset(candidate, frames, focus)
        for i, pose in enumerate(sample_curve(camera, frames, CAMERA)):
            matrix, _ = camera_matrix(source, target, pose, fit, i, shake)
            if subject_cropped(subject, source, (0, 0, *source), matrix, target):
                break
        else:
            return candidate, fit, camera, shake
    return 'hold', 'contain', *camera_preset('hold', frames)


def _pace(text):
    text = text.lower()
    if any(word in text for word in ('追逐', '战斗', '急促', '爆炸', '冲向', 'chase', 'fight', 'explosion')):
        return 'fast'
    if any(word in text for word in ('回忆', '停顿', '沉默', '凝视', 'flashback', 'pause', 'silence')):
        return 'slow'
    return 'normal'


def _local_motion(text, pace, analysis, source):
    if analysis['motion_score'] >= .12:
        return 'hold'
    if analysis['subject'] and any(word in text.lower() for word in ('特写', '表情', 'close-up', 'reaction')):
        return 'close_up'
    if pace == 'fast':
        return 'push_fast'
    if pace == 'slow':
        return 'pull_back'
    if source[1] > source[0] * 1.7:
        return 'pan_vertical'
    if source[0] > source[1] * 1.9:
        return 'pan_horizontal'
    return 'push_slow'


def model_advisor(model):
    """Opt into the configured Ollama/OpenAI-compatible vision endpoint."""
    from pipeline.llm_transport import build_transport

    transport = build_transport()

    def advise(context, images):
        encoded_images = []
        for image in images:
            ok, encoded = cv2.imencode('.jpg', image)
            require(ok, '无法编码送给视觉模型的画面')
            encoded_images.append(base64.b64encode(encoded).decode('ascii'))
        prompt = ('根据分镜、画面、节奏和上一镜头选择保守的剪辑手法。只输出一个 JSON 对象，'
                  '字段 pace 为 fast/normal/slow，motion 为 ' + '/'.join(MOTIONS) +
                  '，transition 为 cut/dissolve/black，仅在回忆、段落切换等情节变化时推荐非硬切。'
                  'subject 为覆盖所有采样帧中同一主体的归一化框 [左,上,宽,高]，无法判断时为 null。'
                  '输入数据：' + json.dumps(context, ensure_ascii=False))
        response = transport.chat(model=model, prompt=prompt, images=encoded_images, temperature=.1, max_tokens=220)
        response = response.strip()
        if response.startswith('```') and response.endswith('```'):
            response = response.split('\n', 1)[-1][:-3].strip()
        return json.loads(response)

    return advise


def plan(shots, *, base='.', width=1920, height=1080, fps=30, advisor=None):
    """Turn ordered storyboard shots into an editable Timeline draft."""
    require(isinstance(shots, list) and shots, '分镜应为非空数组')
    require(all(type(v) is int and 2 <= v <= 4096 and v % 2 == 0 for v in (width, height)) and
            type(fps) is int and 1 <= fps <= 120, '无效画布尺寸或帧率')
    classifier = getattr(cv2, 'CascadeClassifier', None)
    cascade = classifier(str(Path(__file__).resolve().parents[1] / 'lbpcascade_animeface.xml')) if classifier else None
    if cascade is not None and cascade.empty():
        cascade = None
    assets, clips, decisions = {}, [], []
    cursor, prior_motion, prior_transition = 0, None, 0
    for index, shot in enumerate(shots, 1):
        require(isinstance(shot, dict) and isinstance(shot.get('path'), str) and shot['path'].strip(), '分镜缺少素材路径')
        path = (Path(base) / shot['path']).resolve()
        require(path.is_file(), f'素材不存在：{path}')
        kind = shot.get('kind', 'image' if path.suffix.lower() in IMAGE_SUFFIXES else 'video')
        text, pace = shot.get('text', ''), shot.get('pace')
        seconds = shot.get('duration')
        require(kind in ('image', 'video') and isinstance(text, str) and
                (pace is None or isinstance(pace, str) and pace in PACE_SECONDS) and
                (seconds is None or number(seconds, 1 / fps, 3600)) and
                ('subject' not in shot or rect(shot['subject'])) and
                type(shot.get('scene_change', False)) is bool, '分镜字段无效')
        in_seconds = shot.get('source_in', 0)
        require(number(in_seconds, 0, 3600 * 24) and (kind == 'video' or in_seconds == 0), '无效的分镜素材入点')
        source_in = round(in_seconds * fps)
        limit = 432000
        if kind == 'image':
            image = read_image(path)
            source = image.shape[1::-1]
            samples = [thumbnail(image)]
        else:
            info = probe(path)
            limit = math.floor(info['seconds'] * fps + 1e-6) - source_in
            require(info['video'] and limit > 0, f'无法读取视频或入点超出范围：{path}')
            samples, source = [], None
            sample_end = min(limit - 1, max(0, round((seconds or 4.5) * fps) - 1))
            for at in sorted({0, sample_end // 2, sample_end}):
                decoder = VideoFrames(path, fps, source_in + at)
                try:
                    frame = decoder.at(0)
                    source = frame.shape[1::-1]
                    samples.append(thumbnail(frame))
                finally:
                    decoder.close()
        analysis = analyze(samples, cascade)
        subject_source = 'face_cascade' if analysis['subject'] is not None else None
        if 'subject' in shot:
            analysis['subject'] = shot['subject']
            subject_source = 'storyboard'
        pace = pace or _pace(text)
        motion = _local_motion(text, pace, analysis, source)
        scene_change = shot.get('scene_change', False) or any(word in text.lower() for word in ('回忆', 'flashback'))
        transition = ('black' if pace == 'slow' else 'dissolve') if scene_change else 'cut'
        model_error = None
        if advisor is not None:
            try:
                advice = advisor({'text': text, 'pace': pace, 'scene_change': scene_change,
                                  'previous_motion': prior_motion, **analysis}, samples)
                require(isinstance(advice, dict) and isinstance(advice.get('pace'), str) and advice['pace'] in PACE_SECONDS and
                        advice.get('motion') in MOTIONS and advice.get('transition') in ('cut', 'black', 'dissolve'),
                        '视觉模型剪辑建议格式无效')
                pace = shot.get('pace') or advice['pace']
                motion = advice['motion']
                if scene_change:
                    transition = advice['transition']
                if analysis['subject'] is None and rect(advice.get('subject')):
                    analysis['subject'] = advice['subject']
                    subject_source = 'vision_model'
            except Exception as error:
                model_error = type(error).__name__
        if analysis['motion_score'] >= .12 or motion == prior_motion or (motion == 'close_up' and not analysis['subject']):
            motion = 'hold'
        frames = min(limit, max(1, round((seconds or PACE_SECONDS[pace]) * fps)))
        motion, fit, camera, shake = _safe_camera(motion, frames, analysis['subject'], analysis['focus'], source, (width, height))
        overlap = min(round(fps * .28), frames // 4, clips[-1]['duration'] // 4) if clips else 0
        if transition == 'cut' or prior_transition or overlap < (3 if transition == 'black' else 2):
            transition, overlap = 'cut', 0
        name = f'shot-{index}'
        assets[name] = {'kind': kind, 'path': str(path)}
        clip = {'id': name, 'asset': name, 'start': cursor - overlap, 'duration': frames, 'source_in': source_in,
                'camera': camera, 'fit': fit, 'shake': shake, 'transition': {'type': transition, 'frames': overlap}}
        if analysis['subject'] is not None:
            clip['subject'] = analysis['subject']
        clips.append(clip)
        decisions.append({'clip': name, 'text': text, 'pace': pace, 'motion': motion, 'transition': transition,
                          'advisor_error': model_error, 'subject_source': subject_source, **analysis})
        cursor += frames - overlap
        prior_motion, prior_transition = motion, overlap
    timeline = {'format': FORMAT, 'version': 1, 'canvas': {'width': width, 'height': height, 'fps': fps},
                'assets': assets, 'tracks': [{'kind': 'video', 'clips': clips}], 'decisions': decisions}
    validate(timeline)
    return timeline
