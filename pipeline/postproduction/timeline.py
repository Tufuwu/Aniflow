"""Versioned edit decisions and non-destructive project storage."""

from bisect import bisect_right
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

from filelock import FileLock


FORMAT = 'aniflow.postproduction'
EASING = {
    'linear': lambda t: t,
    'ease_in': lambda t: t * t,
    'ease_out': lambda t: 1 - (1 - t) ** 2,
    'ease_in_out': lambda t: t * t * (3 - 2 * t),
}
TRANSITIONS = {'cut', 'dissolve', 'black', 'slide', 'mask', 'zoom'}
CAMERA = {'x': .5, 'y': .5, 'zoom': 1., 'rotation': 0.}


class EditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EditError(message)


def number(value, low, high):
    return type(value) in (int, float) and low <= value <= high


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def rect(value):
    if not isinstance(value, list) or len(value) != 4 or not all(number(v, 0, 1) for v in value):
        return False
    x, y, w, h = value
    return w > 0 and h > 0 and x + w <= 1.000000001 and y + h <= 1.000000001


def keyframes(points, duration, fields):
    require(isinstance(points, list), '关键帧必须是数组')
    previous = -1
    for point in points:
        require(isinstance(point, dict) and integer(point.get('frame'), previous + 1, duration - 1),
                '关键帧必须按镜头内帧号递增')
        require(isinstance(point.get('ease', 'linear'), str) and point.get('ease', 'linear') in EASING,
                '未知运动曲线')
        require(all(number(point.get(key), *bounds) for key, bounds in fields.items()), '关键帧参数超出范围')
        previous = point['frame']
    require(not points or points[0]['frame'] == 0, '首个关键帧必须位于第 0 帧')


def validate(timeline):
    require(isinstance(timeline, dict) and timeline.get('format') == FORMAT and type(timeline.get('version')) is int
            and timeline['version'] == 1, '无效或不受支持的剪辑工程版本')
    canvas = timeline.get('canvas')
    require(isinstance(canvas, dict) and all(integer(canvas.get(k), 2, 4096) and canvas[k] % 2 == 0
               for k in ('width', 'height')) and integer(canvas.get('fps'), 1, 120), '画布须为偶数尺寸且帧率为整数')
    assets = timeline.get('assets')
    require(isinstance(assets, dict) and assets, '工程缺少素材')
    for name, asset in assets.items():
        require(isinstance(name, str) and name and isinstance(asset, dict) and
                asset.get('kind') in ('image', 'video', 'audio') and isinstance(asset.get('path'), str) and
                asset['path'].strip() and '\x00' not in asset['path'], f'素材 {name} 无效')
    tracks = timeline.get('tracks')
    require(isinstance(tracks, list) and tracks, '工程缺少轨道')
    ids, total, has_video = set(), 0, False
    for track in tracks:
        require(isinstance(track, dict) and track.get('kind') in ('video', 'audio') and
                isinstance(track.get('clips'), list) and track['clips'], '无效或为空的轨道')
        visual = track['kind'] == 'video'
        has_video |= visual
        previous_end = incoming_overlap = 0
        for index, clip in enumerate(track['clips']):
            require(isinstance(clip, dict) and isinstance(clip.get('id'), str) and clip['id'] and
                    clip['id'] not in ids and isinstance(clip.get('asset'), str) and clip['asset'] in assets,
                    '镜头 ID 重复或引用了未知素材')
            ids.add(clip['id'])
            kind = assets[clip['asset']]['kind']
            require(kind in (('image', 'video') if visual else ('audio', 'video')), '素材类型与轨道不匹配')
            start, duration = clip.get('start'), clip.get('duration')
            require(integer(start, 0, 432000) and integer(duration, 1, 432000 - start), '无效的镜头时长或起点')
            require(integer(clip.get('source_in', 0), 0, 100000000) and number(clip.get('volume', 1), 0, 2),
                    '无效的素材入点或音量')
            require(kind != 'image' or (not clip.get('source_in') and not clip.get('speed')), '静态图片不接受入点或速度')
            keyframes(clip.get('speed', []), duration, {'rate': (.125, 8)})
            transition = clip.get('transition', {'type': 'cut', 'frames': 0})
            require(isinstance(transition, dict) and isinstance(transition.get('type'), str) and
                    transition['type'] in TRANSITIONS, '未知转场')
            overlap = transition.get('frames')
            if transition['type'] == 'cut':
                require(integer(overlap, 0, 0) and start >= previous_end, '硬切不得重叠')
            else:
                minimum = 3 if transition['type'] == 'black' else 2
                require(visual and index and integer(overlap, minimum, duration - 1) and start == previous_end - overlap and
                        start >= previous_end - track['clips'][index - 1]['duration'] + incoming_overlap + 1,
                        '转场帧数必须等于相邻镜头的重叠，且每个镜头至少保留一帧独立画面')
                require(clip.get('rect', [0, 0, 1, 1]) == track['clips'][index - 1].get('rect', [0, 0, 1, 1]),
                        '转场两侧镜头必须使用相同的画面区域')
            if visual:
                require(rect(clip.get('rect', [0, 0, 1, 1])) and rect(clip.get('crop', [0, 0, 1, 1])) and
                        ('subject' not in clip or rect(clip['subject'])), '画面区域、裁切或主体框超出边界')
                require(clip.get('fit', 'cover') in ('cover', 'contain') and
                        number(clip.get('shake', 0), 0, .05) and number(clip.get('opacity', 1), 0, 1),
                        '无效的画面适配、抖动或透明度')
                keyframes(clip.get('camera', []), duration, {'x': (0, 1), 'y': (0, 1), 'zoom': (1, 4),
                                                           'rotation': (-360, 360)})
            else:
                require(not {'rect', 'crop', 'subject', 'fit', 'shake', 'opacity', 'camera'}.intersection(clip),
                        '音轨不接受画面参数')
            previous_end, incoming_overlap = start + duration, overlap
            total = max(total, previous_end)
    require(has_video, '至少需要一条视频轨')
    return total


def sample_curve(points, frames, defaults):
    """Yield values for each output frame. Ease belongs to the arriving keyframe."""
    if not points:
        for _ in range(frames):
            yield defaults
        return
    times = [p['frame'] for p in points]
    for frame in range(frames):
        i = bisect_right(times, frame) - 1
        left = points[i]
        if i == len(points) - 1:
            yield {key: left[key] for key in defaults}
            continue
        right = points[i + 1]
        t = EASING[right.get('ease', 'linear')]((frame - times[i]) / (times[i + 1] - times[i]))
        yield {key: left[key] + (right[key] - left[key]) * t for key in defaults}


def asset_paths(timeline, base):
    paths = {name: (Path(base) / asset['path']).resolve() for name, asset in timeline['assets'].items()}
    for name, path in paths.items():
        require(path.is_file(), f'素材 {name} 不存在：{path}')
    return paths


def load_project(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        timeline = json.load(stream)
    validate(timeline)
    return timeline


def save_project(timeline, path, *, replace=False):
    validate(timeline)
    path = Path(path).resolve()
    require(path.suffix.lower() == '.json' and path not in asset_paths(timeline, path.parent).values(),
            '工程须保存为独立的 JSON 文件')
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + '.lock'):
        if path.exists():
            require(replace, f'工程文件已存在：{path}')
            load_project(path)
        temporary = None
        try:
            with NamedTemporaryFile('w', dir=path.parent, delete=False, encoding='utf-8') as stream:
                temporary = Path(stream.name)
                json.dump(timeline, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return path
