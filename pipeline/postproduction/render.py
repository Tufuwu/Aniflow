"""Stream a Timeline through OpenCV and encode a separate MP4 with FFmpeg."""

from contextlib import ExitStack, contextmanager
import math
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import wave

import cv2
from filelock import FileLock
import imageio_ffmpeg
import numpy as np

from .media import (VideoFrames, black_edges, camera_matrix, ffmpeg, pixel_rect, probe, read_image,
                    subject_cropped, transform)
from .timeline import CAMERA, asset_paths, require, sample_curve, validate


SAMPLE_RATE = 48000
FULL = [0, 0, 1, 1]


def transition_frame(previous, current, kind, progress):
    if kind == 'dissolve':
        return cv2.addWeighted(previous, 1 - progress, current, progress, 0)
    if kind == 'black':
        weight = abs(2 * progress - 1)
        return cv2.convertScaleAbs(previous if progress < .5 else current, alpha=weight)
    if kind == 'zoom':
        height, width = previous.shape[:2]
        def zoom(frame, factor):
            matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), 0, factor)
            return cv2.warpAffine(frame, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
        return cv2.addWeighted(zoom(previous, 1 + .25 * progress), 1 - progress,
                               zoom(current, 1 + .25 * (1 - progress)), progress, 0)
    width = previous.shape[1]
    split = round(width * progress)
    result = previous.copy()
    if split and kind == 'slide':
        result[:, :width - split] = previous[:, split:]
        result[:, width - split:] = current[:, :split]
    elif kind == 'mask':
        # A horizontal wipe reveals a stationary incoming frame.
        result[:, :split] = current[:, :split]
    return result


class ClipReader:
    def __init__(self, clip, asset, path, fps, report):
        self.clip = clip
        self.report = report
        self.image = self.video = None
        self.position = 0.
        self.fps = fps
        self.camera = sample_curve(clip.get('camera', []), clip['duration'], CAMERA)
        self.speed = sample_curve(clip.get('speed', []), clip['duration'], {'rate': 1.})
        self.kind, self.path = asset['kind'], path

    def picture(self, local, target):
        if self.kind == 'image':
            if self.image is None:
                self.image = read_image(self.path)
            source = self.image
        else:
            if self.video is None:
                self.video = VideoFrames(self.path, self.fps, self.clip.get('source_in', 0))
            source = self.video.at(math.floor(self.position + 1e-8))
        h, w = source.shape[:2]
        crop = pixel_rect(self.clip.get('crop', FULL), w, h)
        left, top, right, bottom = crop
        frame = source[top:bottom, left:right]
        fit = self.clip.get('fit', 'cover')
        matrix, clamped = camera_matrix(frame.shape[1::-1], target, next(self.camera), fit, local,
                                        self.clip.get('shake', 0))
        if clamped:
            self.report['camera_clamped_frames'] += 1
        if subject_cropped(self.clip.get('subject'), (w, h), crop, matrix, target):
            self.report['subject_cropped_frames'] += 1
        rendered = transform(frame, target, matrix, fit)
        last = self.clip['duration'] - 1
        if local % max(1, self.fps // 2) == 0 or local == last:
            key = 'rendered_black_edges'
            self.report[key] = [max(a, b) for a, b in zip(self.report[key], black_edges(rendered))]
            if local == 0 or self.kind == 'video':
                key = 'source_black_edges'
                self.report[key] = [max(a, b) for a, b in zip(self.report[key], black_edges(source))]
        return rendered

    def close(self):
        if self.video is not None:
            self.video.close()
            self.video = None
        self.image = None


def timeline_size(canvas, preview):
    width, height = canvas['width'], canvas['height']
    if preview:
        scale = min(1., 640 / max(width, height))
        return max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2)
    return width, height


@contextmanager
def encoder(path, size, fps, preview):
    with TemporaryDirectory(dir=path.parent) as logs:
        error_path = Path(logs) / 'ffmpeg.txt'
        with error_path.open('w+b') as errors:
            command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-nostdin',
                       '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', f'{size[0]}x{size[1]}',
                       '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '4',
                       '-preset', 'ultrafast' if preview else 'medium', '-crf', '28' if preview else '18',
                       '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
            with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors) as process:
                try:
                    yield process.stdin
                    process.stdin.close()
                    code = process.wait(timeout=120)
                except BrokenPipeError as error:
                    code = process.wait(timeout=120)
                    if code == 0:
                        raise error
                finally:
                    if process.poll() is None:
                        process.kill()
                errors.seek(0)
                require(code == 0 and path.is_file(), '编码失败：' + errors.read().decode(errors='replace')[-2000:])


def active_clips(track, readers):
    clips = iter(track['clips'])
    upcoming = next(clips, None)
    active = []
    frame = yield
    while True:
        for clip in active:
            if frame >= clip['start'] + clip['duration']:
                readers[clip['id']].close()
        active = [clip for clip in active if frame < clip['start'] + clip['duration']]
        while upcoming and upcoming['start'] <= frame:
            active.append(upcoming)
            upcoming = next(clips, None)
        frame = yield active


def draw_track(canvas, active, readers, frame):
    x, y, right, bottom = pixel_rect(active[-1].get('rect', FULL), canvas.shape[1], canvas.shape[0])
    target = right - x, bottom - y
    current = active[-1]
    picture = readers[current['id']].picture(frame - current['start'], target)
    opacity = current.get('opacity', 1)
    progress = 1.
    if len(active) == 2:
        previous = active[0]
        old = readers[previous['id']].picture(frame - previous['start'], target)
        transition = current['transition']
        progress = (frame - current['start']) / (transition['frames'] - 1)
        visual_progress = progress
        if transition['type'] == 'black':
            offset = frame - current['start']
            middle = transition['frames'] // 2
            visual_progress = (offset / middle / 2 if offset <= middle else
                               .5 + (offset - middle) / (transition['frames'] - 1 - middle) / 2)
        picture = transition_frame(old, picture, transition['type'], visual_progress)
        opacity = previous.get('opacity', 1) * (1 - progress) + opacity * progress
    region = canvas[y:bottom, x:right]
    region[:] = picture if opacity == 1 else cv2.addWeighted(region, 1 - opacity, picture, opacity, 0)
    return progress


class AudioMixer:
    def __init__(self, timeline, paths, directory):
        self.fps = timeline['canvas']['fps']
        self.sources = {}
        self.report = set()
        for track in timeline['tracks']:
            for clip in track['clips']:
                name = clip['asset']
                if name in self.sources or timeline['assets'][name]['kind'] == 'image':
                    continue
                path = paths[name]
                if not probe(path)['audio']:
                    require(track['kind'] != 'audio', f'音轨素材没有音频流：{path}')
                    self.sources[name] = None
                    continue
                pcm = directory / f'audio-{len(self.sources)}.pcm'
                ffmpeg(['-v', 'error', '-i', str(path), '-map', '0:a:0', '-ac', '2', '-ar', str(SAMPLE_RATE),
                        '-f', 's16le', str(pcm)])
                self.sources[name] = (pcm, pcm.stat().st_size // 4)
        self.enabled = any(self.sources.values())

    def mix(self, mixed, clip, reader, rate, weight, *, audio_track):
        source = self.sources.get(clip['asset'])
        if source is None:
            require(not audio_track, '音轨素材没有音频流')
            return
        path, count = source
        offset = (clip.get('source_in', 0) + reader.position) * SAMPLE_RATE / self.fps
        samples = offset + np.arange(len(mixed)) * rate * (SAMPLE_RATE / self.fps) / len(mixed)
        first = int(offset)
        end = min(count, int(samples[-1]) + 2)
        if samples[-1] >= count + 1:
            self.report.add(clip['id'])
        if first >= end or not clip.get('volume', 1) or not weight:
            return
        with path.open('rb') as stream:
            stream.seek(first * 4)
            values = np.frombuffer(stream.read((end - first) * 4), dtype='<i2').reshape(-1, 2)
        times = np.arange(first, end)
        for channel in range(2):
            mixed[:, channel] += np.interp(samples, times, values[:, channel], right=0) * weight * clip.get('volume', 1)


def render(timeline, output, *, base='.', preview=False, progress=None):
    """Render to a new path. Return dimensions, duration and per-clip visual diagnostics."""
    total = validate(timeline)
    paths = asset_paths(timeline, base)
    output = Path(output).resolve()
    require(output.suffix.lower() == '.mp4' and output not in paths.values(), '输出必须是独立的 MP4 文件')
    output.parent.mkdir(parents=True, exist_ok=True)
    size = timeline_size(timeline['canvas'], preview)
    fps = timeline['canvas']['fps']
    reports = {}
    readers = {}
    for track in timeline['tracks']:
        for clip in track['clips']:
            name = clip['asset']
            report = {'camera_clamped_frames': 0, 'subject_cropped_frames': 0, 'subject_known': 'subject' in clip,
                      'source_black_edges': [0.] * 4, 'rendered_black_edges': [0.] * 4}
            if track['kind'] == 'video':
                reports[clip['id']] = report
            readers[clip['id']] = ClipReader(clip, timeline['assets'][name], paths[name], fps, report)
    with FileLock(str(output) + '.lock'):
        require(not output.exists(), f'输出文件已存在：{output}')
        with TemporaryDirectory(dir=output.parent, prefix='.postproduction-') as temporary, ExitStack() as resources:
            work = Path(temporary)
            video, audio, final = work / 'video.mp4', work / 'audio.wav', work / 'final.mp4'
            mixer = AudioMixer(timeline, paths, work)
            wav = None
            if mixer.enabled:
                wav = resources.enter_context(wave.open(str(audio), 'wb'))
                wav.setparams((2, 2, SAMPLE_RATE, 0, 'NONE', 'not compressed'))
            for reader in readers.values():
                resources.callback(reader.close)
            tracks = [(track['kind'], active_clips(track, readers)) for track in timeline['tracks']]
            for _, clock in tracks:
                next(clock)
            with encoder(video, size, fps, preview) as sink:
                for frame in range(total):
                    canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                    count = round((frame + 1) * SAMPLE_RATE / fps) - round(frame * SAMPLE_RATE / fps)
                    mixed = np.zeros((count, 2), dtype=np.float32) if wav else None
                    for kind, clock in tracks:
                        active = clock.send(frame)
                        if not active:
                            continue
                        progress_in = draw_track(canvas, active, readers, frame) if kind == 'video' else 1.
                        weights = (1 - progress_in, progress_in) if len(active) == 2 else (1.,)
                        for clip, weight in zip(active, weights):
                            reader = readers[clip['id']]
                            rate = next(reader.speed)['rate']
                            if mixed is not None:
                                mixer.mix(mixed, clip, reader, rate, weight, audio_track=kind == 'audio')
                            reader.position += rate
                    sink.write(canvas.tobytes())
                    if wav:
                        wav.writeframesraw(np.clip(mixed, -32768, 32767).astype('<i2').tobytes())
                    if progress:
                        progress(frame + 1, total)
            if wav:
                wav.close()
                ffmpeg(['-v', 'error', '-i', str(video), '-i', str(audio), '-map', '0:v:0', '-map', '1:a:0',
                        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', str(final)])
            finished = final if wav else video
            # Publish only after encoding, and fail if another producer claimed the path.
            if os.name == 'nt':
                finished.rename(output)
            else:
                os.link(finished, output)
    return {'path': str(output), 'width': size[0], 'height': size[1], 'fps': fps, 'frames': total,
            'audio': mixer.enabled, 'short_audio_clips': sorted(mixer.report), 'checks': reports}
