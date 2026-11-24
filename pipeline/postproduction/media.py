"""Read local media and calculate camera geometry in source coordinates."""

import math
import re
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageOps

from .timeline import require


def ffmpeg(args, *, check=True, timeout=1800):
    result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-nostdin', *args],
                            capture_output=True, timeout=timeout)
    if check:
        require(result.returncode == 0, 'FFmpeg 处理失败：' + result.stderr.decode(errors='replace')[-2000:])
    return result


def probe(path):
    output = ffmpeg(['-i', str(path)], check=False, timeout=30).stderr.decode(errors='replace')
    duration = re.search(r'Duration: (\d+):(\d+):([\d.]+)', output)
    seconds = (int(duration[1]) * 3600 + int(duration[2]) * 60 + float(duration[3])) if duration else 0
    return {'seconds': seconds, 'audio': bool(re.search(r'^\s*Stream #.*: Audio:', output, re.M)),
            'video': bool(re.search(r'^\s*Stream #.*: Video:', output, re.M))}


def read_image(path):
    with Image.open(path) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def video_reader(path, fps, start=0):
    """Return constant-rate, square-pixel, display-oriented BGR frames."""
    filters = f'fps={fps}:start_time=0,scale=trunc(iw*sar+0.5):ih,setsar=1'
    options = ['-ss', f'{start / fps:.9f}'] if start else []
    return imageio_ffmpeg.read_frames(str(path), pix_fmt='bgr24', input_params=options,
                                    output_params=['-map', '0:v:0', '-vf', filters])


class VideoFrames:
    def __init__(self, path, fps, start):
        self.reader = video_reader(path, fps, start)
        try:
            width, height = next(self.reader)['size']
        except Exception:
            self.reader.close()
            raise
        self.shape = (height, width, 3)
        self.index = -1
        self.frame = None

    def at(self, index):
        while self.index < index:
            data = next(self.reader, None)
            require(data is not None, '镜头入点、时长或变速超出视频素材末尾')
            self.frame = np.frombuffer(data, dtype=np.uint8).reshape(self.shape)
            self.index += 1
        return self.frame

    def close(self):
        self.reader.close()


def pixel_rect(rect, width, height):
    x, y, w, h = rect
    left = min(width - 1, round(x * width))
    top = min(height - 1, round(y * height))
    return left, top, max(left + 1, round((x + w) * width)), max(top + 1, round((y + h) * height))


def black_edges(frame):
    """Estimate persistent-looking dark edge bands on an individual sampled frame."""
    gray = cv2.cvtColor(cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    if np.quantile(gray[24:72, 24:72], .75) < 32:
        return (0.,) * 4
    rows, cols = (gray < 16).mean(axis=1) >= .97, (gray < 16).mean(axis=0) >= .97

    def band(edge):
        length = int(np.argmax(~edge))
        return length / 96 if 2 <= length <= 24 else 0.

    return tuple(band(edge) for edge in (cols, rows, cols[::-1], rows[::-1]))


def camera_matrix(source, target, pose, fit, frame=0, shake=0):
    w, h = source
    out_w, out_h = target
    angle = math.radians(pose['rotation'])
    c, s = abs(math.cos(angle)), abs(math.sin(angle))
    span_x, span_y = c * out_w + s * out_h, s * out_w + c * out_h
    if fit == 'cover':
        scale = max(span_x / w, span_y / h) * pose['zoom']
    else:
        scale = min(out_w / (c * w + s * h), out_h / (s * w + c * h)) * pose['zoom']
    x = pose['x'] * w + w * shake * math.sin(frame * 2.4) - .5
    y = pose['y'] * h + h * shake * math.sin(frame * 1.7) - .5
    clamped = False
    if fit == 'cover':
        dx, dy = span_x / (2 * scale), span_y / (2 * scale)
        safe_x = min(max(x, dx - .5), w - dx - .5)
        safe_y = min(max(y, dy - .5), h - dy - .5)
        clamped = abs(x - safe_x) > 1e-6 or abs(y - safe_y) > 1e-6
        x, y = safe_x, safe_y
    matrix = cv2.getRotationMatrix2D((0., 0.), pose['rotation'], scale)
    matrix[:, 2] = np.array([(out_w - 1) / 2, (out_h - 1) / 2]) - matrix[:, :2] @ [x, y]
    return matrix, clamped


def subject_cropped(subject, size, crop, matrix, target):
    if subject is None:
        return False
    x, y, w, h = subject
    width, height = size
    left, top, right, bottom = crop
    if x * width < left - .01 or y * height < top - .01 or (x + w) * width > right + .01 or (y + h) * height > bottom + .01:
        return True
    corners = np.array([[u * width - left - .5, v * height - top - .5] for u in (x, x + w) for v in (y, y + h)])
    projected = corners @ matrix[:, :2].T + matrix[:, 2]
    return bool(np.any(projected < -.51) or np.any(projected > np.array(target) - .49))


def transform(frame, target, matrix, fit):
    if fit == 'cover':
        return cv2.warpAffine(frame, matrix, target, borderMode=cv2.BORDER_REPLICATE)
    h, w = frame.shape[:2]
    bg_size = (max(2, target[0] // 16), max(2, target[1] // 16))
    thumb = cv2.resize(frame, (max(2, min(w, 160)), max(2, min(h, 160))), interpolation=cv2.INTER_AREA)
    bg_matrix, _ = camera_matrix(thumb.shape[1::-1], bg_size, {'x': .5, 'y': .5, 'zoom': 1, 'rotation': 0}, 'cover')
    background = cv2.warpAffine(thumb, bg_matrix, bg_size, borderMode=cv2.BORDER_REPLICATE)
    background = cv2.resize(cv2.GaussianBlur(background, (0, 0), 2), target)
    return cv2.warpAffine(frame, matrix, target, dst=background, borderMode=cv2.BORDER_TRANSPARENT)
