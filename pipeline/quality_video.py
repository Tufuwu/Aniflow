"""Streaming video SR with motion-compensated enhancement detail."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import cv2
from filelock import FileLock
import imageio_ffmpeg
import numpy as np

from pipeline.quality_enhancement import CACHE_DIR, CACHE_VERSION, EnhancementOptions, atomic_bytes
from pipeline.quality_models import LocalModels


class TemporalDetail:
    """Stabilize SR residuals on tracked pixels, leaving base motion intact."""
    def __init__(self, strength=.45):
        self.strength = strength
        self.previous_gray = None
        self.previous_detail = None

    def apply(self, frame, enhanced):
        if not self.strength:
            return enhanced
        h, w = enhanced.shape[:2]
        scale = min(1., 640 / max(frame.shape[:2]))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (max(1, round(gray.shape[1] * scale)),
                                 max(1, round(gray.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        base = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)
        detail = enhanced.astype(np.float32) - base
        old = self.previous_gray
        if old is not None and cv2.absdiff(old, gray).mean() < 38:
            flow = lambda a, b: cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 15, 3, 5, 1.2, 0)
            backward = flow(gray, old)
            forward = flow(old, gray)
            gh, gw = gray.shape
            x, y = np.meshgrid(np.arange(gw, dtype=np.float32), np.arange(gh, dtype=np.float32))
            mx, my = x + backward[:, :, 0], y + backward[:, :, 1]
            moved_gray = cv2.remap(old, mx, my, cv2.INTER_LINEAR)
            reverse = cv2.remap(forward, mx, my, cv2.INTER_LINEAR)
            consistency = np.linalg.norm(backward + reverse, axis=2)
            good = ((mx >= 0) & (my >= 0) & (mx <= gw - 1) & (my <= gh - 1)
                    & (consistency < 1.5) & (cv2.absdiff(gray, moved_gray) < 18))
            if good.mean() > .35:
                if not hasattr(self, "grid"):
                    self.grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
                bx = cv2.resize(backward[:, :, 0], (w, h)) * (w / gw)
                by = cv2.resize(backward[:, :, 1], (w, h)) * (h / gh)
                warped = cv2.remap(self.previous_detail, self.grid[0] + bx, self.grid[1] + by, cv2.INTER_LINEAR)
                weight = cv2.resize(good.astype(np.float32), (w, h))[:, :, None] * self.strength
                detail += (warped - detail) * weight
        self.previous_gray, self.previous_detail = gray, detail
        return np.rint(np.clip(base + detail, 0, 255)).astype(np.uint8)


def _source_signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def enhance_video(source, destination, scale=2, options=None, strength=.45, progress=None, backend=None,
                  cache_dir=CACHE_DIR):
    """Write a new H.264 MP4 with optional audio, using a single SR model per video."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.suffix.lower() != ".mp4" or destination.exists():
        raise ValueError("请使用一个独立且尚不存在的 MP4 输出路径")
    if scale not in (2, 4) or not 0 <= strength <= .8:
        raise ValueError("视频倍率必须为 2 或 4，时序强度必须在 0–0.8 之间")
    if not source.is_file():
        raise ValueError("视频来源不存在")
    options = options or EnhancementOptions()
    backend = backend or LocalModels()
    source_stat = _source_signature(source)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    if _source_signature(source) != source_stat:
        raise RuntimeError("读取期间源视频发生变化，请重新处理")
    identity = [CACHE_VERSION, digest.hexdigest(), scale, strength, asdict(options), backend.signature(scale, False)]
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cached = Path(cache_dir).resolve() / "videos" / (key + ".mp4")
    cached.parent.mkdir(parents=True, exist_ok=True)
    manifest = cached.with_suffix(".json")
    with FileLock(str(cached.with_suffix(".lock"))):
        hit = cached.is_file() and manifest.is_file()
        if hit:
            stats = json.loads(manifest.read_text(encoding="utf-8"))
        else:
            stats = _encode_video(source, cached, scale, options, strength, progress, backend)
            if _source_signature(source) != source_stat:
                cached.unlink()
                raise RuntimeError("增强期间源视频发生变化，请重新处理")
            atomic_bytes(manifest, json.dumps(stats).encode())

        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=destination.parent) as folder:
            prepared = Path(folder) / "result.mp4"
            shutil.copyfile(cached, prepared)
            # Publish without overwriting an existing output, including a concurrent job's result.
            os.link(prepared, destination)
    return {**stats, "path": str(destination), "cache_hit": hit}


def _encode_video(source, destination, scale, options, strength, progress, backend):
    temporal = TemporalDetail(strength)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as work:
        silent, complete = Path(work) / "silent.mp4", Path(work) / "complete.mp4"
        reader = imageio_ffmpeg.read_frames(str(source), pix_fmt="bgr24", input_params=["-xerror"])
        try:
            meta = next(reader)
            w, h = meta["size"]
            fps = meta["fps"]
            if not fps or min(w, h) < 8 or max(w, h) * scale > 8192 or w * h * scale**2 > 36_000_000:
                raise ValueError("视频尺寸或帧率无法处理")
            writer = imageio_ffmpeg.write_frames(str(silent), (w*scale, h*scale), pix_fmt_in="bgr24",
                                                 codec="libx264", fps=fps, macro_block_size=1,
                                                 ffmpeg_log_level="error", output_params=["-movflags", "+faststart"])
            frames = 0
            try:
                writer.send(None)
                for raw in reader:
                    frame = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
                    output = temporal.apply(frame, backend.upscale(frame, scale, options))
                    writer.send(output.tobytes())
                    frames += 1
                    if progress:
                        progress(frames)
            finally:
                writer.close()
        finally:
            reader.close()
        if frames == 0 or not silent.is_file():
            raise RuntimeError("未生成任何视频帧")
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(silent), "-i", str(source),
                        "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac",
                        "-movflags", "+faststart", str(complete)], check=True, capture_output=True)
        if imageio_ffmpeg.count_frames_and_secs(str(complete))[0] != frames:
            raise RuntimeError("视频增强输出帧数校验失败")
        complete.replace(destination)
    return {"width": w * scale, "height": h * scale, "fps": fps, "frames": frames}
