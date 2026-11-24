"""Local, content-addressed enhancement for still images and editor assets."""
from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import cv2
from filelock import FileLock
import numpy as np
from PIL import Image

import config


MODEL_DIR = Path(config.BASE_DIR) / "models" / "enhancement"
CACHE_DIR = Path(config.OUTPUT_DIR) / "enhancement"
CACHE_VERSION = 1


@dataclass(frozen=True)
class EnhancementOptions:
    enabled: bool = False
    face: bool = False
    fidelity: float = 0.9
    device: str = "auto"
    tile: int = 0
    max_zoom: float = 1.0

    @classmethod
    def parse(cls, settings=None):
        if settings is None:
            settings = {"enabled": os.environ.get("ANIFLOW_ENHANCE", "0") == "1"}
        if not isinstance(settings, dict) or set(settings) - set(cls.__dataclass_fields__):
            raise ValueError("无效的画质增强配置")
        result = cls(**settings)
        if (type(result.enabled) is not bool or type(result.face) is not bool
                or result.device not in ("auto", "cuda", "cpu")
                or type(result.tile) is not int or result.tile not in (0, *range(32, 1025, 16))
                or type(result.fidelity) not in (int, float) or not .5 <= result.fidelity <= 1
                or type(result.max_zoom) not in (int, float) or not 1 <= result.max_zoom <= 8):
            raise ValueError("设备、分块大小、人脸保真度或最大缩放超出范围")
        return result


def required_resolution(target, zoom, source_size=None, *, cover=False):
    """Use the dimensions occupied by the foreground in the editor."""
    width, height = target
    if (any(type(n) is not int or not 1 <= n <= 8192 for n in (width, height))
            or type(zoom) not in (int, float) or not 1 <= zoom <= 8):
        raise ValueError("输出尺寸需在 1–8192 像素内，缩放需在 1–8 倍内")
    if source_size:
        sw, sh = source_size
        if (width * sh <= height * sw) != cover:
            height = width * sh / sw
        else:
            width = height * sw / sh
    return math.ceil(width * zoom), math.ceil(height * zoom)


def decode_image(data):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        # Older OpenCV builds cannot decode every format supported by the editor, e.g. GIF.
        try:
            with Image.open(BytesIO(data)) as opened:
                image = cv2.cvtColor(np.asarray(opened.convert("RGBA")), cv2.COLOR_RGBA2BGRA)
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError("无法读取图片") from exc
    if image.dtype == np.uint16:
        image = np.rint(image / 257).astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def analyze(image):
    height, width = image.shape[:2]
    # Sample native pixels so JPEG's 8×8 grid and high-frequency noise are preserved.
    sharpness, noise, blocking = [], [], []
    for y in sorted({0, max(0, height // 2 - 256) // 8 * 8, max(0, height - 512) // 8 * 8}):
        for x in sorted({0, max(0, width // 2 - 256) // 8 * 8, max(0, width - 512) // 8 * 8}):
            patch = image[y:y + 512, x:x + 512]
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
            visible = patch[:, :, 3] > 240 if patch.shape[2] == 4 else np.ones(gray.shape, bool)
            visible = cv2.erode(visible.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            if visible.sum() < 64:
                continue
            laplacian = cv2.Laplacian(gray, cv2.CV_32F)
            if np.std(gray[visible]) > 7:
                sharpness.append(float(np.var(laplacian[visible])))
            residual = gray - cv2.medianBlur(gray, 3)
            smooth = cv2.GaussianBlur(gray, (5, 5), 0)
            flat = visible & (np.abs(cv2.Laplacian(smooth, cv2.CV_32F)) < 3)
            if flat.sum() >= 64:
                noise.append(float(np.median(np.abs(residual[flat])) * 1.4826))
            for values, mask in ((gray, visible), (gray.T, visible.T)):
                diff = np.abs(np.diff(values, axis=1))
                mask = mask[:, 1:] & mask[:, :-1]
                phases = [diff[:, i::8][mask[:, i::8]] for i in range(8)]
                if all(p.size >= 64 for p in phases):
                    means = [np.mean(p) for p in phases]
                    # The maximum also detects a JPEG grid shifted by subsequent crops.
                    blocking.append(max(0., float(max(means) - np.median(means))))
    sharp = float(np.median(sharpness)) if sharpness else None
    sigma = float(np.median(noise)) if noise else 0.
    blocks = float(np.median(blocking)) if blocking else 0.
    return {"width": width, "height": height, "laplacian_variance": sharp,
            "noise_sigma": sigma, "block_score": blocks,
            "blurry": sharp is not None and sharp < 45, "noisy": sigma >= 4, "blocky": blocks >= 2.5}


def plan(quality, target, zoom, options, fit=False):
    size = quality["width"], quality["height"]
    needed = required_resolution(target, zoom, size if fit else None, cover=fit == "cover")
    ratio = max(n / s for n, s in zip(needed, size))
    scale = (4 if ratio > 2 else 2) if options.enabled and ratio > 1 else 1
    filters = []
    degraded = any(quality[key] for key in ("blurry", "noisy", "blocky"))
    if options.enabled and scale == 1:
        if quality["noisy"]:
            filters.append("denoise")
        if quality["blocky"]:
            filters.append("deblock")
        if quality["blurry"] and not quality["blocky"]:
            filters.append("sharpen")
    return {"required": needed, "scale": scale, "filters": filters,
            "face": options.enabled and options.face and (degraded or scale > 1),
            "meets_resolution": ratio <= scale}


def atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{uuid4().hex}.tmp")
    try:
        temp.write_bytes(data)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


class EnhancementService:
    def __init__(self, cache_dir=CACHE_DIR, backend=None):
        from pipeline.quality_models import LocalModels
        self.root = Path(cache_dir).resolve()
        self.backend = backend or LocalModels()

    def image(self, source, target, options, *, zoom=1., fit=False, crop=None):
        source = Path(source)
        data = source.read_bytes()
        original = decode_image(data)
        if crop is not None:
            height, width = original.shape[:2]
            x, y, w, h = crop
            if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1
                    and x + w <= 1.00000001 and y + h <= 1.00000001):
                raise ValueError("无效的归一化裁切坐标")
            original = original[round(y * height):round((y + h) * height),
                                round(x * width):round((x + w) * width)]
            if not original.size:
                raise ValueError("裁切区域小于一个像素")
        quality = analyze(original)
        decision = plan(quality, target, max(zoom, options.max_zoom), options, fit)
        scale = decision["scale"]
        result = {"path": str(source), "quality": quality, "plan": decision,
                  "cache_hit": False, "enhanced": False, "faces_restored": 0}
        if scale == 1 and not (decision["filters"] or decision["face"]):
            return result
        if quality["width"] * quality["height"] * scale ** 2 > 64_000_000:
            raise ValueError("增强结果超过 6400 万像素，请先裁切素材")

        identity = [CACHE_VERSION, hashlib.sha256(data).hexdigest(), crop, decision, asdict(options),
                    self.backend.signature(scale, decision["face"])]
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.root / "images" / (key + ".png")
        metadata = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path.with_suffix(".lock"))):
            if metadata.is_file():
                faces = json.loads(metadata.read_text(encoding="utf-8"))["faces"]
                if path.is_file() or (scale == 1 and not decision["filters"] and not faces):
                    return {**result, "path": str(path if path.is_file() else source), "cache_hit": True,
                            "enhanced": path.is_file(), "faces_restored": faces}
            color = np.ascontiguousarray(original[:, :, :3])
            faces = 0
            if scale > 1:
                color = self.backend.upscale(color, scale, options)
            if decision["face"]:
                color, faces = self.backend.restore_faces(color, options)
            for method in decision["filters"]:
                if method == "denoise":
                    color = cv2.fastNlMeansDenoisingColored(color, None, 4, 4, 7, 21)
                elif method == "deblock":
                    color = cv2.bilateralFilter(color, 5, 12, 4)
                else:
                    color = cv2.addWeighted(color, 1.3, cv2.GaussianBlur(color, (0, 0), 1.1), -.3, 0)
            changed = bool(scale > 1 or decision["filters"] or faces)
            if changed:
                if original.shape[2] == 4:
                    alpha = cv2.resize(original[:, :, 3], (color.shape[1], color.shape[0]), interpolation=cv2.INTER_LINEAR)
                    color = np.dstack((color, alpha))
                ok, encoded = cv2.imencode(".png", color)
                if not ok:
                    raise ValueError("增强图片编码失败")
                atomic_bytes(path, encoded.tobytes())
            atomic_bytes(metadata, json.dumps({"faces": faces}).encode())
            return {**result, "path": str(path if changed else source), "enhanced": changed,
                    "faces_restored": faces}

    def batch(self, sources, target, options, **kwargs):
        return [self.image(source, target, options, **kwargs) for source in sources]


def prepare_panels(panels, target, settings, zoom, progress=None, *, fit=True):
    options = EnhancementOptions.parse(settings)
    if not options.enabled:
        return panels
    service = EnhancementService()
    prepared = []
    for index, panel in enumerate(panels):
        path = panel.get("path")
        if path and Path(path).is_file():
            item = service.image(path, target, options, zoom=zoom, fit=fit)
            panel = {**panel, "path": item["path"], "_preserve_quality_detail": True}
        prepared.append(panel)
        if progress:
            progress(index + 1, len(panels), "正在检查并增强视频素材")
    return prepared
