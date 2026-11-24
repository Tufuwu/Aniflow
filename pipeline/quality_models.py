"""Adapters for local Real-ESRGAN weights and an isolated CodeFormer checkout."""
from pathlib import Path
import os
import subprocess
import threading
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from pipeline.quality_enhancement import MODEL_DIR


_INFERENCE = threading.Lock()
FACE_WEIGHTS = ("weights/CodeFormer/codeformer.pth", "weights/facelib/detection_Resnet50_Final.pth",
                "weights/facelib/parsing_parsenet.pth")


def hardware(device="auto"):
    import torch
    available = torch.cuda.is_available()
    if device == "cuda" and not available:
        raise RuntimeError("当前 PyTorch 环境没有可用的 CUDA 设备")
    result = {"cuda_available": available, "device": "cuda" if available and device != "cpu" else "cpu"}
    if result["device"] == "cuda":
        free, total = torch.cuda.mem_get_info()
        result.update(free_bytes=free, total_bytes=total, name=torch.cuda.get_device_name())
    return result


def _tile_size(info, requested):
    if requested:
        return requested
    if info["device"] == "cpu":
        return 192
    free_gib = info["free_bytes"] / 1024**3
    return 512 if free_gib >= 8 else 320 if free_gib >= 4 else 192 if free_gib >= 2 else 96


class LocalModels:
    def __init__(self):
        self.model_dir = Path(os.environ.get("ANIFLOW_ENHANCE_MODELS", MODEL_DIR)).resolve()
        self.codeformer_dir = Path(os.environ.get("ANIFLOW_CODEFORMER_DIR", self.model_dir / "CodeFormer")).resolve()
        self.codeformer_python = os.environ.get("ANIFLOW_CODEFORMER_PYTHON")
        self._loaded = None

    def _sr_path(self, scale):
        if scale not in (2, 4):
            raise ValueError("Real-ESRGAN 支持 2× 或 4×")
        path = self.model_dir / f"RealESRGAN_x{scale}plus.pth"
        safe = path.with_suffix(".safetensors")
        return safe if safe.is_file() else path

    def signature(self, scale, face):
        paths = [self._sr_path(scale)] if scale > 1 else []
        if face:
            paths.extend(self.codeformer_dir / name for name in (
                "inference_codeformer.py", "basicsr/archs/codeformer_arch.py",
                "facelib/utils/face_restoration_helper.py", *FACE_WEIGHTS))
        files = []
        for path in paths:
            if not path.is_file():
                raise RuntimeError(f"缺少本地增强模型文件 {path}，请参阅 docs/quality-enhancement.md")
            stat = path.stat()
            files.append((str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        return files

    def upscale(self, image, scale, options):
        import torch
        try:
            from spandrel import ImageModelDescriptor, ModelLoader
        except ImportError as exc:
            raise RuntimeError("请先安装 requirements-enhancement.txt") from exc
        with _INFERENCE, torch.inference_mode():
            info = hardware(options.device)
            device = info["device"]
            identity = self.signature(scale, False), device
            if self._loaded is None or self._loaded[0] != identity:
                self._loaded = None
                path = self._sr_path(scale)
                if path.suffix == ".safetensors":
                    from safetensors.torch import load_file
                    weights = load_file(str(path))
                else:
                    weights = torch.load(str(path), map_location="cpu", weights_only=True)
                for key in ("params_ema", "params"):
                    if key in weights:
                        weights = weights[key]
                        break
                model = ModelLoader().load_from_state_dict(weights)
                if (not isinstance(model, ImageModelDescriptor) or model.scale != scale
                        or (model.input_channels, model.output_channels) != (3, 3)):
                    raise RuntimeError(f"权重与 Real-ESRGAN {scale}× 不匹配")
                self._loaded = identity, model.eval().to(device)
            model = self._loaded[1]
            tile = _tile_size(info, options.tile)
            while True:
                try:
                    return self._tiled(image, model, scale, tile, device)
                except torch.cuda.OutOfMemoryError:
                    if device != "cuda" or tile <= 32:
                        raise
                torch.cuda.empty_cache()
                tile = max(32, (tile // 32) * 16)

    @staticmethod
    def _tiled(image, model, scale, tile, device):
        import torch
        height, width = image.shape[:2]
        output = np.empty((height * scale, width * scale, 3), np.uint8)
        padding = 32
        for y in range(0, height, tile):
            for x in range(0, width, tile):
                bottom, right = min(y + tile, height), min(x + tile, width)
                y0, x0 = max(0, y - padding), max(0, x - padding)
                patch = image[y0:min(bottom + padding, height), x0:min(right + padding, width)]
                rgb = np.ascontiguousarray(patch[:, :, ::-1].transpose(2, 0, 1))
                tensor = torch.from_numpy(rgb).unsqueeze(0).to(device=device, dtype=torch.float32) / 255
                result = model(tensor)[0]
                cropped = result[:, (y-y0)*scale:(bottom-y0)*scale, (x-x0)*scale:(right-x0)*scale]
                bgr = (cropped.flip(0).permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()
                output[y*scale:bottom*scale, x*scale:right*scale] = bgr
        return output

    def restore_faces(self, image, options):
        self.signature(1, True)
        if not self.codeformer_python:
            raise RuntimeError("请设置 ANIFLOW_CODEFORMER_PYTHON 指向 CodeFormer 的独立 Python 环境")
        worker = Path(__file__).with_name("quality_codeformer_worker.py").resolve()
        with _INFERENCE, TemporaryDirectory() as directory:
            folder = Path(directory)
            _, encoded = cv2.imencode(".png", image)
            (folder / "input.png").write_bytes(encoded.tobytes())
            command = [self.codeformer_python, str(worker), str(self.codeformer_dir), str(folder),
                       options.device, str(options.fidelity)]
            try:
                run = subprocess.run(command, cwd=self.codeformer_dir, capture_output=True,
                                     encoding="utf-8", errors="replace", timeout=600)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(f"无法完成 CodeFormer 推理：{exc}") from exc
            if run.returncode:
                raise RuntimeError(f"CodeFormer 推理失败：{(run.stderr or run.stdout)[-1600:]}")
            faces = int((folder / "faces.txt").read_text(encoding="utf-8"))
            if not faces:
                return image, 0
            result = cv2.imdecode(np.frombuffer((folder / "output.png").read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if result is None or result.shape != image.shape:
                raise RuntimeError("CodeFormer 未生成有效图片")
            return result, faces
