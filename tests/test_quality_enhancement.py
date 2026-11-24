from io import BytesIO
from pathlib import Path
import subprocess

import cv2
from flask import Flask
import imageio_ffmpeg
import numpy as np
import pytest

from pipeline.quality_api import create_quality_blueprint
from pipeline.quality_enhancement import EnhancementOptions, EnhancementService, analyze, plan, required_resolution
from pipeline.quality_video import TemporalDetail, enhance_video


class StubModels:
    def __init__(self):
        self.calls = []

    def signature(self, scale, face):
        return ["stub", scale, face]

    def upscale(self, image, scale, options):
        self.calls.append(scale)
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    def restore_faces(self, image, options):
        self.calls.append(options.fidelity)
        return image.copy(), 1


def test_quality_signals_and_automatic_decisions():
    flat = np.full((256, 256, 3), 128, np.uint8)
    rng = np.random.default_rng(3)
    noisy = np.clip(flat.astype(float) + rng.normal(0, 11, (256, 256, 1)), 0, 255).astype(np.uint8)
    blocked = np.tile(np.repeat([96, 160], 8), (256, 16)).astype(np.uint8)
    blocked = cv2.cvtColor(blocked, cv2.COLOR_GRAY2BGR)
    blurred = cv2.GaussianBlur(blocked, (0, 0), 4)
    assert analyze(noisy)["noisy"]
    assert analyze(blocked)["blocky"]
    assert analyze(blurred)["blurry"]
    options = EnhancementOptions(enabled=True)
    assert plan(analyze(flat), (256, 256), 1., options)["filters"] == []
    assert plan(analyze(noisy), (256, 256), 1., options)["filters"] == ["denoise"]
    assert plan(analyze(flat), (360, 256), 1., options)["scale"] == 2
    assert plan(analyze(flat), (512, 512), 1.15, options)["scale"] == 4
    assert required_resolution((1920, 1080), 1.15) == (2208, 1242)
    assert required_resolution((1920, 1080), 1.15, (500, 1000)) == (621, 1242)
    assert required_resolution((1080, 1920), 1.15, (500, 1500), cover=True) == (1242, 3726)


def test_source_is_immutable_cache_tracks_crop_model_and_fidelity(tmp_path):
    original = np.full((64, 96, 4), 128, np.uint8)
    source = tmp_path / "source.png"
    source.write_bytes(cv2.imencode(".png", original)[1].tobytes())
    original_bytes = source.read_bytes()
    models = StubModels()
    service = EnhancementService(tmp_path / "cache", backend=models)
    opts = EnhancementOptions(enabled=True, face=True)
    crop = [.25, 0., .5, 1.]
    first = service.image(source, (96, 128), opts, crop=crop)
    second = service.image(source, (96, 128), opts, crop=crop)
    another = service.image(source, (96, 128), EnhancementOptions(enabled=True, face=True, fidelity=.95), crop=crop)
    result = cv2.imread(first["path"], cv2.IMREAD_UNCHANGED)
    assert second["cache_hit"] and first["path"] == second["path"] != another["path"]
    assert result.shape == (128, 96, 4) and np.all(result[:, :, 3] == 128)
    assert models.calls == [2, .9, 2, .95]
    assert source.read_bytes() == original_bytes
    assert service.batch([source], (1, 1), EnhancementOptions())[0]["path"] == str(source)


def test_uploaded_comparison_provides_pngs_without_exposing_paths(tmp_path):
    app = Flask(__name__)
    app.register_blueprint(create_quality_blueprint(tmp_path))
    client = app.test_client()
    image = np.full((64, 64, 3), 170, np.uint8)
    data = cv2.imencode(".png", image)[1].tobytes()
    response = client.post("/api/quality-enhancement/preview",
                           data={"width": 64, "height": 64, "image": (BytesIO(data), "picture.png")})
    assert response.status_code == 200
    result = response.get_json()
    assert result["before_url"] == result["after_url"]
    assert "path" not in result and client.get(result["before_url"]).data.startswith(b"\x89PNG")
    assert client.get("/api/quality-enhancement/assets/images/invalid.png").status_code == 404


def test_temporal_detail_smooths_static_textures_and_resets_at_cut():
    temporal = TemporalDetail(.5)
    dark = np.full((32, 48, 3), 40, np.uint8)
    bright = np.full_like(dark, 200)
    temporal.apply(dark, np.full((64, 96, 3), 60, np.uint8))
    second = temporal.apply(dark, np.full((64, 96, 3), 40, np.uint8))
    assert np.all(second == 50)
    assert np.all(temporal.apply(bright, np.full_like(second, 200)) == 200)


def test_video_interface_preserves_frames_audio_and_source(tmp_path):
    source, out = tmp_path / "clip.mp4", tmp_path / "enhanced.mp4"
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([exe, "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=48x32:r=10:d=0.5",
                    "-f", "lavfi", "-i", "sine=d=0.5", "-c:v", "libx264", "-c:a", "aac",
                    str(source)], check=True)
    before = source.read_bytes()
    models = StubModels()
    result = enhance_video(source, out, backend=models, cache_dir=tmp_path / "cache")
    assert (result["width"], result["height"], result["frames"], result["fps"]) == (96, 64, 5, 10)
    assert Path(result["path"]).is_file() and source.read_bytes() == before
    subprocess.run([exe, "-v", "error", "-i", str(out), "-map", "0:a:0", "-f", "null", "-"], check=True)
    copy = enhance_video(source, tmp_path / "copy.mp4", backend=models, cache_dir=tmp_path / "cache")
    assert copy["cache_hit"] and len(models.calls) == 5
    with pytest.raises(ValueError, match="尚不存在"):
        enhance_video(source, out, backend=StubModels())


@pytest.mark.parametrize(("scale", "shuffle"), [(2, 2), (4, None)])
def test_rrdb_loader_and_tiling_on_cpu(tmp_path, monkeypatch, scale, shuffle):
    pytest.importorskip("spandrel")
    import torch
    from spandrel.architectures.ESRGAN.__arch.RRDB import RRDBNet
    from pipeline.quality_models import LocalModels

    monkeypatch.setenv("ANIFLOW_ENHANCE_MODELS", str(tmp_path))
    tensor_model = RRDBNet(in_nc=3 if shuffle is None else 12, num_filters=8,
                          num_blocks=1, scale=4, shuffle_factor=shuffle)
    torch.save({"params_ema": tensor_model.state_dict()}, tmp_path / f"RealESRGAN_x{scale}plus.pth")
    # Odd-sized inputs exercise RRDB x2's padding and non-divisible tiles.
    image = np.full((13, 35, 3), 128, np.uint8)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        output = LocalModels().upscale(image, scale, EnhancementOptions(device="cpu", tile=32))
    finally:
        torch.set_num_threads(previous_threads)
    assert output.shape == (13 * scale, 35 * scale, 3)
