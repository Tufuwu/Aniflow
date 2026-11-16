"""Offline verification of the selected voice, real filters, timing and cache isolation."""
from array import array
import hashlib
import json
import math
from pathlib import Path
import socket
import struct
import subprocess
import wave

import pytest

import config
from pipeline import tts_engine as tts
from pipeline.timeline_director import ffmpeg_exe
from pipeline.tts_profiles import voice_cache_options
from pipeline.voice_pipeline import voice_options
from test_semantic_alignment import run_mock_stream
from test_shot_editing import chapter, reviewed


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("Voice preset tests must not call a service")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    async def inline(function, *args):
        return function(*args)
    monkeypatch.setattr(tts.asyncio, "to_thread", inline)


def waveform(path):
    result = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", str(path), "-ar", "48000",
                             "-ac", "1", "-f", "s16le", "pipe:1"], capture_output=True, check=True)
    return array("h", result.stdout)


def test_selected_processing_boosts_bass_preserves_timing_and_hashes_final_audio(tmp_path, monkeypatch):
    original = tmp_path / "source.wav"
    with wave.open(str(original), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(b"".join(struct.pack("<h", round(1800 * (
            math.sin(math.tau * 140 * i / 24000) + math.sin(math.tau * 1000 * i / 24000))))
            for i in range(28800)))
    encoded = tmp_path / "original.mp3"
    subprocess.run([ffmpeg_exe(), "-v", "error", "-i", str(original), "-c:a", "libmp3lame",
                    str(encoded)], capture_output=True, check=True)
    raw = encoded.read_bytes()
    calls = []
    class Communicate:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        async def stream(self):
            yield {"type": "audio", "data": raw}
            yield {"type": "WordBoundary", "offset": 0, "duration": 10000000, "text": "test"}
    monkeypatch.setattr(tts.edge_tts, "Communicate", Communicate)
    output = tmp_path / "processed.mp3"
    result = run_mock_stream(tts._generate_speech_edge("test", str(output)))
    assert calls[0]["voice"] == "zh-CN-YunxiNeural" and calls[0]["rate"] == "+35%"
    assert calls[0]["pitch"] == "-2Hz"
    assert output.read_bytes() != raw
    assert result["audio_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    saved = json.loads(output.with_suffix(".boundaries.json").read_text("utf-8"))
    assert saved["tts_options"]["audio_profile"]["id"] == "yunxi_fast_warm_v1"
    assert saved["boundaries"] == [{"text": "test", "start": 0., "end": 1.}]
    before, after = waveform(encoded), waveform(output)
    assert abs(len(before) - len(after)) <= 96
    assert max(abs(value) for value in after) < 32000
    def magnitude(samples, frequency):
        samples = samples[4800:-4800]
        real = sum(value * math.cos(math.tau * frequency * i / 48000) for i, value in enumerate(samples))
        imaginary = sum(value * math.sin(math.tau * frequency * i / 48000) for i, value in enumerate(samples))
        return math.hypot(real, imaginary)
    before_ratio = magnitude(before, 140) / magnitude(before, 1000)
    after_ratio = magnitude(after, 140) / magnitude(after, 1000)
    assert after_ratio > before_ratio * 1.2


def test_processing_failure_does_not_write_success_metadata(tmp_path, monkeypatch):
    class Communicate:
        def __init__(self, **kwargs):
            pass
        async def stream(self):
            yield {"type": "audio", "data": b"mock raw audio"}
    monkeypatch.setattr(tts.edge_tts, "Communicate", Communicate)
    def fail(*args):
        raise RuntimeError("filter failed")
    monkeypatch.setattr(tts, "apply_audio_profile", fail)
    path = tmp_path / "failed.mp3"
    with pytest.raises(RuntimeError, match="filter failed"):
        run_mock_stream(tts._generate_speech_edge("test", str(path)))
    assert not path.with_suffix(".boundaries.json").exists()


def test_audio_profile_change_uses_new_cache_without_overwriting_old_audio(chapter, monkeypatch):
    pipeline = chapter[2]
    state = pipeline.direct(reviewed(chapter)["revision"])
    previous = {nid: dict(audio) for nid, audio in state["audio"].items()}
    monkeypatch.setattr(config, "TTS_AUDIO_PROFILE", "")
    next_state = pipeline.direct(state["revision"])
    for nid, old in previous.items():
        assert next_state["audio"][nid]["key"] != old["key"]
        assert hashlib.sha256(Path(old["path"]).read_bytes()).hexdigest() == old["sha256"]


def test_voice_defaults_and_other_engines_keep_their_own_controls():
    defaults = voice_options({})
    assert (defaults["engine"], defaults["voice"], defaults["rate"], defaults["pitch"]) == (
        "edge", "zh-CN-YunxiNeural", "+35%", "-2Hz")
    eleven = voice_options({"engine": "elevenlabs", "voice": "fixture_voice"})
    assert eleven["rate"] == "+0%" and eleven["pitch"] == "+0Hz"
    assert "audio_profile" not in voice_cache_options(eleven)
    assert "audio_profile" not in voice_cache_options({"engine": "edge", "voice": "zh-CN-XiaoxiaoNeural"})
