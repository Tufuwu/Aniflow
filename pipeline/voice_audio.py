"""Synthesize semantic units, insert exact additional pauses and audit real samples."""
from array import array
from difflib import SequenceMatcher
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import wave

from pipeline.shot_selector import require
from pipeline.timeline_director import ffmpeg_exe
from pipeline.voice_director import expressive_text
from pipeline.tts_profiles import voice_cache_options

SAMPLE_RATE = 48000


def transcribe_optional(path):
    model_name = os.getenv("VOICE_ASR_MODEL", "").strip()
    if not model_name:
        return None
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ValueError("转写校验已启用，但未安装 faster-whisper") from exc
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    parts, _ = model.transcribe(str(path), language="zh", vad_filter=True)
    return ''.join(p.text for p in parts)


def audio_review(path, expected, transcriber=None):
    result = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", str(path), "-map", "0:a:0",
                             "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1"],
                            capture_output=True, timeout=180)
    require(result.returncode == 0 and bool(result.stdout), "试听音频无法完整解码")
    samples = array('h', result.stdout)
    peak = max(abs(x) for x in samples) / 32768
    rms = math.sqrt(sum(x*x for x in samples) / len(samples)) / 32768
    require(rms > .0001, "试听音频完全静音或音量过低")
    transcript = (transcriber or transcribe_optional)(path)
    normalize = lambda s: ''.join(re.findall(r'[\w\u4e00-\u9fff]', s.lower()))
    result = {"decodable": True, "duration": len(samples) / SAMPLE_RATE, "peak": peak, "rms": rms,
              "clipping_suspected": sum(abs(x) >= 32760 for x in samples) / len(samples) > .005,
              "transcript_status": "not_run", "transcript": transcript, "manual_listen_required": True}
    if transcript is not None:
        require(isinstance(transcript, str), "转写服务未返回文字")
        expected_norm, actual = normalize(expected), normalize(transcript)
        matcher = SequenceMatcher(None, expected_norm, actual, autojunk=False)
        differences = [{"kind": op, "expected": expected_norm[a:b], "heard": actual[c:d]}
                       for op, a, b, c, d in matcher.get_opcodes() if op != 'equal']
        result.update(transcript_status="matched" if not differences else "needs_review",
                      text_similarity=matcher.ratio(), differences=differences)
    return result


def synthesize_performance(plan, output_path, options, speech=None):
    from pipeline.tts_engine import generate_speech
    speech = speech or generate_speech
    require(options["engine"] in {"edge", "elevenlabs"}, "情绪配音目前支持 ElevenLabs 和 Edge 基础预演")
    if options["engine"] == "elevenlabs":
        require(options.get("model") in {"eleven_v4", "eleven_v3"}, "情绪配音请选择 Eleven v4 或 v3")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    timings, cursor = [], 0
    with tempfile.TemporaryDirectory(prefix="voice-", dir=target.parent) as folder:
        root = Path(folder)
        combined = root / "combined.wav"
        with wave.open(str(combined), 'wb') as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(SAMPLE_RATE)
            for i, unit in enumerate(plan["units"]):
                unit_path = root / f"unit_{i}.mp3"
                kwargs = dict(options)
                if options["engine"] == "elevenlabs":
                    spoken = expressive_text(unit)
                    kwargs["previous_text"] = plan["units"][i-1]["text"] if i else ""
                    kwargs["next_text"] = plan["units"][i+1]["text"] if i+1 < len(plan["units"]) else ""
                    kwargs["stability"] = round(max(0., min(1., options["stability"] + .15 - .3 * unit["intensity"])), 3)
                else:
                    spoken = unit["text"]
                    base = 1 + int(options.get("rate", "+0%")[:-1]) / 100
                    kwargs["rate"] = f"{round((base * unit['rate'] - 1) * 100):+d}%"
                    for field in ("model", "stability", "similarity_boost"):
                        kwargs.pop(field, None)
                speech(spoken, str(unit_path), **kwargs)
                decoded = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", str(unit_path), "-map", "0:a:0",
                                          "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "pipe:1"],
                                         capture_output=True, timeout=180)
                require(decoded.returncode == 0 and bool(decoded.stdout), "语义段音频解码失败")
                count = len(decoded.stdout) // 2
                pause = round(unit["pause_after_ms"] * SAMPLE_RATE / 1000)
                stream.writeframes(decoded.stdout)
                stream.writeframes(b'\0\0' * pause)
                timings.append({"text": unit["text"], "start_sample": cursor, "speech_end_sample": cursor + count,
                                "end_sample": cursor + count + pause, "pause_samples": pause})
                cursor += count + pause
        # WAV keeps inserted silence and sample boundaries exact. Extension is explicit in the cache manifest.
        target.write_bytes(combined.read_bytes())
    return {"unit_timings": timings, "duration": cursor / SAMPLE_RATE, "sample_rate": SAMPLE_RATE,
            "audio_profile": voice_cache_options(options).get("audio_profile"),
            "delivery_mode": "expressive_tags" if options["engine"] == "elevenlabs" else "basic_rate_only",
            "note": "额外停顿按样本插入；语义段内部自然停顿与情绪标签需试听确认"}
