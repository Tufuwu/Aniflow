"""Approved narration processing, with a cache identity for its exact settings."""
import json
from pathlib import Path
import subprocess
from uuid import uuid4

import config


YUNXI_WARM = {
    "id": "yunxi_fast_warm_v1",
    "filters": (
        "highpass=f=60,equalizer=f=140:t=q:w=0.85:g=3,"
        "equalizer=f=280:t=q:w=1:g=1,equalizer=f=450:t=q:w=1:g=-1.5,"
        "equalizer=f=2800:t=q:w=0.9:g=0.8,"
        "acompressor=threshold=0.125:ratio=1.8:attack=15:release=110:makeup=1"
    ),
    "loudness_lufs": -18,
    "true_peak_db": -2,
    "loudness_range": 7,
    "sample_rate": 48000,
    "bitrate": "192k",
}


def voice_cache_options(options):
    """Keep generation arguments separate from the processing recorded in cache keys."""
    result = dict(options)
    engine = options.get("engine") or config.TTS_ENGINE
    voice = options.get("voice") or config.TTS_VOICE
    if (engine == "edge" and voice == "zh-CN-YunxiNeural"
            and config.TTS_AUDIO_PROFILE == YUNXI_WARM["id"]):
        result["audio_profile"] = dict(YUNXI_WARM)
    return result


def apply_audio_profile(audio_path, profile):
    """Apply the approved EQ/compression and two-pass normalization without retiming."""
    if not profile:
        return
    from pipeline.timeline_director import audio_duration, ffmpeg_exe

    path = Path(audio_path)
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.mp3")
    ffmpeg = ffmpeg_exe()
    prefix = profile["filters"]
    target = (f"loudnorm=I={profile['loudness_lufs']}:TP={profile['true_peak_db']}"
              f":LRA={profile['loudness_range']}")

    def run(arguments):
        result = subprocess.run([ffmpeg, "-hide_banner", "-nostdin", *arguments],
                                capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError("Narration audio processing failed: "
                               + result.stderr.decode("utf-8", errors="replace")[-1500:])
        return result

    try:
        analysis = run(["-i", str(path), "-af", prefix + "," + target + ":print_format=json",
                        "-f", "null", "-"])
        logs = analysis.stderr.decode("utf-8", errors="replace")
        measured = json.loads(logs[logs.rfind("{"):logs.rfind("}") + 1])
        normalization = (target + f":measured_I={measured['input_i']}"
                         f":measured_TP={measured['input_tp']}:measured_LRA={measured['input_lra']}"
                         f":measured_thresh={measured['input_thresh']}:offset={measured['target_offset']}"
                         ":linear=true")
        run(["-v", "error", "-i", str(path), "-af", prefix + "," + normalization,
             "-ar", str(profile["sample_rate"]), "-ac", "1", "-c:a", "libmp3lame",
             "-b:a", profile["bitrate"], str(temporary)])
        # Word boundaries still describe this performance; EQ must not stretch its timeline.
        if abs(audio_duration(path) - audio_duration(temporary)) > .002:
            raise RuntimeError("Narration audio processing changed its timing")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
