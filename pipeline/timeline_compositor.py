"""Render an approved timeline without changing any source image or earlier export."""
from pathlib import Path
import hashlib
import re
import shutil
import subprocess

from PIL import Image, ImageOps

from pipeline.shot_selector import require
from pipeline.timeline_director import FPS, audio_duration, ffmpeg_exe, validate_timeline


def run(args, cwd=None):
    result = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostdin", "-v", "error", *args],
                            cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    require(result.returncode == 0, "视频处理失败：" + result.stderr[-1800:])
    return result


def srt_time(frame):
    ms = round(frame * 1000 / FPS)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def srt_text(timeline):
    return "\n\n".join(f"{i}\n{srt_time(s['start_frame'])} --> {srt_time(s['end_frame'])}\n{s['text']}"
                        for i, s in enumerate(timeline["subtitles"], 1)) + "\n"


def ass_time(frame):
    cs = round(frame*100/FPS)
    return f'{cs//360000}:{cs//6000%60:02d}:{cs//100%60:02d}.{cs%100:02d}'


def ass_text(timeline):
    """Use video pixels explicitly; avoid libass's implicit low-resolution canvas."""
    lines = ['[Script Info]','ScriptType: v4.00+','PlayResX: 1080','PlayResY: 1920','WrapStyle: 0','',
             '[V4+ Styles]',
             'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
             'Style: Caption,Microsoft YaHei,52,&H00FFFFFF,&H00FFFFFF,&H00100C15,&H90000000,-1,0,0,0,100,100,0,0,1,2.5,0,2,70,70,150,1','',
             '[Events]','Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text']
    for sub in timeline['subtitles']:
        text = sub['text'].replace('\\', '＼').replace('{','｛').replace('}','｝').replace('\n',r'\N')
        lines.append(f"Dialogue: 0,{ass_time(sub['start_frame'])},{ass_time(sub['end_frame'])},Caption,,0,0,0,,{text}")
    return '\n'.join(lines)+'\n'


def inspect_export(path, timeline):
    path = Path(path)
    require(path.is_file() and path.stat().st_size > 1024, "导出文件为空或不完整")
    # Probe the actual exported streams. ffmpeg is bundled, ffprobe is not always installed.
    probe = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    metadata = probe.stderr
    require(re.search(r"Video: h264\b", metadata) and "1080x1920" in metadata, "导出不是1080×1920 H.264")
    require(re.search(r"Audio: aac\b", metadata), "导出缺少 AAC 旁白音轨")
    require(re.search(r"\b30 fps\b", metadata), "导出帧率不是30fps")
    # Decode every frame and sample; a header-only probe cannot detect a truncated MP4.
    decoded = run(["-xerror", "-i", str(path), "-map", "0:v:0", "-map", "0:a:0",
                   "-progress", "pipe:1", "-f", "null", "-"])
    values = re.findall(r"^frame=(\d+)", decoded.stdout, flags=re.M)
    require(bool(values) and int(values[-1]) == timeline["total_frames"], "导出帧数与时间轴不符")
    match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", metadata)
    require(match is not None, "无法读取导出时长")
    duration = int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3])
    require(abs(duration - timeline["duration"]) < .12, "导出音画总时长不符")
    decoded_audio = audio_duration(path)
    require(abs(decoded_audio - timeline["duration"]) < .05, "导出音轨不完整或音画不同步")
    return {"complete": True, "width": 1080, "height": 1920, "codec": "h264", "audio_codec": "aac",
            "decoded_frames": int(values[-1]), "duration": duration, "decoded_audio_duration": decoded_audio,
            "audio_duration_error": abs(decoded_audio - timeline["duration"]), "bytes": path.stat().st_size}


def render_timeline(store, source, timeline, output_dir, progress=None, display_assets=None):
    validate_timeline(timeline)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    work = root / "work"
    work.mkdir()
    panels = {p["source_panel_id"]: p for p in source["bible"]["panels"]}
    visual_checks = []
    # Reuse AniFlow's background compositor; detection/cropping remains in existing modules.
    from pipeline.compositor import _apply_bg_mode
    for i, shot in enumerate(timeline["shots"]):
        panel = panels[shot['panel_id']]
        path = store.evidence_path(panel)
        derivative = (display_assets or {}).get(shot['panel_id'])
        if derivative:
            path = Path(derivative['path'])
            require(derivative.get('review_status') == 'approved' and derivative.get('original_sha256') == panel['sha256'],
                    '清图尚未审核或不属于当前来源')
            require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == derivative['sha256'], '清图校验失败')
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            x, y, w, h = shot["crop"]
            box = (round(x * image.width), round(y * image.height),
                   round((x + w) * image.width), round((y + h) * image.height))
            require(box[2] > box[0] and box[3] > box[1], "裁切区域小于一个像素")
            image = image.crop(box)
            base = Image.fromarray(_apply_bg_mode(image, 1080, 1920, "blur", None))
            base = Image.blend(base,Image.new('RGB',base.size,'#0c101b'),.72)
            fitted = ImageOps.contain(image, (1000, 1390), Image.Resampling.LANCZOS)
            base.paste(fitted, ((1080-fitted.width)//2, 180+(1390-fitted.height)//2))
            base.save(work / f"base_{i:04d}.png")
            gray = image.resize((128,128)).convert('L')
            white = sum(v >= 250 for v in gray.getdata())/(128*128)
            visual_checks.append({'shot_id':shot['shot_id'],'panel_id':shot['panel_id'],
                                  'white_fraction':round(white,4),'sparse_content':white > .95,
                                  'boundary_uncertain':panels[shot['panel_id']].get('boundary_uncertain',False)})
        n, motion = shot["frames"], shot["motion"]
        ratio = f"on/{max(1, n - 1)}"
        zoom = "1" if motion == "hold" else f"1+0.06*{ratio}" if motion == "zoom_in" else (
            f"1.06-0.06*{ratio}" if motion == "zoom_out" else "1.06")
        xexpr, yexpr = "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"
        if motion in {"pan_left", "pan_right"}:
            xexpr = f"(iw-iw/zoom)*{'(1-' + ratio + ')' if motion == 'pan_left' else ratio}"
        if motion in {"pan_up", "pan_down"}:
            yexpr = f"(ih-ih/zoom)*{'(1-' + ratio + ')' if motion == 'pan_up' else ratio}"
        vf = f"scale=2160:3840,zoompan=z='{zoom}':x='{xexpr}':y='{yexpr}':d={n}:s=1080x1920:fps={FPS},setsar=1"
        if shot["transition_frames"]:
            vf += f",fade=t=in:st=0:d={shot['transition_frames'] / FPS}"
        run(["-i", f"base_{i:04d}.png", "-vf", vf, "-frames:v", str(n), "-an",
             "-c:v", "libx264", "-threads", "2", "-preset", "ultrafast", "-crf", "22",
             "-pix_fmt", "yuv420p", f"shot_{i:04d}.mp4"], work)
        if progress:
            progress(i + 1, len(timeline["shots"]) + 2, "正在渲染镜头")
    for i, item in enumerate(timeline["narration"]):
        duration = (item["end_frame"] - item["start_frame"]) / FPS
        run(["-i", str(item["audio_path"]), "-af", f"apad,atrim=duration={duration},asetpts=PTS-STARTPTS",
             "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", f"audio_{i:04d}.wav"], work)
    (work / "video.txt").write_text("\n".join(f"file 'shot_{i:04d}.mp4'" for i in range(len(timeline["shots"]))), encoding="utf-8")
    (work / "audio.txt").write_text("\n".join(f"file 'audio_{i:04d}.wav'" for i in range(len(timeline["narration"]))), encoding="utf-8")
    (work / "subtitles.srt").write_text(srt_text(timeline), encoding="utf-8")
    (work / 'captions.ass').write_text(ass_text(timeline),encoding='utf-8-sig')
    shutil.copyfile(work / "subtitles.srt", root / "subtitles.srt")
    run(["-f", "concat", "-safe", "0", "-i", "video.txt", "-c", "copy", "visual.mp4"], work)
    target = root / "video.mp4"
    run(["-i", "visual.mp4", "-f", "concat", "-safe", "0", "-i", "audio.txt",
         "-map", "0:v:0", "-map", "1:a:0", "-vf",
         "ass=captions.ass",
         "-frames:v", str(timeline["total_frames"]), "-r", str(FPS), "-c:v", "libx264",
         "-threads", "2", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(target)], work)
    report = inspect_export(target, timeline)
    report['presentation'] = {'subtitle_canvas':[1080,1920],'subtitle_font_size':52,'subtitle_safe_area':True,
                               'art_box':[40,180,1000,1390], 'shot_checks':visual_checks,
                               'sparse_shots':[x['shot_id'] for x in visual_checks if x['sparse_content']],
                               'semantic_review':'requires_review; local image statistics do not prove story quality'}
    if progress:
        progress(len(timeline["shots"]) + 2, len(timeline["shots"]) + 2, "视频导出及完整解码校验完成")
    # Intermediate files are intentionally kept within this unique export only.
    return report
