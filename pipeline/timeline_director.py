"""Frame based direction measured from real audio, with explicit manual constraints."""
from copy import deepcopy
import math
from pathlib import Path
import re
import subprocess

from pipeline.story_bible import StoryError
from pipeline.shot_selector import require
from pipeline.semantic_grounding import measured_spans
from pipeline.shot_text_ranges import mapped_text_units

FPS = 30
MOTIONS = {"auto", "hold", "zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down"}
TRANSITIONS = {"cut", "fade"}


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def audio_duration(path):
    require(Path(path).is_file(), "TTS 音频不存在")
    # Count decoded samples, rather than trust a caller's estimate or MP3 bitrate header.
    result = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", str(path), "-map", "0:a:0",
                             "-ac", "1", "-ar", "48000", "-f", "s16le", "pipe:1"],
                            capture_output=True, timeout=180)
    require(result.returncode == 0 and bool(result.stdout), "无法解码 TTS 音频")
    return len(result.stdout) / (48000 * 2)


def validate_shot(shot):
    crop = shot.get("crop")
    require(isinstance(crop, list) and len(crop) == 4 and all(
        type(x) in (int, float) and math.isfinite(x) for x in crop), "裁切需提供归一化 x/y/w/h")
    x, y, w, h = crop
    require(x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= 1.000001 and y + h <= 1.000001,
            "裁切范围超出原图")
    require(shot.get("motion") in MOTIONS and shot.get("transition") in TRANSITIONS, "镜头运动或转场无效")
    frames = shot.get("duration_frames")
    require(frames is None or type(frames) is int and 1 <= frames <= 36000, "镜头时长应为1到36000帧或自动")


def subtitle_chunks(text, limit=18):
    parts = re.findall(r"[^，。！？；：,.!?;:]+[，。！？；：,.!?;:]?", text)
    output = []
    for part in parts:
        current, width = '', 0.
        for word in re.findall(r'[A-Za-z0-9]+|.',part):
            # Ordinary Latin words stay together; exceptionally long tokens still fit.
            for offset in range(0,len(word),limit):
                token = word[offset:offset+limit]
                units = len(token)
                if current and width+units > limit:
                    output.append(current); current,width = '',0.
                current += token; width += units
        if current.strip():
            output.append(current)
    return output


class TimelineDirector:
    def build(self, groups, audio, library):
        by_id = {p["panel_id"]: p for p in library}
        entries, narration, subtitles, cursor = [], [], [], 0
        semantic_units, semantic_issues, shot_text_units, shot_text_narration_ids = [], [], [], []
        for group in groups:
            nid = group["narration_id"]
            require(nid in audio, f"缺少旁白音频：{nid}")
            measured = audio[nid]["duration"]
            require(type(measured) in (int, float) and math.isfinite(measured) and measured > 0, "实际音频时长无效")
            frames = math.ceil(measured * FPS - 1e-8)
            shots = group["selected"]
            require(bool(shots), f"旁白 {nid} 没有镜头")
            for shot in shots:
                validate_shot(shot)
            units = group.get('semantic_units', [])
            editorial_units = mapped_text_units(group)
            use_text_ranges = bool(editorial_units) and not units
            if use_text_ranges:
                units = editorial_units
                shot_text_narration_ids.append(nid)
            spans, problem = measured_spans(units, audio[nid])
            if problem:
                require(not use_text_ranges, '图片文字范围无法与实测配音对齐：'+problem)
                semantic_issues.append({'narration_id':nid, 'detail':problem})
            else:
                scheduled = []
                require(all(any(s['panel_id'] in u['display_panel_ids'] and
                                (use_text_ranges or u['unit_id'] in s.get('unit_ids', []))
                                for u in spans) for s in shots), '所选镜头缺少当前语义单元关联')
                if use_text_ranges:
                    require(all(s['duration_frames'] is None or sum(
                        s['panel_id'] in u['display_panel_ids'] for u in spans)==1 for s in shots),
                        '跨多个文字范围的镜头请使用自动时长，或先调整范围后再指定时长')
                for i, span in enumerate(spans):
                    start = round(span['start']*FPS)
                    end = frames if i == len(spans)-1 else round(span['end']*FPS)
                    matching = [s for s in shots if (use_text_ranges or span['unit_id'] in s.get('unit_ids', []))
                                and s['panel_id'] in span['display_panel_ids']]
                    require(matching and end-start >= len(matching), '口播语义单元没有可用镜头或时长')
                    fixed = sum(s['duration_frames'] or 0 for s in matching)
                    automatic = sum(s['duration_frames'] is None for s in matching)
                    require(fixed <= end-start and (automatic or fixed == end-start)
                            and end-start-fixed >= automatic, '人工固定镜头时长与语义单元音频不符')
                    remaining = end-start-fixed
                    for j,s in enumerate(matching):
                        if s['duration_frames'] is None:
                            count = remaining // automatic
                            remaining -= count; automatic -= 1
                        else:
                            count = s['duration_frames']
                        scheduled.append({**deepcopy(s), 'shot_id':s['shot_id']+f':U{i+1}:{j+1}',
                                          'duration_frames':count, 'unit_ids':[span['unit_id']],
                                          'display_event_ids':list(by_id[s['panel_id']]['event_ids'])})
                    target_units = shot_text_units if use_text_ranges else semantic_units
                    target_units.append({**deepcopy(span), 'narration_id':nid,
                                         'start_frame':cursor+start, 'end_frame':cursor+end})
                shots = scheduled
            fixed = sum(s["duration_frames"] or 0 for s in shots)
            automatic = [s for s in shots if s["duration_frames"] is None]
            require(fixed <= frames and (bool(automatic) or fixed == frames),
                    f"旁白 {nid} 的人工固定镜头时长与实际音频不符，请调整时长")
            require(frames - fixed >= len(automatic), f"旁白 {nid} 的音频不足以安排全部镜头")
            importance = {r["panel_id"]: r["scores"]["importance"] for r in group["decisions"]}
            weights = [(1.4 if s.get("motion") in {"pan_up", "pan_down"} else 1.) +
                       .5 * importance.get(s["panel_id"], 0) for s in automatic]
            remaining, allocations = frames - fixed, {}
            for index, shot in enumerate(automatic):
                count = remaining if index == len(automatic) - 1 else max(
                    1, min(remaining - len(automatic) + index + 1,
                           round((frames - fixed) * weights[index] / sum(weights))))
                allocations[shot["shot_id"]] = count
                remaining -= count
            local = cursor
            for index, shot in enumerate(shots):
                count = shot["duration_frames"] or allocations[shot["shot_id"]]
                panel = by_id[shot["panel_id"]]
                motion = shot["motion"]
                if motion == "auto":
                    motion = "pan_down" if panel["height"] / panel["width"] > 2 else (
                        "zoom_in" if index % 2 == 0 else "zoom_out")
                entries.append({**deepcopy(shot), "motion": motion, "narration_id": nid,
                                "event_id": group["event_id"], "start_frame": local,
                                "end_frame": local + count, "frames": count,
                                "transition_frames": min(6, count // 4) if shot["transition"] == "fade" else 0})
                local += count
            narration.append({"narration_id": nid, "start_frame": cursor, "end_frame": cursor + frames,
                              "audio_duration": measured, "audio_path": audio[nid]["path"],
                              "audio_sha256": audio[nid]["sha256"], "text": group["text"]})
            units = audio[nid].get("unit_timings")
            if spans:
                spans = [(u['text'], cursor+round(u['start']*FPS),
                          cursor+frames if i==len(spans)-1 else cursor+round(u['end']*FPS))
                         for i,u in enumerate(spans)]
            elif units:
                sample_rate = audio[nid]["sample_rate"]
                require(''.join(u["text"] for u in units) == group["text"], "配音语义段与字幕原文不符")
                require(abs(units[-1]["end_sample"] / sample_rate - measured) < .002, "配音语义段时长不符")
                spans, sample_cursor = [], 0
                for i, unit in enumerate(units):
                    require(unit["start_sample"] == sample_cursor and
                            unit["start_sample"] < unit["speech_end_sample"] <= unit["end_sample"], "语义段音频边界无效")
                    spans.append((unit["text"], cursor + round(unit["start_sample"] * FPS / sample_rate),
                                  cursor + frames if i == len(units)-1 else cursor + round(unit["end_sample"] * FPS / sample_rate)))
                    sample_cursor = unit["end_sample"]
            else:
                spans = [(group["text"], cursor, cursor + frames)]
            for text, span_start, span_end in spans:
                parts = subtitle_chunks(text)
                span_frames = span_end - span_start
                require(len(parts) <= span_frames, "字幕过长，无法在实际音频内排布")
                total, boundary, cumulative = sum(map(len, parts)), span_start, 0
                for i, part in enumerate(parts):
                    cumulative += len(part)
                    end = span_end if i == len(parts)-1 else min(
                        span_end - (len(parts)-i-1), max(boundary+1, span_start + round(span_frames*cumulative/total)))
                    subtitles.append({"narration_id": nid, "text": part, "start_frame": boundary, "end_frame": end})
                    boundary = end
            cursor += frames
        return {"fps": FPS, "width": 1080, "height": 1920, "codec": "h264", "total_frames": cursor,
                "duration": cursor / FPS, "shots": entries, "narration": narration, "subtitles": subtitles,
                "semantic_units":semantic_units, "semantic_issues":semantic_issues,
                'shot_text_units':shot_text_units, 'shot_text_narration_ids':shot_text_narration_ids,
                "subtitle_alignment": "measured_shot_text_ranges" if shot_text_units else "measured_semantic_units" if semantic_units or any(a.get("unit_timings") for a in audio.values()) else "phrase_proportional_to_measured_audio",
                "subtitle_note": "按图片文字范围和实测配音边界安排字幕；范围内按短语分配" if shot_text_units else "按实测语义段音频安排字幕；段内按短语分配，非字级强制对齐" if semantic_units or any(a.get("unit_timings") for a in audio.values()) else
                                 "短语按实际音频时长分配；不是字级强制对齐", "sync_tolerance": 1 / FPS}


def semantic_alignment(timeline, manual_narration_ids=()):
    manual = set(manual_narration_ids) & {n['narration_id'] for n in timeline['narration']}
    issues = [deepcopy(i) for i in timeline.get('semantic_issues', []) if i['narration_id'] not in manual]
    for item in timeline['narration']:
        nid = item['narration_id']
        if nid in manual:
            continue  # Human-reviewed narration uses the existing measured segment timeline.
        units = [u for u in timeline.get('semantic_units', []) if u['narration_id']==nid]
        if not units:
            issues.append({'narration_id':nid, 'detail':'没有实测的句子—事件—画面对应关系'})
            continue
        if ''.join(u['text'] for u in units) != item['text']:
            issues.append({'narration_id':nid, 'detail':'语义单元与当前正文不符'})
        boundary = item['start_frame']
        for u in units:
            overlapping = [s for s in timeline['shots'] if s['start_frame'] < u['end_frame']
                           and s['end_frame'] > u['start_frame']]
            if u['start_frame'] != boundary or u['end_frame'] <= boundary:
                issues.append({'narration_id':nid, 'unit_id':u['unit_id'], 'detail':'语义区间不连续'})
            boundary = u['end_frame']
            if not overlapping or any(s['narration_id'] != nid or s['panel_id'] not in u['display_panel_ids']
                                      or u['unit_id'] not in s.get('unit_ids', []) for s in overlapping):
                issues.append({'narration_id':nid, 'unit_id':u['unit_id'], 'detail':'当前句子与展示镜头错配'})
            covered = {e for s in overlapping for e in s.get('display_event_ids', [])}
            if not set(u['event_ids']) <= covered:
                issues.append({'narration_id':nid, 'unit_id':u['unit_id'], 'detail':'正文事实缺少区间内展示依据'})
        if boundary != item['end_frame']:
            issues.append({'narration_id':nid, 'detail':'语义区间未覆盖完整配音'})
    return {'status':'needs_review' if issues else 'manual_review' if manual else 'aligned', 'issues':issues,
            'note':'人工审核文案按整段实际配音排轴，镜头内容和节奏由人工确认' if manual else
                   '依据模型来源标注与实测边界检查；仍需人工观看确认内容和节奏'}


def validate_timeline(timeline, require_semantic=False, manual_narration_ids=()):
    cursor = 0
    for shot in timeline["shots"]:
        require(shot["start_frame"] == cursor and shot["end_frame"] > cursor, "时间轴镜头有空洞或重叠")
        cursor = shot["end_frame"]
    require(cursor == timeline["total_frames"] and cursor > 0, "时间轴总帧数不符")
    cursor, max_error = 0, 0.
    for item in timeline["narration"]:
        require(item["start_frame"] == cursor, "旁白时间轴不连续")
        error = (item["end_frame"] - cursor) / FPS - item["audio_duration"]
        require(-1e-6 <= error < 1 / FPS + 1e-6, "音画同步超出一帧")
        max_error = max(max_error, abs(error))
        shots = [s for s in timeline["shots"] if s["narration_id"] == item["narration_id"]]
        require(bool(shots) and shots[0]["start_frame"] == cursor and shots[-1]["end_frame"] == item["end_frame"],
                "镜头未覆盖对应旁白")
        subs = [s for s in timeline["subtitles"] if s["narration_id"] == item["narration_id"]]
        sub_cursor = cursor
        for sub in subs:
            require(sub["start_frame"] == sub_cursor and sub["end_frame"] > sub_cursor, "字幕未连续覆盖旁白")
            sub_cursor = sub["end_frame"]
        require(sub_cursor == item["end_frame"], "字幕未完整覆盖旁白")
        cursor = item["end_frame"]
    require(cursor == timeline["total_frames"], "旁白未完整覆盖成片")
    semantic = semantic_alignment(timeline, manual_narration_ids)
    text_alignment = shot_text_alignment(timeline)
    require(not text_alignment['issues'], '图片与原文范围排轴无效：'+str(text_alignment['issues']))
    require(not require_semantic or not semantic['issues'], '句子与镜头语义对应尚未通过：'+str(semantic['issues']))
    return {"max_audio_visual_error": max_error, "timeline_complete": True, 'semantic_alignment':semantic,
            'shot_text_alignment':text_alignment}


def shot_text_alignment(timeline):
    """Check editorial text intervals separately from factual grounding."""
    required = timeline.get('shot_text_narration_ids', [])
    issues = []
    for nid in required:
        item = next((n for n in timeline['narration'] if n['narration_id']==nid),None)
        units = [u for u in timeline.get('shot_text_units',[]) if u['narration_id']==nid]
        if not item or not units or ''.join(u['text'] for u in units)!=item['text']:
            issues.append({'narration_id':nid,'detail':'图片文字范围未完整覆盖当前正文'})
            continue
        boundary, char_boundary = item['start_frame'], 0
        for unit in units:
            if (unit['start_frame']!=boundary or unit['end_frame']<=boundary or
                    unit['start_char']!=char_boundary or item['text'][unit['start_char']:unit['end_char']]!=unit['text']):
                issues.append({'narration_id':nid,'detail':'图片文字区间或实测时间不连续'})
            boundary, char_boundary = unit['end_frame'], unit['end_char']
            shots = [s for s in timeline['shots'] if s['start_frame']<unit['end_frame'] and s['end_frame']>unit['start_frame']]
            if not shots or any(s['narration_id']!=nid or s['panel_id'] not in unit['display_panel_ids']
                                or unit['unit_id'] not in s.get('unit_ids',[]) for s in shots):
                issues.append({'narration_id':nid,'detail':'图片没有展示在它对应的原文时间范围内'})
        if boundary!=item['end_frame'] or char_boundary!=len(item['text']):
            issues.append({'narration_id':nid,'detail':'图片文字区间未覆盖完整配音'})
    return {'status':'needs_review' if issues else 'aligned' if required else 'not_run','issues':issues,
            'note':'模型图片文字关联与实测配音边界；不表示完成事实核验'}
