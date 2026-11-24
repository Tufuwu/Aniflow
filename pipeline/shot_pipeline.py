"""Single-chapter, revisioned editing layered beside the immutable story store."""
from copy import deepcopy
import hashlib
from pathlib import Path
import re
from uuid import uuid4
import config

from pipeline.story_bible import StoryBible, StoryConflict, StoryError, atomic_json, digest
from pipeline.shot_selector import (ShotSelectorAgent, GlobalShotOptimizer, build_library,
                                    coverage_report, required_events, require)
from pipeline.timeline_director import TimelineDirector, audio_duration, validate_shot, validate_timeline
from pipeline.voice_pipeline import VoicePipelineMixin
from pipeline.semantic_grounding import attach_tts_boundaries
from pipeline.human_review import manually_approved, segment_approved
from pipeline.shot_text_ranges import build_mapping, mapped_text_units, validate_ranges, validate_context_ranges
from pipeline.tts_profiles import voice_cache_options
from pipeline.image_cleanup import ImageCleanupMixin, display_assets


class ShotEditingPipeline(ImageCleanupMixin, VoicePipelineMixin):
    def __init__(self, story_store, model=None, speech=None, transcriber=None):
        require(story_store.root.name == "Chapter_00001", "当前仅支持单话验证，不支持批量章节")
        self.story = story_store
        self.store = StoryBible(story_store.root / "editing")
        self.model, self.speech = model, speech
        self.transcriber = transcriber

    @staticmethod
    def approved(source):
        segments = (source.get("script") or {}).get("segments", [])
        beats = (source.get("plan") or {}).get("beats", [])
        require(source.get("bible") and segments and beats, "请先完成剧情分析与解说稿")
        require(len(segments) == len(beats) and len({s["segment_id"] for s in segments}) == len(segments)
                and {s["event_id"] for s in segments} == {b["event_id"] for b in beats}, "文案与事件规划不完整")
        require(all(segment_approved(source['bible'], source['plan'], s) for s in segments),
                "所有解说段落必须先人工审核锁定；文案或来源变化后需重新审核")
        # A milestone must be in a narrated event or its explicit causal context.
        required = {e for s in segments for e in required_events(source, s)}
        milestones = {eid for eid in (source["plan"].get("milestones") or {}).values() if eid}
        milestones |= {source["plan"].get(k) for k in ("opening_event_id", "climax_event_id", "ending_event_id")
                       if source["plan"].get(k)}
        require(milestones <= required, "文案缺少关键里程碑，请先补充剧情审核")

    def read(self, revision=None):
        state = self.store.read(revision)
        if not state.get("source_revision"):
            source = self.story.read()
            segments = (source.get("script") or {}).get("segments", [])
            state["readiness"] = {"segments": len(segments), "approved": sum(
                s.get("review_status") == "approved" for s in segments),
                "source_revision": source["revision"]}
        return state

    def _current(self, revision):
        state = self.store.read()
        if state["revision"] != revision:
            raise StoryConflict("剪辑版本已变化，请刷新后重试")
        return state

    def source(self, state):
        require(bool(state.get("source_revision")), "请先生成镜头候选")
        source = self.story.read(state["source_revision"])
        self.approved(source)
        return source

    @staticmethod
    def manual_narration_ids(source):
        return [s['segment_id'] for s in source['script']['segments']
                if manually_approved(source['bible'], s)]

    @staticmethod
    def invalidate(state):
        state.update(timeline=None, export=None, stage="shot_review")

    def select(self, revision, source_revision, use_model=False):
        with self.store.lock:
            state = self._current(revision)
            if state.get("source_revision"):
                source = self.source(state)
                require(source_revision == state["source_revision"], "剪辑已绑定剧情版本，不能覆盖为其他文案")
            else:
                source = self.story.read()
                require(source["revision"] == source_revision, "剧情版本已变化，请刷新后重试")
                self.approved(source)
                state.update(schema_version=1, source_revision=source["revision"],
                             bible_id=source["bible"]["bible_id"], chapter=1, groups=[], audio={})
            library = build_library(self.story, source)
            require(not use_model or self.model is not None, "镜头模型未配置")
            previous = {g["narration_id"]: g for g in state.get("groups", [])}
            selector = ShotSelectorAgent(self.model if use_model else None)
            groups = GlobalShotOptimizer().optimize(source, library, selector, previous)
            # Locked human crops must not be changed by a rerun.
            unlocked = [g for g in groups if not g.get('locked')]
            visual_review = selector.review_selected(source,library,unlocked) if unlocked else {'status':'human_locked'}
            state.update(library=library, groups=groups, metrics=coverage_report(groups, library),
                         selector="metadata+model" if use_model else "evidence_metadata", visual_review=visual_review,
                         voice_review_required=bool(self.manual_narration_ids(source)))
            self.invalidate(state)
            return self.store.save(state, "select_shots", revision)

    def review(self, nid, action, revision, data):
        with self.store.lock:
            state = self._current(revision)
            self.source(state)
            group = next((g for g in state["groups"] if g["narration_id"] == nid), None)
            require(group is not None, "旁白镜头组不存在")
            if action == "update":
                proposed = data.get("selected")
                require(isinstance(proposed, list) and 0 < len(proposed) <= 100, "每段需选择1到100个镜头")
                known = {r["panel_id"] for r in group["decisions"] if r["eligible"]}
                existing = {s["panel_id"]: s for s in group["selected"]}
                selected, seen = [], set()
                for item in proposed:
                    require(isinstance(item, dict) and isinstance(item.get("panel_id"), str), "镜头格式无效")
                    pid = item["panel_id"]
                    require(pid in known and pid not in seen, "分镜重复或缺少当前事件证据")
                    seen.add(pid)
                    shot = deepcopy(existing.get(pid) or {"shot_id": "SH_" + digest([nid, pid])[:16],
                                  "panel_id": pid, "crop": [0, 0, 1, 1], "motion": "auto",
                                  "transition": "cut", "duration_frames": None})
                    for field in ("crop", "motion", "transition", "duration_frames"):
                        if field in item:
                            shot[field] = item[field]
                    if 'text_mapping' in group:
                        row = next(r for r in group['decisions'] if r['panel_id']==pid)
                        shot['text_ranges'] = validate_ranges(group['text'], item.get('text_ranges',
                            shot.get('text_ranges',row.get('text_ranges',[]))))
                        for key in ('context_only','context_origins'):
                            shot[key] = deepcopy(row.get(key,False if key=='context_only' else []))
                        validate_context_ranges(group['text'],shot['text_ranges'],shot)
                    validate_shot(shot)
                    panel = next(p for p in state["library"] if p["panel_id"] == pid)
                    require(round(shot["crop"][2] * panel["width"]) >= 1 and
                            round(shot["crop"][3] * panel["height"]) >= 1, "裁切区域小于一个像素")
                    shot["locked"] = True
                    if group.get('semantic_units'):
                        shot['unit_ids'] = [u['unit_id'] for u in group['semantic_units'] if pid in u['display_panel_ids']]
                    selected.append(shot)
                group.update(selected=selected, locked=True, review_status="pending")
                if 'text_mapping' in group:
                    group['text_mapping'] = build_mapping(group,version=2)
                report = coverage_report(state["groups"], state["library"])
                require(not report["missing_by_narration"] and not report['missing_semantic_units'],
                        "不能删除必要剧情画面；请先补足本句关键事件证据")
                for row in group["decisions"]:
                    row.update(selected=row["panel_id"] in seen, reason="人工选择并锁定" if row["panel_id"] in seen else "人工排除并锁定镜头组")
            elif action == "approve":
                mapped_text_units(group)
                group.update(locked=True, review_status="approved")
                for shot in group["selected"]:
                    shot["locked"] = True
                for row in group["decisions"]:
                    row["reason"] += "；人工审核通过并锁定"
            elif action == "unlock":
                group.update(locked=False, review_status="pending")
                for shot in group["selected"]:
                    shot["locked"] = False
            else:
                raise StoryError("不支持的镜头审核操作")
            state["metrics"] = coverage_report(state["groups"], state["library"])
            self.invalidate(state)
            return self.store.save(state, f"{action}:{nid}", revision)

    def direct(self, revision, options=None, progress=None):
        with self.store.lock:
            state = self._current(revision)
            source = self.source(state)
            require(state.get("groups"), "请先选择镜头")
            require(not self.manual_narration_ids(source) or state.get('voice_direction'),
                    '请先完成配音试听与审核，再安排时间轴')
            if state.get("voice_direction"):
                audio = self.accepted_voice_audio(state)
                timeline = TimelineDirector().build(state["groups"], audio, state["library"])
                state.update(audio=audio, timeline=timeline, export=None, stage="timeline_review")
                state["sync"] = validate_timeline(timeline, manual_narration_ids=self.manual_narration_ids(source))
                return self.store.save(state, "direct_reviewed_voice", revision)
            options = options or {}
            require(isinstance(options, dict) and not (set(options) - {"engine", "voice", "rate", "pitch", "speed"}),
                    "语音参数无效")
            opts = {"engine": "edge", "voice": config.TTS_VOICE, "rate": config.TTS_RATE,
                    "pitch": config.TTS_PITCH, "speed": 1., **options}
            require(opts["engine"] in {"edge", "kokoro", "elevenlabs"}, "不支持的 TTS 引擎")
            require(isinstance(opts["voice"], str) and 0 < len(opts["voice"]) <= 120, "语音名称无效")
            for field, pattern in (("rate", r"[+-]\d{1,3}%"), ("pitch", r"[+-]\d{1,3}Hz")):
                require(field not in opts or isinstance(opts[field], str) and re.fullmatch(pattern, opts[field]), "语速或音调格式无效")
            require("speed" not in opts or type(opts["speed"]) in (int, float) and .25 <= opts["speed"] <= 4, "语速无效")
            audio = {}
            for i, group in enumerate(state["groups"]):
                key = digest({"text": group["text"], "options": voice_cache_options(opts), "version": 1})
                path = self.store.root / "media" / "audio" / f"{key}.mp3"
                manifest = path.with_suffix(".json")
                if not (path.exists() and manifest.exists()):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    speech = self.speech
                    if speech is None:
                        from pipeline.tts_engine import generate_speech, KOKORO_AVAILABLE, ELEVENLABS_AVAILABLE
                        require(opts["engine"] != "kokoro" or KOKORO_AVAILABLE, "Kokoro 未安装，不能生成指定语音")
                        require(opts["engine"] != "elevenlabs" or ELEVENLABS_AVAILABLE, "ElevenLabs 未安装或配置")
                        speech = generate_speech
                    temporary = path.with_name(f"{key}.{uuid4().hex}.mp3")
                    try:
                        result = speech(group["text"], str(temporary), **opts)
                        duration = audio_duration(temporary)
                        info = {"duration":duration, "sha256":hashlib.sha256(temporary.read_bytes()).hexdigest(),
                                "options":voice_cache_options(opts), "text_hash":digest(group['text'])}
                        info = attach_tts_boundaries(info, temporary, group['text'], result)
                        temporary.replace(path)
                        atomic_json(manifest, info)
                    finally:
                        temporary.unlink(missing_ok=True)
                        temporary.with_suffix('.boundaries.json').unlink(missing_ok=True)
                        temporary.with_suffix('.srt').unlink(missing_ok=True)
                import json
                info = json.loads(manifest.read_text(encoding="utf-8"))
                require(hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"], "缓存音频校验失败")
                audio[group["narration_id"]] = {**attach_tts_boundaries(info,path,group['text']),
                                               "path": str(path), "key": key, 'duration':audio_duration(path)}
                if progress:
                    progress(i + 1, len(state["groups"]), "正在生成或复用实际旁白音频")
            timeline = TimelineDirector().build(state["groups"], audio, state["library"])
            state.update(audio=audio, tts_options=opts, timeline=timeline, export=None, stage="timeline_review")
            state["sync"] = validate_timeline(timeline, manual_narration_ids=self.manual_narration_ids(source))
            return self.store.save(state, "direct_timeline", revision)

    def export(self, revision, progress=None, quality_enhancement=None):
        with self.store.lock:
            state = self._current(revision)
            source = self.source(state)
            require(all(g["review_status"] == "approved" for g in state["groups"]), "请先逐段审核镜头选择与裁切")
            require(state.get("timeline"), "请先根据 TTS 音频生成时间轴")
            require(not coverage_report(state["groups"], state["library"])["missing_by_narration"], "存在关键剧情覆盖缺口")
            validate_timeline(state["timeline"], require_semantic=source['plan'].get('version',0)>=5,
                              manual_narration_ids=self.manual_narration_ids(source))
            if state.get("voice_direction"):
                require(self.accepted_voice_audio(state) == state["audio"], "已审核配音已变化，请重新安排时间轴")
            else:
                require(not self.manual_narration_ids(source), '请先完成配音试听与审核，再导出视频')
            # Rebuild from immutable inputs so a stale/tampered timeline cannot be exported.
            for audio in state["audio"].values():
                require(hashlib.sha256(Path(audio["path"]).read_bytes()).hexdigest() == audio["sha256"], "音频内容已变化")
            require(TimelineDirector().build(state["groups"], state["audio"], state["library"]) == state["timeline"],
                    "时间轴已过期，请重新生成")
            assets = display_assets(self.story, source, state, self.store.root)
            from pipeline.compositor import compose_story_timeline
            export_id = uuid4().hex
            directory = self.store.root / "exports" / export_id
            directory.parent.mkdir(parents=True, exist_ok=True)
            render_options = {"display_assets": assets} if assets else {}
            if quality_enhancement is not None:
                render_options["quality_enhancement"] = quality_enhancement
            report = compose_story_timeline(self.story, source, state["timeline"], directory, progress, **render_options)
            result = {"export_id": export_id, "editing_revision": revision, "source_revision": state["source_revision"],
                      "report": report, "metrics": state["metrics"], "sync": state["sync"],
                      "video_sha256": hashlib.sha256((directory / "video.mp4").read_bytes()).hexdigest()}
            atomic_json(directory / "manifest.json", {**result, "timeline": state["timeline"], "groups": state["groups"],
                "voice_direction": state.get("voice_direction"), "voice_selected": state.get("voice_selected"),
                "voice_samples": state.get("voice_samples"), "display_assets": assets,
                "image_cleanup": state.get("image_cleanup"), "quality_enhancement": quality_enhancement})
            state.update(export=result, stage="exported")
            return self.store.save(state, "export_video", revision)
