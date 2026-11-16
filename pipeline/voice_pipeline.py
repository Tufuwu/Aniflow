"""Versioned voice planning, auditions and human acceptance for the shot editor."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from uuid import uuid4
import config

from pipeline.shot_selector import require
from pipeline.story_bible import atomic_json, digest, StoryError
from pipeline.voice_director import VoiceDirector, PRESETS, validate_plan, plan_hash
from pipeline.voice_audio import synthesize_performance, audio_review
from pipeline.tts_profiles import voice_cache_options


def voice_options(options):
    require(isinstance(options, dict) and not (set(options) - {
        "engine", "voice", "model", "rate", "pitch", "speed", "stability", "similarity_boost", "seed"}), "配音参数无效")
    edge = options.get('engine', 'edge') == 'edge'
    opts = {"engine": "edge", "model": "eleven_v4", "voice": config.TTS_VOICE if edge else "",
            "rate": config.TTS_RATE if edge else "+0%", "pitch": config.TTS_PITCH if edge else "+0Hz",
            "speed": 1., "stability": .5, "similarity_boost": .75, **options}
    require(opts["engine"] in {"edge", "elevenlabs"}, "情绪配音目前支持 ElevenLabs 或 Edge 基础预演")
    require(isinstance(opts["voice"], str) and re.fullmatch(r"[A-Za-z0-9_-]{1,120}", opts["voice"]), "请提供有效的声音 ID")
    if opts["engine"] == "elevenlabs":
        require(opts["model"] in {"eleven_v4", "eleven_v3"}, "情绪配音请选择 Eleven v4 或 v3")
        require(opts["rate"] == "+0%" and opts["pitch"] == "+0Hz" and opts["speed"] == 1.,
                "情绪模型通过表演计划控制语速，不接收基础语速和音调参数")
    for key in ("stability", "similarity_boost"):
        require(type(opts[key]) in (int, float) and math.isfinite(opts[key]) and 0 <= opts[key] <= 1, "声音参数应在0到1之间")
    for key, pattern in (("rate", r"[+-]\d{1,3}%"), ("pitch", r"[+-]\d{1,3}Hz")):
        require(isinstance(opts[key], str) and re.fullmatch(pattern, opts[key]), "基础语速或音调无效")
    require(type(opts["speed"]) in (int, float) and opts["speed"] == 1., "请在逐句表演计划中设置语速")
    require("seed" not in opts or type(opts["seed"]) is int and 0 <= opts["seed"] <= 4294967295, "随机种子无效")
    return opts


class VoicePipelineMixin:
    def _voice_segment(self, state, nid):
        source = self.source(state)
        segment = next((s for s in source["script"]["segments"] if s["segment_id"] == nid), None)
        require(segment is not None, "旁白不存在")
        return source, segment

    def _voice_invalidate(self, state, nid):
        state.get("audio", {}).pop(nid, None)
        self.invalidate(state)

    def plan_voice(self, revision, style, use_model=True):
        with self.store.lock:
            state = self._current(revision)
            source = self.source(state)
            require(style in PRESETS, "配音风格无效")
            previous = (state.get("voice_direction") or {}).get("plans", {})
            plans = {}
            for segment in source["script"]["segments"]:
                nid = segment["segment_id"]
                old = previous.get(nid)
                if old and old.get("locked"):
                    plans[nid] = deepcopy(old)
                    continue
                plans[nid] = VoiceDirector(self.model).plan(source, segment, style, use_model)
                if not old or plan_hash(old) != plan_hash(plans[nid]):
                    state.get("voice_selected", {}).pop(nid, None)
                    self._voice_invalidate(state, nid)
            state["voice_direction"] = {"version": 1, "style": style, "plans": plans}
            return self.store.save(state, "plan_voice", revision)

    def review_voice(self, revision, nid, action, data):
        with self.store.lock:
            state = self._current(revision)
            source, segment = self._voice_segment(state, nid)
            plan = (state.get("voice_direction") or {}).get("plans", {}).get(nid)
            require(plan is not None, "请先生成配音计划")
            if action == "update":
                require(isinstance(data.get("units"), list), "请提交逐句表演计划")
                # Only performance fields are editable. Original text and event ownership stay immutable.
                units = deepcopy(plan["units"])
                require(len(units) == len(data["units"]), "不能在审核界面增删原文片段")
                for original, proposed in zip(units, data["units"]):
                    require(isinstance(proposed, dict), "表演单元格式无效")
                    if "text" in proposed:
                        require(proposed["text"] == original["text"], "不能改写已审核原文")
                    for field in ("emotion", "intensity", "rate", "emphasis", "pause_after_ms", "reason"):
                        if field in proposed:
                            original[field] = proposed[field]
                plan.update(units=units, locked=True, review_status="pending")
                validate_plan(plan, segment, source)
                state.get("voice_selected", {}).pop(nid, None)
                self._voice_invalidate(state, nid)
            elif action == "approve":
                validate_plan(plan, segment, source)
                plan.update(locked=True, review_status="approved")
            elif action == "unlock":
                plan.update(locked=False, review_status="pending")
                state.get("voice_selected", {}).pop(nid, None)
                self._voice_invalidate(state, nid)
            elif action == "select_take":
                take = self._voice_take(state, nid, data.get("take_id"))
                candidate = deepcopy(take["plan"])
                validate_plan(candidate, segment, source)
                if plan_hash(candidate) == plan_hash(plan):
                    candidate["review_status"] = plan["review_status"]
                else:
                    candidate["review_status"] = "pending"
                candidate["locked"] = True
                state["voice_direction"]["plans"][nid] = candidate
                state.setdefault("voice_selected", {})[nid] = take["take_id"]
                self._voice_invalidate(state, nid)
            elif action == "approve_audio":
                take = self._voice_take(state, nid, state.get("voice_selected", {}).get(nid))
                require(plan["review_status"] == "approved" and take["plan_hash"] == plan_hash(plan), "请先审核对应表演计划")
                require(data.get("listened") is True, "请先试听并确认没有漏读、错读或把情绪标签念出")
                note = data.get("review_note", "")
                require(isinstance(note, str) and len(note) <= 1000, "审核说明过长")
                quality = take["quality"]
                require(not (quality["transcript_status"] == "needs_review" or quality["clipping_suspected"]) or bool(note.strip()),
                        "自动检查发现可疑读音或削波，请填写试听结论后再审核")
                take.update(review_status="approved", locked=True, review_note=note, human_listened=True)
            else:
                raise StoryError("配音审核操作无效")
            return self.store.save(state, f"voice_{action}:{nid}", revision)

    @staticmethod
    def _voice_take(state, nid, take_id):
        take = next((t for t in state.get("voice_samples", {}).get(nid, []) if t["take_id"] == take_id), None)
        require(take is not None, "试听版本不存在，请先生成或选择试听")
        return take

    def audition_voice(self, revision, nid, options, compare=False, fresh=False, progress=None):
        with self.store.lock:
            state = self._current(revision)
            source, segment = self._voice_segment(state, nid)
            current = (state.get("voice_direction") or {}).get("plans", {}).get(nid)
            require(current is not None, "请先生成配音计划")
            opts = voice_options(options)
            require(type(compare) is bool and type(fresh) is bool, "试听选项无效")
            plans = [current]
            if compare:
                # Explicitly labelled rule-based style comparisons; never replace the active/locked plan.
                plans = [VoiceDirector().plan(source, segment, style, False) for style in PRESETS]
            for i, plan in enumerate(plans):
                validate_plan(plan, segment, source)
                key = digest({"voice_plan": plan_hash(plan), "options": voice_cache_options(opts), "version": 1,
                              "nonce": uuid4().hex if fresh else None})
                path = self.store.root / "media" / "audio" / f"{key}.wav"
                manifest = path.with_suffix(".json")
                if not (path.is_file() and manifest.is_file()):
                    temporary = path.with_name(f"{key}.{uuid4().hex}.wav")
                    try:
                        info = synthesize_performance(plan, temporary, opts, self.speech)
                        quality = audio_review(temporary, segment["text"], getattr(self, "transcriber", None))
                        sha = hashlib.sha256(temporary.read_bytes()).hexdigest()
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temporary.replace(path)
                        atomic_json(manifest, {**info, "sha256": sha, "quality": quality})
                    finally:
                        temporary.unlink(missing_ok=True)
                info = json.loads(manifest.read_text(encoding="utf-8"))
                require(hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"], "试听音频缓存校验失败")
                samples = state.setdefault("voice_samples", {}).setdefault(nid, [])
                if not any(t["key"] == key for t in samples):
                    samples.append({**info, "take_id": uuid4().hex, "key": key, "path": str(path), "format": "wav",
                                    "plan": deepcopy(plan), "plan_hash": plan_hash(plan), "options": opts,
                                    "review_status": "pending", "locked": False})
                if progress:
                    progress(i+1, len(plans), "正在生成配音试听；已有审核配音保持不变")
            state["voice_options"] = opts
            return self.store.save(state, f"voice_audition:{nid}", revision)

    def accepted_voice_audio(self, state):
        require(bool(state.get("voice_direction")), "尚无配音计划")
        source = self.source(state)
        audio = {}
        for segment in source["script"]["segments"]:
            nid = segment["segment_id"]
            plan = state["voice_direction"]["plans"].get(nid)
            require(plan is not None, f"缺少配音计划：{nid}")
            validate_plan(plan, segment, source)
            take = self._voice_take(state, nid, state.get("voice_selected", {}).get(nid))
            require(plan["review_status"] == "approved" and take["review_status"] == "approved"
                    and take.get("human_listened") and take["plan_hash"] == plan_hash(plan), f"请完成本段配音试听审核：{nid}")
            require(hashlib.sha256(Path(take["path"]).read_bytes()).hexdigest() == take["sha256"], "已审核配音已变化")
            audio[nid] = {k: deepcopy(take[k]) for k in ("key", "path", "duration", "sha256", "format", "unit_timings", "sample_rate")}
            audio[nid]["take_id"] = take["take_id"]
        return audio
