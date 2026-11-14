"""Single-chapter editing API, with background jobs and revision checked writes."""
from copy import deepcopy
from io import BytesIO
import re
import threading
from uuid import uuid4

from filelock import Timeout
from flask import Blueprint, jsonify, render_template, request, send_file
from PIL import Image

from pipeline.story_api import story_store
from pipeline.story_bible import StoryError, StoryConflict
from pipeline.shot_pipeline import ShotEditingPipeline
from pipeline.timeline_director import validate_shot
from pipeline.voice_director import PRESETS, EMOTIONS


def create_shot_blueprint(model_factory=None, speech=None, transcriber=None):
    bp = Blueprint("shots", __name__)
    jobs, active, guard = {}, set(), threading.Lock()

    @bp.errorhandler(StoryError)
    def error(exc):
        return jsonify(error=str(exc)), 409 if isinstance(exc, StoryConflict) else 422

    def service(project, use_model=False):
        store = story_store(project)
        if use_model:
            from pipeline.story_model import StoryModel
            model = (model_factory or StoryModel)(store)
        else:
            model = None
        return ShotEditingPipeline(store, model, speech, transcriber)

    @bp.get("/api/editing/voice-capabilities")
    def voice_capabilities():
        from pipeline.elevenlabs_tts import get_elevenlabs_tts
        return jsonify(presets=PRESETS, emotions=EMOTIONS, elevenlabs_ready=get_elevenlabs_tts().available,
                       expressive_models=["eleven_v4", "eleven_v3"], edge_note="仅调整语速及额外停顿；不模拟情绪或重音")

    @bp.post("/api/editing/voice-key")
    def voice_key():
        from pipeline.elevenlabs_tts import get_elevenlabs_tts
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not isinstance(data.get("key"), str) or not re.fullmatch(r"\S{1,512}", data["key"]):
            raise StoryError("请输入有效的 ElevenLabs 密钥")
        get_elevenlabs_tts().set_api_key(data["key"])
        return jsonify(configured=True)

    @bp.get("/api/editing/voices")
    def voices():
        from pipeline.elevenlabs_tts import get_elevenlabs_tts
        engine = get_elevenlabs_tts()
        if not engine.available:
            raise StoryError("请先配置 ElevenLabs 密钥")
        return jsonify(voices=[{"voice_id": v["voice_id"], "name": v["name"], "labels": v["labels"]}
                               for v in engine.get_voices(force_refresh=True)])

    @bp.get("/shot-review")
    def page():
        return render_template("shot_review.html")

    @bp.get("/api/editing/<project>")
    def state(project):
        svc = service(project)
        result = svc.read(request.args.get("revision"))
        result["history"] = svc.store.history()
        with guard:
            result["active_job"] = next((deepcopy(j) for j in jobs.values() if
                                        j["project"] == project and j["status"] == "running"), None)
        return jsonify(result)

    @bp.get("/api/editing/<project>/jobs/<job_id>")
    def job(project, job_id):
        with guard:
            item = jobs.get(job_id)
            if not item or item["project"] != project:
                return jsonify(error="任务不存在；重启后请刷新查看已保存版本"), 404
            return jsonify(deepcopy(item))

    def start(project, action, nid=None):
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or "revision" not in data:
            raise StoryError("请提交 JSON 对象及剪辑版本号")
        if data.get("use_model", False) not in (True, False):
            raise StoryError("模型复核开关无效")
        svc = service(project, data.get("use_model", action == "voice_plan"))
        try:
            with svc.store.lock.acquire(timeout=0):
                svc._current(data["revision"])
        except Timeout as exc:
            raise StoryConflict("当前章节正在处理，请稍候") from exc
        key = str(svc.store.root)
        with guard:
            if key in active:
                raise StoryConflict("当前章节正在处理，请稍候")
            active.add(key)
            jid = uuid4().hex
            jobs[jid] = {"job_id": jid, "project": project, "status": "running", "message": "正在准备", "current": 0, "total": 0}

        def progress(current, total, message):
            with guard:
                jobs[jid].update(current=current, total=total, message=message)

        def work():
            try:
                if action == "select":
                    result = svc.select(data["revision"], data.get("source_revision"), data.get("use_model", False))
                elif action == "direct":
                    result = svc.direct(data["revision"], data.get("tts"), progress)
                elif action == "export":
                    result = svc.export(data["revision"], progress)
                elif action == "voice_plan":
                    result = svc.plan_voice(data["revision"], data.get("style", "suspense"), data.get("use_model", True))
                elif action == "voice_audition":
                    result = svc.audition_voice(data["revision"], nid, data.get("tts", {}),
                                               data.get("compare", False), data.get("fresh", False), progress)
                elif action.startswith("voice_"):
                    result = svc.review_voice(data["revision"], nid, action[6:], data)
                else:
                    result = svc.review(nid, action, data["revision"], data)
                with guard:
                    jobs[jid].update(status="completed", revision=result["revision"], message="已保存新剪辑版本")
            except Exception as exc:
                with guard:
                    jobs[jid].update(status="failed", error=str(exc), message="处理失败；先前版本与原图已保留")
            finally:
                with guard:
                    active.discard(key)
        threading.Thread(target=work, daemon=True, name=f"editing-{jid}").start()
        return jsonify(job_id=jid, status="running"), 202

    @bp.post("/api/editing/<project>/select")
    def select(project):
        return start(project, "select")

    @bp.post("/api/editing/<project>/direct")
    def direct(project):
        return start(project, "direct")

    @bp.post("/api/editing/<project>/export")
    def export(project):
        return start(project, "export")

    @bp.post("/api/editing/<project>/groups/<nid>/<action>")
    def review(project, nid, action):
        if action not in {"update", "approve", "unlock"}:
            raise StoryError("不支持的审核操作")
        return start(project, action, nid)

    @bp.post("/api/editing/<project>/voice/plan")
    def plan_voice(project):
        return start(project, "voice_plan")

    @bp.post("/api/editing/<project>/voice/<nid>/<action>")
    def review_voice(project, nid, action):
        if action not in {"update", "approve", "unlock", "audition", "select_take", "approve_audio"}:
            raise StoryError("不支持的配音操作")
        return start(project, "voice_" + action, nid)

    @bp.get("/api/editing/<project>/evidence/<pid>")
    def evidence(project, pid):
        svc = service(project)
        source = svc.source(svc.read(request.args.get("revision")))
        panel = next((p for p in source["bible"]["panels"] if p["source_panel_id"] == pid), None)
        if panel is None:
            return jsonify(error="分镜不存在"), 404
        return send_file(svc.story.evidence_path(panel), conditional=True)

    @bp.get("/api/editing/<project>/audio/<nid>")
    def audio(project, nid):
        svc = service(project)
        item = svc.read(request.args.get("revision")).get("audio", {}).get(nid)
        if not item:
            return jsonify(error="音频不存在"), 404
        path = (svc.store.root / "media" / "audio" / f"{item['key']}.{item.get('format', 'mp3')}").resolve()
        if not path.is_relative_to(svc.store.root / "media" / "audio") or not path.is_file():
            raise StoryError("音频不存在")
        return send_file(path, conditional=True)

    @bp.get("/api/editing/<project>/voice-audio/<nid>/<take_id>")
    def voice_audio(project, nid, take_id):
        svc = service(project)
        take = svc._voice_take(svc.read(request.args.get("revision")), nid, take_id)
        if not re.fullmatch(r"[a-f0-9]{64}", take["key"]):
            raise StoryError("试听资源无效")
        path = svc.store.root / "media" / "audio" / f"{take['key']}.wav"
        if not path.is_file():
            return jsonify(error="试听音频不存在"), 404
        return send_file(path, conditional=True)

    @bp.get("/api/editing/<project>/preview/<nid>/<pid>")
    def crop_preview(project, nid, pid):
        svc = service(project)
        state = svc.read(request.args.get("revision"))
        source = svc.source(state)
        group = next((g for g in state["groups"] if g["narration_id"] == nid), None)
        shot = next((s for s in (group or {}).get("selected", []) if s["panel_id"] == pid), None)
        if shot is None:
            return jsonify(error="已选镜头不存在"), 404
        validate_shot(shot)
        panel = next(p for p in source["bible"]["panels"] if p["source_panel_id"] == pid)
        with Image.open(svc.story.evidence_path(panel)) as image:
            x, y, w, h = shot["crop"]
            cropped = image.crop((round(x * image.width), round(y * image.height),
                                  round((x + w) * image.width), round((y + h) * image.height)))
            stream = BytesIO()
            cropped.convert("RGB").save(stream, format="PNG")
        stream.seek(0)
        return send_file(stream, mimetype="image/png", conditional=True)

    @bp.get("/api/editing/<project>/exports/<export_id>/<asset>")
    def exported(project, export_id, asset):
        if not re.fullmatch(r"[a-f0-9]{32}", export_id) or asset not in {"video.mp4", "subtitles.srt", "manifest.json"}:
            return jsonify(error="导出资源不存在"), 404
        svc = service(project)
        root = svc.store.root / "exports" / export_id
        if not (root / "manifest.json").is_file() or not (root / asset).is_file():
            return jsonify(error="导出未完成或资源不存在"), 404
        return send_file(root / asset, conditional=True, as_attachment=request.args.get("download") == "1")

    return bp
