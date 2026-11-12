"""A reusable Flask blueprint: main AniFlow and lightweight text-only review."""
from __future__ import annotations

import json
from pathlib import Path
import re
import threading
from uuid import uuid4

from flask import Blueprint, jsonify, render_template, request, send_file
from filelock import Timeout
import config
from pipeline.story_bible import StoryBible, StoryConflict, StoryError
from pipeline.story_model import StoryModel
from pipeline.story_pipeline import StoryPipeline


def story_store(project):
    if not re.fullmatch(r"[\w\-\u4e00-\u9fff]{1,100}", project):
        raise StoryError("项目名只能包含中英文、数字、下划线和连字符")
    return StoryBible(Path(config.STORY_DIR) / project / "Chapter_00001")


def local_inputs(project):
    store = story_store(project)
    path = store.root / "inputs.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def create_story_blueprint(panel_provider=None, model_factory=StoryModel):
    bp = Blueprint("story", __name__)
    jobs, active, guard = {}, set(), threading.Lock()

    @bp.errorhandler(StoryError)
    def story_error(exc):
        return jsonify(error=str(exc)), 409 if isinstance(exc, StoryConflict) else 422

    def service(project):
        store = story_store(project)
        return StoryPipeline(store, model_factory(store))

    def start(project, action, segment_id=None):
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            raise StoryError("请求必须是 JSON 对象")
        if "revision" not in data:
            raise StoryError("缺少版本号，请刷新后重试")
        if segment_id is not None:
            data["segment_id"] = segment_id
        store = story_store(project)
        # Refuse stale requests before scheduling, and check again inside the lock.
        try:
            with store.lock.acquire(timeout=0):
                current = store.read()
        except Timeout as exc:
            raise StoryConflict("当前章节正在其他窗口处理，请等待完成") from exc
        if data["revision"] != current["revision"]:
            raise StoryConflict("内容已变化，请刷新后重试")
        if action in {"analyze", "generate", "prepare", "confirm_panels"}:
            StoryPipeline._unlocked(current)
        imported = local_inputs(project)
        if action == "approve" and imported and imported["input_kind"] != "panels":
            raise StoryConflict("请先核对输入完整性与阅读顺序，再确认分镜；当前文案可预览和编辑")
        if action in {"edit", "regenerate", "reject", "approve"} and segment_id:
            segment = next((s for s in (current.get("script") or {}).get("segments", [])
                            if s["segment_id"] == segment_id), None)
            if segment and segment["review_status"] == "approved":
                raise StoryConflict("此段已审核并锁定，不能覆盖")
        key = str(store.root)
        with guard:
            if key in active:
                raise StoryConflict("当前章节正在处理，请等待完成")
            active.add(key)
            job_id = uuid4().hex
            jobs[job_id] = {"job_id": job_id, "project": project, "status": "running",
                            "current": 0, "total": 0, "message": "正在准备"}

        def progress(current, total, message):
            with guard:
                jobs[job_id].update(current=current, total=total, message=message)

        def work():
            if store.read()["revision"] != data["revision"]:
                raise StoryConflict("内容已变化，请刷新后重试")
            pipeline = service(project)
            if action == "prepare":
                from pipeline.story_inputs import prepare_panels
                prepare_panels(store, progress)
                result = store.read()
            elif action == "confirm_panels":
                from pipeline.story_bible import atomic_json
                imported = local_inputs(project)
                if not imported or imported["input_kind"] != "detected_panels":
                    raise StoryError("没有待确认的分镜")
                imported["input_kind"] = "panels"
                atomic_json(store.root / "inputs.json", imported)
                result = store.read()
            elif action == "analyze":
                imported = local_inputs(project)
                if imported:
                    if imported["input_kind"] not in {"panels", "detected_panels"}:
                        panels = panel_provider(project) if panel_provider else []
                        if not panels:
                            raise StoryError("请先确认分镜顺序与完整性，原始切片不能直接作为完整分镜")
                    else:
                        panels = imported["panels"]
                else:
                    panels = panel_provider(project) if panel_provider else []
                result = pipeline.analyze(panels, data["revision"], progress)
            elif action == "generate":
                result = pipeline.generate(data["revision"], data.get("style", "faithful"),
                                           data.get("max_events", 12), progress)
            else:
                result = pipeline.segment_action(data["segment_id"], action, data["revision"], data.get("text"))
            return result

        def run():
            try:
                with store.lock:
                    result = work()
                with guard:
                    jobs[job_id].update(status="completed", revision=result["revision"], message="处理完成")
            except Exception as exc:
                with guard:
                    jobs[job_id].update(status="failed", error=str(exc), message="处理失败；已保存的结果仍可查看")
            finally:
                with guard:
                    active.discard(key)

        threading.Thread(target=run, daemon=True, name=f"story-{job_id}").start()
        return jsonify(job_id=job_id, status="running"), 202

    @bp.get("/story-review")
    def story_review():
        return render_template("story_review.html")

    @bp.get("/api/story/<project>")
    def get_story(project):
        store = story_store(project)
        state = store.read(request.args.get("revision"))
        imported = local_inputs(project)
        # Do not expose arbitrary paths as an image-serving API; only snapshot IDs.
        state["input"] = {"kind": imported["input_kind"], "count": len(imported["panels"])} if imported else None
        state["history"] = store.history()
        with guard:
            state["active_job"] = next((dict(j) for j in jobs.values()
                                          if j["project"] == project and j["status"] == "running"), None)
        return jsonify(state)

    @bp.get("/api/story/<project>/jobs/<job_id>")
    def job_status(project, job_id):
        with guard:
            job = jobs.get(job_id)
            if not job or job["project"] != project:
                return jsonify(error="任务不存在；若服务重启，请刷新查看已保存结果"), 404
            return jsonify(dict(job))

    @bp.post("/api/story/<project>/analyze")
    def analyze(project):
        return start(project, "analyze")

    @bp.post("/api/story/<project>/generate")
    def generate(project):
        return start(project, "generate")

    @bp.post("/api/story/<project>/prepare")
    def prepare(project):
        return start(project, "prepare")

    @bp.post("/api/story/<project>/confirm_panels")
    def confirm_panels(project):
        return start(project, "confirm_panels")

    @bp.get("/api/story/<project>/inputs/<int:index>")
    def input_image(project, index):
        store = story_store(project)
        imported = local_inputs(project)
        if not imported or index < 0 or index >= len(imported["panels"]):
            return jsonify(error="输入图片不存在"), 404
        # Only registered images; never accept an arbitrary client-supplied path.
        image = imported["panels"][index]
        return send_file(store.evidence_path(store.snapshot(image["path"])), conditional=True)

    @bp.post("/api/story/<project>/segments/<segment_id>/<action>")
    def update_segment(project, segment_id, action):
        if action not in {"edit", "approve", "reject", "regenerate", "check"}:
            raise StoryError("不支持的操作")
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            raise StoryError("请求必须是 JSON 对象")
        return start(project, action, segment_id)

    @bp.get("/api/story/<project>/evidence/<panel_id>")
    def evidence(project, panel_id):
        store = story_store(project)
        state = store.read(request.args.get("revision"))
        panel = next((p for p in (state.get("bible") or {}).get("panels", [])
                      if p["source_panel_id"] == panel_id), None)
        if not panel:
            return jsonify(error="分镜证据不存在"), 404
        return send_file(store.evidence_path(panel), conditional=True)

    @bp.get("/api/story/<project>/export")
    def export(project):
        return jsonify(service(project).export())

    return bp
