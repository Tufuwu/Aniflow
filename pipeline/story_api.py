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
from pipeline.narration_polish import polish_options


def story_store(project, chapter=1):
    if not re.fullmatch(r"[\w\-\u4e00-\u9fff]{1,100}", project):
        raise StoryError("项目名只能包含中英文、数字、下划线和连字符")
    if type(chapter) is not int or not 1 <= chapter <= 99999:
        raise StoryError("章节号应为1到99999")
    return StoryBible(Path(config.STORY_DIR) / project / f"Chapter_{chapter:05d}")


def local_inputs(project, chapter=1):
    store = story_store(project,chapter)
    path = store.root / "inputs.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def create_story_blueprint(panel_provider=None, model_factory=StoryModel):
    bp = Blueprint("story", __name__)
    jobs, active, guard = {}, set(), threading.Lock()

    @bp.errorhandler(StoryError)
    def story_error(exc):
        return jsonify(error=str(exc)), 409 if isinstance(exc, StoryConflict) else 422

    def service(project, chapter=1):
        store = story_store(project,chapter)
        return StoryPipeline(store, model_factory(store))

    def start(project, action, segment_id=None, chapter=1):
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            raise StoryError("请求必须是 JSON 对象")
        if action == 'generate' and {'candidate_count', 'candidate_id'} & data.keys():
            raise StoryError('已移除多候选参数，请刷新页面后生成单稿')
        if "revision" not in data:
            raise StoryError("缺少版本号，请刷新后重试")
        if action in {'generate','revise'} and 'polish_options' in data:
            data['polish_options'] = polish_options(data['polish_options'])
        if segment_id is not None:
            data["segment_id"] = segment_id
        store = story_store(project,chapter)
        # Refuse stale requests before scheduling, and check again inside the lock.
        try:
            with store.lock.acquire(timeout=0):
                current = store.read()
        except Timeout as exc:
            raise StoryConflict("当前章节正在其他窗口处理，请等待完成") from exc
        if data["revision"] != current["revision"]:
            raise StoryConflict("内容已变化，请刷新后重试")
        if action in {"analyze", "integrate", "generate", "revise", "prepare", "confirm_panels", "verify"}:
            StoryPipeline._unlocked(current)
        imported = local_inputs(project,chapter)
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
            jobs[job_id] = {"job_id": job_id, "project": project, "chapter":chapter, "status": "running",
                            "current": 0, "total": 0, "message": "正在准备"}

        def progress(current, total, message):
            with guard:
                jobs[job_id].update(current=current, total=total, message=message)

        def work():
            if store.read()["revision"] != data["revision"]:
                raise StoryConflict("内容已变化，请刷新后重试")
            pipeline = service(project,chapter)
            if action == "prepare":
                from pipeline.story_inputs import prepare_panels
                prepare_panels(store, progress)
                result = store.read()
            elif action == "confirm_panels":
                from pipeline.story_bible import atomic_json
                imported = local_inputs(project,chapter)
                if not imported or imported["input_kind"] != "detected_panels":
                    raise StoryError("没有待确认的分镜")
                imported["input_kind"] = "panels"
                atomic_json(store.root / "inputs.json", imported)
                result = store.read()
            elif action == "analyze":
                imported = local_inputs(project,chapter)
                if imported:
                    if imported["input_kind"] not in {"panels", "detected_panels"}:
                        panels = panel_provider(project) if panel_provider and chapter==1 else []
                        if not panels:
                            raise StoryError("请先确认分镜顺序与完整性，原始切片不能直接作为完整分镜")
                    else:
                        panels = imported["panels"]
                else:
                    panels = panel_provider(project) if panel_provider and chapter==1 else []
                result = pipeline.analyze(panels, data["revision"], progress)
            elif action == 'integrate':
                result = pipeline.integrate(data['revision'])
            elif action == "generate":
                result = pipeline.generate(data["revision"], data.get("style", "faithful"),
                                           data.get("max_events"), progress, data.get('polish_options'))
            elif action == 'verify':
                result = pipeline.verify(data['revision'])
            elif action == 'revise':
                result = pipeline.revise(data['revision'],data.get('polish_options'))
            elif action == 'memory':
                result = pipeline.remember(data['revision'])
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

    @bp.get('/api/story/<project>/context')
    def get_context(project):
        from pipeline.chapter_memory import series_notes
        return jsonify(series_notes(story_store(project)))

    @bp.put('/api/story/<project>/context')
    def update_context(project):
        from pipeline.chapter_memory import save_series_notes
        data = request.get_json(silent=True)
        if not isinstance(data,dict) or 'revision' not in data:
            raise StoryError('缺少背景资料版本号')
        return jsonify(save_series_notes(story_store(project),data.get('background',''),data.get('characters',''),data['revision']))

    @bp.post('/api/story/<project>/memory')
    @bp.post('/api/story/<project>/chapters/<int:chapter>/memory')
    def refresh_memory(project, chapter=1):
        return start(project,'memory',chapter=chapter)

    @bp.get("/api/story/<project>/chapters/<int:chapter>")
    @bp.get("/api/story/<project>")
    def get_story(project, chapter=1):
        store = story_store(project,chapter)
        state = store.read(request.args.get("revision"))
        imported = local_inputs(project,chapter)
        # Do not expose arbitrary paths as an image-serving API; only snapshot IDs.
        state["input"] = {"kind": imported["input_kind"], "count": len(imported["panels"])} if imported else None
        state["history"] = store.history()
        from pipeline.chapter_memory import series_notes, memory_usable, context_current
        state['chapter'] = chapter
        state['series_notes'] = series_notes(store)
        state['memory_usable'] = memory_usable(state) and context_current(store,state.get('bible') or {})
        from pipeline.character_profile import character_records, world_background
        bible = state.get('bible') or {}
        memory = state.get('chapter_memory') or {}
        from pipeline.chapter_memory import memory_basis
        current = memory.get('basis_hash') == memory_basis(state)
        # Read-only views also work for legacy revisions without migrating or
        # classifying their free-text records. An old draft's memory is not the
        # current character profile merely because it is still on disk.
        profile_memory = memory if current else bible.get('prior_context',{})
        usable = state['memory_usable'] if current else False
        state['character_records'] = character_records(bible,profile_memory.get('character_states',[]),usable,
                                                       profile_memory.get('main_character_ids',[]))
        state['world_background'] = world_background(bible,profile_memory,usable)
        with guard:
            state["active_job"] = next((dict(j) for j in jobs.values()
                                          if j["project"] == project and j["chapter"]==chapter and j["status"] == "running"), None)
        return jsonify(state)

    @bp.get("/api/story/<project>/chapters/<int:chapter>/jobs/<job_id>")
    @bp.get("/api/story/<project>/jobs/<job_id>")
    def job_status(project, job_id, chapter=1):
        with guard:
            job = jobs.get(job_id)
            if not job or job["project"] != project or job["chapter"]!=chapter:
                return jsonify(error="任务不存在；若服务重启，请刷新查看已保存结果"), 404
            return jsonify(dict(job))

    @bp.post("/api/story/<project>/chapters/<int:chapter>/analyze")
    @bp.post("/api/story/<project>/analyze")
    def analyze(project, chapter=1):
        return start(project, "analyze", chapter=chapter)

    @bp.post('/api/story/<project>/chapters/<int:chapter>/integrate')
    @bp.post('/api/story/<project>/integrate')
    def integrate(project, chapter=1):
        return start(project, 'integrate', chapter=chapter)

    @bp.post("/api/story/<project>/chapters/<int:chapter>/generate")
    @bp.post("/api/story/<project>/generate")
    def generate(project, chapter=1):
        return start(project, "generate", chapter=chapter)

    @bp.post('/api/story/<project>/chapters/<int:chapter>/verify')
    @bp.post('/api/story/<project>/verify')
    def verify(project, chapter=1):
        return start(project, 'verify', chapter=chapter)

    @bp.post('/api/story/<project>/chapters/<int:chapter>/revise')
    @bp.post('/api/story/<project>/revise')
    def revise(project, chapter=1):
        return start(project, 'revise', chapter=chapter)

    @bp.post("/api/story/<project>/chapters/<int:chapter>/prepare")
    @bp.post("/api/story/<project>/prepare")
    def prepare(project, chapter=1):
        return start(project, "prepare", chapter=chapter)

    @bp.post("/api/story/<project>/chapters/<int:chapter>/confirm_panels")
    @bp.post("/api/story/<project>/confirm_panels")
    def confirm_panels(project, chapter=1):
        return start(project, "confirm_panels", chapter=chapter)

    @bp.get("/api/story/<project>/chapters/<int:chapter>/inputs/<int:index>")
    @bp.get("/api/story/<project>/inputs/<int:index>")
    def input_image(project, index, chapter=1):
        store = story_store(project,chapter)
        imported = local_inputs(project,chapter)
        if not imported or index < 0 or index >= len(imported["panels"]):
            return jsonify(error="输入图片不存在"), 404
        # Only registered images; never accept an arbitrary client-supplied path.
        image = imported["panels"][index]
        return send_file(store.evidence_path(store.snapshot(image["path"])), conditional=True)

    @bp.post("/api/story/<project>/chapters/<int:chapter>/segments/<segment_id>/<action>")
    @bp.post("/api/story/<project>/segments/<segment_id>/<action>")
    def update_segment(project, segment_id, action, chapter=1):
        if action not in {"edit", "approve", "reject", "regenerate", "check"}:
            raise StoryError("不支持的操作")
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            raise StoryError("请求必须是 JSON 对象")
        return start(project, action, segment_id,chapter)

    @bp.get("/api/story/<project>/chapters/<int:chapter>/evidence/<panel_id>")
    @bp.get("/api/story/<project>/evidence/<panel_id>")
    def evidence(project, panel_id, chapter=1):
        store = story_store(project,chapter)
        state = store.read(request.args.get("revision"))
        panel = next((p for p in (state.get("bible") or {}).get("panels", [])
                      if p["source_panel_id"] == panel_id), None)
        if not panel:
            return jsonify(error="分镜证据不存在"), 404
        return send_file(store.evidence_path(panel), conditional=True)

    @bp.get("/api/story/<project>/chapters/<int:chapter>/export")
    @bp.get("/api/story/<project>/export")
    def export(project, chapter=1):
        return jsonify(service(project,chapter).export())

    return bp
