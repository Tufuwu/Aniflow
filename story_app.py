"""Lightweight first-chapter importer/reviewer (no torch, TTS or FFmpeg needed).

python story_app.py import --project first-chapter --source <folder>
python story_app.py import --project first-chapter --source <panel-folder> --panels
python story_app.py serve
"""
import argparse
from pathlib import Path

from flask import Flask
from pipeline.downloader import _natural_sort_key, validate_images
from pipeline.story_api import create_story_blueprint, story_store
from pipeline.story_bible import StoryConflict, StoryError, atomic_json


def import_chapter(project, source, confirmed_panels=False):
    source = Path(source).resolve()
    if not source.is_dir():
        raise StoryError("漫画文件夹不存在")
    files = sorted((p for p in source.iterdir() if p.suffix.lower() in
                    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}), key=_natural_sort_key)
    if not files or len(validate_images([str(p) for p in files])) != len(files):
        raise StoryError("目录为空或包含损坏图片")
    store = story_store(project)
    with store.lock:
        if store.read().get("bible"):
            raise StoryConflict("此项目已有剧情分析，请用新的项目名导入不同来源")
        # Snapshot before any processing; the source folder is never written to.
        panels = [{"panel_id": str(i + 1), "path": str(store.evidence_path(store.snapshot(p))),
                   "source_page": str(p)} for i, p in enumerate(files)]
        result = {"input_kind": "panels" if confirmed_panels else "raw_slices",
                  "source_folder": str(source), "panels": panels}
        atomic_json(store.root / "inputs.json", result)
        return result


def create_app():
    app = Flask(__name__)
    app.register_blueprint(create_story_blueprint())
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AniFlow 第一话剧情与解说审核")
    commands = parser.add_subparsers(dest="command", required=True)
    load = commands.add_parser("import")
    load.add_argument("--project", required=True)
    load.add_argument("--source", required=True)
    load.add_argument("--panels", action="store_true", help="确认输入已经是阅读顺序排列的完整分镜")
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8081)
    run = commands.add_parser("run", help="分析并生成待人工审核草稿；不生成音视频")
    run.add_argument("--project", required=True)
    run.add_argument("--style", choices=["faithful", "payoff", "suspense"], default="faithful")
    run.add_argument("--max-events", type=int, default=12)
    args = parser.parse_args()
    if args.command == "import":
        imported = import_chapter(args.project, args.source, args.panels)
        print(f"已只读导入 {len(imported['panels'])} 张图片，类型：{imported['input_kind']}")
    elif args.command == "run":
        from pipeline.story_api import local_inputs
        from pipeline.story_model import StoryModel
        from pipeline.story_pipeline import StoryPipeline
        store = story_store(args.project)
        imported = local_inputs(args.project)
        if not imported or imported["input_kind"] == "raw_slices":
            raise StoryError("请先在审核页面完成分镜准备")
        pipeline = StoryPipeline(store, StoryModel(store))
        report = lambda current, total, message: print(f"{current}/{total} {message}", flush=True)
        state = store.read()
        if not state["bible"]:
            state = pipeline.analyze(imported["panels"], state["revision"], report)
        state = pipeline.generate(state["revision"], args.style, args.max_events, report)
        print(f"草稿已保存，等待人工审核：{state['revision']}")
    else:
        create_app().run(host="127.0.0.1", port=args.port, threaded=True)
