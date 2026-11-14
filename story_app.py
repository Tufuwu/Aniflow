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
from pipeline.shot_api import create_shot_blueprint
from pipeline.story_bible import StoryConflict, StoryError, atomic_json


def import_chapter(project, source, confirmed_panels=False, chapter=1):
    source = Path(source).resolve()
    if not source.is_dir():
        raise StoryError("漫画文件夹不存在")
    files = sorted((p for p in source.iterdir() if p.suffix.lower() in
                    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}), key=_natural_sort_key)
    if not files or len(validate_images([str(p) for p in files])) != len(files):
        raise StoryError("目录为空或包含损坏图片")
    store = story_store(project,chapter)
    with store.lock:
        if store.read().get("bible"):
            raise StoryConflict("此话已有剧情分析，请使用新的章节号或独立草稿项目导入不同来源")
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
    app.register_blueprint(create_shot_blueprint())
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AniFlow 章节剧情、跨话资料与解说审核")
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
    run.add_argument("--max-events", type=int, default=None, help="兼容旧命令；不用于压缩剧情")
    run.add_argument('--candidate-count',type=int,choices=[2,3],default=2)
    integrate = commands.add_parser('integrate', help='本话信息整合与事件还原；复用已有分析，不写稿，会调用文字模型')
    integrate.add_argument('--project', required=True)
    candidates = commands.add_parser('candidates',help='比较、选定或严格核验候选；比较/核验会调用模型')
    candidates.add_argument('--project',required=True)
    candidates.add_argument('--action',choices=['compare','select','verify_selected'],required=True)
    candidates.add_argument('--candidate-id')
    revise = commands.add_parser('revise',help='模型复核当前稿件，只返修问题段落并核验改动；会调用API')
    revise.add_argument('--project',required=True)
    memory = commands.add_parser('memory',help='重新生成并核对本话摘要和人物更新；不改正文，会调用模型')
    memory.add_argument('--project',required=True)
    context = commands.add_parser('context',help='保存生成前的背景和主要人物参考资料；不调用模型')
    context.add_argument('--project',required=True)
    context.add_argument('--background-file',type=Path,required=True)
    context.add_argument('--characters-file',type=Path,required=True)
    context.add_argument('--revision',default=None,help='已有资料的当前版本号，防止覆盖并发修改')
    for command in (load,run,integrate,revise,memory,candidates):
        command.add_argument('--chapter',type=int,default=1)
    preview = commands.add_parser('preview',help='用现有草稿生成待审核样片，不改变人工审核状态')
    preview.add_argument('--project',required=True)
    preview.add_argument('--output',type=Path)
    preview.add_argument('--rate',default='+25%')
    preview.add_argument('--offline',action='store_true',help='只使用已缓存音频；不调用模型和配音API')
    args = parser.parse_args()
    if args.command == "import":
        imported = import_chapter(args.project, args.source, args.panels,args.chapter)
        print(f"已只读导入 {len(imported['panels'])} 张图片，类型：{imported['input_kind']}")
    elif args.command == "run":
        from pipeline.story_api import local_inputs
        from pipeline.story_model import StoryModel
        from pipeline.story_pipeline import StoryPipeline
        store = story_store(args.project,args.chapter)
        imported = local_inputs(args.project,args.chapter)
        if not imported or imported["input_kind"] == "raw_slices":
            raise StoryError("请先在审核页面完成分镜准备")
        pipeline = StoryPipeline(store, StoryModel(store))
        report = lambda current, total, message: print(f"{current}/{total} {message}", flush=True)
        state = store.read()
        if not state["bible"]:
            state = pipeline.analyze(imported["panels"], state["revision"], report)
        state = pipeline.generate(state["revision"], args.style, args.max_events, report,args.candidate_count)
        print(f"候选已保存，请比较并选定后严格核验：{state['revision']}")
    elif args.command == 'integrate':
        from pipeline.story_model import StoryModel
        from pipeline.story_pipeline import StoryPipeline
        store = story_store(args.project, args.chapter)
        state = StoryPipeline(store, StoryModel(store)).integrate(store.read()['revision'])
        print(f"本话事件整合已保存，未生成讲稿：{state['revision']}")
    elif args.command == 'candidates':
        from pipeline.story_model import StoryModel
        from pipeline.story_pipeline import StoryPipeline
        store = story_store(args.project,args.chapter)
        pipeline = StoryPipeline(store,StoryModel(store))
        method = getattr(pipeline,args.action)
        state = method(store.read()['revision'],args.candidate_id) if args.action=='select' else method(store.read()['revision'])
        print(f"候选操作已保存：{state['revision']}；选定不等于核验或人工通过")
    elif args.command == 'revise':
        from pipeline.story_model import StoryModel
        from pipeline.story_pipeline import StoryPipeline
        store = story_store(args.project,args.chapter)
        state = StoryPipeline(store,StoryModel(store)).revise(store.read()['revision'])
        print(f"定向返修已保存，等待人工审核：{state['revision']}")
    elif args.command == 'memory':
        from pipeline.story_model import StoryModel
        from pipeline.story_pipeline import StoryPipeline
        store = story_store(args.project,args.chapter)
        state = StoryPipeline(store,StoryModel(store)).remember(store.read()['revision'])
        print(f"本话摘要已保存：{state['chapter_memory']['status']}；正文和人工审核状态保持原样")
    elif args.command == 'context':
        from pipeline.chapter_memory import save_series_notes
        result = save_series_notes(story_store(args.project),args.background_file.read_text(encoding='utf-8'),
                                   args.characters_file.read_text(encoding='utf-8'),args.revision)
        print(f"生成前资料已保存：{result['revision']}")
    elif args.command == 'preview':
        from pipeline.preview_pipeline import preview_chapter
        store = story_store(args.project)
        result = preview_chapter(store,args.output or store.root/'previews',args.rate,args.offline,
                                 lambda n,total,msg:print(f'{n}/{total} {msg}',flush=True))
        print(result['video'])
    else:
        create_app().run(host="127.0.0.1", port=args.port, threaded=True)
