"""Draft previews use the production components without inventing human approval."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from uuid import uuid4
import config

from pipeline.story_bible import atomic_json,digest
from pipeline.story_model import StoryModel,require
from pipeline.shot_selector import build_library,ShotSelectorAgent,GlobalShotOptimizer,coverage_report
from pipeline.timeline_director import TimelineDirector,audio_duration,validate_timeline
from pipeline.timeline_compositor import render_timeline
from pipeline.semantic_grounding import attach_tts_boundaries
from pipeline.tts_profiles import voice_cache_options


def preview_chapter(store, output_root, rate=None, offline=False, progress=None, model=None, speech=None):
    """Offline mode permits only existing verified audio and metadata selection."""
    source = store.read()
    segments = (source.get('script') or {}).get('segments',[])
    require(segments and len(segments)==len(source['plan']['beats']), '请先完成整稿生成')
    root = Path(output_root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    opts = {'engine':'edge','voice':config.TTS_VOICE,'rate':rate or config.TTS_RATE,'pitch':config.TTS_PITCH}
    cache_options = voice_cache_options(opts)
    # Validate before constructing a model or synthesizer.
    import re
    require(isinstance(opts['rate'],str) and re.fullmatch(r'[+-]\d{1,3}%',opts['rate']), '语速格式无效')
    model = None if offline else model or StoryModel(store)
    library = build_library(store,source)
    selector = ShotSelectorAgent(model)
    groups = GlobalShotOptimizer().optimize(source,library,selector)
    visual_review = selector.review_selected(source,library,groups)
    metrics = coverage_report(groups,library)
    require(not metrics['missing_by_narration'], '预览缺少剧情画面')
    audio = {}
    for i,g in enumerate(groups):
        key = digest({'text':g['text'],'options':cache_options})
        path = root/'audio'/f'{key}.mp3'
        cache = path.with_suffix('.json')
        if path.exists() and cache.exists():
            info = json.loads(cache.read_text('utf-8'))
            require(hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256'], '音频缓存校验失败')
        else:
            require(not offline, '离线模式缺少音频缓存；未调用任何配音API')
            if speech is None:
                from pipeline.tts_engine import generate_speech
                speech = generate_speech
            path.parent.mkdir(exist_ok=True)
            result = speech(g['text'],str(path),**opts)
            info = {'path':str(path),'duration':audio_duration(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            info = attach_tts_boundaries(info, path, g['text'], result)
            atomic_json(cache,info)
        # Measure the cached bytes again, never trust duration metadata alone.
        info = {**info,'path':str(path),'duration':audio_duration(path)}
        info = attach_tts_boundaries(info, path, g['text'])
        audio[g['narration_id']] = info
        if progress:
            progress(i+1,len(groups),'已准备配音')
    timeline = TimelineDirector().build(groups,audio,library)
    sync = validate_timeline(timeline)
    directory = root/'exports'/uuid4().hex
    directory.parent.mkdir(exist_ok=True)
    report = render_timeline(store,source,timeline,directory,progress)
    manifest = {'kind':'unreviewed_story_preview','human_approved':False,'offline':offline,
                'source_revision':source['revision'],'models_used':not offline,
                'fact_checks':dict(Counter((s.get('fact_check') or {}).get('status','not_run') for s in segments)),
                'narrative_review':source['script'].get('narrative_review'),
                'visual_review':visual_review,
                'timeline':timeline,'groups':groups,'metrics':metrics,'sync':sync,'report':report,'tts':cache_options}
    require(store.read()['revision']==source['revision'], '来源已改变；本预览未发布')
    atomic_json(directory/'manifest.json',manifest)
    (directory/'narration.txt').write_text('\n\n'.join(s['text'] for s in segments)+'\n',encoding='utf-8')
    atomic_json(root/'latest.json',{'directory':str(directory),'video':str(directory/'video.mp4')})
    return {**manifest,'directory':str(directory),'video':str(directory/'video.mp4')}
