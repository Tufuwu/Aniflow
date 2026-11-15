"""Explicit, resumable live validation. Never imported by the offline test suite.

Run from the repository root. Uses real configured models; no audio/video or
human approval actions. Each step saves diagnostics in its isolated project.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from pipeline.chapter_memory import generation_context, memory_usable, save_series_notes, context_current
from pipeline.story_api import story_store, local_inputs
from pipeline.story_bible import atomic_json
from pipeline.story_inputs import prepare_panels
from pipeline.story_model import StoryModel, prompt_json
from pipeline.story_pipeline import StoryPipeline
from pipeline.narration_review import review_ready
from pipeline.semantic_grounding import grounded_units
from story_app import import_chapter


class ObservedModel(StoryModel):
    def ask(self,stage,prompt,images=None,fresh=False):
        started=time.monotonic()
        print(f'MODEL {stage} start images={len(images or [])} fresh={fresh}',flush=True)
        try:
            result=super().ask(stage,prompt,images,fresh)
        except Exception as exc:
            print(f'MODEL {stage} failed {time.monotonic()-started:.1f}s {exc}',flush=True)
            raise
        print(f'MODEL {stage} done {time.monotonic()-started:.1f}s',flush=True)
        return result


def progress(n,total,message):
    print(f'{n}/{total} {message}',flush=True)


def export_chapter(store):
    state=store.read()
    if state.get('script'):
        (store.root/'narration.txt').write_text('\n\n'.join(s['text'] for s in state['script']['segments']),encoding='utf-8')
        atomic_json(store.root/'memory.json',state.get('chapter_memory'))
        atomic_json(store.root/'validation_manifest.json',state)


def report(root):
    chapters=[]
    for path in sorted(root.glob('Chapter_[0-9][0-9][0-9][0-9][0-9]')):
        store=story_store(root.name,int(path.name.split('_')[1]));state=store.read()
        segments=(state.get('script') or {}).get('segments',[])
        bible=state.get('bible') or {}
        calls=[json.loads(p.read_text(encoding='utf-8')) for p in (path/'calls').glob('*.json')]
        memory=state.get('chapter_memory') or {}
        chapters.append({'chapter':int(path.name.split('_')[1]),'revision':state['revision'],
            'panels':sum(not p.get('historical') for p in bible.get('panels',[])),
            'historical_panels':sum(bool(p.get('historical')) for p in bible.get('panels',[])),
            'events':sum(not e.get('historical') for e in bible.get('events',[])),
            'characters':[{'character_id':c['character_id'],'name':c['name'],'name_status':c.get('name_status'),
                           'labels':[l['label'] for l in c.get('label_claims',[])]} for c in bible.get('characters',[])],
            'segments':len(segments),'characters_in_script':sum(len(s['text']) for s in segments),
            'fact_statuses':dict(Counter((s.get('fact_check') or {}).get('status','missing') for s in segments)),
            'grounded_segments':sum(bool(grounded_units(bible,s)) for s in segments),
            'narrative_review_ready':bool(segments and review_ready((state.get('script') or {}).get('narrative_review'),bible,state.get('plan'),segments)),
            'memory_status':memory.get('status'),'memory_usable':bool(memory_usable(state) and context_current(store,bible)),
            'memory_error':memory.get('error'),'prior_status':bible.get('prior_context',{}).get('status'),
            'human_statuses':dict(Counter(s['review_status'] for s in segments)),
            'request_records':len(calls),'request_errors':sum('error' in c for c in calls),
            'stages':dict(Counter(c['request']['stage'] for c in calls))})
        export_chapter(store)
    result={'project':root.name,'updated_at':datetime.now(timezone.utc).isoformat(),
            'chapters':chapters,'human_approval_actions':0,'audio_or_video_generated':False,
            'narration_authored_by_code_assistant':False}
    atomic_json(root/'live_validation.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','revise','memory','recheck','probe','report'])
    parser.add_argument('--project',required=True)
    parser.add_argument('--chapter',type=int,default=1)
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--prepared-first',type=Path)
    args=parser.parse_args()
    store=story_store(args.project,args.chapter);root=store.root.parent
    try:
        if args.action=='prepare':
            if (store.root/'inputs.json').exists():
                print('Existing prepared inputs retained.',flush=True);return
            if args.chapter==1 and args.prepared_first:
                imported=json.loads(args.prepared_first.read_text(encoding='utf-8'))
                for panel in imported['panels']:
                    snap=store.snapshot(panel.get('raw_path') or panel['path'])
                    panel['path']=panel['raw_path']=str(store.evidence_path(snap))
                atomic_json(store.root/'inputs.json',imported)
                # Reuse only source-linked analysis as unverified reference notes,
                # never an old narration. The new model still checks the images.
                old_root=args.prepared_first.parent
                from pipeline.story_bible import StoryBible
                old=StoryBible(old_root).read()
                reference={'source_revision':old['revision'],'status':'unverified_reference',
                           'events':[e for e in old['bible']['events'] if any(str(pid) in
                               {str(p) for p in range(23,32)} for pid in e['source_panel_ids'])]}
                save_series_notes(store,prompt_json(reference),prompt_json({'status':'unverified_reference',
                    'characters':old['bible']['characters']}),None,input_role='analysis_reference')
            else:
                sources=sorted(p for p in args.source_root.iterdir() if p.name.startswith(f'{args.chapter:04d}_'))
                if len(sources)!=1:raise ValueError('Expected exactly one source chapter directory')
                import_chapter(args.project,sources[0],False,args.chapter)
                prepare_panels(store,progress)
        elif args.action in {'run','revise','memory','recheck'}:
            model=ObservedModel(store);pipeline=StoryPipeline(store,model);state=store.read()
            if args.action=='run':
                if not state.get('bible'):
                    state=pipeline.analyze(local_inputs(args.project,args.chapter)['panels'],state['revision'],progress)
                if not (state.get('script') or {}).get('segments'):
                    pipeline.generate(state['revision'],progress=progress)
                else:
                    print('Existing draft retained; use revise/memory explicitly.',flush=True)
            elif args.action=='revise':pipeline.revise(state['revision'])
            elif args.action=='recheck':
                for segment in state['script']['segments']:
                    if (segment.get('fact_check') or {}).get('status')!='passed':
                        state=pipeline.segment_action(segment['segment_id'],'check',state['revision'])
            else:pipeline.remember(state['revision'])
        elif args.action=='probe':
            context=generation_context(store)
            if context['status']!='available':raise ValueError('Previous chapter memory is not usable')
            brief={k:v for k,v in context.items() if k not in {'panels','events','characters','relationships'}}
            prompt='''Continuity Input Probe v1。只核对收到的下一话输入，不创作下一话剧情。
返回 {"previous_chapter":数字,"earlier_chapter_evidence":[{"chapter":数字,"event_ids":["原样引用收到的历史事件ID"],"text":"逐字引用累计背景或人物事实"}],
"main_character_ids":["原样引用"],"unresolved":["已知未完成问题"]}。
不能补造未来情节，不能把参考资料当作确认事实。资料：'''+prompt_json(brief)
            raw=ObservedModel(store).ask('continuity_probe',prompt)
            atomic_json(store.root/'continuity_input_probe.json',{'context':brief,'response':raw})
    except Exception as exc:
        atomic_json(root/f'failure-chapter-{args.chapter}-{args.action}-{time.time_ns()}.json',
                    {'action':args.action,'chapter':args.chapter,'error':str(exc)})
        raise
    finally:
        report(root)


if __name__=='__main__':main()
