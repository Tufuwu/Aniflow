"""Read-only character inventory. Never instantiate a transport or return prose.

The capture raises BaseException at ask(), before any model, cache or usage
write. An optional historical digest is a labelled snapshot, not a new result.
"""
import argparse
import json
from pathlib import Path

from pipeline.fact_checker import FactChecker
from pipeline.prompt_material import current_fact_sheet
from pipeline.script_writer import ScriptWriter
from pipeline.story_bible import StoryBible
from pipeline.story_model import prompt_json, WRITER_SYSTEM, SYSTEM, system_for_stage
from pipeline.input_compaction import compact_json


class Captured(BaseException):
    pass


class Capture:
    def ask(self, stage, prompt, images=None, fresh=False):
        self.request = (stage, prompt, images or [])
        raise Captured()


def json_after(prompt, marker):
    return json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])[0]


def inventory(request):
    stage, prompt, images = request
    if stage == 'write_story':
        data = json_after(prompt, '叙事依据：')
        extras = {'output_skeleton':json_after(prompt, '必须严格填写的输出结构：')}
        if '\n本候选的表达任务' in prompt:
            # Accept historical captures as well as the current writing policy.
            label = prompt.split('\n本候选的表达任务', 1)[1].split('：', 1)[0]
            extras['candidate_direction'] = json_after(prompt, '\n本候选的表达任务'+label+'：')
        system = WRITER_SYSTEM
    elif stage in {'script_polish','script_polish_analysis','script_review'}:
        data = json_after(prompt, '资料：')
        extras = ({'repeat_candidates':json_after(prompt, '\nrepeat_candidates：')}
                  if '\nrepeat_candidates：' in prompt else {})
        system = system_for_stage(stage)
    else:
        labels = {'characters':'人物：', 'prior_context':'生成前资料：',
                  'identity_availability':'identity_availability：', 'relationships':'原文已确认的关系：',
                  'events':'\n事件：', 'causal_events':'允许的因果前提：',
                  'continuity_panel_ids':'前文连续性参考图（仅核验，不增加本段事实/展示范围）：',
                  'image_panel_ids':'附图顺序的来源ID：', 'sentences':'待核对句子：'}
        data = {key:json_after(prompt,label) for key,label in labels.items()}
        extras = {}; system = SYSTEM
    serializer = compact_json if stage in {'write_story','script_polish','script_review'} else prompt_json
    fields = {key:len(serializer(value)) for key,value in data.items()}
    fields.update({key:len(prompt_json(value)) for key,value in extras.items()})
    # Same explicit source labels as StoryModel.ask for these verification
    # images. Image bytes/base64 and transport JSON are not text/token counts.
    labels = [f'来源 source_panel_id={p.source_panel_id}。返回这个来源ID，不能用附图序号代替。'
              if hasattr(p,'source_panel_id') else f'附图 {i+1}，对应提示中来源列表的第{i+1}项'
              for i,p in enumerate(images)]
    label_chars = sum(map(len,labels))
    return {'stage':stage, 'prompt_chars':len(prompt), 'system_chars':len(system),
            'image_source_label_chars':label_chars,
            'text_total_chars':len(prompt)+len(system)+label_chars, 'components_chars':fields,
            'instructions_keys_and_layout_chars':len(prompt)-sum(fields.values()),
            'attached_images':len(images)}


def capture(model, operation):
    try:
        operation()
    except Captured:
        return inventory(model.request)
    raise ValueError('Expected to stop before a model request')


def audit(root, historical_digest=None):
    root = Path(root).resolve()
    # Bypass __init__: read/evidence_path only; no mkdir, locks, cache writes.
    store = object.__new__(StoryBible); store.root = root
    state = store.read()
    bible, plan, script = state['bible'], state['plan'], state['script']
    facts = current_fact_sheet(bible,plan,script)
    basis = 'current_plan_bound_fact_sheet' if facts is not None else 'program_sources'
    if facts is None and historical_digest:
        record = json.loads(Path(historical_digest).read_text(encoding='utf-8'))
        require_plan = json_after(record['request']['prompt'], '资料：')['plan']
        if require_plan != plan:
            raise ValueError('Historical digest belongs to a different plan')
        facts = record['result']['nodes']
        basis = 'historical_same_plan_digest_snapshot; not regenerated or validated under new prompts'
    model = Capture(); writer = ScriptWriter(model)
    result = {'unit':'Unicode characters, not tokens; images counted separately',
              'source_revision':state['revision'], 'fact_sheet_basis':basis,
              'model_calls':0, 'tts_calls':0, 'project_writes':0, 'fourth_stage':'performance_script_rewriting'}
    result['writing'] = capture(model,lambda:writer.write_story(bible,plan,facts=facts))
    result['whole_review'] = capture(model,lambda:writer.polish_story(bible,plan,script['segments'],facts=facts))
    if basis.startswith('historical_'):
        result['legacy_review_without_fact_sheet'] = capture(model,
            lambda:writer.polish_story(bible,plan,script['segments']))
    checker = FactChecker(store,model)
    rows = []
    for segment in script['segments']:
        row = capture(model,lambda:checker.check(bible,segment))
        rows.append({'event_id':segment['event_id'], **row})
    result['original_image_verification'] = {'calls_if_all_checked':len(rows), 'per_node':rows,
        'prompt_chars_sum':sum(r['prompt_chars'] for r in rows),
        'system_chars_sum':sum(r['system_chars'] for r in rows),
        'image_source_label_chars_sum':sum(r['image_source_label_chars'] for r in rows),
        'text_total_chars_sum':sum(r['text_total_chars'] for r in rows),
        'prompt_chars_min':min(r['prompt_chars'] for r in rows),
        'prompt_chars_max':max(r['prompt_chars'] for r in rows),
        'attached_images_sum_with_repeated_use':sum(r['attached_images'] for r in rows)}
    result['limits'] = ('Read-only capture only; no digest, prose, polishing output, verification result or video was generated. '
                        'Unresolved legacy reference units remain labelled; no prediction of new model size, cost or quality.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('chapter',type=Path)
    parser.add_argument('--historical-digest',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.chapter,args.historical_digest)
    output = args.output.resolve()
    if output.is_relative_to(args.chapter.resolve()):
        raise ValueError('Audit output must be outside the original chapter')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('writing','whole_review')},ensure_ascii=False,indent=2))
    print({k:v for k,v in result['original_image_verification'].items() if k != 'per_node'})
