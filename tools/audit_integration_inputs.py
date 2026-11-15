"""Read-only first-stage input audit. No transport, model, cache or chapter writes."""
import argparse
import json
from collections import Counter
from pathlib import Path

from pipeline.event_integration import chapter_input, fragments_for, integration_prompt
from pipeline.integration_material import model_material, compact_json
from pipeline.story_bible import StoryBible, digest
from pipeline.story_model import configured_integration_output_policy


def audit(root, baseline_trace=None):
    root = Path(root).resolve()
    store = object.__new__(StoryBible)
    store.root = root
    state = store.read()
    bible = state['bible']
    source = (bible.get('event_integration') or {}).get('source_revision') or state.get('analysis_revision') or state['revision']
    analysis = chapter_input(bible, source)
    fragments = fragments_for(analysis)
    material = model_material(analysis, fragments)
    prompt = integration_prompt(analysis, fragments)
    assert [r[0] for r in material['fragments']] == [f['fragment_id'] for f in fragments]
    report = {'source_revision': source, 'input_view_version': material['input_view_version'],
        'raw_input_hash': digest(analysis), 'prompt_hash': digest(prompt),
        'unit': 'Unicode characters; not estimated tokens or fees',
        'panels': len(analysis['panels']), 'events': len(analysis['events']), 'fragments': len(fragments),
        'fragment_kinds': dict(Counter(r[1] for r in material['fragments'])),
        'new_prompt_chars': len(prompt), 'new_prompt_utf8_bytes': len(prompt.encode('utf-8')),
        'new_component_chars': {k: len(compact_json(v)) for k, v in material.items()},
        'new_prompt_tokens': None, 'model_calls': 0, 'tts_calls': 0, 'project_writes': 0,
        'output_request_capacity': configured_integration_output_policy()['max_tokens'],
        'narrative_length_quota': None, 'output_policy': configured_integration_output_policy(),
        'limits': 'Preserves recorded semantic information; does not prove source image extraction or model integration quality.'}
    if baseline_trace:
        baseline = json.loads(Path(baseline_trace).read_text(encoding='utf-8'))
        request = baseline['request']
        if request.get('stage') != 'event_integration':
            raise ValueError('Baseline is not an event integration request')
        data = json.JSONDecoder().raw_decode(request['prompt'].split('资料：', 1)[1])[0]
        if data.get('analysis') != analysis:
            raise ValueError('Baseline does not contain the exact same complete chapter analysis and source revision')
        old_chars = len(request['prompt'])
        report.update(baseline_trace=str(Path(baseline_trace).resolve()), baseline_prompt_chars=old_chars,
            baseline_provider_reported_usage=baseline.get('response_metadata', {}).get('usage'),
            prompt_chars_removed=old_chars-len(prompt),
            prompt_chars_reduction_percent=round((old_chars-len(prompt))*100/old_chars, 2))
    return {'report': report, 'material': material, 'prompt': prompt}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('chapter', type=Path)
    parser.add_argument('--baseline-trace', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.is_relative_to(args.chapter.resolve()):
        raise ValueError('Audit exports must stay outside the original chapter')
    result = audit(args.chapter, args.baseline_trace)
    out.mkdir(parents=True, exist_ok=True)
    (out/'input-audit.json').write_text(json.dumps(result['report'], ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'model-input.json').write_text(json.dumps(result['material'], ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'prompt.txt').write_text(result['prompt'], encoding='utf-8')
    print(json.dumps(result['report'], ensure_ascii=True))
