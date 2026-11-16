"""Read-only director request and integration checkpoint audit; never calls an API."""
import argparse
import json
import socket
from pathlib import Path
from unittest.mock import patch

from pipeline import event_integration as integration
from pipeline.chapter_memory import planning_settings
from pipeline.narrative_director import NarrativeDirector, PLANNING_CONTRACT_VERSION
from pipeline.planning_material import planning_material
from pipeline.story_bible import StoryBible, digest
from pipeline.story_model import PLAN_SYSTEM, StoryModel


class RequestCaptured(BaseException):
    """Stop before response parsing, retry handling or any transport work."""


class Capture:
    planning_version = 5

    def ask(self, stage, prompt, images=None, fresh=False):
        if stage != 'direct' or images or fresh:
            raise AssertionError('Audit expects only the initial text-only direct request')
        self.request = {'stage': stage, 'system': PLAN_SYSTEM, 'prompt': prompt, 'images': []}
        raise RequestCaptured


def configured_model(store):
    # Construction performs no HTTP or store writes and reflects stage-specific routes.
    # audit() also blocks ask/save/network below. Cache identities never contain keys.
    return StoryModel(store)


def integration_checkpoint(store, bible):
    saved = bible.get('event_integration')
    if not saved:
        return {'available': False, 'reason': 'No saved integration; audit will not generate one'}
    analysis = integration.chapter_input(bible, saved['source_revision'])
    fragments = integration.fragments_for(analysis)
    prompt = integration.integration_prompt(analysis, fragments)
    model = configured_model(store)
    dependencies = {'input': analysis, 'fragments': fragments, 'prompt': prompt,
                    'output_contract': integration.OUTPUT_CONTRACT,
                    'input_view_version': integration.INPUT_VIEW_VERSION,
                    'repair_prompt': integration.field_repair.REPAIR_INSTRUCTIONS,
                    'repair_version': integration.field_repair.REPAIR_VERSION}
    key = digest({'validated_version': 1, 'stage': 'event_integration',
                  'model': model.cache_identity('event_integration'), 'dependencies': dependencies})
    path = store.root/'cache'/f'valid-{key}.json'
    record = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    valid = bool(record) and record.get('result_hash') == digest(record.get('result'))
    return {'available': True, 'source_revision': saved['source_revision'],
            'input_matches_saved': saved['input_hash'] == digest(analysis),
            'prompt_matches_saved': saved['prompt_hash'] == digest(prompt),
            'output_contract_matches_saved': saved['output_contract_hash'] == digest(integration.OUTPUT_CONTRACT),
            'model_identity_matches_saved': saved['model'] == model.cache_identity('event_integration'),
            'checkpoint_key': key, 'validated_cache_exists_and_hash_valid': valid,
            'cached_events_match_saved': valid and record['result']['events'] == [
                e for e in bible['events'] if not e.get('historical')]}


def audit(root, existing_plan=None):
    def deny(*args, **kwargs):
        raise AssertionError('Read-only planning audit cannot call real APIs or save project data')
    with patch.object(socket.socket, 'connect', deny), patch.object(socket, 'create_connection', deny), \
            patch.object(StoryModel, 'ask', deny), patch.object(StoryBible, 'save', deny):
        store = object.__new__(StoryBible)  # No directory or lock creation.
        store.root = Path(root).resolve()
        state = store.read()
        bible = state['bible']
        settings = planning_settings(store)
        model = Capture()
        try:
            NarrativeDirector(model, settings).plan(bible)
        except RequestCaptured:
            pass
        material = planning_material(bible, settings)
        old = state.get('plan') or {}
        if existing_plan is not None:
            old = json.loads(Path(existing_plan).read_text(encoding='utf-8'))
            if old.get('bible_id') != bible['bible_id']:
                raise ValueError('Existing plan must belong to this exact stored integration')
        old_ids = [eid for b in old.get('beats', []) for eid in (b.get('event_ids') or [b['event_id']])]
        request_events = json.JSONDecoder().raw_decode(
            model.request['prompt'].split('整话证据事件（按原作顺序）：', 1)[1])[0]
        report = {'source_revision': state['revision'], 'bible_id': bible['bible_id'],
                  'settings_revision': settings['revision'] if settings else None,
                  'planning_contract': PLANNING_CONTRACT_VERSION,
                  'prompt_hash': digest(model.request['prompt']), 'system_hash': digest(PLAN_SYSTEM),
                  'prompt_chars': len(model.request['prompt']), 'system_chars': len(PLAN_SYSTEM),
                  'events': len(material['events']), 'facets': sum(len(e.get('facets', [])) for e in material['events']),
                  'dialogue': sum(len(e.get('dialogue', [])) for e in material['events']),
                  'stored_integration_reused_for_request': bool(bible.get('event_integration')),
                  'request_events_match_saved_planning_view': request_events == material['events'],
                  'existing_plan': {'source': str(existing_plan) if existing_plan else 'chapter_state',
                                    'beats': len(old.get('beats', [])), 'selected_events': len(old_ids),
                                    'essential_events': len(old.get('essential_event_ids', [])), 'hash': digest(old)},
                  'first_stage_checkpoint_comparison': integration_checkpoint(store, bible),
                  'captured_stage': model.request['stage'], 'attached_images': 0,
                  'model_calls': 0, 'first_stage_reruns': 0, 'project_writes': 0,
                  'limits': 'Request and stored checkpoint inspection only; no new plan or model quality assessment.'}
        return {'report': report, 'material': material, 'request': model.request}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('chapter', type=Path)
    parser.add_argument('--existing-plan', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.is_relative_to(args.chapter.resolve().parent):
        raise ValueError('Audit exports must stay outside the original chapter and series settings')
    result = audit(args.chapter, args.existing_plan)
    out.mkdir(parents=True, exist_ok=False)
    for name in ('report', 'material', 'request'):
        (out/f'{name}.json').write_text(json.dumps(result[name], ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'system.txt').write_text(result['request']['system'], encoding='utf-8')
    (out/'prompt.txt').write_text(result['request']['prompt'], encoding='utf-8')
    print(json.dumps(result['report'], ensure_ascii=False, indent=2))
